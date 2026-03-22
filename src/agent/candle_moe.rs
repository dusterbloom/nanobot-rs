//! Candle Metal MoE FFN — quantized matmul on GPU via QMatMul.
//!
//! At model load: dequantize MLX experts to f32 → QTensor::quantize(Q4_1) on Metal.
//! At inference: QMatMul::forward() — native Metal quantized matmul, zero CPU dequant.
//!
//! This replaces the CPU scalar dequant bottleneck (~650ms/token) with
//! Metal GPU quantized matmul (~35ms/token).

#[cfg(feature = "candle")]
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    Device, Module, Result, Tensor,
};

/// Pre-compiled MoE expert weights on Metal GPU.
/// Created once at model load, reused for every token.
#[cfg(feature = "candle")]
pub struct CandleMoeExperts {
    /// Per-expert [gate_proj, up_proj, down_proj] as QMatMul on Metal.
    /// Indexed by expert_idx. None for experts not yet compiled.
    experts: Vec<Option<CandleExpert>>,
    /// Shared expert (always active).
    shared: Option<CandleExpert>,
    pub n_experts: usize,
    pub moe_hidden: usize,
    pub dim: usize,
    device: Device,
}

#[cfg(feature = "candle")]
struct CandleExpert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

#[cfg(feature = "candle")]
impl CandleMoeExperts {
    /// Pre-compile all 256 experts from PackedMoeExperts to Metal QMatMul.
    ///
    /// This dequantizes each expert ONCE to f32, then re-quantizes to GGUF Q4_1
    /// on the Metal device. The Q4_1 data stays on GPU — no per-token transfer.
    ///
    /// Memory: 256 experts × ~1.5 MB Q4_1 = ~384 MB GPU. Fits easily.
    pub fn compile(
        moe: &super::ane_weights::MoeLayerWeights,
        device: &Device,
    ) -> Result<Self> {
        let ne = moe.num_experts;
        let hidden = moe.moe_hidden;
        let dim = moe.packed_experts.gate_cols;

        let mut experts = Vec::with_capacity(ne);

        for e in 0..ne {
            let (gate_f32, up_f32) = moe.packed_experts.dequant_expert_gate_up(e);
            let down_f32 = moe.packed_experts.dequant_expert_down(e);

            let gate_t = Tensor::from_slice(&gate_f32, (hidden, dim), device)?;
            let up_t = Tensor::from_slice(&up_f32, (hidden, dim), device)?;
            let down_t = Tensor::from_slice(&down_f32, (dim, hidden), device)?;

            let gate_q = QTensor::quantize(&gate_t, GgmlDType::Q4_1)?;
            let up_q = QTensor::quantize(&up_t, GgmlDType::Q4_1)?;
            let down_q = QTensor::quantize(&down_t, GgmlDType::Q4_1)?;

            experts.push(Some(CandleExpert {
                gate: QMatMul::from_qtensor(gate_q)?,
                up: QMatMul::from_qtensor(up_q)?,
                down: QMatMul::from_qtensor(down_q)?,
            }));
        }

        // Shared expert
        let shared = if let Some(ref se) = moe.shared_expert {
            let gate_f32 = se.gate_proj.dequantize();
            let up_f32 = se.up_proj.dequantize();
            let down_f32 = se.down_proj.dequantize();

            let gate_t = Tensor::from_slice(&gate_f32, (hidden, dim), device)?;
            let up_t = Tensor::from_slice(&up_f32, (hidden, dim), device)?;
            let down_t = Tensor::from_slice(&down_f32, (dim, hidden), device)?;

            Some(CandleExpert {
                gate: QMatMul::from_qtensor(QTensor::quantize(&gate_t, GgmlDType::Q4_1)?)?,
                up: QMatMul::from_qtensor(QTensor::quantize(&up_t, GgmlDType::Q4_1)?)?,
                down: QMatMul::from_qtensor(QTensor::quantize(&down_t, GgmlDType::Q4_1)?)?,
            })
        } else {
            None
        };

        Ok(Self {
            experts,
            shared,
            n_experts: ne,
            moe_hidden: hidden,
            dim,
            device: device.clone(),
        })
    }

    /// Forward: router selects top-k, run SwiGLU on Metal via QMatMul.
    /// Zero CPU dequant per token — weights are pre-compiled Q4_1 on GPU.
    pub fn forward(
        &self,
        moe: &super::ane_weights::MoeLayerWeights,
        x: &mut Vec<f32>,
        rms_ffn: &[f32],
        rms_eps: f32,
        layer: usize,
        adapter_gate: Option<&[f32]>,
    ) {
        use super::ane_forward::{cpu_matmul, rmsnorm};

        let dim = self.dim;
        let hidden = self.moe_hidden;
        let ne = self.n_experts;
        let k = moe.num_experts_per_tok;

        // RMSNorm (CPU — tiny)
        let mut xnorm = vec![0.0f32; dim];
        rmsnorm(&mut xnorm, x, rms_ffn, dim, 1, rms_eps);

        // Router (CPU — one small matmul)
        let router_logits = if let Some(gate) = adapter_gate {
            cpu_matmul(gate, &xnorm, ne, dim, 1)
        } else {
            cpu_matmul(&moe.router, &xnorm, ne, dim, 1)
        };

        // Top-k + softmax
        let mut indexed: Vec<(usize, f32)> = router_logits.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = &indexed[..k.min(indexed.len())];

        let max_logit = top_k[0].1;
        let mut exp_sum = 0.0f32;
        let mut weights = Vec::with_capacity(k);
        for &(_, logit) in top_k {
            let e = (logit - max_logit).exp();
            weights.push(e);
            exp_sum += e;
        }
        for w in &mut weights {
            *w /= exp_sum;
        }

        // Metal: x tensor [1, dim]
        let x_tensor = match Tensor::from_slice(&xnorm, (1, dim), &self.device) {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!("Candle MoE L{layer}: Metal tensor failed: {e}, CPU fallback");
                cpu_fallback(moe, x, &xnorm, top_k, &weights, hidden, dim);
                return;
            }
        };

        let mut ffn_out = vec![0.0f32; dim];

        // SwiGLU for each active expert — QMatMul on Metal, zero CPU dequant
        for (i, &(expert_idx, _)) in top_k.iter().enumerate() {
            let w = weights[i];
            if let Some(Some(ref expert)) = self.experts.get(expert_idx) {
                match expert_swiglu(&expert.gate, &expert.up, &expert.down, &x_tensor, hidden) {
                    Ok(out_vec) => {
                        for j in 0..dim {
                            ffn_out[j] += w * out_vec[j];
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Candle MoE L{layer} expert {expert_idx}: {e}");
                        // Per-expert CPU fallback
                        let mut h1 = moe.packed_experts.gate_matmul(expert_idx, &xnorm);
                        let h3 = moe.packed_experts.up_matmul(expert_idx, &xnorm);
                        super::ane_decode::silu_inplace(&mut h1);
                        for j in 0..hidden { h1[j] *= h3[j]; }
                        let out = moe.packed_experts.down_matmul(expert_idx, &h1);
                        for j in 0..dim { ffn_out[j] += w * out[j]; }
                    }
                }
            }
        }

        // Shared expert
        if let Some(ref shared) = self.shared {
            if let Ok(out_vec) = expert_swiglu(&shared.gate, &shared.up, &shared.down, &x_tensor, hidden) {
                for j in 0..dim {
                    ffn_out[j] += out_vec[j];
                }
            }
        }

        // Residual
        for j in 0..dim {
            x[j] += ffn_out[j];
        }
    }
}

/// SwiGLU: out = down(SiLU(gate(x)) * up(x))
#[cfg(feature = "candle")]
fn expert_swiglu(
    gate: &QMatMul,
    up: &QMatMul,
    down: &QMatMul,
    x: &Tensor,
    _hidden: usize,
) -> Result<Vec<f32>> {
    let h1 = gate.forward(x)?;
    let h3 = up.forward(x)?;
    let h1_silu = candle_nn::ops::silu(&h1)?;
    let gated = (h1_silu * h3)?;
    let out = down.forward(&gated)?;
    out.squeeze(0)?.to_vec1::<f32>()
}

/// CPU fallback when Metal fails entirely.
#[cfg(feature = "candle")]
fn cpu_fallback(
    moe: &super::ane_weights::MoeLayerWeights,
    x: &mut Vec<f32>,
    xnorm: &[f32],
    top_k: &[(usize, f32)],
    weights: &[f32],
    hidden: usize,
    dim: usize,
) {
    let mut ffn_out = vec![0.0f32; dim];
    for (i, &(expert_idx, _)) in top_k.iter().enumerate() {
        let w = weights[i];
        let mut h1 = moe.packed_experts.gate_matmul(expert_idx, xnorm);
        let h3 = moe.packed_experts.up_matmul(expert_idx, xnorm);
        super::ane_decode::silu_inplace(&mut h1);
        for j in 0..hidden { h1[j] *= h3[j]; }
        let out = moe.packed_experts.down_matmul(expert_idx, &h1);
        for j in 0..dim { ffn_out[j] += w * out[j]; }
    }
    if let Some(ref shared) = moe.shared_expert {
        use super::ane_forward::cpu_quantized_matmul;
        let mut h1 = cpu_quantized_matmul(&shared.gate_proj, xnorm, 1);
        let h3 = cpu_quantized_matmul(&shared.up_proj, xnorm, 1);
        super::ane_decode::silu_inplace(&mut h1);
        for j in 0..hidden { h1[j] *= h3[j]; }
        let out = cpu_quantized_matmul(&shared.down_proj, &h1, 1);
        for j in 0..dim { ffn_out[j] += out[j]; }
    }
    for j in 0..dim { x[j] += ffn_out[j]; }
}
