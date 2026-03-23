//! Lazy MoE expert cache — keeps top-K experts on Metal, evicts LRU.
//!
//! Memory budget: 32 experts × 3 projections × ~1.5 MB × 40 layers ≈ 5.6 GB.
//! Cache hit: QMatMul on Metal (zero CPU dequant, ~1ms/expert).
//! Cache miss: dequant → QTensor::quantize(Q4_1) → cache + evict LRU (~5ms one-time).

#[cfg(feature = "candle")]
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    Device, Module, Result, Tensor,
};

#[cfg(feature = "candle")]
use std::collections::HashMap;

/// Cached expert on Metal GPU.
#[cfg(feature = "candle")]
struct CachedExpert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    last_used: u64, // token counter for LRU eviction
}

/// Per-layer lazy expert cache.
#[cfg(feature = "candle")]
pub struct LayerExpertCache {
    cache: HashMap<usize, CachedExpert>, // expert_idx → cached
    max_cached: usize,
    token_counter: u64,
    hidden: usize,
    dim: usize,
    device: Device,
}

#[cfg(feature = "candle")]
impl LayerExpertCache {
    fn new(max_cached: usize, hidden: usize, dim: usize, device: &Device) -> Self {
        Self {
            cache: HashMap::with_capacity(max_cached),
            max_cached,
            token_counter: 0,
            hidden,
            dim,
            device: device.clone(),
        }
    }

    /// Get or compile an expert. Returns the QMatMul projections.
    fn get_or_compile(
        &mut self,
        expert_idx: usize,
        packed: &super::ane_weights::PackedMoeExperts,
    ) -> Result<(&QMatMul, &QMatMul, &QMatMul)> {
        self.token_counter += 1;
        let counter = self.token_counter;

        if self.cache.contains_key(&expert_idx) {
            let entry = self.cache.get_mut(&expert_idx).unwrap();
            entry.last_used = counter;
            let entry = self.cache.get(&expert_idx).unwrap();
            return Ok((&entry.gate, &entry.up, &entry.down));
        }

        // Cache miss — compile expert
        if self.cache.len() >= self.max_cached {
            // Evict LRU
            let lru_key = self.cache.iter()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, _)| *k)
                .unwrap();
            self.cache.remove(&lru_key);
        }

        let (gate_f32, up_f32) = packed.dequant_expert_gate_up(expert_idx);
        let down_f32 = packed.dequant_expert_down(expert_idx);

        let gate_t = Tensor::from_slice(&gate_f32, (self.hidden, self.dim), &self.device)?;
        let up_t = Tensor::from_slice(&up_f32, (self.hidden, self.dim), &self.device)?;
        let down_t = Tensor::from_slice(&down_f32, (self.dim, self.hidden), &self.device)?;

        let entry = CachedExpert {
            gate: QMatMul::from_qtensor(QTensor::quantize(&gate_t, GgmlDType::Q4_1)?)?,
            up: QMatMul::from_qtensor(QTensor::quantize(&up_t, GgmlDType::Q4_1)?)?,
            down: QMatMul::from_qtensor(QTensor::quantize(&down_t, GgmlDType::Q4_1)?)?,
            last_used: counter,
        };

        self.cache.insert(expert_idx, entry);
        let entry = self.cache.get(&expert_idx).unwrap();
        Ok((&entry.gate, &entry.up, &entry.down))
    }

    /// Cache stats: (cached_count, hit_rate estimate)
    pub fn stats(&self) -> (usize, usize) {
        (self.cache.len(), self.max_cached)
    }
}

/// All layers' expert caches.
#[cfg(feature = "candle")]
pub struct MoeExpertCache {
    layers: Vec<Option<LayerExpertCache>>,
    shared_experts: Vec<Option<CachedExpert>>,
    device: Device,
}

#[cfg(feature = "candle")]
impl MoeExpertCache {
    /// Create caches for all MoE layers. Non-MoE layers get None.
    pub fn new(
        model: &super::ane_weights::ModelWeights,
        max_cached_per_layer: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(model.layers.len());
        let mut shared_experts = Vec::with_capacity(model.layers.len());

        for lw in &model.layers {
            if let Some(ref moe) = lw.moe {
                let hidden = moe.moe_hidden;
                let dim = moe.packed_experts.gate_cols;
                layers.push(Some(LayerExpertCache::new(max_cached_per_layer, hidden, dim, device)));

                // Pre-compile shared expert (always used, small, one-time)
                let shared = if let Some(ref se) = moe.shared_expert {
                    let g = Tensor::from_slice(&se.gate_proj.dequantize(), (hidden, dim), device)?;
                    let u = Tensor::from_slice(&se.up_proj.dequantize(), (hidden, dim), device)?;
                    let d = Tensor::from_slice(&se.down_proj.dequantize(), (dim, hidden), device)?;
                    Some(CachedExpert {
                        gate: QMatMul::from_qtensor(QTensor::quantize(&g, GgmlDType::Q4_1)?)?,
                        up: QMatMul::from_qtensor(QTensor::quantize(&u, GgmlDType::Q4_1)?)?,
                        down: QMatMul::from_qtensor(QTensor::quantize(&d, GgmlDType::Q4_1)?)?,
                        last_used: 0,
                    })
                } else {
                    None
                };
                shared_experts.push(shared);
            } else {
                layers.push(None);
                shared_experts.push(None);
            }
        }

        let n_moe = layers.iter().filter(|l| l.is_some()).count();
        tracing::info!("MoeExpertCache: {n_moe} layers, {max_cached_per_layer} experts/layer on Metal");

        Ok(Self {
            layers,
            shared_experts,
            device: device.clone(),
        })
    }

    /// Forward MoE FFN for one layer. Uses cached Metal QMatMul.
    pub fn forward(
        &mut self,
        layer: usize,
        moe: &super::ane_weights::MoeLayerWeights,
        x: &mut Vec<f32>,
        rms_ffn: &[f32],
        rms_eps: f32,
        adapter_gate: Option<&[f32]>,
    ) {
        use super::ane_forward::{cpu_matmul, rmsnorm};

        let Some(ref mut cache) = self.layers[layer] else { return };
        let dim = cache.dim;
        let hidden = cache.hidden;
        let ne = moe.num_experts;
        let k = moe.num_experts_per_tok;

        // RMSNorm
        let mut xnorm = vec![0.0f32; dim];
        rmsnorm(&mut xnorm, x, rms_ffn, dim, 1, rms_eps);

        // Router
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
        let mut weights: Vec<f32> = top_k.iter().map(|&(_, l)| (l - max_logit).exp()).collect();
        let sum: f32 = weights.iter().sum();
        for w in &mut weights { *w /= sum; }

        // x tensor on Metal
        let x_tensor = match Tensor::from_slice(&xnorm, (1, dim), &self.device) {
            Ok(t) => t,
            Err(_) => {
                // CPU fallback
                for (i, &(eidx, _)) in top_k.iter().enumerate() {
                    let mut h1 = moe.packed_experts.gate_matmul(eidx, &xnorm);
                    let h3 = moe.packed_experts.up_matmul(eidx, &xnorm);
                    super::ane_decode::silu_inplace(&mut h1);
                    for j in 0..hidden { h1[j] *= h3[j]; }
                    let out = moe.packed_experts.down_matmul(eidx, &h1);
                    for j in 0..dim { x[j] += weights[i] * out[j]; }
                }
                return;
            }
        };

        let mut ffn_out = vec![0.0f32; dim];

        // Expert SwiGLU via cached QMatMul
        for (i, &(expert_idx, _)) in top_k.iter().enumerate() {
            let w = weights[i];
            match cache.get_or_compile(expert_idx, &moe.packed_experts) {
                Ok((gate, up, down)) => {
                    if let Ok(out) = swiglu(gate, up, down, &x_tensor) {
                        for j in 0..dim { ffn_out[j] += w * out[j]; }
                        continue;
                    }
                }
                Err(_) => {}
            }
            // CPU fallback for this expert
            let mut h1 = moe.packed_experts.gate_matmul(expert_idx, &xnorm);
            let h3 = moe.packed_experts.up_matmul(expert_idx, &xnorm);
            super::ane_decode::silu_inplace(&mut h1);
            for j in 0..hidden { h1[j] *= h3[j]; }
            let out = moe.packed_experts.down_matmul(expert_idx, &h1);
            for j in 0..dim { ffn_out[j] += w * out[j]; }
        }

        // Shared expert
        if let Some(ref shared) = self.shared_experts[layer] {
            if let Ok(out) = swiglu(&shared.gate, &shared.up, &shared.down, &x_tensor) {
                for j in 0..dim { ffn_out[j] += out[j]; }
            }
        }

        for j in 0..dim { x[j] += ffn_out[j]; }
    }

    pub fn device(&self) -> &Device { &self.device }

    pub fn n_moe_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.is_some()).count()
    }
}

#[cfg(feature = "candle")]
fn swiglu(gate: &QMatMul, up: &QMatMul, down: &QMatMul, x: &Tensor) -> Result<Vec<f32>> {
    let h1 = gate.forward(x)?;
    let h3 = up.forward(x)?;
    let h1_silu = candle_nn::ops::silu(&h1)?;
    let gated = (h1_silu * h3)?;
    let out = down.forward(&gated)?;
    out.squeeze(0)?.to_vec1::<f32>()
}
