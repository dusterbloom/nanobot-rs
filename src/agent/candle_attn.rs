//! Candle Metal GQA attention for Qwen3.5's 10 full attention layers.
//!
//! The 30 GDN layers run on ANE (conv1x1 projections + CPU recurrence).
//! The 10 GQA layers need variable-length KV cache and softmax attention —
//! this is what GPUs excel at. Candle's Metal backend handles it.
//!
//! This module provides a minimal GQA implementation: just the attention
//! computation, no FFN (MoE FFN runs on CPU with the router adapter).

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Result, Tensor};

/// Single GQA attention layer for Candle Metal inference.
#[cfg(feature = "candle")]
pub struct CandleGqaLayer {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    rms_att: Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    partial_rotary_factor: f32,
    attn_output_gate: bool,
    // KV cache
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
    cache_len: usize,
}

#[cfg(feature = "candle")]
impl CandleGqaLayer {
    /// Create from dequantized f32 weights (loaded by ane_weights.rs).
    pub fn from_weights(
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        rms_att: &[f32],
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f32,
        attn_output_gate: bool,
        device: &Device,
    ) -> Result<Self> {
        let dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Q projection may be 2x dim for output gating (Qwen3.5)
        let q_out_dim = if attn_output_gate { dim * 2 } else { dim };

        Ok(Self {
            wq: Tensor::from_slice(wq, (q_out_dim, dim), device)?,
            wk: Tensor::from_slice(wk, (kv_dim, dim), device)?,
            wv: Tensor::from_slice(wv, (kv_dim, dim), device)?,
            wo: Tensor::from_slice(wo, (dim, dim), device)?,
            rms_att: Tensor::from_slice(rms_att, dim, device)?,
            q_norm: q_norm.map(|v| Tensor::from_slice(v, head_dim, device)).transpose()?,
            k_norm: k_norm.map(|v| Tensor::from_slice(v, head_dim, device)).transpose()?,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor,
            attn_output_gate,
            k_cache: None,
            v_cache: None,
            cache_len: 0,
        })
    }

    /// Forward pass: single token GQA attention with KV cache.
    ///
    /// Input: hidden state `h` as &[f32] of length `dim`.
    /// Output: attention output as Vec<f32> of length `dim`.
    pub fn forward(&mut self, h: &[f32], pos: usize) -> Result<Vec<f32>> {
        let device = self.wq.device().clone();
        let dim = self.n_heads * self.head_dim;

        // h → Tensor [1, dim]
        let h_t = Tensor::from_slice(h, (1, dim), &device)?;

        // RMSNorm
        let h_normed = rms_norm(&h_t, &self.rms_att, 1e-6)?;

        // Q, K, V projections
        let q = h_normed.matmul(&self.wq.t()?)?; // [1, q_out_dim]
        let k = h_normed.matmul(&self.wk.t()?)?; // [1, kv_dim]
        let v = h_normed.matmul(&self.wv.t()?)?; // [1, kv_dim]

        // Handle output gate (Qwen3.5: Q projection is [2*dim, dim])
        let (q_attn, gate) = if self.attn_output_gate {
            let q_and_gate = q.reshape((1, 2, self.n_heads, self.head_dim))?;
            let q_part = q_and_gate.narrow(1, 0, 1)?.reshape((1, self.n_heads, self.head_dim))?;
            let gate_part = q_and_gate.narrow(1, 1, 1)?.reshape((1, self.n_heads, self.head_dim))?;
            (q_part, Some(gate_part))
        } else {
            let q_reshaped = q.reshape((1, self.n_heads, self.head_dim))?;
            (q_reshaped, None)
        };

        // Reshape K, V for GQA
        let k_reshaped = k.reshape((1, self.n_kv_heads, self.head_dim))?;
        let v_reshaped = v.reshape((1, self.n_kv_heads, self.head_dim))?;

        // Apply per-head Q/K RMSNorm if available
        let q_normed = if let Some(ref qn) = self.q_norm {
            per_head_rms_norm(&q_attn, qn)?
        } else {
            q_attn
        };
        let k_normed = if let Some(ref kn) = self.k_norm {
            per_head_rms_norm(&k_reshaped, kn)?
        } else {
            k_reshaped
        };

        // Apply RoPE (partial rotary for Qwen3.5)
        let rotary_dim = (self.head_dim as f32 * self.partial_rotary_factor) as usize;
        let q_rope = apply_rope(&q_normed, pos, rotary_dim, self.rope_theta)?;
        let k_rope = apply_rope(&k_normed, pos, rotary_dim, self.rope_theta)?;

        // Update KV cache
        self.k_cache = Some(if let Some(ref existing) = self.k_cache {
            Tensor::cat(&[existing, &k_rope], 0)?  // concat along seq dim
        } else {
            k_rope
        });
        self.v_cache = Some(if let Some(ref existing) = self.v_cache {
            Tensor::cat(&[existing, &v_reshaped], 0)?
        } else {
            v_reshaped
        });
        self.cache_len = pos + 1;

        let k_full = self.k_cache.as_ref().unwrap();
        let v_full = self.v_cache.as_ref().unwrap();

        // GQA: expand KV heads to match Q heads
        let hpg = self.n_heads / self.n_kv_heads;
        let k_expanded = if hpg > 1 {
            k_full.unsqueeze(2)?.expand((self.cache_len, self.n_kv_heads, hpg, self.head_dim))?
                .reshape((self.cache_len, self.n_heads, self.head_dim))?
        } else {
            k_full.clone()
        };
        let v_expanded = if hpg > 1 {
            v_full.unsqueeze(2)?.expand((self.cache_len, self.n_kv_heads, hpg, self.head_dim))?
                .reshape((self.cache_len, self.n_heads, self.head_dim))?
        } else {
            v_full.clone()
        };

        // Attention scores: Q @ K^T / sqrt(head_dim)
        // q_rope: [1, n_heads, head_dim], k_expanded: [seq, n_heads, head_dim]
        let scale = (self.head_dim as f64).sqrt();
        let q_scaled = (q_rope / scale)?;

        // Per-head dot product: [n_heads, 1, seq]
        let scores = q_scaled
            .transpose(0, 1)?  // [n_heads, 1, head_dim]
            .matmul(&k_expanded.transpose(0, 1)?.transpose(1, 2)?)?; // [n_heads, 1, seq]

        // Softmax (causal: all positions visible for single-token decode)
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // Weighted sum: [n_heads, 1, head_dim]
        let attn_output = attn_weights
            .matmul(&v_expanded.transpose(0, 1)?)?  // [n_heads, 1, head_dim]
            .transpose(0, 1)?  // [1, n_heads, head_dim]
            .reshape((1, dim))?;

        // Apply output gate (Qwen3.5)
        let gated_output = if let Some(ref gate_t) = gate {
            let gate_sigmoid = candle_nn::ops::sigmoid(gate_t)?;
            let gate_flat = gate_sigmoid.reshape((1, dim))?;
            (attn_output * gate_flat)?
        } else {
            attn_output
        };

        // Output projection
        let output = gated_output.matmul(&self.wo.t()?)?;

        // Back to f32 vec
        let result = output.squeeze(0)?.to_vec1::<f32>()?;
        Ok(result)
    }

    /// Reset KV cache.
    pub fn reset_cache(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
        self.cache_len = 0;
    }
}

/// RMSNorm: x * rsqrt(mean(x²) + eps) * weight
#[cfg(feature = "candle")]
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(1)?;
    let rsqrt = (mean_sq + eps)?.sqrt()?.recip()?;
    let normed = x.broadcast_mul(&rsqrt)?;
    normed.broadcast_mul(&weight.unsqueeze(0)?)
}

/// Per-head RMSNorm (for Q/K norms in Qwen3.5)
#[cfg(feature = "candle")]
fn per_head_rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    // x: [1, n_heads, head_dim], weight: [head_dim]
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(2)?;
    let rsqrt = (mean_sq + 1e-6)?.sqrt()?.recip()?;
    let normed = x.broadcast_mul(&rsqrt)?;
    normed.broadcast_mul(&weight.unsqueeze(0)?.unsqueeze(0)?)
}

/// Apply partial RoPE to Q or K tensor.
#[cfg(feature = "candle")]
fn apply_rope(x: &Tensor, pos: usize, rotary_dim: usize, theta: f64) -> Result<Tensor> {
    let (_batch, n_heads, head_dim) = x.dims3()?;

    if rotary_dim == 0 {
        return Ok(x.clone());
    }

    // Split into rotary and passthrough parts
    let x_rot = x.narrow(2, 0, rotary_dim)?;
    let x_pass = if rotary_dim < head_dim {
        Some(x.narrow(2, rotary_dim, head_dim - rotary_dim)?)
    } else {
        None
    };

    // Compute rotation angles
    let device = x.device().clone();
    let half_rot = rotary_dim / 2;
    let freqs: Vec<f32> = (0..half_rot)
        .map(|i| {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / rotary_dim as f64);
            (pos as f64 * freq) as f32
        })
        .collect();

    let cos_vals: Vec<f32> = freqs.iter().map(|f| f.cos()).collect();
    let sin_vals: Vec<f32> = freqs.iter().map(|f| f.sin()).collect();

    let cos_t = Tensor::from_slice(&cos_vals, (1, 1, half_rot), &device)?;
    let sin_t = Tensor::from_slice(&sin_vals, (1, 1, half_rot), &device)?;

    // Split rotary part into first half and second half
    let x1 = x_rot.narrow(2, 0, half_rot)?;
    let x2 = x_rot.narrow(2, half_rot, half_rot)?;

    // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = (x1.broadcast_mul(&cos_t)? - x2.broadcast_mul(&sin_t)?)?;
    let r2 = (x1.broadcast_mul(&sin_t)? + x2.broadcast_mul(&cos_t)?)?;
    let rotated = Tensor::cat(&[&r1, &r2], 2)?;

    // Recombine with passthrough
    if let Some(pass) = x_pass {
        Tensor::cat(&[&rotated, &pass], 2)
    } else {
        Ok(rotated)
    }
}

/// Collection of Candle GQA layers for the 10 full attention layers.
#[cfg(feature = "candle")]
pub struct CandleGqaLayers {
    layers: Vec<(usize, CandleGqaLayer)>, // (layer_index, layer)
    device: Device,
}

#[cfg(feature = "candle")]
impl CandleGqaLayers {
    /// Initialize from ModelWeights for all non-GDN layers.
    pub fn from_model(model: &super::ane_weights::ModelWeights) -> Result<Self> {
        let device = Device::new_metal(0)?;
        let cfg = &model.cfg;
        let mut layers = Vec::new();

        for (l, lw) in model.layers.iter().enumerate() {
            if lw.gdn.is_some() {
                continue; // GDN layer → ANE, skip
            }
            if lw.wq.is_empty() {
                continue; // No attention weights
            }

            let layer = CandleGqaLayer::from_weights(
                &lw.wq, &lw.wk, &lw.wv, &lw.wo,
                &lw.rms_att,
                lw.q_norm.as_deref(),
                lw.k_norm.as_deref(),
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim(),
                cfg.rope_theta,
                0.25, // partial_rotary_factor for Qwen3.5
                cfg.attn_output_gate,
                &device,
            )?;
            layers.push((l, layer));
        }

        tracing::info!("CandleGqaLayers: {} layers on Metal", layers.len());
        Ok(Self { layers, device })
    }

    /// Forward one token through a specific attention layer.
    /// Returns None if this layer isn't a GQA layer.
    pub fn forward(&mut self, layer_idx: usize, h: &[f32], pos: usize) -> Option<Result<Vec<f32>>> {
        self.layers.iter_mut()
            .find(|(l, _)| *l == layer_idx)
            .map(|(_, layer)| layer.forward(h, pos))
    }

    /// Reset all KV caches.
    pub fn reset_caches(&mut self) {
        for (_, layer) in &mut self.layers {
            layer.reset_cache();
        }
    }

    /// Number of GQA layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
#[cfg(feature = "candle")]
mod tests {
    use super::*;

    #[test]
    fn test_candle_metal_device() {
        match Device::new_metal(0) {
            Ok(d) => eprintln!("Metal device: {:?}", d),
            Err(e) => eprintln!("No Metal: {e} (expected on non-macOS)"),
        }
    }

    #[test]
    fn test_candle_gqa_smoke() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => Device::Cpu, // fallback for CI
        };

        let dim = 64;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = dim / n_heads;

        let wq: Vec<f32> = (0..dim * dim * 2).map(|i| (i as f32 * 0.001).sin() * 0.1).collect(); // 2x for gate
        let wk: Vec<f32> = (0..n_kv_heads * head_dim * dim).map(|i| (i as f32 * 0.002).cos() * 0.1).collect();
        let wv: Vec<f32> = (0..n_kv_heads * head_dim * dim).map(|i| (i as f32 * 0.003).sin() * 0.1).collect();
        let wo: Vec<f32> = (0..dim * dim).map(|i| (i as f32 * 0.004).cos() * 0.1).collect();
        let rms: Vec<f32> = vec![1.0; dim];

        let mut layer = CandleGqaLayer::from_weights(
            &wq, &wk, &wv, &wo, &rms,
            None, None,
            n_heads, n_kv_heads, head_dim,
            10000.0, 1.0, true,
            &device,
        ).expect("layer creation");

        let h: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let result = layer.forward(&h, 0).expect("forward");
        assert_eq!(result.len(), dim);

        // Second token should also work (KV cache grows)
        let result2 = layer.forward(&h, 1).expect("forward pos=1");
        assert_eq!(result2.len(), dim);
        assert_ne!(result, result2, "Different positions should give different outputs");

        eprintln!("Candle GQA smoke test passed on {:?}", device);
    }
}
