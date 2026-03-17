// Split-Silicon Training: MLX forward (GPU) + ANE/CPU backward
//
// The key insight from autoresearch-ANE: "Both accelerators run simultaneously
// with zero interference." ANE and GPU are independent hardware blocks.
//
// Instead of running both forward and backward on ANE (CPU-heavy, slow matmuls),
// we run forward on MLX (GPU — fast quantized matmul via Metal) and backward on
// CPU/ANE (pre-packed weights, fused kernels, LoRA-only gradients).
//
// This module provides `mlx_forward_for_training` which runs an MLX forward pass
// and produces a `ForwardResultWithLora` compatible with the existing backward
// functions in `ane_backward`.

#[cfg(feature = "mlx")]
use super::ane_forward::{
    self, cross_entropy_loss, embed_lookup, rmsnorm, ForwardResult, ForwardResultWithLora,
    LayerActivations,
};
#[cfg(feature = "mlx")]
use super::ane_lora::LoraLayerActivations;
#[cfg(feature = "mlx")]
use super::ane_mil::MilConfig;
#[cfg(feature = "mlx")]
use super::ane_weights::{ModelWeights, WeightSource};
#[cfg(feature = "mlx")]
use super::mlx_lora::MlxLoraModel;

/// Run the model forward on MLX (GPU) and produce a `ForwardResultWithLora`
/// that the CPU/ANE backward pass can consume.
///
/// This is the split-silicon bridge: GPU does the bandwidth-heavy forward pass
/// (quantized matmul on Metal), then we extract per-layer activations into the
/// same format that `forward_cpu_generic` produces. The backward pass doesn't
/// know or care which hardware produced the activations.
///
/// # Arguments
/// - `mlx_model`: The MLX model (runs on GPU via Metal)
/// - `ane_model`: The ANE weight source (for dimensions and embedding lookup)
/// - `lora`: Optional LoRA adapters (LoRA deltas applied during forward)
/// - `tokens`: Input token IDs
/// - `targets`: Target token IDs for loss computation
/// - `mil_cfg`: Model configuration (dimensions, head counts, etc.)
#[cfg(feature = "mlx")]
pub fn mlx_forward_for_training(
    _mlx_model: &MlxLoraModel,
    ane_model: &ModelWeights,
    lora: Option<&super::ane_lora::LoraModel>,
    tokens: &[u32],
    targets: &[u32],
    _mil_cfg: &MilConfig,
) -> ForwardResultWithLora {
    // Phase 1 (TDD RED→GREEN): delegate to CPU forward to prove the interface.
    // Phase 2: implement actual MLX forward with activation capture on GPU.
    //
    // The contract: produce a ForwardResultWithLora with correct activations
    // so backward_lora_cpu_generic can compute valid LoRA gradients.
    ane_forward::forward_cpu_generic(ane_model, lora, tokens, targets)
}
