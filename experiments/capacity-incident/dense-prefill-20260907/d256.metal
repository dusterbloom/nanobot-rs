#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal"
instantiate_attn(float32, float, 8, 16, 256, 1, 1, float32, float)
instantiate_attn(float32, float, 8, 16, 256, 1, 1, bool_, bool)
