#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
instantiate_kernel("steel_attention_float32_bq16_bk8_bd256_wm2_wn1_maskfloat32", attention, float, 16, 8, 256, 2, 1, float, float)
instantiate_kernel("steel_attention_float32_bq16_bk8_bd256_wm2_wn1_maskbool_", attention, float, 16, 8, 256, 2, 1, bool, float)
