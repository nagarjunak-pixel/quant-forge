// Native C++20 (-O3) Bit-Packed INT4 (2 weights/byte) Group-Quantize, Dequantize & Fused
// Register Dequant-GEMM Kernel for QuantForge.
// Packs two asymmetric 4-bit integers q0, q1 in [0..15] into one uint8_t:
//   packed = (q0 & 0x0F) | ((q1 & 0x0F) << 4)

#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" {

// Pack group-wise asymmetric INT4 weights from row-major float32 W [out_features, in_features].
// in_features and group_size must be even and divisible (in_features % group_size == 0).
void qforge_pack_int4_grouped_f32(
    const float* __restrict__ w,
    int32_t out_features,
    int32_t in_features,
    int32_t group_size,
    uint8_t* __restrict__ packed_out,
    float* __restrict__ scales_out,
    float* __restrict__ zeros_out
) {
    const int32_t num_groups = in_features / group_size;
    const int32_t packed_cols = in_features / 2;

    for (int32_t row = 0; row < out_features; ++row) {
        const float* w_row = w + static_cast<size_t>(row) * in_features;
        uint8_t* p_row = packed_out + static_cast<size_t>(row) * packed_cols;
        float* s_row = scales_out + static_cast<size_t>(row) * num_groups;
        float* z_row = zeros_out + static_cast<size_t>(row) * num_groups;

        for (int32_t g = 0; g < num_groups; ++g) {
            const int32_t base = g * group_size;
            float w_min = w_row[base];
            float w_max = w_row[base];
            for (int32_t i = 1; i < group_size; ++i) {
                const float v = w_row[base + i];
                w_min = std::min(w_min, v);
                w_max = std::max(w_max, v);
            }
            const float span = std::max(w_max - w_min, 1e-8f);
            const float scale = span / 15.0f;
            const float zero = std::clamp(std::round(-w_min / scale), 0.0f, 15.0f);

            s_row[g] = scale;
            z_row[g] = zero;

            for (int32_t i = 0; i < group_size; i += 2) {
                const float v0 = w_row[base + i];
                const float v1 = w_row[base + i + 1];
                const int32_t q0 = static_cast<int32_t>(
                    std::clamp(std::round(v0 / scale + zero), 0.0f, 15.0f)
                );
                const int32_t q1 = static_cast<int32_t>(
                    std::clamp(std::round(v1 / scale + zero), 0.0f, 15.0f)
                );
                p_row[(base + i) / 2] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            }
        }
    }
}

// Fused INT4 Dequantize + Matrix Multiply: Y [batch, out_features] = X [batch, in_features] @ W_hat^T
// Never materializes the dequantized float32 weight matrix in memory!
void qforge_int4_fused_gemm_f32(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int32_t batch_size,
    int32_t out_features,
    int32_t in_features,
    int32_t group_size,
    float* __restrict__ y_out
) {
    const int32_t num_groups = in_features / group_size;
    const int32_t packed_cols = in_features / 2;
    const int32_t half_group = group_size / 2;

    for (int32_t b = 0; b < batch_size; ++b) {
        const float* x_row = x + static_cast<size_t>(b) * in_features;
        float* y_row = y_out + static_cast<size_t>(b) * out_features;

        for (int32_t o = 0; o < out_features; ++o) {
            const uint8_t* p_row = packed_w + static_cast<size_t>(o) * packed_cols;
            const float* s_row = scales + static_cast<size_t>(o) * num_groups;
            const float* z_row = zeros + static_cast<size_t>(o) * num_groups;

            double total_acc = 0.0;
            for (int32_t g = 0; g < num_groups; ++g) {
                const float sc = s_row[g];
                const float zp = z_row[g];
                const uint8_t* p_grp = p_row + g * half_group;
                const float* x_grp = x_row + g * group_size;

                double grp_q_dot_x = 0.0;
                double grp_sum_x = 0.0;
                for (int32_t j = 0; j < half_group; ++j) {
                    const uint8_t byte_val = p_grp[j];
                    const float q0 = static_cast<float>(byte_val & 0x0F);
                    const float q1 = static_cast<float>((byte_val >> 4) & 0x0F);
                    const float x0 = x_grp[2 * j];
                    const float x1 = x_grp[2 * j + 1];
                    grp_q_dot_x += static_cast<double>(q0 * x0 + q1 * x1);
                    grp_sum_x += static_cast<double>(x0 + x1);
                }
                // Algebraic fusion: sum_i sc * (q_i - zp) * x_i == sc * (sum_i q_i x_i - zp * sum_i x_i)
                total_acc += static_cast<double>(sc) * (grp_q_dot_x - static_cast<double>(zp) * grp_sum_x);
            }
            y_row[o] = static_cast<float>(total_acc);
        }
    }
}

}  // extern "C"
