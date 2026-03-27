/*
 * fully_connected.c — RVV-optimized INT8 fully connected layer.
 *
 * Weight layout: [out_features, in_features] (row-major, TFLite FC format).
 * Vectorizes the dot product across in_features using strip-mined RVV ops.
 *
 * This is the most performance-critical kernel: the 1556→256 FC layer
 * accounts for 63% of all inference MACs (398,336 MACs).
 *
 * Target: RISC-V RV32IMF_Zve32x_Zve32f, VLEN=128.
 */

#include <riscv_vector.h>

#include "../kernel_fixedpoint.h"
#include "../kernel_ops.h"

void kernel_fully_connected_int8(const int8_t* input, int in_features,
                                 const int8_t* weights, int out_features,
                                 const int32_t* bias, const quant_param_t* in_q,
                                 const quant_param_per_ch_t* wt_q,
                                 const quant_param_t* out_q, int8_t* output,
                                 int fused_activation) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;
  const size_t vl = 16;

  /* Precompute per-channel fixed-point multipliers. */
  int32_t multipliers[256];
  int shifts[256];
  int32_t act_min, act_max;
  compute_activation_range(fused_activation, out_zp, out_q->scale, &act_min,
                           &act_max);
  for (int oc = 0; oc < out_features; oc++) {
    double eff_scale =
        (double)in_q->scale * (double)wt_q->scales[oc] / (double)out_q->scale;
    quantize_multiplier(eff_scale, &multipliers[oc], &shifts[oc]);
  }

  for (int oc = 0; oc < out_features; oc++) {
    const int8_t* row = weights + oc * in_features;

    /*
     * RVV strip-mined dot product.
     *
     * For 1556→256: 97 full 16-element iterations + 4-element scalar tail.
     * For 256→64:   16 full iterations, no tail.
     * For 64→6:      4 full iterations, no tail.
     *
     * Uses vredsum to reduce each 16-element product vector into a
     * running scalar accumulator.
     */
    vint32m1_t vacc_s = __riscv_vmv_v_x_i32m1(0, 1);
    int ic = 0;

    for (; ic + 16 <= in_features; ic += 16) {
      vint8m1_t vin = __riscv_vle8_v_i8m1(input + ic, vl);
      vint8m1_t vwt = __riscv_vle8_v_i8m1(row + ic, vl);

      /* int8 → int16 (sign extend), subtract input zero point. */
      vint16m2_t vin16 = __riscv_vsext_vf2_i16m2(vin, vl);
      vin16 = __riscv_vsub_vx_i16m2(vin16, (int16_t)in_zp, vl);
      vint16m2_t vwt16 = __riscv_vsext_vf2_i16m2(vwt, vl);

      /* Widening multiply: int16 × int16 → int32. */
      vint32m4_t vprod = __riscv_vwmul_vv_i32m4(vin16, vwt16, vl);

      /* Reduce: sum 16 int32 products into scalar accumulator. */
      vacc_s = __riscv_vredsum_vs_i32m4_i32m1(vprod, vacc_s, vl);
    }

    int32_t acc = __riscv_vmv_x_s_i32m1_i32(vacc_s);

    /* Scalar tail for remainder (e.g. 4 elements for 1556 % 16 = 12,
     * actually 1556 = 97*16 + 4). */
    for (; ic < in_features; ic++) {
      acc += ((int32_t)input[ic] - in_zp) * (int32_t)row[ic];
    }

    acc += bias[oc];

    int32_t result =
        multiply_by_quantized_multiplier(acc, multipliers[oc], shifts[oc]);
    result += out_zp;
    result = apply_fused_activation(result, fused_activation, act_min, act_max);
    if (result < -128) result = -128;
    if (result > 127) result = 127;

    output[oc] = (int8_t)result;
  }
}
