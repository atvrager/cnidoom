/*
 * fully_connected.c — Reference INT8 fully connected layer (generic C).
 *
 * Weight layout: [out_features, in_features] (row-major, TFLite FC format).
 * Per-channel quantization on the output axis.
 * Uses TFLM-compatible fixed-point requantization.
 */

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
    int32_t acc = 0;

    const int8_t* row = weights + oc * in_features;
    for (int ic = 0; ic < in_features; ic++) {
      acc += (int32_t)(input[ic] - in_zp) * (int32_t)row[ic];
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
