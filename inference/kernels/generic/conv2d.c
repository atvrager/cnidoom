/*
 * conv2d.c — Reference INT8 2D convolution (generic C).
 *
 * NHWC layout.  Filter layout: [out_c, filt_h, filt_w, in_c].
 * Accumulates in INT32, adds INT32 bias, then requantizes to INT8
 * using TFLM-compatible fixed-point math.
 */

#include "../kernel_fixedpoint.h"
#include "../kernel_ops.h"

void kernel_conv2d_int8(const int8_t* input, int in_h, int in_w, int in_c,
                        const int8_t* filter, int filt_h, int filt_w, int out_c,
                        const int32_t* bias, int stride_h, int stride_w,
                        int pad_t, int pad_l, int pad_b, int pad_r,
                        const quant_param_t* in_q,
                        const quant_param_per_ch_t* filt_q,
                        const quant_param_t* out_q, int8_t* output, int out_h,
                        int out_w, int fused_activation) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;

  (void)pad_b;
  (void)pad_r;

  /* Precompute per-channel fixed-point multipliers. */
  int32_t multipliers[256];
  int shifts[256];
  int32_t act_min, act_max;

  compute_activation_range(fused_activation, out_zp, out_q->scale, &act_min,
                           &act_max);

  for (int oc = 0; oc < out_c; oc++) {
    double eff_scale =
        (double)in_q->scale * (double)filt_q->scales[oc] / (double)out_q->scale;
    quantize_multiplier(eff_scale, &multipliers[oc], &shifts[oc]);
  }

  for (int oh = 0; oh < out_h; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
      for (int oc = 0; oc < out_c; oc++) {
        int32_t acc = 0;

        for (int kh = 0; kh < filt_h; kh++) {
          for (int kw = 0; kw < filt_w; kw++) {
            int ih = oh * stride_h + kh - pad_t;
            int iw = ow * stride_w + kw - pad_l;

            for (int ic = 0; ic < in_c; ic++) {
              int32_t in_val;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                in_val = input[ih * in_w * in_c + iw * in_c + ic];
              } else {
                in_val = in_zp;
              }

              /* Filter: [out_c, filt_h, filt_w, in_c] */
              int32_t f_val = filter[oc * filt_h * filt_w * in_c +
                                     kh * filt_w * in_c + kw * in_c + ic];

              acc += (in_val - in_zp) * f_val;
            }
          }
        }

        acc += bias[oc];

        int32_t result =
            multiply_by_quantized_multiplier(acc, multipliers[oc], shifts[oc]);
        result += out_zp;
        result =
            apply_fused_activation(result, fused_activation, act_min, act_max);

        if (result < -128) result = -128;
        if (result > 127) result = 127;

        output[oh * out_w * out_c + ow * out_c + oc] = (int8_t)result;
      }
    }
  }
}
