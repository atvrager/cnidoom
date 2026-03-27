/*
 * depthwise_conv2d.c — Reference INT8 depthwise convolution (generic C).
 *
 * NHWC layout.  Accumulates in INT32, adds INT32 bias, then requantizes
 * to INT8 using TFLM-compatible fixed-point math for byte-identical output.
 */

#include "../kernel_fixedpoint.h"
#include "../kernel_ops.h"

void kernel_depthwise_conv2d_int8(
    const int8_t* input, int in_h, int in_w, int channels, const int8_t* filter,
    int filt_h, int filt_w, const int32_t* bias, int stride_h, int stride_w,
    int pad_t, int pad_l, int pad_b, int pad_r, const quant_param_t* in_q,
    const quant_param_per_ch_t* filt_q, const quant_param_t* out_q,
    int8_t* output, int out_h, int out_w, int fused_activation) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;

  (void)pad_b;
  (void)pad_r;

  /* Precompute per-channel fixed-point multipliers. */
  int32_t multipliers[256]; /* max channels we support */
  int shifts[256];
  int32_t act_min, act_max;

  compute_activation_range(fused_activation, out_zp, out_q->scale, &act_min,
                           &act_max);

  for (int c = 0; c < channels; c++) {
    double eff_scale =
        (double)in_q->scale * (double)filt_q->scales[c] / (double)out_q->scale;
    quantize_multiplier(eff_scale, &multipliers[c], &shifts[c]);
  }

  for (int oh = 0; oh < out_h; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
      for (int c = 0; c < channels; c++) {
        int32_t acc = 0;

        for (int kh = 0; kh < filt_h; kh++) {
          for (int kw = 0; kw < filt_w; kw++) {
            int ih = oh * stride_h + kh - pad_t;
            int iw = ow * stride_w + kw - pad_l;

            int32_t in_val;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
              in_val = input[ih * in_w * channels + iw * channels + c];
            } else {
              in_val = in_zp; /* zero-padding in quantized domain */
            }

            /* Filter layout: [1, filt_h, filt_w, channels] */
            int32_t f_val = filter[kh * filt_w * channels + kw * channels + c];

            acc += (in_val - in_zp) * f_val;
          }
        }

        acc += bias[c];

        /* Fixed-point requantization matching TFLM. */
        int32_t result =
            multiply_by_quantized_multiplier(acc, multipliers[c], shifts[c]);
        result += out_zp;
        result =
            apply_fused_activation(result, fused_activation, act_min, act_max);

        if (result < -128) result = -128;
        if (result > 127) result = 127;

        output[oh * out_w * channels + ow * channels + c] = (int8_t)result;
      }
    }
  }
}
