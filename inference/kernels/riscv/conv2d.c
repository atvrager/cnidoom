/*
 * conv2d.c — RVV-optimized INT8 2D convolution.
 *
 * NHWC layout.  Filter layout: [out_c, filt_h, filt_w, in_c].
 *
 * For 1×1 pointwise convolutions with in_c >= 16, uses RVV strip-mined
 * dot products with vredsum reduction.  Falls back to scalar for spatial
 * convolutions (where the overhead of bounds checks dominates).
 *
 * Target: RISC-V RV32IMF_Zve32x_Zve32f, VLEN=128.
 */

#include <riscv_vector.h>

#include "../kernel_fixedpoint.h"
#include "../kernel_ops.h"

/*
 * Scalar fallback for general spatial convolutions.
 * Identical to generic/conv2d.c.
 */
static void conv2d_general(const int8_t* input, int in_h, int in_w, int in_c,
                           const int8_t* filter, int filt_h, int filt_w,
                           int out_c, const int32_t* bias, int stride_h,
                           int stride_w, int pad_t, int pad_l,
                           const quant_param_t* in_q,
                           const quant_param_per_ch_t* filt_q,
                           const quant_param_t* out_q, int8_t* output,
                           int out_h, int out_w, int fused_activation) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;

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

/*
 * RVV pointwise (1×1) convolution path (Nodes 3, 5).
 *
 * Strip-mines the in_c dot product in chunks of 16 using:
 *   int8(m1) → int16(m2) via vsext, subtract zp, vwmul → int32(m4),
 *   then vredsum to accumulate partial sums into a running scalar.
 *
 * For in_c=16: one iteration, no tail.
 * For in_c=32: two iterations, no tail.
 */
static void conv2d_pointwise_rvv(const int8_t* input, int in_h, int in_w,
                                 int in_c, const int8_t* filter, int out_c,
                                 const int32_t* bias, int stride_h,
                                 int stride_w, int pad_t, int pad_l,
                                 const quant_param_t* in_q,
                                 const quant_param_per_ch_t* filt_q,
                                 const quant_param_t* out_q, int8_t* output,
                                 int out_h, int out_w, int fused_activation) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;
  const size_t vl = 16;

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

  /* 1×1 conv with any stride: for each output pixel, compute the
   * corresponding input pixel and run the in_c dot product. */
  for (int oh = 0; oh < out_h; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
      int ih = oh * stride_h - pad_t;
      int iw = ow * stride_w - pad_l;

      /* For 1×1 kernels the pixel is either fully in-bounds or not.
       * If out-of-bounds, the entire contribution is zero (padding). */
      const int8_t* in_pixel = NULL;
      int in_bounds = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w);
      if (in_bounds) {
        in_pixel = input + (ih * in_w + iw) * in_c;
      }

      for (int oc = 0; oc < out_c; oc++) {
        const int8_t* f_row = filter + oc * in_c;

        int32_t acc = 0;

        if (in_bounds) {
          /* RVV strip-mined dot product over in_c. */
          vint32m1_t vacc_s = __riscv_vmv_v_x_i32m1(0, 1);
          int ic = 0;

          for (; ic + 16 <= in_c; ic += 16) {
            vint8m1_t vin = __riscv_vle8_v_i8m1(in_pixel + ic, vl);
            vint8m1_t vwt = __riscv_vle8_v_i8m1(f_row + ic, vl);

            vint16m2_t vin16 = __riscv_vsext_vf2_i16m2(vin, vl);
            vin16 = __riscv_vsub_vx_i16m2(vin16, (int16_t)in_zp, vl);
            vint16m2_t vwt16 = __riscv_vsext_vf2_i16m2(vwt, vl);

            vint32m4_t vprod = __riscv_vwmul_vv_i32m4(vin16, vwt16, vl);
            vacc_s = __riscv_vredsum_vs_i32m4_i32m1(vprod, vacc_s, vl);
          }

          acc = __riscv_vmv_x_s_i32m1_i32(vacc_s);

          /* Scalar tail for in_c not divisible by 16. */
          for (; ic < in_c; ic++) {
            acc += ((int32_t)in_pixel[ic] - in_zp) * (int32_t)f_row[ic];
          }
        }
        /* else: padding pixel, acc stays 0. */

        acc += bias[oc];

        int32_t result =
            multiply_by_quantized_multiplier(acc, multipliers[oc], shifts[oc]);
        result += out_zp;
        result =
            apply_fused_activation(result, fused_activation, act_min, act_max);
        if (result < -128) result = -128;
        if (result > 127) result = 127;

        output[(oh * out_w + ow) * out_c + oc] = (int8_t)result;
      }
    }
  }
}

/* Dispatch: RVV for 1×1 pointwise with in_c ≥ 16, scalar otherwise. */
void kernel_conv2d_int8(const int8_t* input, int in_h, int in_w, int in_c,
                        const int8_t* filter, int filt_h, int filt_w, int out_c,
                        const int32_t* bias, int stride_h, int stride_w,
                        int pad_t, int pad_l, int pad_b, int pad_r,
                        const quant_param_t* in_q,
                        const quant_param_per_ch_t* filt_q,
                        const quant_param_t* out_q, int8_t* output, int out_h,
                        int out_w, int fused_activation) {
  (void)pad_b;
  (void)pad_r;

  if (filt_h == 1 && filt_w == 1 && in_c >= 16) {
    conv2d_pointwise_rvv(input, in_h, in_w, in_c, filter, out_c, bias, stride_h,
                         stride_w, pad_t, pad_l, in_q, filt_q, out_q, output,
                         out_h, out_w, fused_activation);
  } else {
    conv2d_general(input, in_h, in_w, in_c, filter, filt_h, filt_w, out_c, bias,
                   stride_h, stride_w, pad_t, pad_l, in_q, filt_q, out_q,
                   output, out_h, out_w, fused_activation);
  }
}
