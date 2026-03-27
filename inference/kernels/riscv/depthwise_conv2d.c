/*
 * depthwise_conv2d.c — RVV-optimized INT8 depthwise convolution.
 *
 * NHWC layout.  Vectorizes across the channel dimension using RVV intrinsics.
 * For channels < 16, falls back to scalar (too few lanes to benefit).
 * For channels == 16: single VLMAX pass per kernel position.
 * For channels == 32: two 16-channel chunks per kernel position.
 *
 * Target: RISC-V RV32IMF_Zve32x_Zve32f, VLEN=128.
 */

#include <riscv_vector.h>

#include "../kernel_fixedpoint.h"
#include "../kernel_ops.h"

/*
 * Scalar fallback for small channel counts (e.g. 4ch in Node 0).
 * Identical to generic/depthwise_conv2d.c.
 */
static void depthwise_conv2d_scalar(
    const int8_t* input, int in_h, int in_w, int channels, const int8_t* filter,
    int filt_h, int filt_w, const int32_t* bias, int stride_h, int stride_w,
    int pad_t, int pad_l, const quant_param_t* in_q,
    const quant_param_per_ch_t* filt_q, const quant_param_t* out_q,
    int8_t* output, int out_h, int out_w, int fused_activation) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;

  int32_t multipliers[256];
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
              in_val = in_zp;
            }
            int32_t f_val = filter[kh * filt_w * channels + kw * channels + c];
            acc += (in_val - in_zp) * f_val;
          }
        }
        acc += bias[c];
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

/*
 * RVV path for 16-channel depthwise conv (Node 2: 25,920 MACs).
 *
 * Strategy: 16 channels = VLMAX for SEW=8/LMUL=1.  Each vector lane
 * accumulates independently (one channel per lane).  The widening chain
 * int8(m1) → int16(m2) → int32(m4) produces a 16-element int32 accumulator
 * in a single m4 register group.
 */
static void depthwise_rvv_16ch(const int8_t* input, int in_h, int in_w,
                               const int8_t* filter, int filt_h, int filt_w,
                               const int32_t* bias, int stride_h, int stride_w,
                               int pad_t, int pad_l, const quant_param_t* in_q,
                               const quant_param_per_ch_t* filt_q,
                               const quant_param_t* out_q, int8_t* output,
                               int out_h, int out_w, int fused_activation) {
  const int channels = 16;
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;
  const size_t vl = 16;

  int32_t multipliers[16];
  int shifts_arr[16];
  int32_t act_min, act_max;
  compute_activation_range(fused_activation, out_zp, out_q->scale, &act_min,
                           &act_max);
  for (int c = 0; c < 16; c++) {
    double eff_scale =
        (double)in_q->scale * (double)filt_q->scales[c] / (double)out_q->scale;
    quantize_multiplier(eff_scale, &multipliers[c], &shifts_arr[c]);
  }

  for (int oh = 0; oh < out_h; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
      /* Zero the 16-lane int32 accumulator. */
      vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);

      for (int kh = 0; kh < filt_h; kh++) {
        for (int kw = 0; kw < filt_w; kw++) {
          int ih = oh * stride_h + kh - pad_t;
          int iw = ow * stride_w + kw - pad_l;

          if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            const int8_t* in_ptr = input + (ih * in_w + iw) * channels;
            const int8_t* f_ptr = filter + (kh * filt_w + kw) * channels;

            /* Load 16 × int8 input and filter. */
            vint8m1_t vin = __riscv_vle8_v_i8m1(in_ptr, vl);
            vint8m1_t vfilt = __riscv_vle8_v_i8m1(f_ptr, vl);

            /* Widen to int16, subtract input zero point. */
            vint16m2_t vin16 = __riscv_vsext_vf2_i16m2(vin, vl);
            vin16 = __riscv_vsub_vx_i16m2(vin16, (int16_t)in_zp, vl);
            vint16m2_t vfilt16 = __riscv_vsext_vf2_i16m2(vfilt, vl);

            /* Widening MAC: int16 × int16 → accumulate into int32. */
            vacc = __riscv_vwmacc_vv_i32m4(vacc, vin16, vfilt16, vl);
          }
          /* Out-of-bounds: (in_val - in_zp) == 0, no contribution. */
        }
      }

      /* Store accumulator to memory for scalar requantization. */
      int32_t acc[16];
      __riscv_vse32_v_i32m4(acc, vacc, vl);

      int8_t* out_ptr = output + (oh * out_w + ow) * channels;
      for (int c = 0; c < 16; c++) {
        int32_t a = acc[c] + bias[c];
        int32_t result =
            multiply_by_quantized_multiplier(a, multipliers[c], shifts_arr[c]);
        result += out_zp;
        result =
            apply_fused_activation(result, fused_activation, act_min, act_max);
        if (result < -128) result = -128;
        if (result > 127) result = 127;
        out_ptr[c] = (int8_t)result;
      }
    }
  }
}

/*
 * RVV path for 32-channel depthwise conv (Node 4: 13,824 MACs).
 *
 * Processes in two chunks of 16 channels each to avoid LMUL=8 pressure.
 */
static void depthwise_rvv_32ch(const int8_t* input, int in_h, int in_w,
                               const int8_t* filter, int filt_h, int filt_w,
                               const int32_t* bias, int stride_h, int stride_w,
                               int pad_t, int pad_l, const quant_param_t* in_q,
                               const quant_param_per_ch_t* filt_q,
                               const quant_param_t* out_q, int8_t* output,
                               int out_h, int out_w, int fused_activation) {
  const int channels = 32;
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;
  const size_t vl = 16;

  int32_t multipliers[32];
  int shifts_arr[32];
  int32_t act_min, act_max;
  compute_activation_range(fused_activation, out_zp, out_q->scale, &act_min,
                           &act_max);
  for (int c = 0; c < 32; c++) {
    double eff_scale =
        (double)in_q->scale * (double)filt_q->scales[c] / (double)out_q->scale;
    quantize_multiplier(eff_scale, &multipliers[c], &shifts_arr[c]);
  }

  for (int oh = 0; oh < out_h; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
      /* Two 16-lane accumulators for channels [0..15] and [16..31]. */
      vint32m4_t vacc_lo = __riscv_vmv_v_x_i32m4(0, vl);
      vint32m4_t vacc_hi = __riscv_vmv_v_x_i32m4(0, vl);

      for (int kh = 0; kh < filt_h; kh++) {
        for (int kw = 0; kw < filt_w; kw++) {
          int ih = oh * stride_h + kh - pad_t;
          int iw = ow * stride_w + kw - pad_l;

          if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            const int8_t* in_ptr = input + (ih * in_w + iw) * channels;
            const int8_t* f_ptr = filter + (kh * filt_w + kw) * channels;

            /* Chunk 0: channels [0..15]. */
            vint8m1_t vin0 = __riscv_vle8_v_i8m1(in_ptr, vl);
            vint8m1_t vf0 = __riscv_vle8_v_i8m1(f_ptr, vl);
            vint16m2_t vin16_0 = __riscv_vsext_vf2_i16m2(vin0, vl);
            vin16_0 = __riscv_vsub_vx_i16m2(vin16_0, (int16_t)in_zp, vl);
            vint16m2_t vf16_0 = __riscv_vsext_vf2_i16m2(vf0, vl);
            vacc_lo = __riscv_vwmacc_vv_i32m4(vacc_lo, vin16_0, vf16_0, vl);

            /* Chunk 1: channels [16..31]. */
            vint8m1_t vin1 = __riscv_vle8_v_i8m1(in_ptr + 16, vl);
            vint8m1_t vf1 = __riscv_vle8_v_i8m1(f_ptr + 16, vl);
            vint16m2_t vin16_1 = __riscv_vsext_vf2_i16m2(vin1, vl);
            vin16_1 = __riscv_vsub_vx_i16m2(vin16_1, (int16_t)in_zp, vl);
            vint16m2_t vf16_1 = __riscv_vsext_vf2_i16m2(vf1, vl);
            vacc_hi = __riscv_vwmacc_vv_i32m4(vacc_hi, vin16_1, vf16_1, vl);
          }
        }
      }

      /* Store accumulators and requantize. */
      int32_t acc[32];
      __riscv_vse32_v_i32m4(acc, vacc_lo, vl);
      __riscv_vse32_v_i32m4(acc + 16, vacc_hi, vl);

      int8_t* out_ptr = output + (oh * out_w + ow) * channels;
      for (int c = 0; c < 32; c++) {
        int32_t a = acc[c] + bias[c];
        int32_t result =
            multiply_by_quantized_multiplier(a, multipliers[c], shifts_arr[c]);
        result += out_zp;
        result =
            apply_fused_activation(result, fused_activation, act_min, act_max);
        if (result < -128) result = -128;
        if (result > 127) result = 127;
        out_ptr[c] = (int8_t)result;
      }
    }
  }
}

/* Dispatch: scalar for small channels, RVV for 16 and 32. */
void kernel_depthwise_conv2d_int8(
    const int8_t* input, int in_h, int in_w, int channels, const int8_t* filter,
    int filt_h, int filt_w, const int32_t* bias, int stride_h, int stride_w,
    int pad_t, int pad_l, int pad_b, int pad_r, const quant_param_t* in_q,
    const quant_param_per_ch_t* filt_q, const quant_param_t* out_q,
    int8_t* output, int out_h, int out_w, int fused_activation) {
  (void)pad_b;
  (void)pad_r;

  if (channels == 32) {
    depthwise_rvv_32ch(input, in_h, in_w, filter, filt_h, filt_w, bias,
                       stride_h, stride_w, pad_t, pad_l, in_q, filt_q, out_q,
                       output, out_h, out_w, fused_activation);
  } else if (channels == 16) {
    depthwise_rvv_16ch(input, in_h, in_w, filter, filt_h, filt_w, bias,
                       stride_h, stride_w, pad_t, pad_l, in_q, filt_q, out_q,
                       output, out_h, out_w, fused_activation);
  } else {
    depthwise_conv2d_scalar(input, in_h, in_w, channels, filter, filt_h, filt_w,
                            bias, stride_h, stride_w, pad_t, pad_l, in_q,
                            filt_q, out_q, output, out_h, out_w,
                            fused_activation);
  }
}
