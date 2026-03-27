/*
 * conv2d.c — AVX2-optimized INT8 2D convolution.
 *
 * For 1x1 pointwise convolutions (our model's case), this is effectively
 * a GEMM: each output pixel is an independent dot product of the input
 * channels with the filter.  AVX2 _mm256_maddubs_epi16 computes 32
 * unsigned*signed products and pairwise sums in one instruction.
 */

#include <immintrin.h>
#include <math.h>

#include "../kernel_fixedpoint.h"
#include "../kernel_ops.h"

static inline int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

/*
 * AVX2 pointwise (1x1) convolution.
 * Vectorizes the input-channel reduction for each output channel.
 */
static void conv2d_pointwise_avx2(const int8_t* input, int in_h, int in_w,
                                  int in_c, const int8_t* filter, int out_c,
                                  const int32_t* bias,
                                  const quant_param_t* in_q,
                                  const quant_param_per_ch_t* filt_q,
                                  const quant_param_t* out_q, int8_t* output,
                                  int out_h, int out_w, int fused_activation) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;
  const int num_pixels = out_h * out_w;

  /* Precompute per-channel fixed-point multipliers. */
  int32_t* multipliers = (int32_t*)_mm_malloc(out_c * sizeof(int32_t), 32);
  int* fp_shifts = (int*)_mm_malloc(out_c * sizeof(int), 32);
  int32_t act_min, act_max;
  compute_activation_range(fused_activation, out_zp, out_q->scale, &act_min,
                           &act_max);
  for (int oc = 0; oc < out_c; oc++) {
    double eff_scale =
        (double)in_q->scale * (double)filt_q->scales[oc] / (double)out_q->scale;
    quantize_multiplier(eff_scale, &multipliers[oc], &fp_shifts[oc]);
  }

  for (int p = 0; p < num_pixels; p++) {
    const int8_t* in_ptr = input + p * in_c;

    for (int oc = 0; oc < out_c; oc++) {
      const int8_t* f_ptr = filter + oc * in_c;
      int32_t acc = 0;

      /* Vectorized dot product with zero-point subtraction. */
      int ic = 0;
      __m256i vacc = _mm256_setzero_si256();

      for (; ic + 32 <= in_c; ic += 32) {
        __m256i vin = _mm256_loadu_si256((const __m256i*)(in_ptr + ic));
        __m256i vfilt = _mm256_loadu_si256((const __m256i*)(f_ptr + ic));

        /* Extend to 16-bit for zero-point subtraction. */
        /* Process low 16 bytes. */
        __m256i vin_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vin));
        __m256i vfilt_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vfilt));
        __m256i vzp = _mm256_set1_epi16((int16_t)in_zp);
        vin_lo = _mm256_sub_epi16(vin_lo, vzp);
        /* madd: adjacent pairs multiplied and summed → 8 x int32. */
        __m256i p_lo = _mm256_madd_epi16(vin_lo, vfilt_lo);
        vacc = _mm256_add_epi32(vacc, p_lo);

        /* High 16 bytes. */
        __m256i vin_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vin, 1));
        __m256i vfilt_hi =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vfilt, 1));
        vin_hi = _mm256_sub_epi16(vin_hi, vzp);
        __m256i p_hi = _mm256_madd_epi16(vin_hi, vfilt_hi);
        vacc = _mm256_add_epi32(vacc, p_hi);
      }

      /* Horizontal sum of vacc (8 x int32 → scalar). */
      __m128i lo128 = _mm256_castsi256_si128(vacc);
      __m128i hi128 = _mm256_extracti128_si256(vacc, 1);
      __m128i sum128 = _mm_add_epi32(lo128, hi128);
      sum128 = _mm_hadd_epi32(sum128, sum128);
      sum128 = _mm_hadd_epi32(sum128, sum128);
      acc = _mm_cvtsi128_si32(sum128);

      /* Scalar tail. */
      for (; ic < in_c; ic++) {
        acc += ((int32_t)in_ptr[ic] - in_zp) * (int32_t)f_ptr[ic];
      }

      acc += bias[oc];

      int32_t result =
          multiply_by_quantized_multiplier(acc, multipliers[oc], fp_shifts[oc]);
      result += out_zp;
      result =
          apply_fused_activation(result, fused_activation, act_min, act_max);
      if (result < -128) result = -128;
      if (result > 127) result = 127;

      output[p * out_c + oc] = (int8_t)result;
    }
  }

  _mm_free(multipliers);
  _mm_free(fp_shifts);
}

/*
 * General convolution fallback (non-1x1 or non-aligned).
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

  if (filt_h == 1 && filt_w == 1 && stride_h == 1 && stride_w == 1) {
    conv2d_pointwise_avx2(input, in_h, in_w, in_c, filter, out_c, bias, in_q,
                          filt_q, out_q, output, out_h, out_w,
                          fused_activation);
  } else {
    conv2d_general(input, in_h, in_w, in_c, filter, filt_h, filt_w, out_c, bias,
                   stride_h, stride_w, pad_t, pad_l, in_q, filt_q, out_q,
                   output, out_h, out_w, fused_activation);
  }
}
