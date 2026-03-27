/*
 * depthwise_conv2d.c — AVX2-optimized INT8 depthwise convolution.
 *
 * Strategy: vectorize across the channel dimension using _mm256 ops.
 * For channels < 32, falls back to scalar where AVX2 doesn't help.
 * For channels == 32 (our model's largest DW layer), processes all
 * channels in a single 256-bit register.
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
 * Scalar fallback for small channel counts (4, 16).
 * Same algorithm as generic/depthwise_conv2d.c.
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
 * AVX2 path for 32-channel depthwise conv (3x3 kernel).
 * Processes all 32 channels in parallel using _mm256 integer ops.
 */
static void depthwise_conv2d_avx2_32ch(
    const int8_t* input, int in_h, int in_w, const int8_t* filter, int filt_h,
    int filt_w, const int32_t* bias, int stride_h, int stride_w, int pad_t,
    int pad_l, const quant_param_t* in_q, const quant_param_per_ch_t* filt_q,
    const quant_param_t* out_q, int8_t* output, int out_h, int out_w,
    int fused_activation) {
  const int channels = 32;
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;

  /* Precompute per-channel fixed-point multipliers. */
  int32_t multipliers[32];
  int shifts[32];
  int32_t act_min, act_max;
  compute_activation_range(fused_activation, out_zp, out_q->scale, &act_min,
                           &act_max);
  for (int c = 0; c < 32; c++) {
    double eff_scale =
        (double)in_q->scale * (double)filt_q->scales[c] / (double)out_q->scale;
    quantize_multiplier(eff_scale, &multipliers[c], &shifts[c]);
  }

  const __m256i vzp = _mm256_set1_epi16((int16_t)in_zp);

  for (int oh = 0; oh < out_h; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
      /* Accumulate in 8 x 32-bit = 256 bits at a time.
       * Process 32 channels in 4 groups of 8. */
      int32_t acc[32];
      for (int c = 0; c < 32; c++) acc[c] = 0;

      for (int kh = 0; kh < filt_h; kh++) {
        for (int kw = 0; kw < filt_w; kw++) {
          int ih = oh * stride_h + kh - pad_t;
          int iw = ow * stride_w + kw - pad_l;

          const int8_t* f_ptr = filter + (kh * filt_w + kw) * channels;

          if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            const int8_t* in_ptr = input + (ih * in_w + iw) * channels;

            /* Load 32 int8 input values and 32 int8 filter values. */
            __m256i vin = _mm256_loadu_si256((const __m256i*)in_ptr);
            __m256i vfilt = _mm256_loadu_si256((const __m256i*)f_ptr);

            /* Extend to 16-bit, subtract zero point, multiply. */
            /* Low 16 channels. */
            __m256i vin_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vin));
            __m256i vfilt_lo =
                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vfilt));
            vin_lo = _mm256_sub_epi16(vin_lo, vzp);
            __m256i prod_lo = _mm256_madd_epi16(vin_lo, vfilt_lo);

            /* For madd_epi16: adjacent pairs are added, giving
             * 8 x int32.  But we need per-channel, not paired.
             * Use mullo_epi16 instead and accumulate manually. */
            __m256i vprod16_lo = _mm256_mullo_epi16(vin_lo, vfilt_lo);

            /* Extend 16→32 and accumulate. */
            __m256i p32_0 =
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod16_lo));
            __m256i p32_1 =
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vprod16_lo, 1));

            __m256i a0 = _mm256_loadu_si256((const __m256i*)(acc + 0));
            __m256i a1 = _mm256_loadu_si256((const __m256i*)(acc + 8));
            a0 = _mm256_add_epi32(a0, p32_0);
            a1 = _mm256_add_epi32(a1, p32_1);
            _mm256_storeu_si256((__m256i*)(acc + 0), a0);
            _mm256_storeu_si256((__m256i*)(acc + 8), a1);

            /* High 16 channels. */
            __m256i vin_hi =
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vin, 1));
            __m256i vfilt_hi =
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vfilt, 1));
            vin_hi = _mm256_sub_epi16(vin_hi, vzp);
            __m256i vprod16_hi = _mm256_mullo_epi16(vin_hi, vfilt_hi);

            __m256i p32_2 =
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vprod16_hi));
            __m256i p32_3 =
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vprod16_hi, 1));

            __m256i a2 = _mm256_loadu_si256((const __m256i*)(acc + 16));
            __m256i a3 = _mm256_loadu_si256((const __m256i*)(acc + 24));
            a2 = _mm256_add_epi32(a2, p32_2);
            a3 = _mm256_add_epi32(a3, p32_3);
            _mm256_storeu_si256((__m256i*)(acc + 16), a2);
            _mm256_storeu_si256((__m256i*)(acc + 24), a3);
          }
          /* else: zero-padding → (in_val - in_zp) == 0, no contribution. */
        }
      }

      /* Add bias and requantize. */
      for (int c = 0; c < 32; c++) {
        acc[c] += bias[c];
        int32_t result =
            multiply_by_quantized_multiplier(acc[c], multipliers[c], shifts[c]);
        result += out_zp;
        result =
            apply_fused_activation(result, fused_activation, act_min, act_max);
        if (result < -128) result = -128;
        if (result > 127) result = 127;
        output[(oh * out_w + ow) * 32 + c] = (int8_t)result;
      }
    }
  }
}

void kernel_depthwise_conv2d_int8(
    const int8_t* input, int in_h, int in_w, int channels, const int8_t* filter,
    int filt_h, int filt_w, const int32_t* bias, int stride_h, int stride_w,
    int pad_t, int pad_l, int pad_b, int pad_r, const quant_param_t* in_q,
    const quant_param_per_ch_t* filt_q, const quant_param_t* out_q,
    int8_t* output, int out_h, int out_w, int fused_activation) {
  (void)pad_b;
  (void)pad_r;

  if (channels == 32) {
    depthwise_conv2d_avx2_32ch(input, in_h, in_w, filter, filt_h, filt_w, bias,
                               stride_h, stride_w, pad_t, pad_l, in_q, filt_q,
                               out_q, output, out_h, out_w, fused_activation);
  } else {
    depthwise_conv2d_scalar(input, in_h, in_w, channels, filter, filt_h, filt_w,
                            bias, stride_h, stride_w, pad_t, pad_l, in_q,
                            filt_q, out_q, output, out_h, out_w,
                            fused_activation);
  }
}
