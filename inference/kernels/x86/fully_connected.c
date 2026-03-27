/*
 * fully_connected.c — AVX2-optimized INT8 fully connected layer.
 *
 * Vectorizes the dot product across the input feature dimension.
 * Uses _mm256_cvtepi8_epi16 + _mm256_madd_epi16 for zero-point-aware
 * multiply-accumulate, then horizontal sum.
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

void kernel_fully_connected_int8(const int8_t* input, int in_features,
                                 const int8_t* weights, int out_features,
                                 const int32_t* bias, const quant_param_t* in_q,
                                 const quant_param_per_ch_t* wt_q,
                                 const quant_param_t* out_q, int8_t* output,
                                 int fused_activation) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;
  const __m256i vzp = _mm256_set1_epi16((int16_t)in_zp);

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
    int32_t acc = 0;
    int ic = 0;

    __m256i vacc = _mm256_setzero_si256();

    /* Process 32 elements at a time. */
    for (; ic + 32 <= in_features; ic += 32) {
      __m256i vin = _mm256_loadu_si256((const __m256i*)(input + ic));
      __m256i vwt = _mm256_loadu_si256((const __m256i*)(row + ic));

      /* Low 16 elements: extend to int16, subtract zp, madd. */
      __m256i vin_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vin));
      __m256i vwt_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vwt));
      vin_lo = _mm256_sub_epi16(vin_lo, vzp);
      __m256i p_lo = _mm256_madd_epi16(vin_lo, vwt_lo);
      vacc = _mm256_add_epi32(vacc, p_lo);

      /* High 16 elements. */
      __m256i vin_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vin, 1));
      __m256i vwt_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vwt, 1));
      vin_hi = _mm256_sub_epi16(vin_hi, vzp);
      __m256i p_hi = _mm256_madd_epi16(vin_hi, vwt_hi);
      vacc = _mm256_add_epi32(vacc, p_hi);
    }

    /* Process 16 elements. */
    if (ic + 16 <= in_features) {
      __m128i vin128 = _mm_loadu_si128((const __m128i*)(input + ic));
      __m128i vwt128 = _mm_loadu_si128((const __m128i*)(row + ic));

      __m256i vin16 = _mm256_cvtepi8_epi16(vin128);
      __m256i vwt16 = _mm256_cvtepi8_epi16(vwt128);
      vin16 = _mm256_sub_epi16(vin16, vzp);
      __m256i prod = _mm256_madd_epi16(vin16, vwt16);
      vacc = _mm256_add_epi32(vacc, prod);
      ic += 16;
    }

    /* Horizontal sum: 8 x int32 → scalar. */
    __m128i lo128 = _mm256_castsi256_si128(vacc);
    __m128i hi128 = _mm256_extracti128_si256(vacc, 1);
    __m128i sum128 = _mm_add_epi32(lo128, hi128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    acc = _mm_cvtsi128_si32(sum128);

    /* Scalar tail. */
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
