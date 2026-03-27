/*
 * mean.c — AVX2-optimized INT8 MEAN (Global Average Pooling) kernel.
 *
 * Reduces [1, H, W, C] → [1, C] by averaging over spatial dimensions.
 *
 * Strategy: For each channel, accumulate H*W int8 values into int32.
 * Since input is NHWC (channels interleaved), we process one channel
 * at a time.  Within a channel we load 32 spatial values at a time
 * (stepping by stride=channels), widen to int16 with zero-point
 * subtraction, then accumulate into int32.
 *
 * For typical V2 dimensions (4×5=20 spatial, 192 channels), the inner
 * loop is short and the scalar tail handles any remainder.
 */

#include <immintrin.h>

#include "../kernel_fixedpoint.h"
#include "../kernel_ops.h"

void kernel_mean_int8(const int8_t* input, int height, int width, int channels,
                      const quant_param_t* in_q, const quant_param_t* out_q,
                      int8_t* output) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;
  const int count = height * width;

  /* Effective scale: input_scale / (H * W) / output_scale. */
  double eff_scale =
      (double)in_q->scale / ((double)count * (double)out_q->scale);
  int32_t multiplier;
  int shift;
  quantize_multiplier(eff_scale, &multiplier, &shift);

  const __m256i vzp16 = _mm256_set1_epi16((int16_t)in_zp);

  for (int c = 0; c < channels; c++) {
    const int8_t* ch_base = input + c;
    int32_t acc = 0;
    int i = 0;

    __m256i vacc = _mm256_setzero_si256();

    /* Process 16 spatial positions at a time.
     * Gather int8 values at stride=channels into a 128-bit register,
     * widen to int16 in a 256-bit register, subtract zp, madd with
     * ones to accumulate pairs into int32, then add to vacc. */
    const __m256i vones = _mm256_set1_epi16(1);

    for (; i + 16 <= count; i += 16) {
      /* Scalar gather into a 128-bit buffer (no AVX2 byte gather). */
      int8_t buf[16];
      for (int j = 0; j < 16; j++) {
        buf[j] = ch_base[(i + j) * channels];
      }
      __m128i vraw = _mm_loadu_si128((const __m128i*)buf);

      /* int8 → int16, subtract zero point. */
      __m256i v16 = _mm256_cvtepi8_epi16(vraw);
      v16 = _mm256_sub_epi16(v16, vzp16);

      /* Pairwise horizontal add via madd with 1s: int16 → int32. */
      __m256i v32 = _mm256_madd_epi16(v16, vones);
      vacc = _mm256_add_epi32(vacc, v32);
    }

    /* Horizontal sum of 8 × int32. */
    __m128i lo = _mm256_castsi256_si128(vacc);
    __m128i hi = _mm256_extracti128_si256(vacc, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    acc = _mm_cvtsi128_si32(sum128);

    /* Scalar tail. */
    for (; i < count; i++) {
      acc += (int32_t)ch_base[i * channels] - in_zp;
    }

    int32_t result = multiply_by_quantized_multiplier(acc, multiplier, shift);
    result += out_zp;

    if (result < -128) result = -128;
    if (result > 127) result = 127;

    output[c] = (int8_t)result;
  }
}
