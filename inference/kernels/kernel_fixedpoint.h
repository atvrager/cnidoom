/*
 * kernel_fixedpoint.h — Fixed-point requantization matching TFLM exactly.
 *
 * Implements the same QuantizeMultiplier / MultiplyByQuantizedMultiplier
 * pipeline as TFLM so that our kernels produce byte-identical output.
 *
 * Reference:
 *   tflite-micro/tensorflow/lite/kernels/internal/quantization_util.cc
 *   tflite-micro/tensorflow/lite/kernels/internal/common.h
 */

#ifndef KERNEL_FIXEDPOINT_H
#define KERNEL_FIXEDPOINT_H

#include <math.h>
#include <stdint.h>

/*
 * Decompose a floating-point multiplier into (quantized_multiplier, shift)
 * exactly as TFLM's QuantizeMultiplier does.
 *
 * The multiplier is normalized to [0.5, 1.0) and scaled to int32.
 */
static inline void quantize_multiplier(double double_multiplier,
                                       int32_t* quantized_multiplier,
                                       int* shift) {
  if (double_multiplier == 0.0) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }

  int s = 0;
  /* Bring into [0.5, 1.0). */
  while (double_multiplier < 0.5) {
    double_multiplier *= 2.0;
    s--;
  }
  while (double_multiplier >= 1.0) {
    double_multiplier /= 2.0;
    s++;
  }

  int64_t q = (int64_t)round(double_multiplier * (1LL << 31));
  if (q == (1LL << 31)) {
    q /= 2;
    s++;
  }

  *quantized_multiplier = (int32_t)q;
  *shift = s;
}

/*
 * SaturatingRoundingDoublingHighMul — compute (a * b + rounding) >> 31.
 * Matches gemmlowp::SaturatingRoundingDoublingHighMul.
 */
static inline int32_t sat_round_doubling_high_mul(int32_t a, int32_t b) {
  int64_t ab = (int64_t)a * (int64_t)b;
  int32_t nudge = ab >= 0 ? (1 << 30) : ((1 << 30) - 1);
  return (int32_t)((ab + nudge) >> 31);
}

/*
 * RoundingDivideByPOT — divide by 2^exponent with rounding.
 * Matches gemmlowp::RoundingDivideByPOT.
 */
static inline int32_t rounding_divide_by_pot(int32_t x, int exponent) {
  int32_t mask = (1 << exponent) - 1;
  int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + ((x & mask) > threshold ? 1 : 0);
}

/*
 * MultiplyByQuantizedMultiplier — apply fixed-point multiplier with shift.
 * Matches tflite::MultiplyByQuantizedMultiplier.
 *
 * Computes: round(x * multiplier * 2^shift)
 */
static inline int32_t multiply_by_quantized_multiplier(
    int32_t x, int32_t quantized_multiplier, int shift) {
  int32_t y = sat_round_doubling_high_mul(x, quantized_multiplier);
  if (shift >= 0) {
    /* Left shift with saturation. */
    int64_t result = (int64_t)y * (1LL << shift);
    /* Clamp to int32 range. */
    if (result > INT32_MAX) return INT32_MAX;
    if (result < INT32_MIN) return INT32_MIN;
    return (int32_t)result;
  } else {
    return rounding_divide_by_pot(y, -shift);
  }
}

/*
 * Full requantization pipeline: INT32 accumulator → INT8 output.
 *
 * effective_scale = input_scale * weight_scale / output_scale
 *
 * This function:
 *   1. Decomposes effective_scale → (multiplier, shift) [once per call]
 *   2. Applies fixed-point multiply: result = MBQM(acc, multiplier, shift)
 *   3. Adds output zero point
 *   4. Clamps to [-128, 127]
 */
static inline int8_t requantize(int32_t acc, int32_t out_multiplier,
                                int out_shift, int32_t out_zp) {
  int32_t result =
      multiply_by_quantized_multiplier(acc, out_multiplier, out_shift);
  result += out_zp;
  if (result < -128) result = -128;
  if (result > 127) result = 127;
  return (int8_t)result;
}

/*
 * Apply fused activation in the quantized domain.
 *
 * For RELU: clamp min to the quantized representation of 0.
 *   quantized_zero = out_zp (since 0.0 dequantizes to out_zp in int8)
 *
 * Note: This matches TFLM's activation clamping which operates on the
 * int32 value BEFORE clamping to [-128, 127].
 */
static inline int32_t apply_fused_activation(int32_t val, int fused_activation,
                                             int32_t act_min, int32_t act_max) {
  if (val < act_min) val = act_min;
  if (val > act_max) val = act_max;
  return val;
}

/*
 * Compute activation bounds for the quantized domain.
 * TFLM computes these from the output quant params.
 */
static inline void compute_activation_range(int fused_activation,
                                            int32_t out_zp, float out_scale,
                                            int32_t* act_min,
                                            int32_t* act_max) {
  *act_min = -128;
  *act_max = 127;
  if (fused_activation == 1) { /* RELU */
    /* Quantized zero = round(0.0 / out_scale) + out_zp = out_zp */
    int32_t zero_q = out_zp;
    if (zero_q > *act_min) *act_min = zero_q;
  } else if (fused_activation == 2) { /* RELU6 */
    int32_t zero_q = out_zp;
    int32_t six_q = (int32_t)round(6.0 / (double)out_scale) + out_zp;
    if (zero_q > *act_min) *act_min = zero_q;
    if (six_q < *act_max) *act_max = six_q;
  }
}

#endif /* KERNEL_FIXEDPOINT_H */
