/*
 * mean.c — Reference INT8 MEAN (Global Average Pooling) kernel (generic C).
 *
 * Reduces [1, H, W, C] → [1, C] by averaging over the spatial dimensions.
 * Per-tensor quantization on both input and output.
 *
 * Accumulates in INT32, then requantizes to output scale/zero_point using
 * the same fixed-point pipeline as other kernels.
 */

#include "../kernel_fixedpoint.h"
#include "../kernel_ops.h"

void kernel_mean_int8(const int8_t* input, int height, int width, int channels,
                      const quant_param_t* in_q, const quant_param_t* out_q,
                      int8_t* output) {
  const int32_t in_zp = in_q->zero_point;
  const int32_t out_zp = out_q->zero_point;
  const int count = height * width;

  /* Effective scale: input_scale / (H * W) / output_scale.
   * The division by count is the "weight" of the mean operation. */
  double eff_scale =
      (double)in_q->scale / ((double)count * (double)out_q->scale);
  int32_t multiplier;
  int shift;
  quantize_multiplier(eff_scale, &multiplier, &shift);

  for (int c = 0; c < channels; c++) {
    int32_t acc = 0;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        acc += (int32_t)input[(h * width + w) * channels + c] - in_zp;
      }
    }

    int32_t result = multiply_by_quantized_multiplier(acc, multiplier, shift);
    result += out_zp;

    if (result < -128) result = -128;
    if (result > 127) result = 127;

    output[c] = (int8_t)result;
  }
}
