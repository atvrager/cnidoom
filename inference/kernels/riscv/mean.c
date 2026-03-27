/*
 * mean.c — RVV-optimized INT8 MEAN (Global Average Pooling) kernel.
 *
 * Reduces [1, H, W, C] → [1, C] by averaging over spatial dimensions.
 *
 * Strategy: For each channel, accumulate H*W int8 values into int32 using
 * RVV widening.  Since the input is NHWC (channels interleaved), we process
 * one channel at a time with stride-C loads, falling back to scalar when
 * channels are small.
 *
 * For GAP the hot loop is the spatial sum — we vectorize across the spatial
 * dimension by loading contiguous int8 values with stride = channels, then
 * widening to int32 and using vredsum.
 *
 * Target: RISC-V RV32IMF_Zve32x_Zve32f, VLEN=128.
 */

#include <riscv_vector.h>

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

  for (int c = 0; c < channels; c++) {
    int32_t acc = 0;

    /*
     * Accumulate across spatial positions for this channel.
     * Input layout is NHWC, so values for channel c are at
     * offsets c, c+C, c+2C, ..., c+(count-1)*C.
     *
     * We use strided loads to gather channel c across spatial positions,
     * widen to int16, subtract zero point, widen to int32, and reduce.
     */
    const int8_t* ch_base = input + c;
    int i = 0;

    vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, 1);

    /* Process 16 spatial positions at a time with strided load. */
    for (; i + 16 <= count; i += 16) {
      /* Strided load: gather 16 int8 values at stride=channels. */
      vint8m1_t vin =
          __riscv_vlse8_v_i8m1(ch_base + i * channels, (ptrdiff_t)channels, 16);

      /* Widen to int16, subtract zero point, widen to int32, reduce. */
      vint16m2_t vin16 = __riscv_vsext_vf2_i16m2(vin, 16);
      vin16 = __riscv_vsub_vx_i16m2(vin16, (int16_t)in_zp, 16);

      vint32m4_t vin32 = __riscv_vsext_vf2_i32m4(vin16, 16);
      vacc = __riscv_vredsum_vs_i32m4_i32m1(vin32, vacc, 16);
    }

    acc = __riscv_vmv_x_s_i32m1_i32(vacc);

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
