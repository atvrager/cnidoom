/*
 * kernel_ops.h — Kernel function signatures for static inference engine.
 *
 * All kernels operate on INT8 tensors with per-tensor or per-channel
 * quantization.  Weights are extern const arrays emitted by codegen.
 * Requantization (INT32 accumulator → INT8 output) uses float math
 * in the generic backend and fixed-point in target-specific backends.
 */

#ifndef KERNEL_OPS_H
#define KERNEL_OPS_H

#include <stdint.h>

/* Per-tensor quantization parameters. */
typedef struct {
  float scale;
  int32_t zero_point;
} quant_param_t;

/* Per-channel quantization parameters (conv/FC weights, biases). */
typedef struct {
  const float* scales;
  int num_channels;
} quant_param_per_ch_t;

/* Fused activation function codes. */
#define KERNEL_ACT_NONE 0
#define KERNEL_ACT_RELU 1
#define KERNEL_ACT_RELU6 2

/*
 * Depthwise convolution — INT8, NHWC layout.
 *
 * Filter layout: [1, filt_h, filt_w, channels] (TFLite depthwise format).
 * Bias: int32 per-channel.
 * Padding: explicit top/left/bottom/right pixel counts (SAME computed by
 * codegen).
 */
void kernel_depthwise_conv2d_int8(
    const int8_t* input, int in_h, int in_w, int channels, const int8_t* filter,
    int filt_h, int filt_w, const int32_t* bias, int stride_h, int stride_w,
    int pad_t, int pad_l, int pad_b, int pad_r, const quant_param_t* in_q,
    const quant_param_per_ch_t* filt_q, const quant_param_t* out_q,
    int8_t* output, int out_h, int out_w, int fused_activation);

/*
 * Standard 2D convolution — INT8, NHWC layout.
 *
 * Filter layout: [out_c, filt_h, filt_w, in_c] (TFLite Conv2D format).
 */
void kernel_conv2d_int8(const int8_t* input, int in_h, int in_w, int in_c,
                        const int8_t* filter, int filt_h, int filt_w, int out_c,
                        const int32_t* bias, int stride_h, int stride_w,
                        int pad_t, int pad_l, int pad_b, int pad_r,
                        const quant_param_t* in_q,
                        const quant_param_per_ch_t* filt_q,
                        const quant_param_t* out_q, int8_t* output, int out_h,
                        int out_w, int fused_activation);

/*
 * Fully connected — INT8.
 *
 * Weight layout: [out_features, in_features] (row-major, TFLite FC format).
 */
void kernel_fully_connected_int8(const int8_t* input, int in_features,
                                 const int8_t* weights, int out_features,
                                 const int32_t* bias, const quant_param_t* in_q,
                                 const quant_param_per_ch_t* wt_q,
                                 const quant_param_t* out_q, int8_t* output,
                                 int fused_activation);

/*
 * Element-wise tanh via 256-entry LUT.
 * LUT is indexed by (uint8_t)in[i] — i.e. input interpreted as unsigned.
 */
void kernel_tanh_int8(const int8_t* in, int count, const int8_t lut[256],
                      int8_t* out);

/*
 * Element-wise logistic (sigmoid) via 256-entry LUT.
 */
void kernel_logistic_int8(const int8_t* in, int count, const int8_t lut[256],
                          int8_t* out);

/*
 * Concatenation of two 1-D INT8 vectors along the last axis.
 * Assumes both inputs share the same quantization domain (no requant).
 */
void kernel_concatenation_int8(const int8_t* a, int a_len, const int8_t* b,
                               int b_len, int8_t* out);

#endif /* KERNEL_OPS_H */
