/*
 * test_rvv_bitexact.c — Verify RVV kernels are bit-identical to generic C.
 *
 * Runs each kernel with random INT8 inputs through both the generic (scalar)
 * reference and the RVV-optimized implementation, then compares byte-for-byte.
 *
 * Tests every dispatch path with the actual model dimensions:
 *   - Depthwise conv: 4ch (scalar), 16ch (RVV), 32ch (RVV)
 *   - Conv2d 1×1:     in_c=16→out_c=32 (RVV), in_c=32→out_c=32 (RVV)
 *   - Conv2d 3×3:     in_c=4→out_c=16 (scalar fallback)
 *   - FC:             1556→256, 256→64, 64→6 (all RVV)
 *   - Tanh LUT:       count=64 (RVV)
 *   - Logistic LUT:   count=6 (RVV)
 *
 * Build (as part of the rv32 target):
 *   cmake -B build-rv32 -S inference \
 *     -DCMAKE_TOOLCHAIN_FILE=../cmake/riscv32-elf-clang.cmake \
 *     -DDOOM_AGENT_STATIC=ON -DDOOM_AGENT_RV32=ON \
 *     -DDOOM_AGENT_KERNEL_TARGET=riscv
 *   cmake --build build-rv32 --target test_rvv_bitexact
 *
 * Run (under QEMU):
 *   qemu-system-riscv32 -machine virt \
 *     -cpu rv32,v=true,vlen=128,zve32f=true -m 128M \
 *     -nographic -bios none \
 *     -semihosting-config enable=on,target=native \
 *     -kernel build-rv32/test_rvv_bitexact.elf
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kernels/kernel_ops.h"

/* Reference (generic C) kernel declarations. */
extern void ref_depthwise_conv2d_int8(
    const int8_t* input, int in_h, int in_w, int channels, const int8_t* filter,
    int filt_h, int filt_w, const int32_t* bias, int stride_h, int stride_w,
    int pad_t, int pad_l, int pad_b, int pad_r, const quant_param_t* in_q,
    const quant_param_per_ch_t* filt_q, const quant_param_t* out_q,
    int8_t* output, int out_h, int out_w, int fused_activation);

extern void ref_conv2d_int8(const int8_t* input, int in_h, int in_w, int in_c,
                            const int8_t* filter, int filt_h, int filt_w,
                            int out_c, const int32_t* bias, int stride_h,
                            int stride_w, int pad_t, int pad_l, int pad_b,
                            int pad_r, const quant_param_t* in_q,
                            const quant_param_per_ch_t* filt_q,
                            const quant_param_t* out_q, int8_t* output,
                            int out_h, int out_w, int fused_activation);

extern void ref_fully_connected_int8(const int8_t* input, int in_features,
                                     const int8_t* weights, int out_features,
                                     const int32_t* bias,
                                     const quant_param_t* in_q,
                                     const quant_param_per_ch_t* wt_q,
                                     const quant_param_t* out_q, int8_t* output,
                                     int fused_activation);

extern void ref_tanh_int8(const int8_t* in, int count, const int8_t lut[256],
                          int8_t* out);

extern void ref_logistic_int8(const int8_t* in, int count,
                              const int8_t lut[256], int8_t* out);

/* ------------------------------------------------------------------ */
/* PRNG (xoshiro128+ — same as test_static_vs_tflm)                   */
/* ------------------------------------------------------------------ */

static uint32_t rng_s[4];

static void rng_seed(uint32_t seed) {
  rng_s[0] = seed;
  rng_s[1] = seed ^ 0x12345678u;
  rng_s[2] = seed ^ 0xDEADBEEFu;
  rng_s[3] = seed ^ 0xCAFEBABEu;
}

static uint32_t rng_next(void) {
  uint32_t t = rng_s[1] << 9;
  rng_s[2] ^= rng_s[0];
  rng_s[3] ^= rng_s[1];
  rng_s[1] ^= rng_s[2];
  rng_s[0] ^= rng_s[3];
  rng_s[2] ^= t;
  rng_s[3] = (rng_s[3] << 11) | (rng_s[3] >> 21);
  return rng_s[0] + rng_s[3];
}

static int8_t rng_int8(void) { return (int8_t)(rng_next() & 0xFF); }

static void fill_random_i8(int8_t* buf, int count) {
  for (int i = 0; i < count; i++) buf[i] = rng_int8();
}

static void fill_random_i32(int32_t* buf, int count) {
  for (int i = 0; i < count; i++) {
    /* Bias values: keep them small-ish to be realistic. */
    buf[i] = (int32_t)(rng_next() % 2001) - 1000;
  }
}

/* ------------------------------------------------------------------ */
/* Comparison helper                                                   */
/* ------------------------------------------------------------------ */

static int compare_outputs(const int8_t* ref, const int8_t* rvv, int count,
                           const char* name) {
  int mismatches = 0;
  for (int i = 0; i < count; i++) {
    if (ref[i] != rvv[i]) {
      if (mismatches < 5) {
        printf("  MISMATCH %s[%d]: ref=%d rvv=%d\n", name, i, ref[i], rvv[i]);
      }
      mismatches++;
    }
  }
  return mismatches;
}

/* ------------------------------------------------------------------ */
/* Per-channel quantization parameter helpers                         */
/* ------------------------------------------------------------------ */

static void make_quant_params(float in_scale, int32_t in_zp, float out_scale,
                              int32_t out_zp, const float* ch_scales,
                              int num_ch, quant_param_t* in_q,
                              quant_param_per_ch_t* filt_q,
                              quant_param_t* out_q) {
  in_q->scale = in_scale;
  in_q->zero_point = in_zp;
  out_q->scale = out_scale;
  out_q->zero_point = out_zp;
  filt_q->scales = ch_scales;
  filt_q->num_channels = num_ch;
}

/* ------------------------------------------------------------------ */
/* Depthwise conv test (one trial per config)                         */
/* ------------------------------------------------------------------ */

/* Maximum buffer sizes across all test configs. */
#define MAX_DW_IN (45 * 60 * 32)
#define MAX_DW_FILT (3 * 3 * 32)
#define MAX_DW_OUT (23 * 30 * 32)

static int8_t dw_input[MAX_DW_IN];
static int8_t dw_filter[MAX_DW_FILT];
static int32_t dw_bias[32];
static float dw_ch_scales[32];
static int8_t dw_ref_out[MAX_DW_OUT];
static int8_t dw_rvv_out[MAX_DW_OUT];

static int test_depthwise_conv(int in_h, int in_w, int channels, int filt_h,
                               int filt_w, int stride_h, int stride_w,
                               int pad_t, int pad_l, int pad_b, int pad_r,
                               int out_h, int out_w, int num_trials) {
  int total_fail = 0;

  for (int trial = 0; trial < num_trials; trial++) {
    int in_size = in_h * in_w * channels;
    int filt_size = filt_h * filt_w * channels;
    int out_size = out_h * out_w * channels;

    fill_random_i8(dw_input, in_size);
    fill_random_i8(dw_filter, filt_size);
    fill_random_i32(dw_bias, channels);

    for (int c = 0; c < channels; c++) {
      dw_ch_scales[c] = 0.001f + (float)(rng_next() % 100) * 0.0001f;
    }

    quant_param_t in_q, out_q;
    quant_param_per_ch_t filt_q;
    make_quant_params(0.0078f, -1, 0.0039f, 0, dw_ch_scales, channels, &in_q,
                      &filt_q, &out_q);

    memset(dw_ref_out, 0xAA, (size_t)out_size);
    memset(dw_rvv_out, 0xBB, (size_t)out_size);

    ref_depthwise_conv2d_int8(dw_input, in_h, in_w, channels, dw_filter, filt_h,
                              filt_w, dw_bias, stride_h, stride_w, pad_t, pad_l,
                              pad_b, pad_r, &in_q, &filt_q, &out_q, dw_ref_out,
                              out_h, out_w, KERNEL_ACT_NONE);

    kernel_depthwise_conv2d_int8(
        dw_input, in_h, in_w, channels, dw_filter, filt_h, filt_w, dw_bias,
        stride_h, stride_w, pad_t, pad_l, pad_b, pad_r, &in_q, &filt_q, &out_q,
        dw_rvv_out, out_h, out_w, KERNEL_ACT_NONE);

    char name[64];
    snprintf(name, sizeof(name), "dw_%dch_t%d", channels, trial);
    int m = compare_outputs(dw_ref_out, dw_rvv_out, out_size, name);
    if (m > 0) total_fail++;
  }

  return total_fail;
}

/* ------------------------------------------------------------------ */
/* Conv2d test                                                         */
/* ------------------------------------------------------------------ */

#define MAX_CONV_IN (23 * 30 * 32)
#define MAX_CONV_FILT (32 * 3 * 3 * 32)
#define MAX_CONV_OUT (23 * 30 * 32)

static int8_t conv_input[MAX_CONV_IN];
static int8_t conv_filter[MAX_CONV_FILT];
static int32_t conv_bias[32];
static float conv_ch_scales[32];
static int8_t conv_ref_out[MAX_CONV_OUT];
static int8_t conv_rvv_out[MAX_CONV_OUT];

static int test_conv2d(int in_h, int in_w, int in_c, int filt_h, int filt_w,
                       int out_c, int stride_h, int stride_w, int pad_t,
                       int pad_l, int pad_b, int pad_r, int out_h, int out_w,
                       int num_trials) {
  int total_fail = 0;

  for (int trial = 0; trial < num_trials; trial++) {
    int in_size = in_h * in_w * in_c;
    int filt_size = out_c * filt_h * filt_w * in_c;
    int out_size = out_h * out_w * out_c;

    fill_random_i8(conv_input, in_size);
    fill_random_i8(conv_filter, filt_size);
    fill_random_i32(conv_bias, out_c);

    for (int c = 0; c < out_c; c++) {
      conv_ch_scales[c] = 0.001f + (float)(rng_next() % 100) * 0.0001f;
    }

    quant_param_t in_q, out_q;
    quant_param_per_ch_t filt_q;
    make_quant_params(0.0078f, -1, 0.0039f, 0, conv_ch_scales, out_c, &in_q,
                      &filt_q, &out_q);

    memset(conv_ref_out, 0xAA, (size_t)out_size);
    memset(conv_rvv_out, 0xBB, (size_t)out_size);

    ref_conv2d_int8(conv_input, in_h, in_w, in_c, conv_filter, filt_h, filt_w,
                    out_c, conv_bias, stride_h, stride_w, pad_t, pad_l, pad_b,
                    pad_r, &in_q, &filt_q, &out_q, conv_ref_out, out_h, out_w,
                    KERNEL_ACT_NONE);

    kernel_conv2d_int8(conv_input, in_h, in_w, in_c, conv_filter, filt_h,
                       filt_w, out_c, conv_bias, stride_h, stride_w, pad_t,
                       pad_l, pad_b, pad_r, &in_q, &filt_q, &out_q,
                       conv_rvv_out, out_h, out_w, KERNEL_ACT_NONE);

    char name[64];
    snprintf(name, sizeof(name), "conv_%dx%d_%din_%dout_t%d", filt_h, filt_w,
             in_c, out_c, trial);
    int m = compare_outputs(conv_ref_out, conv_rvv_out, out_size, name);
    if (m > 0) total_fail++;
  }

  return total_fail;
}

/* ------------------------------------------------------------------ */
/* Fully connected test                                                */
/* ------------------------------------------------------------------ */

#define MAX_FC_IN 1556
#define MAX_FC_WT (256 * 1556)
#define MAX_FC_OUT 256

static int8_t fc_input[MAX_FC_IN];
static int8_t fc_weights[MAX_FC_WT];
static int32_t fc_bias_buf[MAX_FC_OUT];
static float fc_ch_scales[MAX_FC_OUT];
static int8_t fc_ref_out[MAX_FC_OUT];
static int8_t fc_rvv_out[MAX_FC_OUT];

static int test_fully_connected(int in_features, int out_features,
                                int num_trials) {
  int total_fail = 0;

  for (int trial = 0; trial < num_trials; trial++) {
    fill_random_i8(fc_input, in_features);
    fill_random_i8(fc_weights, in_features * out_features);
    fill_random_i32(fc_bias_buf, out_features);

    for (int c = 0; c < out_features; c++) {
      fc_ch_scales[c] = 0.001f + (float)(rng_next() % 100) * 0.0001f;
    }

    quant_param_t in_q, out_q;
    quant_param_per_ch_t wt_q;
    make_quant_params(0.0078f, -1, 0.0039f, 0, fc_ch_scales, out_features,
                      &in_q, &wt_q, &out_q);

    memset(fc_ref_out, 0xAA, (size_t)out_features);
    memset(fc_rvv_out, 0xBB, (size_t)out_features);

    ref_fully_connected_int8(fc_input, in_features, fc_weights, out_features,
                             fc_bias_buf, &in_q, &wt_q, &out_q, fc_ref_out,
                             KERNEL_ACT_NONE);

    kernel_fully_connected_int8(fc_input, in_features, fc_weights, out_features,
                                fc_bias_buf, &in_q, &wt_q, &out_q, fc_rvv_out,
                                KERNEL_ACT_NONE);

    char name[64];
    snprintf(name, sizeof(name), "fc_%d_%d_t%d", in_features, out_features,
             trial);
    int m = compare_outputs(fc_ref_out, fc_rvv_out, out_features, name);
    if (m > 0) total_fail++;
  }

  return total_fail;
}

/* ------------------------------------------------------------------ */
/* LUT kernel tests                                                    */
/* ------------------------------------------------------------------ */

static int8_t lut_table[256];
static int8_t lut_input[256];
static int8_t lut_ref_out[256];
static int8_t lut_rvv_out[256];

static int test_tanh_lut(int count, int num_trials) {
  int total_fail = 0;

  for (int trial = 0; trial < num_trials; trial++) {
    fill_random_i8(lut_table, 256);
    fill_random_i8(lut_input, count);

    memset(lut_ref_out, 0xAA, (size_t)count);
    memset(lut_rvv_out, 0xBB, (size_t)count);

    ref_tanh_int8(lut_input, count, lut_table, lut_ref_out);
    kernel_tanh_int8(lut_input, count, lut_table, lut_rvv_out);

    char name[32];
    snprintf(name, sizeof(name), "tanh_%d_t%d", count, trial);
    int m = compare_outputs(lut_ref_out, lut_rvv_out, count, name);
    if (m > 0) total_fail++;
  }

  return total_fail;
}

static int test_logistic_lut(int count, int num_trials) {
  int total_fail = 0;

  for (int trial = 0; trial < num_trials; trial++) {
    fill_random_i8(lut_table, 256);
    fill_random_i8(lut_input, count);

    memset(lut_ref_out, 0xAA, (size_t)count);
    memset(lut_rvv_out, 0xBB, (size_t)count);

    ref_logistic_int8(lut_input, count, lut_table, lut_ref_out);
    kernel_logistic_int8(lut_input, count, lut_table, lut_rvv_out);

    char name[32];
    snprintf(name, sizeof(name), "logistic_%d_t%d", count, trial);
    int m = compare_outputs(lut_ref_out, lut_rvv_out, count, name);
    if (m > 0) total_fail++;
  }

  return total_fail;
}

/* ------------------------------------------------------------------ */
/* Main test driver                                                    */
/* ------------------------------------------------------------------ */

#define TRIALS 10

int main(void) {
  rng_seed(42);

  int total_tests = 0;
  int total_fails = 0;
  int f;

  printf("=== RVV vs Generic Bit-Accuracy Test ===\n\n");

  /* ---- Depthwise conv ---- */

  /* Node 0: DW 4ch 3×3 s2 pad1 (scalar path). */
  printf("DW_CONV 4ch  3x3 s2 [45x60] -> [23x30] ... ");
  f = test_depthwise_conv(45, 60, 4, 3, 3, 2, 2, 1, 1, 1, 1, 23, 30, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Node 2: DW 16ch 3×3 s2 pad1 (RVV 16ch path). */
  printf("DW_CONV 16ch 3x3 s2 [23x30] -> [12x15] ... ");
  f = test_depthwise_conv(23, 30, 16, 3, 3, 2, 2, 1, 1, 1, 1, 12, 15, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Node 4: DW 32ch 3×3 s2 pad(1,0,1,1) → [6x8] (RVV 32ch path). */
  printf("DW_CONV 32ch 3x3 s2 [12x15] -> [6x8]   ... ");
  f = test_depthwise_conv(12, 15, 32, 3, 3, 2, 2, 1, 1, 0, 1, 6, 8, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* ---- Conv2d ---- */

  /* Node 1: 1×1 in_c=4 out_c=16 (scalar fallback — in_c < 16). */
  printf("CONV    1x1  4in 16out [23x30]           ... ");
  f = test_conv2d(23, 30, 4, 1, 1, 16, 1, 1, 0, 0, 0, 0, 23, 30, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Node 3: 1×1 in_c=16 out_c=32 (RVV pointwise). */
  printf("CONV    1x1 16in 32out [12x15]           ... ");
  f = test_conv2d(12, 15, 16, 1, 1, 32, 1, 1, 0, 0, 0, 0, 12, 15, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Node 5: 1×1 in_c=32 out_c=32 (RVV pointwise). */
  printf("CONV    1x1 32in 32out [6x8]             ... ");
  f = test_conv2d(6, 8, 32, 1, 1, 32, 1, 1, 0, 0, 0, 0, 6, 8, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* ---- Fully connected ---- */

  /* Node 7: 1556→256 (dominant — 63% of MACs). */
  printf("FC      1556 -> 256                       ... ");
  f = test_fully_connected(1556, 256, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Node 8: 256→64. */
  printf("FC      256 -> 64                         ... ");
  f = test_fully_connected(256, 64, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Node 10: 64→6. */
  printf("FC      64 -> 6                           ... ");
  f = test_fully_connected(64, 6, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* ---- LUT kernels ---- */

  /* Node 9: tanh count=64. */
  printf("TANH    count=64                          ... ");
  f = test_tanh_lut(64, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Node 11: logistic count=6. */
  printf("LOGISTIC count=6                          ... ");
  f = test_logistic_lut(6, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* ---- Also test with fused RELU activation ---- */
  printf("\n--- With fused RELU activation ---\n");

  /* DW 16ch with RELU. */
  printf("DW_CONV 16ch 3x3 s2 RELU                 ... ");
  {
    int fail_count = 0;
    for (int trial = 0; trial < TRIALS; trial++) {
      int in_size = 23 * 30 * 16;
      int filt_size = 3 * 3 * 16;
      int out_size = 12 * 15 * 16;

      fill_random_i8(dw_input, in_size);
      fill_random_i8(dw_filter, filt_size);
      fill_random_i32(dw_bias, 16);
      for (int c = 0; c < 16; c++)
        dw_ch_scales[c] = 0.001f + (float)(rng_next() % 100) * 0.0001f;

      quant_param_t in_q, out_q;
      quant_param_per_ch_t filt_q;
      make_quant_params(0.0078f, -1, 0.0039f, 0, dw_ch_scales, 16, &in_q,
                        &filt_q, &out_q);

      memset(dw_ref_out, 0xAA, (size_t)out_size);
      memset(dw_rvv_out, 0xBB, (size_t)out_size);

      ref_depthwise_conv2d_int8(dw_input, 23, 30, 16, dw_filter, 3, 3, dw_bias,
                                2, 2, 1, 1, 1, 1, &in_q, &filt_q, &out_q,
                                dw_ref_out, 12, 15, KERNEL_ACT_RELU);
      kernel_depthwise_conv2d_int8(dw_input, 23, 30, 16, dw_filter, 3, 3,
                                   dw_bias, 2, 2, 1, 1, 1, 1, &in_q, &filt_q,
                                   &out_q, dw_rvv_out, 12, 15, KERNEL_ACT_RELU);

      char name[64];
      snprintf(name, sizeof(name), "dw_16ch_relu_t%d", trial);
      if (compare_outputs(dw_ref_out, dw_rvv_out, out_size, name) > 0)
        fail_count++;
    }
    f = fail_count;
  }
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* FC 256→64 with RELU. */
  printf("FC      256 -> 64 RELU                    ... ");
  {
    int fail_count = 0;
    for (int trial = 0; trial < TRIALS; trial++) {
      fill_random_i8(fc_input, 256);
      fill_random_i8(fc_weights, 256 * 64);
      fill_random_i32(fc_bias_buf, 64);
      for (int c = 0; c < 64; c++)
        fc_ch_scales[c] = 0.001f + (float)(rng_next() % 100) * 0.0001f;

      quant_param_t in_q, out_q;
      quant_param_per_ch_t wt_q;
      make_quant_params(0.0078f, -1, 0.0039f, 0, fc_ch_scales, 64, &in_q, &wt_q,
                        &out_q);

      memset(fc_ref_out, 0xAA, 64);
      memset(fc_rvv_out, 0xBB, 64);

      ref_fully_connected_int8(fc_input, 256, fc_weights, 64, fc_bias_buf,
                               &in_q, &wt_q, &out_q, fc_ref_out,
                               KERNEL_ACT_RELU);
      kernel_fully_connected_int8(fc_input, 256, fc_weights, 64, fc_bias_buf,
                                  &in_q, &wt_q, &out_q, fc_rvv_out,
                                  KERNEL_ACT_RELU);

      char name[64];
      snprintf(name, sizeof(name), "fc_256_64_relu_t%d", trial);
      if (compare_outputs(fc_ref_out, fc_rvv_out, 64, name) > 0) fail_count++;
    }
    f = fail_count;
  }
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* ---- Summary ---- */
  printf("\n=== SUMMARY: %d/%d tests passed", total_tests - total_fails,
         total_tests);
  if (total_fails > 0) {
    printf(" (%d FAILED) ===\n", total_fails);
  } else {
    printf(" — ALL BIT-EXACT ===\n");
  }

  return total_fails > 0 ? 1 : 0;
}
