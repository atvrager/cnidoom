/*
 * test_x86_bitexact.c — Verify AVX2 kernels are bit-identical to generic C.
 *
 * Same approach as test_rvv_bitexact.c but for the x86 AVX2 backend.
 * Runs natively on the host — no QEMU needed.
 *
 * Tests the MEAN kernel with V2 model dimensions:
 *   - MEAN 8×10×128 (V2 Block 4 output)
 *   - MEAN 4×5×192  (V2 Block 6 output)
 *   - MEAN 1×1×64   (edge case)
 *
 * Build:
 *   cmake -B build-x86 -S inference \
 *     -DDOOM_AGENT_STATIC=ON -DDOOM_AGENT_KERNEL_TARGET=x86
 *   cmake --build build-x86 --target test_x86_bitexact
 *
 * Run:
 *   ./build-x86/test_x86_bitexact
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kernels/kernel_ops.h"

/* Reference (generic C) kernel declarations. */
extern void ref_mean_int8(const int8_t* input, int height, int width,
                          int channels, const quant_param_t* in_q,
                          const quant_param_t* out_q, int8_t* output);

/* ------------------------------------------------------------------ */
/* PRNG (xoshiro128+)                                                  */
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

/* ------------------------------------------------------------------ */
/* Comparison helper                                                   */
/* ------------------------------------------------------------------ */

static int compare_outputs(const int8_t* ref, const int8_t* opt, int count,
                           const char* name) {
  int mismatches = 0;
  for (int i = 0; i < count; i++) {
    if (ref[i] != opt[i]) {
      if (mismatches < 5) {
        printf("  MISMATCH %s[%d]: ref=%d avx2=%d\n", name, i, ref[i], opt[i]);
      }
      mismatches++;
    }
  }
  return mismatches;
}

/* ------------------------------------------------------------------ */
/* MEAN test                                                           */
/* ------------------------------------------------------------------ */

#define MAX_MEAN_IN (15 * 20 * 192)
#define MAX_MEAN_OUT 192

static int8_t mean_input[MAX_MEAN_IN];
static int8_t mean_ref_out[MAX_MEAN_OUT];
static int8_t mean_avx2_out[MAX_MEAN_OUT];

static int test_mean(int height, int width, int channels, int num_trials) {
  int total_fail = 0;

  for (int trial = 0; trial < num_trials; trial++) {
    int in_size = height * width * channels;
    fill_random_i8(mean_input, in_size);

    quant_param_t in_q = {.scale = 0.0078f, .zero_point = -1};
    quant_param_t out_q = {.scale = 0.0039f, .zero_point = 0};

    memset(mean_ref_out, 0xAA, (size_t)channels);
    memset(mean_avx2_out, 0xBB, (size_t)channels);

    ref_mean_int8(mean_input, height, width, channels, &in_q, &out_q,
                  mean_ref_out);
    kernel_mean_int8(mean_input, height, width, channels, &in_q, &out_q,
                     mean_avx2_out);

    char name[64];
    snprintf(name, sizeof(name), "mean_%dx%dx%d_t%d", height, width, channels,
             trial);
    int m = compare_outputs(mean_ref_out, mean_avx2_out, channels, name);
    if (m > 0) total_fail++;
  }

  return total_fail;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

#define TRIALS 10

int main(void) {
  rng_seed(42);

  int total_tests = 0;
  int total_fails = 0;
  int f;

  printf("=== AVX2 vs Generic Bit-Accuracy Test ===\n\n");

  /* V2 Block 4 output: 8×10×128. */
  printf("MEAN    8x10x128                            ... ");
  f = test_mean(8, 10, 128, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* V2 Block 6 output: 4×5×192. */
  printf("MEAN    4x5x192                             ... ");
  f = test_mean(4, 5, 192, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Small edge case: 1×1×64. */
  printf("MEAN    1x1x64                              ... ");
  f = test_mean(1, 1, 64, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Larger spatial for baseline comparison. */
  printf("MEAN    15x20x64                            ... ");
  f = test_mean(15, 20, 64, TRIALS);
  printf("%s (%d/%d)\n", f == 0 ? "PASS" : "FAIL", TRIALS - f, TRIALS);
  total_tests += TRIALS;
  total_fails += f;

  /* Summary. */
  printf("\n=== SUMMARY: %d/%d tests passed", total_tests - total_fails,
         total_tests);
  if (total_fails > 0) {
    printf(" (%d FAILED) ===\n", total_fails);
  } else {
    printf(" — ALL BIT-EXACT ===\n");
  }

  return total_fails > 0 ? 1 : 0;
}
