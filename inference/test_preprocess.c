/*
 * test_preprocess.c — Unit tests for agent_preprocess_frame().
 *
 * Tests the RGBA → grayscale downsampling and INT8 quantization
 * without needing TFLite Micro or doomgeneric.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "doom_agent.h"

static int tests_run = 0;
static int tests_failed = 0;

#define ASSERT_EQ(a, b, msg)                                                \
  do {                                                                      \
    if ((a) != (b)) {                                                       \
      fprintf(stderr, "FAIL [%s:%d]: %s — expected %d, got %d\n", __FILE__, \
              __LINE__, (msg), (int)(b), (int)(a));                         \
      tests_failed++;                                                       \
    }                                                                       \
    tests_run++;                                                            \
  } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                                          \
  do {                                                                       \
    int diff = (a) - (b);                                                    \
    if (diff < 0) diff = -diff;                                              \
    if (diff > (tol)) {                                                      \
      fprintf(stderr, "FAIL [%s:%d]: %s — expected ~%d, got %d\n", __FILE__, \
              __LINE__, (msg), (int)(b), (int)(a));                          \
      tests_failed++;                                                        \
    }                                                                        \
    tests_run++;                                                             \
  } while (0)

/* ------------------------------------------------------------------ */
/* Helper: create a solid-color RGBA framebuffer.                     */
/* ------------------------------------------------------------------ */

static uint32_t* make_solid_frame(int w, int h, uint8_t r, uint8_t g,
                                  uint8_t b) {
  uint32_t* buf = (uint32_t*)malloc((size_t)w * (size_t)h * sizeof(uint32_t));
  const uint32_t pixel = ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
  for (int i = 0; i < w * h; i++) {
    buf[i] = pixel;
  }
  return buf;
}

/* ------------------------------------------------------------------ */
/* Tests                                                              */
/* ------------------------------------------------------------------ */

static void test_output_shape(void) {
  /* Output should be exactly AGENT_FRAME_H × AGENT_FRAME_W. */
  int8_t out[AGENT_FRAME_H][AGENT_FRAME_W];
  memset(out, 0x42, sizeof(out)); /* fill with sentinel */

  uint32_t* frame = make_solid_frame(320, 200, 128, 128, 128);
  agent_preprocess_frame(frame, 320, 200, out);

  /* Every cell should have been written (no 0x42 sentinels). */
  int untouched = 0;
  for (int y = 0; y < AGENT_FRAME_H; y++)
    for (int x = 0; x < AGENT_FRAME_W; x++)
      if (out[y][x] == 0x42) untouched++;

  ASSERT_EQ(untouched, 0, "all output pixels should be written");
  free(frame);
}

static void test_black_frame(void) {
  /* Pure black (0,0,0) → gray=0 → INT8 = 0 - 128 = -128. */
  int8_t out[AGENT_FRAME_H][AGENT_FRAME_W];
  uint32_t* frame = make_solid_frame(320, 200, 0, 0, 0);
  agent_preprocess_frame(frame, 320, 200, out);

  ASSERT_EQ(out[0][0], -128, "black pixel → INT8 -128");
  ASSERT_EQ(out[AGENT_FRAME_H - 1][AGENT_FRAME_W - 1], -128,
            "black pixel (corner) → INT8 -128");
  free(frame);
}

static void test_white_frame(void) {
  /* Pure white (255,255,255) → gray≈255 → INT8 = 255-128 = 127. */
  int8_t out[AGENT_FRAME_H][AGENT_FRAME_W];
  uint32_t* frame = make_solid_frame(320, 200, 255, 255, 255);
  agent_preprocess_frame(frame, 320, 200, out);

  /* Allow ±1 for rounding in the luma computation. */
  ASSERT_NEAR(out[0][0], 127, 1, "white pixel → INT8 ~127");
  free(frame);
}

static void test_red_frame(void) {
  /* Pure red (255,0,0) → gray = (255*77)>>8 = 76 → INT8 = 76-128 = -52. */
  int8_t out[AGENT_FRAME_H][AGENT_FRAME_W];
  uint32_t* frame = make_solid_frame(320, 200, 255, 0, 0);
  agent_preprocess_frame(frame, 320, 200, out);

  ASSERT_NEAR(out[0][0], -52, 2, "red pixel → INT8 ~-52");
  free(frame);
}

static void test_green_frame(void) {
  /* Pure green (0,255,0) → gray = (255*150)>>8 = 149 → INT8 = 149-128 = 21. */
  int8_t out[AGENT_FRAME_H][AGENT_FRAME_W];
  uint32_t* frame = make_solid_frame(320, 200, 0, 255, 0);
  agent_preprocess_frame(frame, 320, 200, out);

  ASSERT_NEAR(out[0][0], 21, 2, "green pixel → INT8 ~21");
  free(frame);
}

static void test_uniform_output(void) {
  /* A solid-color input should produce a uniform output. */
  int8_t out[AGENT_FRAME_H][AGENT_FRAME_W];
  uint32_t* frame = make_solid_frame(320, 200, 100, 100, 100);
  agent_preprocess_frame(frame, 320, 200, out);

  const int8_t expected = out[0][0];
  int mismatches = 0;
  for (int y = 0; y < AGENT_FRAME_H; y++)
    for (int x = 0; x < AGENT_FRAME_W; x++)
      if (out[y][x] != expected) mismatches++;

  ASSERT_EQ(mismatches, 0, "solid input → uniform output");
  free(frame);
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
  test_output_shape();
  test_black_frame();
  test_white_frame();
  test_red_frame();
  test_green_frame();
  test_uniform_output();

  printf("%d tests, %d passed, %d failed\n", tests_run,
         tests_run - tests_failed, tests_failed);
  return tests_failed > 0 ? 1 : 0;
}
