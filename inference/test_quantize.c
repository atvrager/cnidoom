/*
 * test_quantize.c — Unit tests for agent_quantize_state().
 *
 * Verifies INT8 quantization of the 20-float game state vector
 * with various scale/zero_point combinations.
 */

#include <math.h>
#include <stdio.h>
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
/* Tests                                                              */
/* ------------------------------------------------------------------ */

static void test_zero_state(void) {
  /* All-zero state with symmetric quant (scale=1/127, zp=0). */
  agent_game_state_t gs;
  memset(&gs, 0, sizeof(gs));

  int8_t out[20];
  agent_quantize_state(&gs, out, 1.0f / 127.0f, 0);

  for (int i = 0; i < 20; i++) {
    ASSERT_EQ(out[i], 0, "zero state → quantized 0");
  }
}

static void test_max_state(void) {
  /* Health=1.0 with scale=1/127, zp=0 → q = round(1.0 * 127) = 127. */
  agent_game_state_t gs;
  memset(&gs, 0, sizeof(gs));
  gs.health = 1.0f;

  int8_t out[20];
  agent_quantize_state(&gs, out, 1.0f / 127.0f, 0);

  ASSERT_EQ(out[0], 127, "health=1.0 → quantized 127");
}

static void test_negative_clamp(void) {
  /* Value that would underflow INT8 should clamp to -128. */
  agent_game_state_t gs;
  memset(&gs, 0, sizeof(gs));
  gs.health = -2.0f; /* way below normal range */

  int8_t out[20];
  agent_quantize_state(&gs, out, 1.0f / 127.0f, 0);

  ASSERT_EQ(out[0], -128, "underflow clamps to -128");
}

static void test_positive_clamp(void) {
  /* Value that would overflow INT8 should clamp to 127. */
  agent_game_state_t gs;
  memset(&gs, 0, sizeof(gs));
  gs.health = 2.0f; /* above normal range */

  int8_t out[20];
  agent_quantize_state(&gs, out, 1.0f / 127.0f, 0);

  ASSERT_EQ(out[0], 127, "overflow clamps to 127");
}

static void test_asymmetric_quant(void) {
  /* Asymmetric: scale=1/255, zp=-128 (common for [0,1] range).
   * val=0.5 → q = round(0.5/scale) + zp = round(127.5) + (-128) = 128 - 128 =
   * 0. */
  agent_game_state_t gs;
  memset(&gs, 0, sizeof(gs));
  gs.health = 0.5f;

  int8_t out[20];
  agent_quantize_state(&gs, out, 1.0f / 255.0f, -128);

  ASSERT_NEAR(out[0], 0, 1, "asymmetric quant: 0.5 → ~0");
}

static void test_weapon_onehot(void) {
  /* Weapon 3 selected: weapon_onehot[3]=1.0, rest=0.0.
   * With scale=1/127, zp=0: 1.0 → 127, 0.0 → 0. */
  agent_game_state_t gs;
  memset(&gs, 0, sizeof(gs));
  gs.weapon_onehot[3] = 1.0f;

  int8_t out[20];
  agent_quantize_state(&gs, out, 1.0f / 127.0f, 0);

  /* weapon_onehot starts at offset 6 in the struct. */
  ASSERT_EQ(out[6 + 3], 127, "weapon_onehot[3]=1.0 → 127");
  ASSERT_EQ(out[6 + 0], 0, "weapon_onehot[0]=0.0 → 0");
  ASSERT_EQ(out[6 + 8], 0, "weapon_onehot[8]=0.0 → 0");
}

static void test_all_fields(void) {
  /* Smoke test: set every field and verify no crash. */
  agent_game_state_t gs = {
      .health = 0.75f,
      .armor = 0.5f,
      .ammo = {0.1f, 0.2f, 0.3f, 0.4f},
      .weapon_onehot = {0, 0, 0, 0, 0, 1.0f, 0, 0, 0},
      .velocity_xy = {-0.5f, 0.3f},
      ._reserved = {0, 0, 0},
  };

  int8_t out[20];
  agent_quantize_state(&gs, out, 1.0f / 127.0f, 0);

  /* Just verify health rounds correctly. */
  ASSERT_NEAR(out[0], 95, 1, "health=0.75 → ~95");
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
  test_zero_state();
  test_max_state();
  test_negative_clamp();
  test_positive_clamp();
  test_asymmetric_quant();
  test_weapon_onehot();
  test_all_fields();

  printf("%d tests, %d passed, %d failed\n", tests_run,
         tests_run - tests_failed, tests_failed);
  return tests_failed > 0 ? 1 : 0;
}
