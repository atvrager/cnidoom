/*
 * doom_agent.c — RL agent inference orchestration.
 *
 * Manages the frame stack, calls preprocessing (doom_agent_preprocess.c),
 * and delegates to the TFLM bridge (doom_agent_tflm.cc) for inference.
 */

#include "doom_agent.h"

#include <string.h>

/* Frame stack: 4 most recent preprocessed frames (INT8, quantized). */
static int8_t frame_stack[AGENT_FRAME_STACK][AGENT_FRAME_H][AGENT_FRAME_W];
static int frame_count = 0;

/* ------------------------------------------------------------------ */
/* TFLM bridge — implemented in doom_agent_tflm.cc                    */
/* ------------------------------------------------------------------ */

/*
 * tflm_init: load model, allocate tensors. Returns 0 on success.
 * tflm_get_state_quant: retrieve the state input tensor's quantization
 *                       params so we can quantize on the C side.
 * tflm_infer: copy pre-quantized data into input tensors, invoke the
 *             model, decode output to action bitfield.
 * tflm_destroy: tear down (no-op on bare metal).
 */
extern int tflm_init(void);
extern void tflm_get_state_quant(float* scale, int32_t* zero_point);
extern uint8_t tflm_infer(
    const int8_t frame_nhwc[AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK],
    const int8_t state_q[20]);
extern void tflm_destroy(void);

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

int agent_init(void) {
  memset(frame_stack, -128, sizeof(frame_stack)); /* quantized zero */
  frame_count = 0;
  return tflm_init();
}

uint8_t agent_infer(const uint32_t* framebuf, int fb_width, int fb_height,
                    const agent_game_state_t* gs) {
  /* 1. Preprocess new frame into the ring buffer. */
  const int slot = frame_count % AGENT_FRAME_STACK;
  agent_preprocess_frame(framebuf, fb_width, fb_height, frame_stack[slot]);
  frame_count++;

  /* 2. Flatten frame stack to NHWC layout for the model.
   *    Model expects (45, 60, 4) — height, width, channels.
   *    frame_stack is [channel][height][width]. */
  int8_t nhwc[AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK];
  for (int y = 0; y < AGENT_FRAME_H; y++) {
    for (int x = 0; x < AGENT_FRAME_W; x++) {
      for (int c = 0; c < AGENT_FRAME_STACK; c++) {
        nhwc[y * AGENT_FRAME_W * AGENT_FRAME_STACK + x * AGENT_FRAME_STACK +
             c] = frame_stack[c][y][x];
      }
    }
  }

  /* 3. Quantize game state using model's input tensor params. */
  float state_scale;
  int32_t state_zp;
  tflm_get_state_quant(&state_scale, &state_zp);

  int8_t state_q[20];
  agent_quantize_state(gs, state_q, state_scale, state_zp);

  /* 4. Run inference → action bitfield. */
  return tflm_infer(nhwc, state_q);
}

void agent_destroy(void) { tflm_destroy(); }
