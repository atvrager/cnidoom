/*
 * doom_agent_host.cc — TFLM-based host backend for x86 inference.
 *
 * Loads a .tflite model from file at runtime (unlike doom_agent_tflm.cc
 * which uses a compiled-in model). Uses the same MicroInterpreter —
 * identical code path as the embedded target, just running on x86.
 *
 * Auto-detects input dtype (float32 vs int8) to handle both FP32/FP16
 * and full-integer INT8 models.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "doom_agent.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Larger arena for host — memory is cheap on x86. */
static constexpr int kArenaSize = 256 * 1024;
static uint8_t tensor_arena[kArenaSize];

/* Model data loaded from file. */
static uint8_t* model_data = nullptr;
static size_t model_data_len = 0;

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_visual = nullptr;
static TfLiteTensor* input_state = nullptr;
static TfLiteTensor* output_action = nullptr;

/* Whether the model uses INT8 I/O. */
static bool is_int8 = false;

static int64_t time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

static uint8_t* load_file(const char* path, size_t* out_len) {
  FILE* f = fopen(path, "rb");
  if (f == nullptr) return nullptr;

  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);

  auto* buf = static_cast<uint8_t*>(malloc(len));
  if (buf == nullptr) {
    fclose(f);
    return nullptr;
  }
  if (fread(buf, 1, len, f) != static_cast<size_t>(len)) {
    free(buf);
    fclose(f);
    return nullptr;
  }
  fclose(f);

  *out_len = static_cast<size_t>(len);
  return buf;
}

extern "C" int agent_init_host(const char* model_path) {
  model_data = load_file(model_path, &model_data_len);
  if (model_data == nullptr) {
    fprintf(stderr, "ERROR: Failed to load model from %s\n", model_path);
    return -1;
  }
  fprintf(stderr, "Loaded model: %s (%zu bytes)\n", model_path, model_data_len);

  const tflite::Model* model = tflite::GetModel(model_data);
  if (model == nullptr) {
    fprintf(stderr, "ERROR: Invalid model data\n");
    return -1;
  }

  /* Register ops — superset of what any of our models use. */
  static tflite::MicroMutableOpResolver<16> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddConcatenation();
  resolver.AddLogistic(); /* sigmoid */
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddShape();
  resolver.AddSlice();
  resolver.AddTranspose();
  resolver.AddCast();
  resolver.AddTanh();

  /* Heap-allocate to avoid static destructor issues when doomgeneric
   * calls exit() — the destructor would run after atexit handlers and
   * crash trying to free already-invalid subgraph state. */
  interpreter =
      new tflite::MicroInterpreter(model, resolver, tensor_arena, kArenaSize);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "ERROR: Failed to allocate tensors\n");
    return -1;
  }

  /* Bind input/output tensors by index.
   * After onnx2tf, the SavedModel signature orders alphabetically:
   * index 0 = state (2D), index 1 = visual (4D). */
  input_state = interpreter->input(0);
  input_visual = interpreter->input(1);
  output_action = interpreter->output(0);

  if (input_state == nullptr || input_visual == nullptr ||
      output_action == nullptr) {
    fprintf(stderr, "ERROR: null tensor pointer\n");
    return -1;
  }

  /* Swap if our guess was wrong (visual is 4D, state is 2D). */
  if (input_state->dims->size == 4) {
    TfLiteTensor* tmp = input_state;
    input_state = input_visual;
    input_visual = tmp;
  }

  is_int8 = (input_visual->type == kTfLiteInt8);
  fprintf(stderr, "Host backend: %s model, arena used: %zu/%d bytes\n",
          is_int8 ? "INT8" : "FP32", interpreter->arena_used_bytes(),
          kArenaSize);

  return 0;
}

extern "C" uint8_t agent_infer_host(
    const float frame_nhwc[AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK],
    const float state[20], float out_probs[AGENT_NUM_ACTIONS],
    int64_t* out_inference_us) {
  const int vis_size = AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK;

  /* Copy inputs, quantizing if INT8 model. */
  if (is_int8) {
    const float vis_scale = input_visual->params.scale;
    const int32_t vis_zp = input_visual->params.zero_point;
    for (int i = 0; i < vis_size; i++) {
      int32_t q =
          static_cast<int32_t>(roundf(frame_nhwc[i] / vis_scale)) + vis_zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input_visual->data.int8[i] = static_cast<int8_t>(q);
    }

    const float st_scale = input_state->params.scale;
    const int32_t st_zp = input_state->params.zero_point;
    for (int i = 0; i < 20; i++) {
      int32_t q = static_cast<int32_t>(roundf(state[i] / st_scale)) + st_zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input_state->data.int8[i] = static_cast<int8_t>(q);
    }
  } else {
    std::memcpy(input_visual->data.f, frame_nhwc, vis_size * sizeof(float));
    std::memcpy(input_state->data.f, state, 20 * sizeof(float));
  }

  /* Invoke. */
  int64_t t0 = time_us();
  TfLiteStatus status = interpreter->Invoke();
  int64_t t1 = time_us();

  if (out_inference_us != nullptr) {
    *out_inference_us = t1 - t0;
  }

  if (status != kTfLiteOk) {
    fprintf(stderr, "WARNING: inference failed\n");
    return 0;
  }

  /* Decode output → action probabilities. */
  float probs[AGENT_NUM_ACTIONS];
  if (is_int8) {
    const float out_scale = output_action->params.scale;
    const int32_t out_zp = output_action->params.zero_point;
    for (int i = 0; i < AGENT_NUM_ACTIONS; i++) {
      probs[i] = (output_action->data.int8[i] - out_zp) * out_scale;
    }
  } else {
    std::memcpy(probs, output_action->data.f,
                AGENT_NUM_ACTIONS * sizeof(float));
  }

  if (out_probs != nullptr) {
    std::memcpy(out_probs, probs, AGENT_NUM_ACTIONS * sizeof(float));
  }

  uint8_t actions = 0;
  for (int i = 0; i < AGENT_NUM_ACTIONS; i++) {
    if (probs[i] > 0.5f) {
      actions |= (1u << i);
    }
  }

  return actions;
}

extern "C" void agent_destroy_host() {
  delete interpreter;
  interpreter = nullptr;
  input_visual = nullptr;
  input_state = nullptr;
  output_action = nullptr;

  if (model_data != nullptr) {
    free(model_data);
    model_data = nullptr;
    model_data_len = 0;
  }
}
