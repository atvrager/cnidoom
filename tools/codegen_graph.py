#!/usr/bin/env python3
"""
codegen_graph.py — Compile a TFLite INT8 model into static C inference code.

Reads the model flatbuffer, extracts weights/quant params, performs
TRANSPOSE elimination and liveness-based scratch allocation, then emits:
  - doom_agent_weights.h / .c  (const weight arrays in .rodata)
  - doom_agent_graph.h / .c    (run_graph() with sequential kernel calls)

Usage:
    uv run python tools/codegen_graph.py \\
        --model models/doom_agent_int8.tflite \\
        --output-dir inference/generated
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# TFLite FlatBuffer parsing (via tflite-micro submodule schema)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def _add_tflm_to_path() -> None:
    """Add tflite-micro submodule to sys.path for schema imports."""
    tflm_dir = _REPO_ROOT / "tflite-micro"
    if tflm_dir.exists() and str(tflm_dir) not in sys.path:
        sys.path.insert(0, str(tflm_dir))


_add_tflm_to_path()

from tensorflow.lite.python import schema_py_generated as tflite_schema  # noqa: E402

# TFLite builtin operator codes.
OP_CONV_2D = 3
OP_DEPTHWISE_CONV_2D = 4
OP_FULLY_CONNECTED = 9
OP_LOGISTIC = 14
OP_RESHAPE = 22
OP_CONCATENATION = 2
OP_TANH = 28
OP_MEAN = 25
OP_TRANSPOSE = 39

OP_NAMES = {
    OP_CONV_2D: "CONV_2D",
    OP_DEPTHWISE_CONV_2D: "DEPTHWISE_CONV_2D",
    OP_FULLY_CONNECTED: "FULLY_CONNECTED",
    OP_LOGISTIC: "LOGISTIC",
    OP_RESHAPE: "RESHAPE",
    OP_CONCATENATION: "CONCATENATION",
    OP_MEAN: "MEAN",
    OP_TANH: "TANH",
    OP_TRANSPOSE: "TRANSPOSE",
}

# TFLite tensor type codes.
TFLITE_INT8 = 9
TFLITE_INT32 = 2

# TFLite padding enum.
PADDING_SAME = 0
PADDING_VALID = 1

# Fused activation.
ACT_NONE = 0
ACT_RELU = 1
ACT_RELU6 = 3

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuantParam:
    scale: float
    zero_point: int


@dataclass
class QuantParamPerCh:
    scales: np.ndarray  # float32
    zero_points: np.ndarray  # int64


@dataclass
class TensorInfo:
    index: int
    name: str
    shape: list[int]
    dtype: int
    buffer_index: int
    data: np.ndarray | None  # None for activation tensors
    quant: QuantParam | None
    quant_per_ch: QuantParamPerCh | None

    @property
    def size_bytes(self) -> int:
        s = 1
        for d in self.shape:
            s *= d
        return s

    @property
    def is_const(self) -> bool:
        return self.data is not None


@dataclass
class OpInfo:
    index: int
    opcode: int
    inputs: list[int]
    outputs: list[int]
    stride_h: int = 1
    stride_w: int = 1
    padding: int = PADDING_SAME
    fused_activation: int = ACT_NONE
    depth_multiplier: int = 1
    axis: int = 0


@dataclass
class ModelGraph:
    tensors: list[TensorInfo]
    ops: list[OpInfo]
    inputs: list[int]
    outputs: list[int]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _np_dtype(tflite_type: int) -> np.dtype:
    return {TFLITE_INT8: np.int8, TFLITE_INT32: np.int32}.get(tflite_type, np.uint8)


def parse_model(model_path: str) -> ModelGraph:
    """Parse a TFLite model file into our intermediate representation."""
    buf = bytearray(Path(model_path).read_bytes())
    model = tflite_schema.ModelT.InitFromPackedBuf(buf, 0)
    sg = model.subgraphs[0]
    op_codes = model.operatorCodes

    tensors: list[TensorInfo] = []
    for i, t in enumerate(sg.tensors):
        shape = [int(x) for x in t.shape] if t.shape is not None else []
        name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)

        # Buffer data.
        buf_data = model.buffers[t.buffer].data
        data = None
        if buf_data is not None and len(buf_data) > 0:
            data = np.frombuffer(bytes(buf_data), dtype=_np_dtype(t.type)).copy()
            expected = 1
            for d in shape:
                expected *= d
            if len(data) == expected:
                data = data.reshape(shape)

        # Quantization.
        q = t.quantization
        quant = None
        quant_per_ch = None
        if q is not None and q.scale is not None and len(q.scale) > 0:
            scales = np.array(q.scale, dtype=np.float32)
            zps = np.array(q.zeroPoint, dtype=np.int64)
            if len(scales) == 1:
                quant = QuantParam(float(scales[0]), int(zps[0]))
            else:
                quant_per_ch = QuantParamPerCh(scales, zps)
                # Also store per-tensor (first scale) for convenience.
                quant = QuantParam(float(scales[0]), int(zps[0]))

        tensors.append(
            TensorInfo(
                index=i,
                name=name,
                shape=shape,
                dtype=t.type,
                buffer_index=t.buffer,
                data=data,
                quant=quant,
                quant_per_ch=quant_per_ch,
            )
        )

    ops: list[OpInfo] = []
    for i, op in enumerate(sg.operators):
        oc = op_codes[op.opcodeIndex]
        code = (
            int(oc.builtinCode)
            if oc.builtinCode != 0
            else int(oc.deprecatedBuiltinCode)
        )
        inputs = [int(x) for x in op.inputs] if op.inputs is not None else []
        outputs = [int(x) for x in op.outputs] if op.outputs is not None else []

        info = OpInfo(index=i, opcode=code, inputs=inputs, outputs=outputs)

        opts = op.builtinOptions
        if opts is not None:
            if hasattr(opts, "strideH"):
                info.stride_h = int(opts.strideH)
                info.stride_w = int(opts.strideW)
            if hasattr(opts, "padding"):
                info.padding = int(opts.padding)
            if hasattr(opts, "fusedActivationFunction"):
                info.fused_activation = int(opts.fusedActivationFunction)
            if hasattr(opts, "depthMultiplier"):
                info.depth_multiplier = int(opts.depthMultiplier)
            if hasattr(opts, "axis"):
                info.axis = int(opts.axis)

        ops.append(info)

    graph_inputs = [int(x) for x in sg.inputs]
    graph_outputs = [int(x) for x in sg.outputs]

    return ModelGraph(
        tensors=tensors, ops=ops, inputs=graph_inputs, outputs=graph_outputs
    )


# ---------------------------------------------------------------------------
# TRANSPOSE elimination
# ---------------------------------------------------------------------------


def eliminate_transpose(graph: ModelGraph) -> ModelGraph:
    """Remove TRANSPOSE + RESHAPE nodes by permuting FC weights.

    The TRANSPOSE converts NHWC→NCHW (perm [0,3,1,2]) and RESHAPE
    flattens to [1, C*H*W].  We eliminate both by permuting the first
    C*H*W columns of the subsequent FC weight matrix from NCHW to NHWC
    index order.
    """
    new_ops = []
    skip_indices: set[int] = set()

    for i, op in enumerate(graph.ops):
        if i in skip_indices:
            continue

        if op.opcode == OP_TRANSPOSE and i + 1 < len(graph.ops):
            next_op = graph.ops[i + 1]
            if next_op.opcode == OP_RESHAPE:
                # Find the FC op that consumes the reshape output.
                reshape_out = next_op.outputs[0]
                fc_op = None
                for future_op in graph.ops[i + 2 :]:
                    if (
                        future_op.opcode in (OP_FULLY_CONNECTED, OP_CONCATENATION)
                        and reshape_out in future_op.inputs
                    ):
                        fc_op = future_op
                        break

                # Get the NHWC shape from TRANSPOSE input.
                transpose_in = graph.tensors[op.inputs[0]]
                # Shape: [1, H, W, C] (NHWC)
                _, h, w, c = transpose_in.shape

                # Find the first FC after concat that uses these features.
                # The concat output feeds into the FC.
                concat_op = None
                for future_op in graph.ops[i + 2 :]:
                    if future_op.opcode == OP_CONCATENATION:
                        concat_op = future_op
                        break

                if concat_op is not None:
                    # Find FC that consumes concat output.
                    concat_out = concat_op.outputs[0]
                    for future_op in graph.ops[i + 2 :]:
                        if (
                            future_op.opcode == OP_FULLY_CONNECTED
                            and concat_out in future_op.inputs
                        ):
                            fc_op = future_op
                            break

                if fc_op is not None:
                    # Permute FC weights: NCHW column order → NHWC column order.
                    weight_tensor = graph.tensors[fc_op.inputs[1]]
                    w_data = weight_tensor.data  # [out_features, in_features]

                    flat_size = h * w * c
                    assert w_data.shape[1] >= flat_size

                    # Build permutation: nhwc_col → nchw_col
                    perm = np.zeros(flat_size, dtype=np.intp)
                    for hh in range(h):
                        for ww in range(w):
                            for cc in range(c):
                                nhwc_j = hh * w * c + ww * c + cc
                                nchw_j = cc * h * w + hh * w + ww
                                perm[nhwc_j] = nchw_j

                    new_weights = w_data.copy()
                    new_weights[:, :flat_size] = w_data[:, perm]
                    weight_tensor.data = new_weights

                # Rewire: make TRANSPOSE input available as RESHAPE output.
                # The concat (or FC) that consumed reshape_out now gets
                # the transpose input directly, reinterpreted as flat.
                transpose_in_idx = op.inputs[0]
                reshape_out_idx = next_op.outputs[0]

                # Update the transpose input tensor shape to be the flat shape.
                flat_tensor = graph.tensors[reshape_out_idx]
                flat_tensor.quant = transpose_in.quant

                # Rewire: any op using reshape_out should use transpose_in.
                for future_op in graph.ops:
                    future_op.inputs = [
                        transpose_in_idx if x == reshape_out_idx else x
                        for x in future_op.inputs
                    ]

                skip_indices.add(i)
                skip_indices.add(i + 1)

                print(
                    f"  Eliminated TRANSPOSE (node {i}) + RESHAPE (node {i + 1}), "
                    f"permuted FC weights for {h}x{w}x{c} NHWC layout"
                )
                continue

        new_ops.append(op)

    graph.ops = new_ops
    # Re-index ops.
    for i, op in enumerate(graph.ops):
        op.index = i

    return graph


# ---------------------------------------------------------------------------
# Liveness analysis & scratch buffer allocation
# ---------------------------------------------------------------------------


@dataclass
class ScratchAlloc:
    offsets: dict[int, int]  # tensor_index → byte offset in scratch[]
    total_size: int


def compute_scratch(graph: ModelGraph) -> ScratchAlloc:
    """Compute scratch buffer offsets via greedy liveness-based packing."""
    # Determine which tensors need scratch space (activations only).
    # Exclude graph outputs — they use the caller-provided output buffer.
    activation_tensors: set[int] = set()
    for op in graph.ops:
        for t_idx in op.outputs:
            t = graph.tensors[t_idx]
            if (
                not t.is_const
                and t_idx not in graph.inputs
                and t_idx not in graph.outputs
            ):
                activation_tensors.add(t_idx)

    if not activation_tensors:
        return ScratchAlloc(offsets={}, total_size=0)

    # Compute live ranges: (first_written, last_read).
    first_write: dict[int, int] = {}
    last_read: dict[int, int] = {}

    for op_idx, op in enumerate(graph.ops):
        for t_idx in op.outputs:
            if t_idx in activation_tensors and t_idx not in first_write:
                first_write[t_idx] = op_idx
        for t_idx in op.inputs:
            if t_idx in activation_tensors:
                last_read[t_idx] = op_idx

    # Output tensors are read "at infinity".
    for t_idx in graph.outputs:
        if t_idx in activation_tensors:
            last_read[t_idx] = len(graph.ops)

    # Greedy allocation: process in birth order.
    alloc_order = sorted(activation_tensors, key=lambda t: first_write.get(t, 0))

    # Active allocations: list of (offset, size, last_read_op).
    active: list[tuple[int, int, int]] = []
    offsets: dict[int, int] = {}
    peak = 0

    for t_idx in alloc_order:
        born = first_write.get(t_idx, 0)
        dies = last_read.get(t_idx, len(graph.ops))
        size = graph.tensors[t_idx].size_bytes

        # Free tensors that died before this one was born.
        active = [(o, s, lr) for o, s, lr in active if lr >= born]

        # Find lowest offset that doesn't overlap any active tensor.
        # Sort active by offset.
        active_sorted = sorted(active, key=lambda x: x[0])
        best_offset = 0
        for a_off, a_size, _ in active_sorted:
            if best_offset + size <= a_off:
                break
            best_offset = max(best_offset, a_off + a_size)

        # Align to 4 bytes.
        best_offset = (best_offset + 3) & ~3

        offsets[t_idx] = best_offset
        active.append((best_offset, size, dies))
        peak = max(peak, best_offset + size)

    # Align total to 16 bytes.
    peak = (peak + 15) & ~15

    return ScratchAlloc(offsets=offsets, total_size=peak)


# ---------------------------------------------------------------------------
# LUT generation
# ---------------------------------------------------------------------------


def generate_tanh_lut(
    in_scale: float, in_zp: int, out_scale: float, out_zp: int
) -> np.ndarray:
    """Generate 256-entry int8 tanh LUT."""
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        # Interpret as uint8 index → signed int8 value.
        signed_val = np.int8(np.uint8(i))
        x = (float(signed_val) - in_zp) * in_scale
        y = math.tanh(x)
        q = int(round(y / out_scale)) + out_zp
        lut[i] = np.int8(max(-128, min(127, q)))
    return lut


def generate_logistic_lut(
    in_scale: float, in_zp: int, out_scale: float, out_zp: int
) -> np.ndarray:
    """Generate 256-entry int8 logistic (sigmoid) LUT."""
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        signed_val = np.int8(np.uint8(i))
        x = (float(signed_val) - in_zp) * in_scale
        y = 1.0 / (1.0 + math.exp(-x))
        q = int(round(y / out_scale)) + out_zp
        lut[i] = np.int8(max(-128, min(127, q)))
    return lut


# ---------------------------------------------------------------------------
# SAME padding computation
# ---------------------------------------------------------------------------


def compute_same_padding(
    in_h: int, in_w: int, filt_h: int, filt_w: int, stride_h: int, stride_w: int
) -> tuple[int, int, int, int, int, int]:
    """Compute SAME padding values and output dimensions.

    Returns (out_h, out_w, pad_t, pad_l, pad_b, pad_r).
    """
    out_h = (in_h + stride_h - 1) // stride_h  # ceil(in_h / stride_h)
    out_w = (in_w + stride_w - 1) // stride_w
    total_h = max(0, (out_h - 1) * stride_h + filt_h - in_h)
    total_w = max(0, (out_w - 1) * stride_w + filt_w - in_w)
    pad_t = total_h // 2
    pad_b = total_h - pad_t
    pad_l = total_w // 2
    pad_r = total_w - pad_l
    return out_h, out_w, pad_t, pad_l, pad_b, pad_r


# ---------------------------------------------------------------------------
# C code emission helpers
# ---------------------------------------------------------------------------


def _format_int8_array(data: np.ndarray, per_line: int = 16) -> str:
    """Format an int8 array as comma-separated C initializer values."""
    flat = data.flatten().astype(np.int8)
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i : i + per_line]
        lines.append("    " + ", ".join(str(int(x)) for x in chunk) + ",")
    return "\n".join(lines)


def _format_int32_array(data: np.ndarray, per_line: int = 8) -> str:
    flat = data.flatten().astype(np.int32)
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i : i + per_line]
        lines.append("    " + ", ".join(str(int(x)) for x in chunk) + ",")
    return "\n".join(lines)


def _format_float_array(data: np.ndarray, per_line: int = 8) -> str:
    flat = data.flatten().astype(np.float32)
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i : i + per_line]
        lines.append("    " + ", ".join(f"{float(x):.10e}f" for x in chunk) + ",")
    return "\n".join(lines)


def _c_ident(name: str) -> str:
    """Sanitize a tensor name into a valid C identifier."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


# ---------------------------------------------------------------------------
# Weight naming
# ---------------------------------------------------------------------------


def _weight_name(op: OpInfo, role: str) -> str:
    """Generate a C variable name for a weight tensor."""
    op_name = OP_NAMES.get(op.opcode, f"op{op.opcode}").lower()
    return f"node{op.index}_{op_name}_{role}"


def _get_weight_tensors(graph: ModelGraph) -> list[tuple[OpInfo, str, TensorInfo]]:
    """Collect (op, role, tensor) for all weight/bias tensors."""
    result = []
    for op in graph.ops:
        if op.opcode in (OP_CONV_2D, OP_DEPTHWISE_CONV_2D):
            # inputs: [input, filter, bias]
            result.append((op, "filter", graph.tensors[op.inputs[1]]))
            result.append((op, "bias", graph.tensors[op.inputs[2]]))
        elif op.opcode == OP_FULLY_CONNECTED:
            result.append((op, "weights", graph.tensors[op.inputs[1]]))
            result.append((op, "bias", graph.tensors[op.inputs[2]]))
    return result


# ---------------------------------------------------------------------------
# Emit weights
# ---------------------------------------------------------------------------


def emit_weights_h(graph: ModelGraph, path: str) -> None:
    """Emit doom_agent_weights.h — weight/bias/scale declarations."""
    lines = [
        "/* Auto-generated by codegen_graph.py — DO NOT EDIT. */",
        "",
        "#ifndef DOOM_AGENT_WEIGHTS_H",
        "#define DOOM_AGENT_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
    ]

    for op, role, tensor in _get_weight_tensors(graph):
        name = _weight_name(op, role)
        size = tensor.size_bytes
        if tensor.dtype == TFLITE_INT8:
            lines.append(f"extern const int8_t {name}[{size}];")
        elif tensor.dtype == TFLITE_INT32:
            count = 1
            for d in tensor.shape:
                count *= d
            lines.append(f"extern const int32_t {name}[{count}];")

        # Per-channel scales.
        if tensor.quant_per_ch is not None:
            n_ch = len(tensor.quant_per_ch.scales)
            lines.append(f"extern const float {name}_scales[{n_ch}];")

    lines += ["", "#endif /* DOOM_AGENT_WEIGHTS_H */", ""]
    Path(path).write_text("\n".join(lines))


def emit_weights_c(graph: ModelGraph, path: str) -> None:
    """Emit doom_agent_weights.c — weight data in .rodata."""
    lines = [
        "/* Auto-generated by codegen_graph.py — DO NOT EDIT. */",
        "",
        '#include "doom_agent_weights.h"',
        "",
    ]

    sec = '__attribute__((section(".cnidoom.weights")))\n'

    for op, role, tensor in _get_weight_tensors(graph):
        name = _weight_name(op, role)
        if tensor.dtype == TFLITE_INT8:
            size = tensor.size_bytes
            lines.append(f"{sec}const int8_t {name}[{size}] = {{")
            lines.append(_format_int8_array(tensor.data))
            lines.append("};")
            lines.append("")
        elif tensor.dtype == TFLITE_INT32:
            count = 1
            for d in tensor.shape:
                count *= d
            lines.append(f"{sec}const int32_t {name}[{count}] = {{")
            lines.append(_format_int32_array(tensor.data))
            lines.append("};")
            lines.append("")

        if tensor.quant_per_ch is not None:
            n_ch = len(tensor.quant_per_ch.scales)
            lines.append(f"{sec}const float {name}_scales[{n_ch}] = {{")
            lines.append(_format_float_array(tensor.quant_per_ch.scales))
            lines.append("};")
            lines.append("")

    Path(path).write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Emit graph
# ---------------------------------------------------------------------------


def emit_graph_h(graph: ModelGraph, path: str) -> None:
    """Emit doom_agent_graph.h — run_graph() declaration."""
    # Determine visual input shape from the graph.
    state_idx = graph.inputs[0]
    visual_idx = graph.inputs[1]
    if len(graph.tensors[state_idx].shape) == 4:
        state_idx, visual_idx = visual_idx, state_idx
    vis_shape = graph.tensors[visual_idx].shape  # [1, H, W, C]
    vis_h, vis_w, vis_c = vis_shape[1], vis_shape[2], vis_shape[3]
    out_shape = graph.tensors[graph.outputs[0]].shape
    out_dim = out_shape[-1] if len(out_shape) >= 2 else out_shape[0]

    content = f"""\
/* Auto-generated by codegen_graph.py — DO NOT EDIT. */

#ifndef DOOM_AGENT_GRAPH_H
#define DOOM_AGENT_GRAPH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

/* Model input/output dimensions (from TFLite model). */
#define GRAPH_VISUAL_H {vis_h}
#define GRAPH_VISUAL_W {vis_w}
#define GRAPH_VISUAL_C {vis_c}
#define GRAPH_OUTPUT_DIM {out_dim}

/*
 * Run the full model graph.
 *
 * visual: INT8 [1, {vis_h}, {vis_w}, {vis_c}] NHWC frame stack
 * state:  INT8 [1, 20] quantized game state
 * output: INT8 [1, {out_dim}] raw logistic output (before dequant + threshold)
 */
void run_graph(const int8_t *visual, const int8_t *state, int8_t *output);

/* Retrieve quantization params for the state input tensor. */
void graph_get_state_quant(float *scale, int32_t *zero_point);

/* Retrieve quantization params for the output tensor. */
void graph_get_output_quant(float *scale, int32_t *zero_point);

/* Retrieve quantization params for the visual input tensor. */
void graph_get_visual_quant(float *scale, int32_t *zero_point);

#ifdef __cplusplus
}}
#endif

#endif /* DOOM_AGENT_GRAPH_H */
"""
    Path(path).write_text(content)


def _activation_name(act: int) -> str:
    return {
        ACT_NONE: "KERNEL_ACT_NONE",
        ACT_RELU: "KERNEL_ACT_RELU",
        ACT_RELU6: "KERNEL_ACT_RELU6",
    }.get(act, "KERNEL_ACT_NONE")


def emit_graph_c(graph: ModelGraph, scratch: ScratchAlloc, path: str) -> None:
    """Emit doom_agent_graph.c — sequential kernel calls + scratch buffer."""
    needs_memcpy = graph.outputs[0] in scratch.offsets

    lines = [
        "/* Auto-generated by codegen_graph.py — DO NOT EDIT. */",
        "",
        '#include "doom_agent_graph.h"',
        '#include "doom_agent_weights.h"',
        '#include "../kernels/kernel_ops.h"',
        "",
    ]
    if needs_memcpy:
        lines.insert(5, "#include <string.h>")
        lines.insert(6, "")

    # Scratch buffer.
    lines.append(
        f'__attribute__((section(".cnidoom.scratch")))\n'
        f"static int8_t scratch[{scratch.total_size}];  "
        f"/* peak scratch: {scratch.total_size} bytes */",
    )
    lines.append("")

    # Scratch offset defines.
    lines.append("/* Scratch buffer offsets (from liveness analysis). */")
    for t_idx in sorted(scratch.offsets.keys()):
        t = graph.tensors[t_idx]
        short_name = f"t{t_idx}"
        lines.append(
            f"#define OFF_{short_name} {scratch.offsets[t_idx]}  "
            f"/* {t.shape} = {t.size_bytes}B */"
        )
    lines.append("")

    # Quant param structs for activation tensors.
    lines.append("/* Per-tensor quant params for activation tensors. */")
    emitted_qp: set[int] = set()
    for op in graph.ops:
        for t_idx in list(op.inputs) + list(op.outputs):
            t = graph.tensors[t_idx]
            if t_idx not in emitted_qp and t.quant is not None and not t.is_const:
                lines.append(
                    f"static const quant_param_t qp_t{t_idx} = "
                    f"{{ .scale = {t.quant.scale:.10e}f, "
                    f".zero_point = {t.quant.zero_point} }};"
                )
                emitted_qp.add(t_idx)
    lines.append("")

    # Per-channel quant param structs for weight tensors.
    lines.append("/* Per-channel quant params for weight tensors. */")
    for op, role, tensor in _get_weight_tensors(graph):
        if tensor.quant_per_ch is not None:
            name = _weight_name(op, role)
            n_ch = len(tensor.quant_per_ch.scales)
            lines.append(
                f"static const quant_param_per_ch_t qpc_{name} = "
                f"{{ .scales = {name}_scales, .num_channels = {n_ch} }};"
            )
    lines.append("")

    # LUTs for TANH and LOGISTIC.
    for op in graph.ops:
        if op.opcode == OP_TANH:
            in_t = graph.tensors[op.inputs[0]]
            out_t = graph.tensors[op.outputs[0]]
            lut = generate_tanh_lut(
                in_t.quant.scale,
                in_t.quant.zero_point,
                out_t.quant.scale,
                out_t.quant.zero_point,
            )
            lines.append(f"static const int8_t tanh_lut_node{op.index}[256] = {{")
            lines.append(_format_int8_array(lut))
            lines.append("};")
            lines.append("")
        elif op.opcode == OP_LOGISTIC:
            in_t = graph.tensors[op.inputs[0]]
            out_t = graph.tensors[op.outputs[0]]
            lut = generate_logistic_lut(
                in_t.quant.scale,
                in_t.quant.zero_point,
                out_t.quant.scale,
                out_t.quant.zero_point,
            )
            lines.append(f"static const int8_t logistic_lut_node{op.index}[256] = {{")
            lines.append(_format_int8_array(lut))
            lines.append("};")
            lines.append("")

    # State and output quant accessors.
    state_idx = graph.inputs[0]  # state is input 0
    visual_idx = graph.inputs[1]  # visual is input 1
    # Auto-detect: state is the 2D input, visual is the 4D input.
    if len(graph.tensors[state_idx].shape) == 4:
        state_idx, visual_idx = visual_idx, state_idx

    state_t = graph.tensors[state_idx]
    output_t = graph.tensors[graph.outputs[0]]

    lines.append("void graph_get_state_quant(float *scale, int32_t *zero_point) {")
    lines.append(f"  *scale = {state_t.quant.scale:.10e}f;")
    lines.append(f"  *zero_point = {state_t.quant.zero_point};")
    lines.append("}")
    lines.append("")

    lines.append("void graph_get_output_quant(float *scale, int32_t *zero_point) {")
    lines.append(f"  *scale = {output_t.quant.scale:.10e}f;")
    lines.append(f"  *zero_point = {output_t.quant.zero_point};")
    lines.append("}")
    lines.append("")

    visual_t = graph.tensors[visual_idx]
    lines.append("void graph_get_visual_quant(float *scale, int32_t *zero_point) {")
    lines.append(f"  *scale = {visual_t.quant.scale:.10e}f;")
    lines.append(f"  *zero_point = {visual_t.quant.zero_point};")
    lines.append("}")
    lines.append("")

    # run_graph().
    lines.append(
        "void run_graph(const int8_t *visual, const int8_t *state, int8_t *output) {"
    )

    def _tensor_ptr(t_idx: int) -> str:
        """Return C expression for a tensor's data pointer."""
        if t_idx == visual_idx:
            return "visual"
        if t_idx == state_idx:
            return "state"
        if t_idx in graph.outputs:
            return "output"
        if t_idx in scratch.offsets:
            return f"scratch + {scratch.offsets[t_idx]}"
        # Const tensor — shouldn't be used as activation pointer.
        return f"/* ERROR: tensor {t_idx} not allocated */"

    for op in graph.ops:
        op_name = OP_NAMES.get(op.opcode, f"OP_{op.opcode}")
        lines.append(f"  /* Node {op.index}: {op_name} */")

        if op.opcode == OP_DEPTHWISE_CONV_2D:
            in_t = graph.tensors[op.inputs[0]]
            filt_t = graph.tensors[op.inputs[1]]
            out_t = graph.tensors[op.outputs[0]]
            in_h, in_w, channels = in_t.shape[1], in_t.shape[2], in_t.shape[3]
            filt_h, filt_w = filt_t.shape[1], filt_t.shape[2]
            out_h, out_w, pad_t, pad_l, pad_b, pad_r = compute_same_padding(
                in_h, in_w, filt_h, filt_w, op.stride_h, op.stride_w
            )
            w_name = _weight_name(op, "filter")
            b_name = _weight_name(op, "bias")
            act = _activation_name(op.fused_activation)

            lines.append("  kernel_depthwise_conv2d_int8(")
            lines.append(
                f"      {_tensor_ptr(op.inputs[0])}, {in_h}, {in_w}, {channels},"
            )
            lines.append(f"      {w_name}, {filt_h}, {filt_w}, {b_name},")
            lines.append(
                f"      {op.stride_h}, {op.stride_w}, "
                f"{pad_t}, {pad_l}, {pad_b}, {pad_r},"
            )
            lines.append(
                f"      &qp_t{op.inputs[0]}, &qpc_{w_name}, &qp_t{op.outputs[0]},"
            )
            lines.append(
                f"      {_tensor_ptr(op.outputs[0])}, {out_h}, {out_w}, {act});"
            )

        elif op.opcode == OP_CONV_2D:
            in_t = graph.tensors[op.inputs[0]]
            filt_t = graph.tensors[op.inputs[1]]
            out_t = graph.tensors[op.outputs[0]]
            in_h, in_w, in_c = in_t.shape[1], in_t.shape[2], in_t.shape[3]
            out_c = filt_t.shape[0]
            filt_h, filt_w = filt_t.shape[1], filt_t.shape[2]
            out_h, out_w, pad_t, pad_l, pad_b, pad_r = compute_same_padding(
                in_h, in_w, filt_h, filt_w, op.stride_h, op.stride_w
            )
            w_name = _weight_name(op, "filter")
            b_name = _weight_name(op, "bias")
            act = _activation_name(op.fused_activation)

            lines.append("  kernel_conv2d_int8(")
            lines.append(f"      {_tensor_ptr(op.inputs[0])}, {in_h}, {in_w}, {in_c},")
            lines.append(f"      {w_name}, {filt_h}, {filt_w}, {out_c}, {b_name},")
            lines.append(
                f"      {op.stride_h}, {op.stride_w}, "
                f"{pad_t}, {pad_l}, {pad_b}, {pad_r},"
            )
            lines.append(
                f"      &qp_t{op.inputs[0]}, &qpc_{w_name}, &qp_t{op.outputs[0]},"
            )
            lines.append(
                f"      {_tensor_ptr(op.outputs[0])}, {out_h}, {out_w}, {act});"
            )

        elif op.opcode == OP_FULLY_CONNECTED:
            in_t = graph.tensors[op.inputs[0]]
            wt_t = graph.tensors[op.inputs[1]]
            out_t = graph.tensors[op.outputs[0]]
            in_features = wt_t.shape[1]
            out_features = wt_t.shape[0]
            w_name = _weight_name(op, "weights")
            b_name = _weight_name(op, "bias")
            act = _activation_name(op.fused_activation)

            lines.append("  kernel_fully_connected_int8(")
            lines.append(f"      {_tensor_ptr(op.inputs[0])}, {in_features},")
            lines.append(f"      {w_name}, {out_features}, {b_name},")
            lines.append(
                f"      &qp_t{op.inputs[0]}, &qpc_{w_name}, &qp_t{op.outputs[0]},"
            )
            lines.append(f"      {_tensor_ptr(op.outputs[0])}, {act});")

        elif op.opcode == OP_CONCATENATION:
            a_idx = op.inputs[0]
            b_idx = op.inputs[1]
            a_t = graph.tensors[a_idx]
            b_t = graph.tensors[b_idx]
            a_len = a_t.size_bytes
            b_len = b_t.size_bytes

            lines.append("  kernel_concatenation_int8(")
            lines.append(f"      {_tensor_ptr(a_idx)}, {a_len},")
            lines.append(f"      {_tensor_ptr(b_idx)}, {b_len},")
            lines.append(f"      {_tensor_ptr(op.outputs[0])});")

        elif op.opcode == OP_TANH:
            in_t = graph.tensors[op.inputs[0]]
            count = in_t.size_bytes
            lines.append(
                f"  kernel_tanh_int8({_tensor_ptr(op.inputs[0])}, {count}, "
                f"tanh_lut_node{op.index}, {_tensor_ptr(op.outputs[0])});"
            )

        elif op.opcode == OP_LOGISTIC:
            in_t = graph.tensors[op.inputs[0]]
            count = in_t.size_bytes
            lines.append(
                f"  kernel_logistic_int8({_tensor_ptr(op.inputs[0])}, {count}, "
                f"logistic_lut_node{op.index}, {_tensor_ptr(op.outputs[0])});"
            )

        elif op.opcode == OP_MEAN:
            in_t = graph.tensors[op.inputs[0]]
            out_t = graph.tensors[op.outputs[0]]
            # Input shape: [1, H, W, C] (NHWC)
            in_h, in_w, channels = in_t.shape[1], in_t.shape[2], in_t.shape[3]

            lines.append("  kernel_mean_int8(")
            lines.append(
                f"      {_tensor_ptr(op.inputs[0])}, {in_h}, {in_w}, {channels},"
            )
            lines.append(f"      &qp_t{op.inputs[0]}, &qp_t{op.outputs[0]},")
            lines.append(f"      {_tensor_ptr(op.outputs[0])});")

        else:
            lines.append(f"  /* WARNING: unhandled op {op_name} — skipped */")

        lines.append("")

    # Copy final output if it's in the scratch buffer (not written directly).
    out_idx = graph.outputs[0]
    if out_idx in scratch.offsets:
        out_size = graph.tensors[out_idx].size_bytes
        lines.append(
            f"  memcpy(output, scratch + {scratch.offsets[out_idx]}, {out_size});"
        )
    # else: last op already wrote to output via _tensor_ptr

    lines.append("}")
    lines.append("")

    Path(path).write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile TFLite INT8 model to static C inference code."
    )
    parser.add_argument("--model", required=True, help="Path to .tflite model file")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for generated C files"
    )
    args = parser.parse_args()

    print(f"Reading model: {args.model}")
    graph = parse_model(args.model)
    print(
        f"  {len(graph.tensors)} tensors, {len(graph.ops)} ops, "
        f"inputs={graph.inputs}, outputs={graph.outputs}"
    )

    print("Eliminating TRANSPOSE + RESHAPE...")
    graph = eliminate_transpose(graph)
    print(f"  {len(graph.ops)} ops after elimination")

    print("Computing scratch buffer allocation...")
    scratch = compute_scratch(graph)
    print(f"  Peak scratch: {scratch.total_size} bytes")
    for t_idx in sorted(scratch.offsets.keys()):
        t = graph.tensors[t_idx]
        print(
            f"    t{t_idx}: offset={scratch.offsets[t_idx]}, "
            f"size={t.size_bytes}, shape={t.shape}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    emit_weights_h(graph, os.path.join(args.output_dir, "doom_agent_weights.h"))
    emit_weights_c(graph, os.path.join(args.output_dir, "doom_agent_weights.c"))
    emit_graph_h(graph, os.path.join(args.output_dir, "doom_agent_graph.h"))
    emit_graph_c(graph, scratch, os.path.join(args.output_dir, "doom_agent_graph.c"))

    print(f"Generated files in {args.output_dir}/:")
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"  {f}: {size:,} bytes")

    print("Done.")


if __name__ == "__main__":
    main()
