#!/usr/bin/env python3
"""Compare a stateful Lyra ONNX graph with its source TFLite model.

The TFLite graphs keep recurrent tensors in resource variables.  The ONNX
conversion exposes each variable read as an input and each assigned value as
an output.  This utility derives that mapping from the TFLite flatbuffer and
checks several consecutive frames, so state-order mistakes cannot hide behind
a successful one-frame import.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def _name(tensor):
    return tensor.Name().decode("utf-8")


def _shape(tensor):
    return tuple(tensor.Shape(i) for i in range(tensor.ShapeLength()))


def state_io(model_path):
    model = schema.Model.GetRootAsModel(bytearray(Path(model_path).read_bytes()), 0)
    graph = model.Subgraphs(0)
    reads = {}
    writes = {}
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        if opcode == schema.BuiltinOperator.READ_VARIABLE:
            reads[op.Inputs(0)] = op.Outputs(0)
        elif opcode == schema.BuiltinOperator.ASSIGN_VARIABLE:
            writes[op.Inputs(0)] = op.Inputs(1)

    pairs = []
    for handle, read_index in reads.items():
        if handle not in writes:
            raise ValueError(f"variable tensor {handle} is read but never assigned")
        read_tensor = graph.Tensors(read_index)
        write_tensor = graph.Tensors(writes[handle])
        pairs.append(
            {
                "input": _name(read_tensor),
                "output": _name(write_tensor),
                "read_index": read_index,
                "write_index": writes[handle],
                "shape": _shape(read_tensor),
            }
        )
    return graph, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-error", type=float)
    args = parser.parse_args()

    graph, pairs = state_io(args.tflite)
    interpreter = tf.lite.Interpreter(
        model_path=str(args.tflite),
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    tflite_input = interpreter.get_input_details()[0]
    tflite_output = interpreter.get_output_details()[0]

    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    onnx_inputs = {item.name: item for item in session.get_inputs()}
    onnx_outputs = [item.name for item in session.get_outputs()]
    original_input_name = _name(graph.Tensors(graph.Inputs(0)))
    original_output_name = _name(graph.Tensors(graph.Outputs(0)))
    main_input_name = (
        "lyra_input" if "lyra_input" in onnx_inputs else original_input_name
    )
    main_output_name = (
        "lyra_output" if "lyra_output" in onnx_outputs else original_output_name
    )
    for index, pair in enumerate(pairs):
        stable_input = f"lyra_state_{index:02d}_input"
        stable_output = f"lyra_state_{index:02d}_output"
        if stable_input in onnx_inputs:
            pair["input"] = stable_input
        if stable_output in onnx_outputs:
            pair["output"] = stable_output

    print("ONNX inputs:")
    for item in session.get_inputs():
        print(f"  {item.name}: {item.shape}")
    print("ONNX outputs:")
    for item in session.get_outputs():
        print(f"  {item.name}: {item.shape}")

    missing = [main_input_name] + [p["input"] for p in pairs]
    missing = [name for name in missing if name not in onnx_inputs]
    if missing:
        raise ValueError(f"missing ONNX inputs: {missing}")
    missing = [main_output_name] + [p["output"] for p in pairs]
    missing = [name for name in missing if name not in onnx_outputs]
    if missing:
        raise ValueError(f"missing ONNX outputs: {missing}")

    states = {
        p["input"]: np.zeros(p["shape"], dtype=np.float32) for p in pairs
    }
    rng = np.random.default_rng(args.seed)
    worst = 0.0
    for frame in range(args.frames):
        sample = rng.uniform(-1.0, 1.0, tflite_input["shape"]).astype(np.float32)
        interpreter.set_tensor(tflite_input["index"], sample)
        interpreter.invoke()
        expected = interpreter.get_tensor(tflite_output["index"])

        feeds = {main_input_name: sample, **states}
        actual_values = session.run(onnx_outputs, feeds)
        actual = actual_values[onnx_outputs.index(main_output_name)]
        error = float(np.max(np.abs(expected - actual)))
        worst = max(worst, error)
        print(
            f"frame {frame:02}: max_abs_error={error:.8g} "
            f"tflite=[{expected.min():.5g},{expected.max():.5g}] "
            f"onnx=[{actual.min():.5g},{actual.max():.5g}]"
        )
        values = dict(zip(onnx_outputs, actual_values))
        states = {p["input"]: values[p["output"]] for p in pairs}

    print(f"worst max_abs_error: {worst:.8g}")
    if args.max_error is not None and worst > args.max_error:
        raise SystemExit(
            f"maximum error {worst:.8g} exceeds limit {args.max_error:.8g}"
        )


if __name__ == "__main__":
    main()
