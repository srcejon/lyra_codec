#!/usr/bin/env python3
"""Convert a stateful Lyra TFLite graph into an OpenCV-friendly float ONNX graph.

TFLite resource variables are not part of ONNX.  This converter exposes each
variable read as a graph input and each assigned value as a graph output.  The
caller feeds those outputs back on the next 20 ms frame.  Float export avoids
OpenCV's unreliable handling of quantized recurrent graph outputs.
"""

import argparse
import copy
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import onnx
from tensorflow.lite.python import schema_py_generated as schema


def _name(tensor):
    return tensor.Name().decode("utf-8")


def _shape(tensor):
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]


def graph_description(model_path):
    model = schema.Model.GetRootAsModel(bytearray(model_path.read_bytes()), 0)
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

    states = []
    for handle, read_index in reads.items():
        if handle not in writes:
            raise ValueError(f"variable tensor {handle} is read but never assigned")
        input_tensor = graph.Tensors(read_index)
        output_tensor = graph.Tensors(writes[handle])
        states.append(
            {
                "input": _name(input_tensor),
                "output": _name(output_tensor),
                "shape": _shape(input_tensor),
            }
        )

    input_tensor = graph.Tensors(graph.Inputs(0))
    output_tensor = graph.Tensors(graph.Outputs(0))
    return {
        "input": {"name": _name(input_tensor), "shape": _shape(input_tensor)},
        "output": {"name": _name(output_tensor), "shape": _shape(output_tensor)},
        "states": states,
    }


def split_shared_initializers(model):
    """Work around OpenCV DNN importing a shared initializer more than once.

    OpenCV 4.13 can try to create the same Const layer twice when an initializer
    feeds multiple Concat nodes. Giving every additional use its own initializer
    preserves the graph while keeping its internal layer names unique.
    """
    initializers = {value.name: value for value in model.graph.initializer}
    use_count = {}
    additions = []
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            initializer = initializers.get(input_name)
            if initializer is None:
                continue
            count = use_count.get(input_name, 0)
            use_count[input_name] = count + 1
            if count == 0:
                continue
            duplicate = copy.deepcopy(initializer)
            duplicate.name = f"{input_name}__opencv_use_{count}"
            node.input[index] = duplicate.name
            additions.append(duplicate)
    model.graph.initializer.extend(additions)


def stabilize_streaming_io(model, description):
    """Replace TensorFlow-generated graph boundary names with stable names."""
    renames = {
        description["input"]["name"]: "lyra_input",
        description["output"]["name"]: "lyra_output",
    }
    description["input"]["name"] = "lyra_input"
    description["output"]["name"] = "lyra_output"
    for index, state in enumerate(description["states"]):
        input_name = f"lyra_state_{index:02d}_input"
        output_name = f"lyra_state_{index:02d}_output"
        renames[state["input"]] = input_name
        renames[state["output"]] = output_name
        state["input"] = input_name
        state["output"] = output_name

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = renames.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = renames.get(name, name)
    for collection in (
        model.graph.input,
        model.graph.output,
        model.graph.value_info,
        model.graph.initializer,
    ):
        for value in collection:
            value.name = renames.get(value.name, value.name)


@contextmanager
def tf2onnx_environment():
    """Provide a local workaround for tf2onnx 1.16.1 empty-output ops."""
    spec = importlib.util.find_spec("tf2onnx")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("tf2onnx is not installed")
    package = Path(next(iter(spec.submodule_search_locations)))
    source = package / "tflite_utils.py"
    text = source.read_text(encoding="utf-8")
    unsafe = "        node_name = output_names[0]\n"
    safe = (
        "        node_name = output_names[0] if output_names else "
        'utils.make_name(f"{optype}_Output")\n'
    )
    # Released tf2onnx 1.16.1 indexes the empty output list used by TFLite's
    # ASSIGN_VARIABLE op. Run a patched temporary package without modifying the
    # user's Python installation. Newer/fixed installations pass through.
    if unsafe + safe not in text:
        yield os.environ.copy()
        return
    with tempfile.TemporaryDirectory(prefix="lyra_tf2onnx_") as directory:
        temporary_package = Path(directory) / "tf2onnx"
        shutil.copytree(package, temporary_package)
        patched = text.replace(unsafe + safe, safe, 1)
        (temporary_package / "tflite_utils.py").write_text(
            patched, encoding="utf-8"
        )
        environment = os.environ.copy()
        old_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = directory + (
            os.pathsep + old_path if old_path else ""
        )
        yield environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument(
        "--keep-quantized",
        action="store_true",
        help=(
            "retain Q/DQ nodes for ONNX Runtime experiments; OpenCV 4.13 "
            "and 4.14 corrupt several recurrent outputs in this form"
        ),
    )
    args = parser.parse_args()

    description = graph_description(args.tflite)
    conversion_inputs = [description["input"]["name"]]
    conversion_inputs.extend(state["input"] for state in description["states"])
    conversion_outputs = [description["output"]["name"]]
    conversion_outputs.extend(state["output"] for state in description["states"])

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--tflite",
        str(args.tflite),
        "--output",
        str(args.onnx),
        "--opset",
        str(args.opset),
        "--inputs",
        ",".join(conversion_inputs),
        "--outputs",
        ",".join(conversion_outputs),
    ]
    if not args.keep_quantized:
        command.append("--dequantize")
    with tf2onnx_environment() as environment:
        subprocess.run(command, check=True, env=environment)

    converted = onnx.load(str(args.onnx))
    split_shared_initializers(converted)
    stabilize_streaming_io(converted, description)
    onnx.checker.check_model(converted)
    inputs = [description["input"]["name"]]
    inputs.extend(state["input"] for state in description["states"])
    outputs = [description["output"]["name"]]
    outputs.extend(state["output"] for state in description["states"])
    actual_inputs = {value.name for value in converted.graph.input}
    actual_outputs = {value.name for value in converted.graph.output}
    if set(inputs) != actual_inputs or set(outputs) != actual_outputs:
        raise ValueError("converted model did not preserve the requested state I/O")
    onnx.save(converted, str(args.onnx))

    manifest_path = args.onnx.with_suffix(".json")
    manifest_path.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.onnx}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
