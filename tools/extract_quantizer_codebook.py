#!/usr/bin/env python3
"""Extracts Lyra's residual-vector-quantizer codebooks from TFLite.

TensorFlow is a conversion-time dependency only. The generated binary is read
by the dependency-free NativeQuantizer implementation.
"""

import argparse
import pathlib
import re
import struct

import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


MAGIC = b"LYRAQV1\0"
MODEL_VERSION = 3
NUM_STAGES = 46
NUM_ENTRIES = 16
NUM_FEATURES = 64
BITS_PER_STAGE = 4


def tensor_shape(tensor):
    return tuple(tensor.Shape(i) for i in range(tensor.ShapeLength()))


def extract_codebooks(model_path):
    model_data = model_path.read_bytes()
    model = schema.Model.GetRootAsModel(model_data, 0)
    if model.SubgraphsLength() < 2:
        raise ValueError("quantizer model does not contain the encode subgraph")

    encode = model.Subgraphs(1)
    codebooks = []
    seen_buffers = set()
    pattern = re.compile(
        r"vector_quantizer_ema_(\d+)/(?:transpose|ExpandDims)$")

    for index in range(encode.TensorsLength()):
        tensor = encode.Tensors(index)
        name = tensor.Name().decode("utf-8") if tensor.Name() else ""
        match = pattern.search(name)
        if not match or tensor_shape(tensor) not in {
            (NUM_ENTRIES, NUM_FEATURES),
            (1, NUM_ENTRIES, NUM_FEATURES),
        }:
            continue

        buffer_index = tensor.Buffer()
        if buffer_index in seen_buffers:
            continue
        seen_buffers.add(buffer_index)

        raw = model.Buffers(buffer_index).DataAsNumpy().tobytes()
        values = np.frombuffer(raw, dtype="<f4")
        expected = NUM_ENTRIES * NUM_FEATURES
        if values.size != expected:
            raise ValueError(f"{name} has {values.size} values, expected {expected}")
        codebooks.append((int(match.group(1)), values.reshape(NUM_ENTRIES,
                                                               NUM_FEATURES)))

    codebooks.sort(key=lambda item: item[0])
    if len(codebooks) != NUM_STAGES:
        raise ValueError(
            f"found {len(codebooks)} codebooks, expected {NUM_STAGES}")

    stage_numbers = [stage for stage, _ in codebooks]
    expected_stages = list(range(stage_numbers[0], stage_numbers[0] + NUM_STAGES))
    if stage_numbers != expected_stages:
        raise ValueError("quantizer codebook stages are not contiguous")

    return np.stack([values for _, values in codebooks]).astype("<f4")


def write_codebooks(output_path, codebooks):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<8sIIIII",
        MAGIC,
        MODEL_VERSION,
        NUM_STAGES,
        NUM_ENTRIES,
        NUM_FEATURES,
        BITS_PER_STAGE,
    )
    output_path.write_bytes(header + codebooks.tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    codebooks = extract_codebooks(args.model)
    write_codebooks(args.output, codebooks)
    print(
        f"wrote {codebooks.shape[0]}x{codebooks.shape[1]}x"
        f"{codebooks.shape[2]} codebook to {args.output}"
    )


if __name__ == "__main__":
    main()
