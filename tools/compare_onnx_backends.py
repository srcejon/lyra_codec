#!/usr/bin/env python3
"""Compare OpenCV DNN and ONNX Runtime on a streaming Lyra graph."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    input_names = [manifest["input"]["name"]]
    input_names.extend(state["input"] for state in manifest["states"])
    output_names = [manifest["output"]["name"]]
    output_names.extend(state["output"] for state in manifest["states"])

    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    net = cv2.dnn.readNetFromONNX(str(args.model))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    state_shapes = {
        state["input"]: tuple(state["shape"]) for state in manifest["states"]
    }
    ort_states = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in state_shapes.items()
    }
    cv_states = {name: value.copy() for name, value in ort_states.items()}
    rng = np.random.default_rng(args.seed)
    worst = {name: 0.0 for name in output_names}
    quantization = {
        state["output"]: state.get("quantization")
        for state in manifest["states"]
    }

    def dequantize(name, value):
        parameters = quantization.get(name)
        if parameters is None:
            return np.asarray(value, dtype=np.float32)
        return (
            np.asarray(value, dtype=np.float32) - parameters["zero_point"]
        ) * parameters["scale"]

    for frame in range(args.frames):
        sample = rng.uniform(
            -1.0, 1.0, tuple(manifest["input"]["shape"])
        ).astype(np.float32)
        ort_feeds = {input_names[0]: sample, **ort_states}
        cv_feeds = {input_names[0]: sample, **cv_states}
        ort_values = session.run(output_names, ort_feeds)
        for name, value in cv_feeds.items():
            net.setInput(value, name)
        cv_values = net.forward(output_names)

        print(f"frame {frame:02d}")
        for name, expected, actual in zip(output_names, ort_values, cv_values):
            expected_flat = dequantize(name, expected).reshape(-1)
            actual_flat = dequantize(name, actual).reshape(-1)
            if expected_flat.size != actual_flat.size:
                raise ValueError(
                    f"{name}: output sizes differ: "
                    f"ORT={expected_flat.size}, OpenCV={actual_flat.size}"
                )
            difference = np.abs(expected_flat - actual_flat)
            error = float(np.nanmax(difference))
            worst[name] = max(worst[name], error)
            print(
                f"  {name}: max_abs={error:.8g} "
                f"mean_abs={float(np.nanmean(difference)):.8g}"
            )

        ort_by_name = dict(zip(output_names, ort_values))
        cv_by_name = dict(zip(output_names, cv_values))
        ort_states = {
            state["input"]: dequantize(
                state["output"], ort_by_name[state["output"]]
            ).reshape(
                state_shapes[state["input"]]
            )
            for state in manifest["states"]
        }
        cv_states = {
            state["input"]: dequantize(
                state["output"], cv_by_name[state["output"]]
            ).reshape(
                state_shapes[state["input"]]
            )
            for state in manifest["states"]
        }

    print("worst max_abs_error by output")
    for name, error in worst.items():
        print(f"  {name}: {error:.8g}")


if __name__ == "__main__":
    main()
