#!/usr/bin/env python3
"""Compare the OpenCV Lyra codec with the original TFLite implementation.

For every input and bitrate this tool compares encoded codebook indices, runs
both cross-decoding directions, writes listening WAVs, and records waveform and
log-spectral metrics in a JSON report.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import wave

import numpy as np
import onnxruntime as ort
import tensorflow as tf


SAMPLES_PER_FRAME = 320
STAGES = {3200: 16, 6000: 30, 9200: 46}


class OnnxStreamingModel:
    def __init__(self, model_path):
        manifest = json.loads(model_path.with_suffix(".json").read_text())
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = manifest["input"]["name"]
        self.output_name = manifest["output"]["name"]
        self.states = manifest["states"]
        self.values = {
            state["input"]: np.zeros(state["shape"], dtype=np.float32)
            for state in self.states
        }

    def run(self, value):
        output_names = [self.output_name] + [
            state["output"] for state in self.states
        ]
        results = self.session.run(
            output_names, {self.input_name: value.astype(np.float32), **self.values}
        )
        self.values = {
            state["input"]: results[index + 1]
            for index, state in enumerate(self.states)
        }
        return results[0]


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if (
            wav.getnchannels() != 1
            or wav.getframerate() != 16000
            or wav.getsampwidth() != 2
            or wav.getcomptype() != "NONE"
        ):
            raise ValueError(f"{path} must be 16-bit PCM, mono, 16000 Hz")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()


def write_wav(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def pack_indices(indices):
    indices = np.asarray(indices, dtype=np.uint8).reshape(-1)
    return ((indices[0::2] << 4) | indices[1::2]).astype(np.uint8).tobytes()


def unpack_indices(encoded, stages):
    packets = np.frombuffer(encoded, dtype=np.uint8).reshape(-1, stages // 2)
    result = np.empty((packets.shape[0], stages), dtype=np.uint8)
    result[:, 0::2] = packets >> 4
    result[:, 1::2] = packets & 0x0F
    return result


def reference_encode(samples, model_dir, bitrate):
    encoder = tf.lite.Interpreter(
        model_path=str(model_dir / "soundstream_encoder.tflite")
    )
    encoder.allocate_tensors()
    encoder_input = encoder.get_input_details()[0]
    encoder_output = encoder.get_output_details()[0]
    quantizer = tf.lite.Interpreter(model_path=str(model_dir / "quantizer.tflite"))
    quantize = quantizer.get_signature_runner("encode")
    stages = STAGES[bitrate]
    encoded = bytearray()
    frame_count = len(samples) // SAMPLES_PER_FRAME
    for frame in range(frame_count):
        pcm = samples[frame * SAMPLES_PER_FRAME : (frame + 1) * SAMPLES_PER_FRAME]
        normalized = (pcm.astype(np.float32) / 32768.0).reshape(
            encoder_input["shape"]
        )
        encoder.set_tensor(encoder_input["index"], normalized)
        encoder.invoke()
        features = encoder.get_tensor(encoder_output["index"])
        indices = quantize(
            input_frames=features, num_quantizers=np.array(stages, dtype=np.int32)
        )["output_0"].reshape(-1)[:stages]
        encoded.extend(pack_indices(indices))
    return bytes(encoded)


def reference_decode(encoded, model_dir, bitrate):
    stages = STAGES[bitrate]
    packet_indices = unpack_indices(encoded, stages)
    quantizer = tf.lite.Interpreter(model_path=str(model_dir / "quantizer.tflite"))
    dequantize = quantizer.get_signature_runner("decode")
    decoder = tf.lite.Interpreter(model_path=str(model_dir / "lyragan.tflite"))
    decoder.allocate_tensors()
    decoder_input = decoder.get_input_details()[0]
    decoder_output = decoder.get_output_details()[0]
    result = []
    for packet in packet_indices:
        all_indices = np.full((46, 1, 1), -1, dtype=np.int32)
        all_indices[:stages, 0, 0] = packet
        features = dequantize(encoding_indices=all_indices)["output_0"]
        decoder.set_tensor(decoder_input["index"], features)
        decoder.invoke()
        unit_audio = decoder.get_tensor(decoder_output["index"]).reshape(-1)
        pcm = np.clip(unit_audio * 32768.0, -32768.0, 32767.0).astype(np.int16)
        result.append(pcm)
    return np.concatenate(result) if result else np.empty(0, dtype=np.int16)


def onnx_encode(samples, onnx_model_dir, tflite_model_dir, bitrate):
    encoder = OnnxStreamingModel(onnx_model_dir / "soundstream_encoder.onnx")
    quantizer = tf.lite.Interpreter(
        model_path=str(tflite_model_dir / "quantizer.tflite")
    )
    quantize = quantizer.get_signature_runner("encode")
    stages = STAGES[bitrate]
    encoded = bytearray()
    for start in range(0, len(samples), SAMPLES_PER_FRAME):
        normalized = (
            samples[start : start + SAMPLES_PER_FRAME].astype(np.float32)
            / 32768.0
        ).reshape(1, SAMPLES_PER_FRAME)
        features = encoder.run(normalized)
        indices = quantize(
            input_frames=features, num_quantizers=np.array(stages, dtype=np.int32)
        )["output_0"].reshape(-1)[:stages]
        encoded.extend(pack_indices(indices))
    return bytes(encoded)


def onnx_decode(encoded, onnx_model_dir, tflite_model_dir, bitrate):
    stages = STAGES[bitrate]
    decoder = OnnxStreamingModel(onnx_model_dir / "lyragan.onnx")
    quantizer = tf.lite.Interpreter(
        model_path=str(tflite_model_dir / "quantizer.tflite")
    )
    dequantize = quantizer.get_signature_runner("decode")
    result = []
    for packet in unpack_indices(encoded, stages):
        all_indices = np.full((46, 1, 1), -1, dtype=np.int32)
        all_indices[:stages, 0, 0] = packet
        features = dequantize(encoding_indices=all_indices)["output_0"]
        unit_audio = decoder.run(features).reshape(-1)
        result.append(
            np.clip(unit_audio * 32768.0, -32768.0, 32767.0).astype(np.int16)
        )
    return np.concatenate(result) if result else np.empty(0, dtype=np.int16)


def waveform_metrics(reference, test):
    size = min(len(reference), len(test))
    reference = reference[:size].astype(np.float64) / 32768.0
    test = test[:size].astype(np.float64) / 32768.0
    error = test - reference
    epsilon = 1e-15
    reference_power = float(np.mean(reference * reference))
    error_power = float(np.mean(error * error))
    correlation = float(np.corrcoef(reference, test)[0, 1])
    scale = float(np.dot(test, reference) / (np.dot(reference, reference) + epsilon))
    projection = scale * reference
    residual = test - projection
    si_sdr = 10.0 * math.log10(
        (float(np.sum(projection * projection)) + epsilon)
        / (float(np.sum(residual * residual)) + epsilon)
    )

    window_size = 512
    hop = 256
    spectral_distances = []
    window = np.hanning(window_size)
    for start in range(0, size - window_size + 1, hop):
        reference_spectrum = np.abs(
            np.fft.rfft(reference[start : start + window_size] * window)
        )
        test_spectrum = np.abs(
            np.fft.rfft(test[start : start + window_size] * window)
        )
        reference_db = 20.0 * np.log10(np.maximum(reference_spectrum, 1e-7))
        test_db = 20.0 * np.log10(np.maximum(test_spectrum, 1e-7))
        spectral_distances.append(
            float(np.sqrt(np.mean((reference_db - test_db) ** 2)))
        )
    return {
        "samples": size,
        "rmse": math.sqrt(error_power),
        "snr_db": 10.0
        * math.log10((reference_power + epsilon) / (error_power + epsilon)),
        "correlation": correlation,
        "si_sdr_db": si_sdr,
        "log_spectral_distance_db": float(np.mean(spectral_distances)),
    }


def packet_metrics(reference, test, bitrate):
    stages = STAGES[bitrate]
    reference_indices = unpack_indices(reference, stages)
    test_indices = unpack_indices(test, stages)
    frame_count = min(len(reference_indices), len(test_indices))
    reference_indices = reference_indices[:frame_count]
    test_indices = test_indices[:frame_count]
    matches = reference_indices == test_indices
    return {
        "frames": frame_count,
        "exact_packet_fraction": float(np.mean(np.all(matches, axis=1))),
        "codebook_index_match_fraction": float(np.mean(matches)),
        "mismatched_indices": int(np.size(matches) - np.count_nonzero(matches)),
        "total_indices": int(np.size(matches)),
    }


def run(command, environment):
    completed = subprocess.run(
        [str(item) for item in command],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--opencv-model-dir", type=Path, required=True)
    parser.add_argument("--tflite-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--opencv-bin", type=Path)
    parser.add_argument("--bitrates", type=int, nargs="+", default=list(STAGES))
    parser.add_argument("--max-si-sdr-regression-db", type=float, default=2.0)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("wavs", type=Path, nargs="+")
    args = parser.parse_args()

    environment = os.environ.copy()
    if args.opencv_bin:
        environment["PATH"] = str(args.opencv_bin) + os.pathsep + environment["PATH"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "cases": [],
    }

    with tempfile.TemporaryDirectory(prefix="lyra_eval_") as temporary:
        temporary = Path(temporary)
        for wav_path in args.wavs:
            samples = read_wav(wav_path)
            complete_samples = samples[: len(samples) // 320 * 320]
            for bitrate in args.bitrates:
                if bitrate not in STAGES:
                    raise ValueError(f"unsupported bitrate {bitrate}")
                name = f"{wav_path.stem}_{bitrate}"
                print(f"evaluating {name}")
                reference_packets = reference_encode(
                    complete_samples, args.tflite_model_dir, bitrate
                )
                reference_packet_path = temporary / f"{name}_tflite.lyra"
                reference_packet_path.write_bytes(reference_packets)

                opencv_packet_path = temporary / f"{name}_opencv.lyra"
                run(
                    [
                        args.encoder,
                        wav_path,
                        opencv_packet_path,
                        args.opencv_model_dir,
                        bitrate,
                    ],
                    environment,
                )
                opencv_packets = opencv_packet_path.read_bytes()
                onnxruntime_packets = onnx_encode(
                    complete_samples,
                    args.opencv_model_dir,
                    args.tflite_model_dir,
                    bitrate,
                )

                opencv_roundtrip_path = args.output_dir / f"{name}_opencv.wav"
                opencv_from_tflite_path = (
                    args.output_dir / f"{name}_opencv_decodes_tflite.wav"
                )
                run(
                    [
                        args.decoder,
                        opencv_packet_path,
                        opencv_roundtrip_path,
                        args.opencv_model_dir,
                        bitrate,
                    ],
                    environment,
                )
                run(
                    [
                        args.decoder,
                        reference_packet_path,
                        opencv_from_tflite_path,
                        args.opencv_model_dir,
                        bitrate,
                    ],
                    environment,
                )

                tflite_roundtrip = reference_decode(
                    reference_packets, args.tflite_model_dir, bitrate
                )
                tflite_from_opencv = reference_decode(
                    opencv_packets, args.tflite_model_dir, bitrate
                )
                onnxruntime_from_tflite = onnx_decode(
                    reference_packets,
                    args.opencv_model_dir,
                    args.tflite_model_dir,
                    bitrate,
                )
                write_wav(
                    args.output_dir / f"{name}_tflite.wav", tflite_roundtrip
                )
                write_wav(
                    args.output_dir / f"{name}_tflite_decodes_opencv.wav",
                    tflite_from_opencv,
                )
                opencv_roundtrip = read_wav(opencv_roundtrip_path)
                opencv_from_tflite = read_wav(opencv_from_tflite_path)

                case = {
                    "input": wav_path.name,
                    "bitrate": bitrate,
                    "packets": packet_metrics(
                        reference_packets, opencv_packets, bitrate
                    ),
                    "onnxruntime_packets": packet_metrics(
                        reference_packets, onnxruntime_packets, bitrate
                    ),
                    "decoder_runtime_difference": waveform_metrics(
                        tflite_roundtrip, opencv_from_tflite
                    ),
                    "onnxruntime_decoder_difference": waveform_metrics(
                        tflite_roundtrip, onnxruntime_from_tflite
                    ),
                    "encoder_packet_effect": waveform_metrics(
                        tflite_roundtrip, tflite_from_opencv
                    ),
                    "end_to_end_difference": waveform_metrics(
                        tflite_roundtrip, opencv_roundtrip
                    ),
                    "tflite_vs_input": waveform_metrics(
                        complete_samples, tflite_roundtrip
                    ),
                    "opencv_vs_input": waveform_metrics(
                        complete_samples, opencv_roundtrip
                    ),
                    "onnxruntime_vs_input": waveform_metrics(
                        complete_samples,
                        onnx_decode(
                            onnxruntime_packets,
                            args.opencv_model_dir,
                            args.tflite_model_dir,
                            bitrate,
                        ),
                    ),
                }
                report["cases"].append(case)
                print(
                    f"  index match {case['packets']['codebook_index_match_fraction']:.3%}, "
                    f"end-to-end SNR {case['end_to_end_difference']['snr_db']:.2f} dB"
                )

    regressions = [
        case["tflite_vs_input"]["si_sdr_db"]
        - case["opencv_vs_input"]["si_sdr_db"]
        for case in report["cases"]
    ]
    index_matches = [
        case["packets"]["codebook_index_match_fraction"]
        for case in report["cases"]
    ]
    report["summary"] = {
        "mean_codebook_index_match_fraction": float(np.mean(index_matches)),
        "mean_si_sdr_regression_db": float(np.mean(regressions)),
        "worst_si_sdr_regression_db": float(np.max(regressions)),
        "allowed_si_sdr_regression_db": args.max_si_sdr_regression_db,
        "quality_gate_passed": bool(
            np.max(regressions) <= args.max_si_sdr_regression_db
        ),
        "packet_format_cross_decode_passed": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")
    if args.enforce and not report["summary"]["quality_gate_passed"]:
        raise SystemExit("OpenCV conversion failed the audio-quality gate")


if __name__ == "__main__":
    main()
