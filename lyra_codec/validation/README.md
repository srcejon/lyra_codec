# Lyra Codec validation

This directory records compatibility and audio-quality measurements for the
OpenCV implementation. The detailed per-file results are in
`windows_opencv_4_13.json`; listening WAVs are generated into the evaluator's
output directory and are intentionally not committed.

## Windows OpenCV 4.13 and 4.14 results

Test corpus: `testdata/sample1_16kHz.wav` and
`testdata/sample2_16kHz.wav`, at 3200, 6000, and 9200 bit/s (six cases, 948
encoded/decoded frames in total).

| Measurement | Mean result |
| --- | ---: |
| OpenCV encoder codebook-index match vs TFLite | 58.2% |
| ONNX Runtime encoder codebook-index match vs TFLite | 58.2% |
| OpenCV decoder difference vs TFLite, SI-SDR | 25.7 dB |
| ONNX Runtime decoder difference vs TFLite, SI-SDR | 25.7 dB |
| TFLite decoded quality vs input, SI-SDR | 6.6 dB |
| ONNX Runtime decoded quality vs input, SI-SDR | 6.8 dB |
| OpenCV decoded quality vs input, SI-SDR | 6.8 dB |

Both cross-decoding directions accepted every packet, so the packet layout and
native quantizer are interoperable. The OpenCV CPU backend passes the configured
quality gate: mean SI-SDR is 0.22 dB better than the TFLite reference and the
worst regression in any case is 0.31 dB, versus an allowed 2 dB.

The checked-in models are exported as float graphs. On these graphs OpenCV and
ONNX Runtime agree closely over recurrent execution: in a separate 10-frame
random-input comparison, the largest state difference was below 0.001 for the
encoder and below 0.000003 for the decoder.

Quantized Q/DQ graphs are not safe with OpenCV 4.13 or 4.14. ONNX Runtime
executes them correctly, but OpenCV corrupts graph outputs produced by
`DequantizeLinear`;
one state contained values on the order of 1e33. Adding arithmetic or layout
consumers and exposing the raw integer tensor did not make output handling
reliable. Exporting float graphs removes this backend-specific path. It grows
the two ONNX files to about 2.7 MB each, but preserves decoded quality.

The test was repeated with the C++ tools linked directly to
`opencv_core4140.dll` and `opencv_dnn4140.dll`. The quantized graphs still
failed with a 10.04 dB mean SI-SDR regression and only 17.66% mean codebook
index agreement. The float graphs retained the same 0.31 dB worst regression
and passed all four C++ tests, so the workaround remains necessary.

## Performance

Windows x64 Release, OpenCV CPU backend, 1000 frames per bitrate:

| Bitrate | Encode mean | Decode mean | Encode realtime | Decode realtime |
| ---: | ---: | ---: | ---: | ---: |
| 3200 | 1.223 ms | 1.710 ms | 16.35x | 11.70x |
| 6000 | 1.227 ms | 1.705 ms | 16.30x | 11.73x |
| 9200 | 1.271 ms | 1.747 ms | 15.74x | 11.45x |

Model loading took approximately 18 ms for both networks together.

## Packet-loss behavior

`lyra_codec_packet_loss_test` covers a one-second loss burst and recovery. The
decoder uses neural zero-feature concealment for 80 ms, crossfades for 40 ms,
then emits locally generated comfort noise based on low-energy recently decoded
frames. A speech listening fixture confirmed why the fallback is needed: neural
concealment alone converged to an exactly repeating 20 ms waveform during the
one-second outage. With comfort noise enabled, adjacent long-loss frames remain
non-identical, no samples clip, and received audio resumes through a 40 ms
crossfade.

## Reproducing

Build with `LYRA_CODEC_BUILD_TOOLS=ON`, put the OpenCV DLL directory on `PATH`,
and run `tools/evaluate_codec.py`. Add `--enforce` to return a failing
exit status when the 2 dB quality gate is exceeded. Use
`tools/compare_onnx_backends.py` to compare every recurrent output from OpenCV
and ONNX Runtime directly.
