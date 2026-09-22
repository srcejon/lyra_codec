# Lyra Codec: a generative low-bitrate speech codec

Lyra Codec is a fork of https://github.com/google/lyra

Lyra is a high-quality, low-bitrate speech codec at 3.2k, 6 and 9.2kbps.

This fork:
* uses OpenCV 4 with ONNX models instead of TFLite
* uses CMake instead of Bazel
* has removed many unnecessary dependencies so it compiles easily on Windows, Mac, Linux and Android.

### Building the Lyra Codec static library with CMake

The default CMake target is `lyra::codec`. It contains the streaming
encoder, decoder, packet packing, and a native residual vector quantizer. It
supports 16 kHz mono PCM in 320-sample (20 ms) frames at 3200, 6000, or 9200
bit/s. The decoder provides frame-level packet-loss concealment and transitions
to locally generated comfort noise during longer loss bursts. DTX and
sample-rate conversion are deliberately outside this small build.

OpenCV 4 with its `core` and `dnn` modules is the only external C++ dependency.
Point `OpenCV_DIR` at the directory containing `OpenCVConfig.cmake`.

On Linux and macOS, configure and build a release archive with:

```shell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake --build build --target lyra_codec --parallel
```

Add `-DLYRA_CODEC_BUILD_TESTS=ON` to build the model-import, quantizer, and
end-to-end codec tests. On Windows, those test executables also need the OpenCV
`x64/vc17/bin` directory on `PATH`. The supplied OpenCV package has release
libraries, so configure this build with `CMAKE_BUILD_TYPE=Release`.

On Windows, run these commands from a Visual Studio developer prompt:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DOpenCV_DIR=C:/Users/jon/source/repos/sdrangel-windows-libraries/opencv4
cmake --build build --target lyra_codec --parallel
```

The result is `build/lyra_codec/liblyra_codec.a` on Linux, macOS, and Android,
or `build/lyra_codec/lyra_codec.lib` with Ninja on Windows. Applications must
package the three files in `lyra_codec/model` and pass that directory to
`lyra::Encoder::Create` and `lyra::Decoder::Create`.

For Android, configure with the toolchain supplied by the NDK. For example, an
arm64 build targeting Android API 21 is:

```shell
cmake -S . -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21 \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/OpenCV-android-sdk/sdk/native/jni
cmake --build build-android --target lyra_codec --parallel
```

Other supported NDK ABIs can be selected with `ANDROID_ABI`. Each ABI needs its
own build directory. The Windows OpenCV installation shown above contains x64
Windows libraries and cannot be linked into an Android build; use an OpenCV 4
Android SDK or an Android build of OpenCV.

To consume Lyra in another CMake project, add this repository and link the
namespaced target:

```cmake
set(LYRA_CODEC_BUILD_TESTS OFF)
add_subdirectory(path/to/lyra)
target_link_libraries(my_target PRIVATE lyra::codec)
```

For an installed copy, use the exported `LyraCodec` package:

```cmake
find_package(LyraCodec CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE lyra::codec)
```

`LyraCodec_MODEL_DIR` contains the installed model directory after
`find_package` succeeds.

The public API is in `lyra_codec/lyra_codec.h`.

A frame-level encode/decode loop looks like this (error handling abbreviated):

```cpp
#include "lyra_codec/lyra_codec.h"

std::string error;
auto encoder = lyra::Encoder::Create(
    model_directory, 3200, &error);
auto decoder = lyra::Decoder::Create(
    model_directory, &error);

std::vector<std::uint8_t> packet;
encoder->Encode(pcm_16khz_mono.data(), 320, &packet, &error);

std::vector<std::int16_t> decoded;
decoder->Decode(packet.data(), packet.size(), &decoded, &error);

// Advance the decoder by one 20 ms frame when a packet is missing.
decoder->DecodeLostFrame(&decoded, &error);

if (decoder->is_comfort_noise()) {
  // The loss burst has moved beyond neural concealment.
}
```

To rebuild the checked-in ONNX and native quantizer assets, install the Python
packages in `tools/requirements-onnx.txt`, then run:

```shell
python tools/convert_streaming_tflite_to_onnx.py \
  tools/source_models/soundstream_encoder.tflite \
  lyra_codec/model/soundstream_encoder.onnx
python tools/convert_streaming_tflite_to_onnx.py \
  tools/source_models/lyragan.tflite lyra_codec/model/lyragan.onnx
python tools/extract_quantizer_codebook.py \
  tools/source_models/quantizer.tflite \
  lyra_codec/model/quantizer_codebook.bin
```

The conversion tools use TensorFlow, tf2onnx, ONNX, and Protobuf only while
regenerating assets; end users do not need them. The converter exports float
ONNX graphs by default. OpenCV 4.13 and 4.14 corrupt several recurrent outputs
in the Q/DQ form even though ONNX Runtime executes that form correctly. The
`--keep-quantized` option is retained for backend investigation, not for the
OpenCV runtime.

### WAV tools and benchmark

Set `LYRA_CODEC_BUILD_TOOLS=ON` to build three dependency-free command-line front
ends. The option defaults to `OFF`. The tools accept only 16-bit, 16 kHz, mono
WAV files:

```shell
lyra_encode input.wav output.lyra /path/to/lyra_codec/model 3200
lyra_decode output.lyra decoded.wav /path/to/lyra_codec/model 3200
lyra_benchmark /path/to/lyra_codec/model 3200 1000
```

For listening tests, the decoder tool can simulate a contiguous loss burst by
adding its zero-based start frame and frame count. This example conceals 500 ms
beginning one second into the stream:

```shell
lyra_decode input.lyra loss.wav /path/to/lyra_codec/model 3200 50 25
```

The `.lyra` output is a raw, fixed-size packet stream. Detailed
TFLite/ONNX/OpenCV compatibility and audio-quality results are
recorded in `lyra_codec/validation`. The float ONNX graphs pass the current
OpenCV 4.13 and 4.14 CPU quality gate: their worst SI-SDR regression against
TFLite is 0.31 dB with a 2 dB limit.

### Continuous integration and releases

GitHub Actions builds and tests the desktop library on Windows, macOS, and
Linux, and compile/link-checks the tests for Android arm64 (they cannot run on
the hosted Linux runner). CI uses OpenCV 4.14.0 on all platforms. The workflow
also builds the optional command-line tools so their sources cannot silently
become stale; they remain disabled by default for normal library builds and
release packages.

Each successful CI job uploads its install tree as a downloadable GitHub
Actions artifact, which GitHub delivers as a ZIP file. CI artifacts are retained
for 14 days and are named for their target platform. Desktop CI artifacts
include the optional command-line tools; the Android artifact contains the
static library, header, models, CMake package files, README, and license.

Pushing a tag whose name starts with `v` (for example, `v1.3.2`) creates a
GitHub release with install-tree ZIPs for Linux x64, Windows x64, macOS x64,
macOS arm64, Android arm64-v8a, and Android x86_64. Running the Release workflow
manually builds the same ZIPs as workflow artifacts without publishing a
release. Release archives contain the library, public header, model data,
CMake package files, README, and license. Each published release also includes
SHA-256 checksums. OpenCV remains a separate dependency.

#### Creating a release

1. Update the version in the top-level `CMakeLists.txt`, for example:

   ```cmake
   project(lyra_codec VERSION 1.3.3 LANGUAGES CXX)
   ```

2. Commit the release changes, push the commit, and wait for the CMake CI
   workflow to pass:

   ```shell
   git add CMakeLists.txt README.md
   git commit -m "Prepare release 1.3.3"
   git push origin HEAD
   ```

3. Create an annotated tag whose version matches the CMake project version,
   then push it:

   ```shell
   git tag -a v1.3.3 -m "Lyra 1.3.3"
   git push origin v1.3.3
   ```

The tag push starts the Release workflow. It rebuilds and validates every
package before creating the GitHub release, so no release is published if a
platform fails. Treat published tags as immutable; make corrections with a new
version rather than moving an existing tag.

For a pre-release check, run the Release workflow manually from the repository's
Actions page. A manual run creates downloadable workflow artifacts for every
platform but does not create a GitHub release. After a tagged release, download
the archives and `SHA256SUMS` from its GitHub release page and verify a package
with:

```shell
sha256sum -c SHA256SUMS
```

## License

Use of this source code is governed by a Apache v2.0 license that can be found
in the LICENSE file.
