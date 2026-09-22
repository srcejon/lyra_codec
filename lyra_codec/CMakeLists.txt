find_package(OpenCV 4 REQUIRED CONFIG COMPONENTS core dnn)

if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS
    "A Debug consumer requires OpenCV debug libraries. The supplied Windows "
    "OpenCV package may contain release libraries only; Release builds are "
    "recommended for that package.")
endif()

add_library(lyra_codec STATIC
  comfort_noise.cc
  comfort_noise.h
  lyra_codec.cc
  lyra_codec.h
  native_quantizer.cc
  native_quantizer.h
  opencv_model.cc
  opencv_model.h
  streaming_model.cc
  streaming_model.h)
add_library(lyra::codec ALIAS lyra_codec)

target_include_directories(lyra_codec PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(lyra_codec PUBLIC cxx_std_17)
target_link_libraries(lyra_codec PRIVATE opencv_core opencv_dnn)

set_target_properties(lyra_codec PROPERTIES
  OUTPUT_NAME lyra_codec
  EXPORT_NAME codec
  POSITION_INDEPENDENT_CODE ON
  CXX_EXTENSIONS OFF
  DEBUG_POSTFIX d)

if(LYRA_CODEC_BUILD_TESTS)
  add_executable(lyra_codec_quantizer_test native_quantizer_test.cc)
  target_link_libraries(lyra_codec_quantizer_test PRIVATE lyra_codec)
  add_test(NAME lyra_codec_quantizer_test
    COMMAND lyra_codec_quantizer_test
      "${PROJECT_SOURCE_DIR}/lyra_codec/model/quantizer_codebook.bin")

  add_executable(lyra_codec_opencv_model_test opencv_model_test.cc)
  target_link_libraries(lyra_codec_opencv_model_test PRIVATE lyra_codec)
  add_test(NAME lyra_codec_encoder_model_test
    COMMAND lyra_codec_opencv_model_test
      "${PROJECT_SOURCE_DIR}/lyra_codec/model/soundstream_encoder.onnx")
  add_test(NAME lyra_codec_decoder_model_test
    COMMAND lyra_codec_opencv_model_test
      "${PROJECT_SOURCE_DIR}/lyra_codec/model/lyragan.onnx")

  add_executable(lyra_codec_test lyra_codec_test.cc)
  target_link_libraries(lyra_codec_test PRIVATE lyra_codec)
  add_test(NAME lyra_codec_test
    COMMAND lyra_codec_test "${PROJECT_SOURCE_DIR}/lyra_codec/model")

  add_executable(lyra_codec_packet_loss_test packet_loss_test.cc)
  target_link_libraries(lyra_codec_packet_loss_test PRIVATE lyra_codec)
  add_test(NAME lyra_codec_packet_loss_test
    COMMAND lyra_codec_packet_loss_test "${PROJECT_SOURCE_DIR}/lyra_codec/model")
endif()

if(LYRA_CODEC_BUILD_TOOLS)
  add_library(lyra_codec_wav_io STATIC tools/wav_io.cc tools/wav_io.h)
  target_include_directories(lyra_codec_wav_io PRIVATE "${PROJECT_SOURCE_DIR}")
  target_compile_features(lyra_codec_wav_io PUBLIC cxx_std_17)
  set_target_properties(lyra_codec_wav_io PROPERTIES CXX_EXTENSIONS OFF)

  add_executable(lyra_encode tools/encode_main.cc)
  target_link_libraries(lyra_encode PRIVATE lyra_codec lyra_codec_wav_io)

  add_executable(lyra_decode tools/decode_main.cc)
  target_link_libraries(lyra_decode PRIVATE lyra_codec lyra_codec_wav_io)

  add_executable(lyra_benchmark tools/benchmark_main.cc)
  target_link_libraries(lyra_benchmark PRIVATE lyra_codec)

  install(TARGETS lyra_encode lyra_decode lyra_benchmark
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

install(TARGETS lyra_codec EXPORT LyraCodecTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES lyra_codec.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lyra_codec")
install(FILES
  model/soundstream_encoder.onnx
  model/lyragan.onnx
  model/quantizer_codebook.bin
  DESTINATION "${CMAKE_INSTALL_DATADIR}/lyra_codec/model")

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/LyraCodecConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/LyraCodecConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/LyraCodec")
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/LyraCodecConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)
install(EXPORT LyraCodecTargets
  FILE LyraCodecTargets.cmake
  NAMESPACE lyra::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/LyraCodec")
install(FILES
  "${PROJECT_BINARY_DIR}/LyraCodecConfig.cmake"
  "${PROJECT_BINARY_DIR}/LyraCodecConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/LyraCodec")
