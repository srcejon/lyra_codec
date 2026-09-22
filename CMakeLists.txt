cmake_minimum_required(VERSION 3.20)

project(lyra_codec VERSION 1.3.2 LANGUAGES CXX)

include(GNUInstallDirs)

option(LYRA_CODEC_BUILD_TESTS "Build Lyra Codec tests" OFF)
option(LYRA_CODEC_BUILD_TOOLS "Build Lyra Codec WAV tools and benchmark" OFF)
option(LYRA_CODEC_BUILD_RUNTIME
  "Build the dynamically loadable C API runtime" OFF)
set(LYRA_CODEC_RUNTIME_MODEL_DIR "${PROJECT_SOURCE_DIR}/lyra_codec/model"
  CACHE PATH "Model directory reported by the Lyra Codec runtime")

if(LYRA_CODEC_BUILD_TESTS)
  enable_testing()
endif()
add_subdirectory(lyra_codec)

install(FILES LICENSE README.md
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/lyra_codec")
