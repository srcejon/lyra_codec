// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/c_api.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "lyra_codec/lyra_codec.h"

#ifndef LYRA_CODEC_VERSION
#define LYRA_CODEC_VERSION "unknown"
#endif

#ifndef LYRA_CODEC_RUNTIME_MODEL_DIR
#define LYRA_CODEC_RUNTIME_MODEL_DIR ""
#endif

namespace {

thread_local std::string last_error;

void SetExceptionError() {
  try {
    throw;
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "unknown C++ exception";
  }
}

}  // namespace

struct lyracodec_encoder {
  std::unique_ptr<lyra::Encoder> encoder;
  int packet_bytes;
};

struct lyracodec_decoder {
  std::unique_ptr<lyra::Decoder> decoder;
};

int lyracodec_abi_version(void) { return LYRACODEC_ABI_VERSION; }

const char* lyracodec_version(void) {
  return "Lyra Codec " LYRA_CODEC_VERSION;
}

const char* lyracodec_built_model_directory(void) {
  return LYRA_CODEC_RUNTIME_MODEL_DIR;
}

const char* lyracodec_last_error(void) { return last_error.c_str(); }

int lyracodec_packet_bytes(int bitrate) {
  switch (bitrate) {
    case 3200:
      return 8;
    case 6000:
      return 15;
    case 9200:
      return 23;
    default:
      return 0;
  }
}

lyracodec_encoder* lyracodec_encoder_create(const char* model_directory,
                                             int bitrate) {
  last_error.clear();
  if (lyracodec_packet_bytes(bitrate) == 0) {
    last_error = "unsupported bit rate";
    return nullptr;
  }

  try {
    std::string error;
    auto encoder = lyra::Encoder::Create(
        std::filesystem::path(model_directory ? model_directory : ""), bitrate,
        &error);
    if (!encoder) {
      last_error = error;
      return nullptr;
    }

    auto result = std::make_unique<lyracodec_encoder>();
    result->encoder = std::move(encoder);
    result->packet_bytes = lyracodec_packet_bytes(bitrate);
    return result.release();
  } catch (...) {
    SetExceptionError();
    return nullptr;
  }
}

int lyracodec_encode(lyracodec_encoder* encoder, const int16_t* pcm,
                     uint8_t* packet, int packet_capacity) {
  last_error.clear();
  if (!encoder || !pcm || !packet) {
    last_error = "invalid encoder or buffer";
    return 0;
  }
  if (packet_capacity < encoder->packet_bytes) {
    last_error = "packet buffer too small";
    return 0;
  }

  try {
    std::vector<uint8_t> output;
    std::string error;
    if (!encoder->encoder->Encode(pcm, LYRACODEC_SAMPLES_PER_FRAME, &output,
                                  &error)) {
      last_error = error;
      return 0;
    }
    if (static_cast<int>(output.size()) > packet_capacity) {
      last_error = "packet buffer too small";
      return 0;
    }
    std::copy(output.begin(), output.end(), packet);
    return static_cast<int>(output.size());
  } catch (...) {
    SetExceptionError();
    return 0;
  }
}

void lyracodec_encoder_reset(lyracodec_encoder* encoder) {
  if (!encoder) return;
  try {
    encoder->encoder->Reset();
  } catch (...) {
    SetExceptionError();
  }
}

void lyracodec_encoder_destroy(lyracodec_encoder* encoder) {
  try {
    delete encoder;
  } catch (...) {
    SetExceptionError();
  }
}

lyracodec_decoder* lyracodec_decoder_create(const char* model_directory) {
  last_error.clear();
  try {
    std::string error;
    auto decoder = lyra::Decoder::Create(
        std::filesystem::path(model_directory ? model_directory : ""), &error);
    if (!decoder) {
      last_error = error;
      return nullptr;
    }

    auto result = std::make_unique<lyracodec_decoder>();
    result->decoder = std::move(decoder);
    return result.release();
  } catch (...) {
    SetExceptionError();
    return nullptr;
  }
}

int lyracodec_decode(lyracodec_decoder* decoder, const uint8_t* packet,
                     int packet_bytes, int16_t* pcm, int pcm_capacity) {
  last_error.clear();
  if (!decoder || !packet || !pcm || packet_bytes <= 0) {
    last_error = "invalid decoder, packet, or buffer";
    return 0;
  }

  try {
    std::vector<int16_t> output;
    std::string error;
    if (!decoder->decoder->Decode(packet, static_cast<size_t>(packet_bytes),
                                  &output, &error)) {
      last_error = error;
      return 0;
    }
    if (static_cast<int>(output.size()) > pcm_capacity) {
      last_error = "audio buffer too small";
      return 0;
    }
    std::copy(output.begin(), output.end(), pcm);
    return static_cast<int>(output.size());
  } catch (...) {
    SetExceptionError();
    return 0;
  }
}

int lyracodec_decode_lost(lyracodec_decoder* decoder, int16_t* pcm,
                          int pcm_capacity) {
  last_error.clear();
  if (!decoder || !pcm) {
    last_error = "invalid decoder or audio buffer";
    return 0;
  }

  try {
    std::vector<int16_t> output;
    std::string error;
    if (!decoder->decoder->DecodeLostFrame(&output, &error)) {
      last_error = error;
      return 0;
    }
    if (static_cast<int>(output.size()) > pcm_capacity) {
      last_error = "audio buffer too small";
      return 0;
    }
    std::copy(output.begin(), output.end(), pcm);
    return static_cast<int>(output.size());
  } catch (...) {
    SetExceptionError();
    return 0;
  }
}

int lyracodec_decoder_is_comfort_noise(lyracodec_decoder* decoder) {
  return decoder && decoder->decoder->is_comfort_noise() ? 1 : 0;
}

void lyracodec_decoder_reset(lyracodec_decoder* decoder) {
  if (!decoder) return;
  try {
    decoder->decoder->Reset();
  } catch (...) {
    SetExceptionError();
  }
}

void lyracodec_decoder_destroy(lyracodec_decoder* decoder) {
  try {
    delete decoder;
  } catch (...) {
    SetExceptionError();
  }
}
