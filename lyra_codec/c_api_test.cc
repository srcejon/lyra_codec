// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/c_api.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: c_api_test MODEL_DIRECTORY\n";
    return 2;
  }
  if (lyracodec_abi_version() != LYRACODEC_ABI_VERSION ||
      lyracodec_packet_bytes(3200) != 8 ||
      lyracodec_packet_bytes(6000) != 15 ||
      lyracodec_packet_bytes(9200) != 23 ||
      lyracodec_packet_bytes(1234) != 0) {
    std::cerr << "C API constants do not match the codec\n";
    return 1;
  }

  lyracodec_encoder* encoder = lyracodec_encoder_create(argv[1], 3200);
  lyracodec_decoder* decoder = lyracodec_decoder_create(argv[1]);
  if (!encoder || !decoder) {
    std::cerr << "create failed: " << lyracodec_last_error() << '\n';
    lyracodec_encoder_destroy(encoder);
    lyracodec_decoder_destroy(decoder);
    return 1;
  }

  std::array<int16_t, LYRACODEC_SAMPLES_PER_FRAME> input{};
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<int16_t>(
        8000.0 * std::sin(2.0 * 3.141592653589793 * i / 32.0));
  }
  std::array<uint8_t, LYRACODEC_MAX_PACKET_BYTES> packet{};
  std::array<int16_t, LYRACODEC_SAMPLES_PER_FRAME> output{};

  const int packet_size = lyracodec_encode(
      encoder, input.data(), packet.data(), static_cast<int>(packet.size()));
  if (packet_size != 8 ||
      lyracodec_decode(decoder, packet.data(), packet_size, output.data(),
                       static_cast<int>(output.size())) !=
          LYRACODEC_SAMPLES_PER_FRAME) {
    std::cerr << "round trip failed: " << lyracodec_last_error() << '\n';
    lyracodec_encoder_destroy(encoder);
    lyracodec_decoder_destroy(decoder);
    return 1;
  }

  for (int frame = 0; frame < 8; ++frame) {
    if (lyracodec_decode_lost(decoder, output.data(),
                              static_cast<int>(output.size())) !=
        LYRACODEC_SAMPLES_PER_FRAME) {
      std::cerr << "packet-loss concealment failed: "
                << lyracodec_last_error() << '\n';
      lyracodec_encoder_destroy(encoder);
      lyracodec_decoder_destroy(decoder);
      return 1;
    }
  }
  if (!lyracodec_decoder_is_comfort_noise(decoder)) {
    std::cerr << "comfort-noise state was not reached\n";
    lyracodec_encoder_destroy(encoder);
    lyracodec_decoder_destroy(decoder);
    return 1;
  }

  lyracodec_encoder_destroy(encoder);
  lyracodec_decoder_destroy(decoder);
  return 0;
}
