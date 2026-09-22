// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/lyra_codec.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: lyra_codec_test <model-directory>\n";
    return 2;
  }
  std::string error;
  auto encoder = lyra::Encoder::Create(argv[1], 3200,
                                                               &error);
  auto decoder =
      lyra::Decoder::Create(argv[1], &error);
  if (encoder == nullptr || decoder == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }

  std::vector<std::int16_t> input(320);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<std::int16_t>(
        12000.0 * std::sin(2.0 * 3.141592653589793 * i / 80.0));
  }
  for (int bitrate : {3200, 6000, 9200}) {
    if (!encoder->SetBitrate(bitrate, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    std::vector<std::uint8_t> packet;
    std::vector<std::int16_t> output;
    if (!encoder->Encode(input.data(), input.size(), &packet, &error) ||
        !decoder->Decode(packet.data(), packet.size(), &output, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    if (packet.size() != static_cast<std::size_t>(bitrate / 400) ||
        output.size() != input.size()) {
      std::cerr << "unexpected packet or audio frame size\n";
      return 1;
    }
  }
  std::cout << "encoded and decoded all supported bitrates\n";
  return 0;
}
