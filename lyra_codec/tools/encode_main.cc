// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "lyra_codec/lyra_codec.h"
#include "lyra_codec/tools/wav_io.h"

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: lyra_encode <input.wav> <output.lyra> "
                 "<model-directory> [3200|6000|9200]\n";
    return 2;
  }
  int bitrate = 3200;
  try {
    if (argc == 5) {
      bitrate = std::stoi(argv[4]);
    }
  } catch (const std::exception&) {
    std::cerr << "invalid bitrate\n";
    return 2;
  }

  std::string error;
  std::vector<std::int16_t> samples;
  if (!lyra::tools::ReadMono16KhzWav(
          argv[1], &samples, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  auto encoder = lyra::Encoder::Create(
      argv[3], bitrate, &error);
  if (encoder == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  std::ofstream output(argv[2], std::ios::binary);
  if (!output) {
    std::cerr << "could not create encoded file: " << argv[2] << '\n';
    return 1;
  }

  const std::size_t frame_count =
      samples.size() / lyra::kSamplesPerFrame;
  std::vector<std::uint8_t> packet;
  for (std::size_t frame = 0; frame < frame_count; ++frame) {
    const std::int16_t* input =
        samples.data() + frame * lyra::kSamplesPerFrame;
    if (!encoder->Encode(input, lyra::kSamplesPerFrame,
                         &packet, &error)) {
      std::cerr << "frame " << frame << ": " << error << '\n';
      return 1;
    }
    output.write(reinterpret_cast<const char*>(packet.data()), packet.size());
  }
  if (!output || frame_count == 0) {
    std::cerr << "no complete 20 ms frames were encoded\n";
    return 1;
  }
  std::cout << "encoded " << frame_count << " frames ("
            << frame_count * 20 << " ms), " << bitrate << " bit/s\n";
  const std::size_t discarded =
      samples.size() % lyra::kSamplesPerFrame;
  if (discarded != 0) {
    std::cout << "discarded " << discarded << " trailing samples\n";
  }
  return 0;
}
