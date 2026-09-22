// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "lyra_codec/lyra_codec.h"
#include "lyra_codec/tools/wav_io.h"

namespace {

std::size_t PacketSize(int bitrate) {
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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5 && argc != 7) {
    std::cerr << "usage: lyra_decode <input.lyra> <output.wav> "
                 "<model-directory> [3200|6000|9200] "
                 "[loss-start-frame loss-frame-count]\n";
    return 2;
  }
  int bitrate = 3200;
  std::size_t loss_start = std::numeric_limits<std::size_t>::max();
  std::size_t loss_count = 0;
  try {
    if (argc >= 5) {
      bitrate = std::stoi(argv[4]);
    }
    if (argc == 7) {
      loss_start = std::stoull(argv[5]);
      loss_count = std::stoull(argv[6]);
    }
  } catch (const std::exception&) {
    std::cerr << "invalid bitrate or loss range\n";
    return 2;
  }
  const std::size_t packet_size = PacketSize(bitrate);
  if (packet_size == 0) {
    std::cerr << "bitrate must be 3200, 6000, or 9200\n";
    return 2;
  }

  std::ifstream input(argv[1], std::ios::binary);
  if (!input) {
    std::cerr << "could not open encoded file: " << argv[1] << '\n';
    return 1;
  }
  const std::vector<std::uint8_t> encoded{
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  if (encoded.empty() || encoded.size() % packet_size != 0) {
    std::cerr << "encoded file size is not a whole number of packets\n";
    return 1;
  }

  std::string error;
  auto decoder =
      lyra::Decoder::Create(argv[3], &error);
  if (decoder == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  std::vector<std::int16_t> samples;
  std::vector<std::int16_t> frame_samples;
  const std::size_t frame_count = encoded.size() / packet_size;
  samples.reserve(frame_count * lyra::kSamplesPerFrame);
  for (std::size_t frame = 0; frame < frame_count; ++frame) {
    const bool packet_lost =
        frame >= loss_start && frame - loss_start < loss_count;
    const bool decoded =
        packet_lost
            ? decoder->DecodeLostFrame(&frame_samples, &error)
            : decoder->Decode(encoded.data() + frame * packet_size,
                              packet_size, &frame_samples, &error);
    if (!decoded) {
      std::cerr << "frame " << frame << ": " << error << '\n';
      return 1;
    }
    samples.insert(samples.end(), frame_samples.begin(), frame_samples.end());
  }
  if (!lyra::tools::WriteMono16KhzWav(
          argv[2], samples, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  std::cout << "decoded " << frame_count << " frames ("
            << frame_count * 20 << " ms), " << bitrate << " bit/s\n";
  if (loss_count != 0) {
    const std::size_t concealed =
        loss_start < frame_count
            ? std::min(loss_count, frame_count - loss_start)
            : 0;
    std::cout << "concealed " << concealed << " missing frames\n";
  }
  return 0;
}
