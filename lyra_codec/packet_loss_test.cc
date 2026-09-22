// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/lyra_codec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using lyra::Decoder;
using lyra::Encoder;
using lyra::kSamplesPerFrame;

std::vector<std::int16_t> TestFrame(int frame_index) {
  std::vector<std::int16_t> samples(kSamplesPerFrame);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const double sample_index =
        static_cast<double>(frame_index * kSamplesPerFrame + i);
    samples[i] = static_cast<std::int16_t>(
        9000.0 * std::sin(2.0 * 3.141592653589793 * sample_index / 80.0) +
        3000.0 * std::sin(2.0 * 3.141592653589793 * sample_index / 37.0));
  }
  return samples;
}

bool IsValidFrame(const std::vector<std::int16_t>& samples) {
  return samples.size() == kSamplesPerFrame;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: lyra_codec_packet_loss_test <model-directory>\n";
    return 2;
  }

  std::string error;
  auto encoder = Encoder::Create(argv[1], 3200, &error);
  auto decoder = Decoder::Create(argv[1], &error);
  if (encoder == nullptr || decoder == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }

  std::vector<std::uint8_t> packet;
  std::vector<std::int16_t> decoded;

  // Warm up the recurrent encoder and decoder with received packets.
  for (int frame = 0; frame < 8; ++frame) {
    const auto input = TestFrame(frame);
    if (!encoder->Encode(input.data(), input.size(), &packet, &error) ||
        !decoder->Decode(packet.data(), packet.size(), &decoded, &error) ||
        !IsValidFrame(decoded)) {
      std::cerr << "warm-up frame " << frame << ": " << error << '\n';
      return 1;
    }
  }

  // Exercise one second of consecutive loss. This is intentionally longer
  // than Lyra's original 80 ms pure-concealment interval so instability is
  // caught even before a comfort-noise fallback is introduced.
  int peak = 0;
  long long total_energy = 0;
  long long comfort_noise_difference = 0;
  std::vector<std::int16_t> previous_lost_frame;
  for (int frame = 0; frame < 50; ++frame) {
    if (!decoder->DecodeLostFrame(&decoded, &error) ||
        !IsValidFrame(decoded)) {
      std::cerr << "lost frame " << frame << ": " << error << '\n';
      return 1;
    }
    for (std::int16_t sample : decoded) {
      peak = std::max(peak, std::abs(static_cast<int>(sample)));
      total_energy += static_cast<long long>(sample) * sample;
    }
    if (frame == 3 && decoder->is_comfort_noise()) {
      std::cerr << "comfort noise started before 80 ms of loss\n";
      return 1;
    }
    if (frame >= 6 && !decoder->is_comfort_noise()) {
      std::cerr << "comfort noise did not start after the 40 ms fade\n";
      return 1;
    }
    if (frame >= 7) {
      for (std::size_t i = 0; i < decoded.size(); ++i) {
        comfort_noise_difference +=
            std::abs(static_cast<int>(decoded[i]) - previous_lost_frame[i]);
      }
    }
    previous_lost_frame = decoded;
  }
  if (peak == 0 || total_energy == 0 || comfort_noise_difference == 0) {
    std::cerr << "concealment or comfort noise was unexpectedly silent or "
                 "repetitive\n";
    return 1;
  }

  // A real packet must still decode after the loss burst.
  const auto recovery_input = TestFrame(58);
  if (!encoder->Encode(recovery_input.data(), recovery_input.size(), &packet,
                       &error) ||
      !decoder->Decode(packet.data(), packet.size(), &decoded, &error) ||
      !IsValidFrame(decoded)) {
    std::cerr << "recovery frame: " << error << '\n';
    return 1;
  }
  if (decoder->is_comfort_noise()) {
    std::cerr << "decoder remained in pure comfort noise during recovery\n";
    return 1;
  }

  if (decoder->DecodeLostFrame(nullptr, &error) || error.empty()) {
    std::cerr << "null output was not rejected\n";
    return 1;
  }

  decoder->Reset();
  if (decoder->is_comfort_noise() ||
      !decoder->DecodeLostFrame(&decoded, &error) || !IsValidFrame(decoded) ||
      decoder->is_comfort_noise()) {
    std::cerr << "loss immediately after reset: " << error << '\n';
    return 1;
  }

  std::cout << "concealed 50 frames and recovered; peak=" << peak << '\n';
  return 0;
}
