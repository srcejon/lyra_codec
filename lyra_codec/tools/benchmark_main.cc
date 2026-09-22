// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "lyra_codec/lyra_codec.h"

namespace {

using Clock = std::chrono::steady_clock;

void PrintStats(const char* label, std::vector<double> times_ms) {
  std::sort(times_ms.begin(), times_ms.end());
  const double mean =
      std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
  const double p50 = times_ms[times_ms.size() / 2];
  const double p95 = times_ms[static_cast<std::size_t>(
      std::floor(0.95 * static_cast<double>(times_ms.size() - 1)))];
  const double maximum = times_ms.back();
  std::cout << std::left << std::setw(9) << label << " mean " << std::setw(8)
            << mean << " p50 " << std::setw(8) << p50 << " p95 "
            << std::setw(8) << p95 << " max " << std::setw(8) << maximum
            << " ms, " << 20.0 / mean << "x realtime\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::cerr << "usage: lyra_benchmark <model-directory> "
                 "[3200|6000|9200] [frames]\n";
    return 2;
  }
  int bitrate = 3200;
  int frames = 500;
  try {
    if (argc >= 3) {
      bitrate = std::stoi(argv[2]);
    }
    if (argc == 4) {
      frames = std::stoi(argv[3]);
    }
  } catch (const std::exception&) {
    std::cerr << "invalid numeric argument\n";
    return 2;
  }
  if (frames <= 0) {
    std::cerr << "frame count must be positive\n";
    return 2;
  }

  std::string error;
  const auto load_start = Clock::now();
  auto encoder = lyra::Encoder::Create(
      argv[1], bitrate, &error);
  auto decoder =
      lyra::Decoder::Create(argv[1], &error);
  const double load_ms = std::chrono::duration<double, std::milli>(
                             Clock::now() - load_start)
                             .count();
  if (encoder == nullptr || decoder == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }

  std::vector<std::int16_t> samples(320);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    samples[i] = static_cast<std::int16_t>(
        10000.0 * std::sin(2.0 * 3.141592653589793 * i / 53.0));
  }
  std::vector<std::uint8_t> packet;
  std::vector<std::int16_t> decoded;
  for (int i = 0; i < 10; ++i) {
    if (!encoder->Encode(samples.data(), samples.size(), &packet, &error) ||
        !decoder->Decode(packet.data(), packet.size(), &decoded, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
  }
  encoder->Reset();
  decoder->Reset();

  std::vector<double> encode_times;
  std::vector<double> decode_times;
  encode_times.reserve(frames);
  decode_times.reserve(frames);
  for (int frame = 0; frame < frames; ++frame) {
    auto start = Clock::now();
    if (!encoder->Encode(samples.data(), samples.size(), &packet, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    encode_times.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    start = Clock::now();
    if (!decoder->Decode(packet.data(), packet.size(), &decoded, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    decode_times.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "model load: " << load_ms << " ms\n";
  PrintStats("encode", std::move(encode_times));
  PrintStats("decode", std::move(decode_times));
  return 0;
}
