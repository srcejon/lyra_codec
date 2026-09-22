// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef LYRA_CODEC_COMFORT_NOISE_H_
#define LYRA_CODEC_COMFORT_NOISE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lyra {

// Small low-percentile-statistics comfort-noise generator. It tracks quiet
// recently decoded frames and uses their level and first-order spectral tilt
// to colour locally generated noise.
class ComfortNoise {
 public:
  void Observe(const std::vector<std::int16_t>& samples);
  std::vector<std::int16_t> Generate(std::size_t sample_count);
  void Reset();

 private:
  struct Estimate {
    float rms = 0.0f;
    float correlation = 0.0f;
  };

  static constexpr std::size_t kHistoryFrames = 50;

  std::vector<Estimate> history_;
  std::size_t history_position_ = 0;
  std::uint32_t random_state_ = 0x6d2b79f5u;
  float filter_state_ = 0.0f;
};

}  // namespace lyra

#endif  // LYRA_CODEC_COMFORT_NOISE_H_
