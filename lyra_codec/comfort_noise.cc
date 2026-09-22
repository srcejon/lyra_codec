// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/comfort_noise.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lyra {
namespace {

std::int16_t ClipToInt16(float sample) {
  const float clipped = std::clamp(
      sample, static_cast<float>(std::numeric_limits<std::int16_t>::min()),
      static_cast<float>(std::numeric_limits<std::int16_t>::max()));
  return static_cast<std::int16_t>(clipped);
}

}  // namespace

void ComfortNoise::Observe(const std::vector<std::int16_t>& samples) {
  if (samples.empty()) {
    return;
  }

  double mean = 0.0;
  for (std::int16_t sample : samples) {
    mean += sample;
  }
  mean /= static_cast<double>(samples.size());

  double energy = 0.0;
  double lag_one = 0.0;
  double previous = static_cast<double>(samples.front()) - mean;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const double current = static_cast<double>(samples[i]) - mean;
    energy += current * current;
    if (i != 0) {
      lag_one += current * previous;
    }
    previous = current;
  }

  Estimate estimate;
  estimate.rms = static_cast<float>(
      std::sqrt(energy / static_cast<double>(samples.size())));
  if (energy > 1.0) {
    estimate.correlation = static_cast<float>(lag_one / energy);
  }
  estimate.correlation =
      std::clamp(estimate.correlation, -0.9f, 0.9f);

  if (history_.size() < kHistoryFrames) {
    history_.push_back(estimate);
  } else {
    history_[history_position_] = estimate;
    history_position_ = (history_position_ + 1) % kHistoryFrames;
  }
}

std::vector<std::int16_t> ComfortNoise::Generate(std::size_t sample_count) {
  std::vector<std::int16_t> result(sample_count, 0);
  if (history_.empty()) {
    return result;
  }

  std::vector<Estimate> ordered = history_;
  const std::size_t low_percentile = ordered.size() / 10;
  std::nth_element(
      ordered.begin(), ordered.begin() + low_percentile, ordered.end(),
      [](const Estimate& left, const Estimate& right) {
        return left.rms < right.rms;
      });
  const Estimate& background = ordered[low_percentile];
  const float target_rms = std::min(background.rms, 2000.0f);
  if (target_rms < 1.0f) {
    return result;
  }

  const float correlation = background.correlation;
  const float innovation_scale =
      std::sqrt(3.0f * (1.0f - correlation * correlation)) * target_rms;
  for (std::size_t i = 0; i < sample_count; ++i) {
    // xorshift32 provides deterministic, allocation-free local randomness.
    random_state_ ^= random_state_ << 13;
    random_state_ ^= random_state_ >> 17;
    random_state_ ^= random_state_ << 5;
    const float white =
        static_cast<float>(random_state_) /
            static_cast<float>(std::numeric_limits<std::uint32_t>::max()) *
            2.0f -
        1.0f;
    filter_state_ = correlation * filter_state_ + innovation_scale * white;
    result[i] = ClipToInt16(filter_state_);
  }
  return result;
}

void ComfortNoise::Reset() {
  history_.clear();
  history_position_ = 0;
  random_state_ = 0x6d2b79f5u;
  filter_state_ = 0.0f;
}

}  // namespace lyra
