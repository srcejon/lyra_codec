// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/native_quantizer.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using lyra::NativeQuantizer;

bool CheckCase(const NativeQuantizer& quantizer,
               const std::array<float, 64>& features,
               const std::vector<std::uint8_t>& expected) {
  std::string error;
  const auto actual = quantizer.Quantize(features.data(), features.size(),
                                         expected.size(), &error);
  if (!error.empty()) {
    std::cerr << error << '\n';
    return false;
  }
  if (actual != expected) {
    std::cerr << "quantizer indices differ\nexpected:";
    for (auto value : expected) std::cerr << ' ' << static_cast<int>(value);
    std::cerr << "\nactual:";
    for (auto value : actual) std::cerr << ' ' << static_cast<int>(value);
    std::cerr << '\n';
    return false;
  }

  const auto decoded = quantizer.Decode(actual.data(), actual.size(), &error);
  if (!error.empty() || decoded.size() != NativeQuantizer::kNumFeatures) {
    std::cerr << (error.empty() ? "decode returned wrong feature count" : error)
              << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: native_quantizer_test CODEBOOK\n";
    return 2;
  }
  std::string error;
  const auto quantizer = NativeQuantizer::Create(argv[1], &error);
  if (!quantizer) {
    std::cerr << error << '\n';
    return 1;
  }

  std::array<float, 64> zeros{};
  const std::vector<std::uint8_t> zero_expected = {
      8, 3, 8, 9, 3, 11, 13, 12, 4, 15, 11, 11, 4, 3, 6, 8};

  std::array<float, 64> ramp{};
  for (std::size_t i = 0; i < ramp.size(); ++i) {
    ramp[i] = (static_cast<float>(i) - 32.0f) / 32.0f;
  }
  const std::vector<std::uint8_t> ramp_expected = {
      8, 3, 15, 3, 14, 11, 13, 14, 6, 15, 7, 11, 4, 3, 6,
      8, 11, 8, 15, 0, 11, 8, 0, 8, 6, 6, 9, 3, 13, 7};

  std::array<float, 64> sine{};
  for (std::size_t i = 0; i < sine.size(); ++i) {
    sine[i] = std::sin(static_cast<float>(i) * 0.17f);
  }
  const std::vector<std::uint8_t> sine_expected = {
      8, 3, 1, 9, 7, 11, 15, 12, 7, 15, 13, 11, 4, 9, 10, 5,
      9, 8, 6, 4, 2, 7, 9, 6, 2, 7, 9, 9, 0, 7, 9, 5,
      2, 3, 2, 5, 5, 10, 7, 13, 10, 12, 0, 11, 13, 10};

  if (!CheckCase(*quantizer, zeros, zero_expected) ||
      !CheckCase(*quantizer, ramp, ramp_expected) ||
      !CheckCase(*quantizer, sine, sine_expected)) {
    return 1;
  }
  std::cout << "native quantizer matches TFLite reference indices\n";
  return 0;
}
