// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/native_quantizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

namespace lyra {
namespace {

constexpr std::array<char, 8> kMagic = {'L', 'Y', 'R', 'A', 'Q', 'V', '1', '\0'};
constexpr std::uint32_t kModelVersion = 3;

struct CodebookHeader {
  std::array<char, 8> magic;
  std::uint32_t model_version;
  std::uint32_t num_stages;
  std::uint32_t entries_per_stage;
  std::uint32_t num_features;
  std::uint32_t bits_per_stage;
};

static_assert(sizeof(CodebookHeader) == 28, "unexpected codebook header layout");

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

std::unique_ptr<NativeQuantizer> NativeQuantizer::Create(
    const std::filesystem::path& codebook_path, std::string* error) {
  std::ifstream input(codebook_path, std::ios::binary | std::ios::ate);
  if (!input) {
    SetError(error, "could not open quantizer codebook: " +
                        codebook_path.string());
    return nullptr;
  }

  const std::streamsize file_size = input.tellg();
  const std::size_t value_count =
      kNumStages * kEntriesPerStage * kNumFeatures;
  const std::size_t expected_size =
      sizeof(CodebookHeader) + value_count * sizeof(float);
  if (file_size < 0 || static_cast<std::size_t>(file_size) != expected_size) {
    SetError(error, "quantizer codebook has an unexpected size");
    return nullptr;
  }
  input.seekg(0);

  CodebookHeader header{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input || header.magic != kMagic ||
      header.model_version != kModelVersion ||
      header.num_stages != kNumStages ||
      header.entries_per_stage != kEntriesPerStage ||
      header.num_features != kNumFeatures ||
      header.bits_per_stage != kBitsPerStage) {
    SetError(error, "quantizer codebook header is incompatible");
    return nullptr;
  }

  std::vector<float> codebooks(value_count);
  input.read(reinterpret_cast<char*>(codebooks.data()),
             static_cast<std::streamsize>(value_count * sizeof(float)));
  if (!input) {
    SetError(error, "could not read quantizer codebook values");
    return nullptr;
  }
  return std::unique_ptr<NativeQuantizer>(
      new NativeQuantizer(std::move(codebooks)));
}

NativeQuantizer::NativeQuantizer(std::vector<float> codebooks)
    : codebooks_(std::move(codebooks)) {}

const float* NativeQuantizer::CodebookEntry(std::size_t stage,
                                             std::size_t entry) const {
  return codebooks_.data() +
         (stage * kEntriesPerStage + entry) * kNumFeatures;
}

std::vector<std::uint8_t> NativeQuantizer::Quantize(
    const float* features, std::size_t feature_count, std::size_t num_stages,
    std::string* error) const {
  if (features == nullptr || feature_count != kNumFeatures) {
    SetError(error, "quantizer input must contain exactly 64 features");
    return {};
  }
  if (num_stages == 0 || num_stages > kNumStages) {
    SetError(error, "quantizer stage count must be between 1 and 46");
    return {};
  }

  std::array<float, kNumFeatures> residual{};
  std::copy(features, features + kNumFeatures, residual.begin());
  std::vector<std::uint8_t> result;
  result.reserve(num_stages);

  for (std::size_t stage = 0; stage < num_stages; ++stage) {
    float best_distance = std::numeric_limits<float>::infinity();
    std::size_t best_entry = 0;
    for (std::size_t entry = 0; entry < kEntriesPerStage; ++entry) {
      const float* code = CodebookEntry(stage, entry);
      float distance = 0.0f;
      for (std::size_t feature = 0; feature < kNumFeatures; ++feature) {
        const float difference = residual[feature] - code[feature];
        distance += difference * difference;
      }
      if (distance < best_distance) {
        best_distance = distance;
        best_entry = entry;
      }
    }

    result.push_back(static_cast<std::uint8_t>(best_entry));
    const float* selected = CodebookEntry(stage, best_entry);
    for (std::size_t feature = 0; feature < kNumFeatures; ++feature) {
      residual[feature] -= selected[feature];
    }
  }
  return result;
}

std::vector<float> NativeQuantizer::Decode(const std::uint8_t* indices,
                                            std::size_t num_indices,
                                            std::string* error) const {
  if (indices == nullptr || num_indices == 0 || num_indices > kNumStages) {
    SetError(error, "quantizer indices must contain between 1 and 46 values");
    return {};
  }

  std::vector<float> features(kNumFeatures, 0.0f);
  for (std::size_t stage = 0; stage < num_indices; ++stage) {
    if (indices[stage] >= kEntriesPerStage) {
      SetError(error, "quantizer index is outside the codebook");
      return {};
    }
    const float* selected = CodebookEntry(stage, indices[stage]);
    for (std::size_t feature = 0; feature < kNumFeatures; ++feature) {
      features[feature] += selected[feature];
    }
  }
  return features;
}

}  // namespace lyra
