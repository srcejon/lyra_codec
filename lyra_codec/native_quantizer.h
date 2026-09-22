// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef LYRA_CODEC_NATIVE_QUANTIZER_H_
#define LYRA_CODEC_NATIVE_QUANTIZER_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lyra {

class NativeQuantizer {
 public:
  static constexpr std::size_t kNumFeatures = 64;
  static constexpr std::size_t kNumStages = 46;
  static constexpr std::size_t kEntriesPerStage = 16;
  static constexpr int kBitsPerStage = 4;

  static std::unique_ptr<NativeQuantizer> Create(
      const std::filesystem::path& codebook_path, std::string* error);

  std::vector<std::uint8_t> Quantize(const float* features,
                                     std::size_t feature_count,
                                     std::size_t num_stages,
                                     std::string* error) const;

  std::vector<float> Decode(const std::uint8_t* indices,
                            std::size_t num_indices,
                            std::string* error) const;

 private:
  explicit NativeQuantizer(std::vector<float> codebooks);

  const float* CodebookEntry(std::size_t stage, std::size_t entry) const;

  std::vector<float> codebooks_;
};

}  // namespace lyra

#endif  // LYRA_CODEC_NATIVE_QUANTIZER_H_
