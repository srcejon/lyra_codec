// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef LYRA_CODEC_STREAMING_MODEL_H_
#define LYRA_CODEC_STREAMING_MODEL_H_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "lyra_codec/opencv_model.h"

namespace lyra {

// Stateful frame-by-frame wrapper for the converted Lyra neural networks.
class StreamingModel {
 public:
  struct StateSpec {
    std::string input_name;
    std::string output_name;
    std::vector<int> shape;
  };

  static std::unique_ptr<StreamingModel> CreateEncoder(
      const std::filesystem::path& model_path, std::string* error);
  static std::unique_ptr<StreamingModel> CreateDecoder(
      const std::filesystem::path& model_path, std::string* error);

  bool Run(const float* input, std::size_t input_size,
           std::vector<float>* output, std::string* error);
  void Reset();

 private:
  StreamingModel(std::unique_ptr<OpenCvModel> model, std::string input_name,
                 std::vector<int> input_shape, std::string output_name,
                 std::vector<StateSpec> states);

  static std::unique_ptr<StreamingModel> Create(
      const std::filesystem::path& model_path, bool encoder,
      std::string* error);

  std::unique_ptr<OpenCvModel> model_;
  std::string input_name_;
  std::vector<int> input_shape_;
  std::string output_name_;
  std::vector<StateSpec> state_specs_;
  std::vector<OpenCvModel::Tensor> state_values_;
};

}  // namespace lyra

#endif  // LYRA_CODEC_STREAMING_MODEL_H_
