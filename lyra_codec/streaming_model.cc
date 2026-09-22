// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/streaming_model.h"

#include <algorithm>
#include <numeric>
#include <utility>

namespace lyra {
namespace {

using StateSpec = StreamingModel::StateSpec;

StateSpec State(std::size_t index, std::vector<int> shape) {
  const std::string number = (index < 10 ? "0" : "") + std::to_string(index);
  return {"lyra_state_" + number + "_input",
          "lyra_state_" + number + "_output", std::move(shape)};
}

std::vector<StateSpec> EncoderStates() {
  return {
      State(0, {1, 2, 1, 64}),    State(1, {1, 6, 1, 64}),
      State(2, {1, 18, 1, 64}),   State(3, {1, 5, 1, 64}),
      State(4, {1, 2, 1, 128}),   State(5, {1, 6, 1, 128}),
      State(6, {1, 18, 1, 128}),  State(7, {1, 2, 1, 128}),
      State(8, {1, 2, 1, 256}),   State(9, {1, 48, 1, 1}),
      State(10, {1, 2, 1, 512}),  State(11, {1, 6, 1, 256}),
      State(12, {1, 18, 1, 256}), State(13, {1, 2, 1, 256}),
  };
}

std::vector<StateSpec> DecoderStates() {
  return {
      State(0, {1, 2, 1, 64}),    State(1, {1, 2, 1, 128}),
      State(2, {1, 6, 1, 128}),   State(3, {1, 18, 1, 128}),
      State(4, {1, 2, 1, 64}),    State(5, {1, 6, 1, 64}),
      State(6, {1, 18, 1, 64}),   State(7, {1, 5, 1, 64}),
      State(8, {1, 48, 1, 1}),    State(9, {1, 2, 1, 256}),
      State(10, {1, 6, 1, 256}),  State(11, {1, 18, 1, 256}),
      State(12, {1, 2, 1, 64}),   State(13, {1, 2, 1, 64}),
      State(14, {1, 2, 1, 64}),   State(15, {1, 2, 1, 64}),
      State(16, {1, 2, 1, 64}),   State(17, {1, 2, 1, 64}),
  };
}

std::size_t ElementCount(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         [](std::size_t product, int dimension) {
                           return product * static_cast<std::size_t>(dimension);
                         });
}

}  // namespace

std::unique_ptr<StreamingModel> StreamingModel::CreateEncoder(
    const std::filesystem::path& model_path, std::string* error) {
  return Create(model_path, true, error);
}

std::unique_ptr<StreamingModel> StreamingModel::CreateDecoder(
    const std::filesystem::path& model_path, std::string* error) {
  return Create(model_path, false, error);
}

std::unique_ptr<StreamingModel> StreamingModel::Create(
    const std::filesystem::path& model_path, bool encoder,
    std::string* error) {
  auto model = OpenCvModel::Create(model_path, error);
  if (model == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<StreamingModel>(new StreamingModel(
      std::move(model), "lyra_input",
      encoder ? std::vector<int>{1, 320} : std::vector<int>{1, 1, 64},
      "lyra_output",
      encoder ? EncoderStates() : DecoderStates()));
}

StreamingModel::StreamingModel(std::unique_ptr<OpenCvModel> model,
                               std::string input_name,
                               std::vector<int> input_shape,
                               std::string output_name,
                               std::vector<StateSpec> states)
    : model_(std::move(model)),
      input_name_(std::move(input_name)),
      input_shape_(std::move(input_shape)),
      output_name_(std::move(output_name)),
      state_specs_(std::move(states)) {
  Reset();
}

void StreamingModel::Reset() {
  state_values_.clear();
  state_values_.reserve(state_specs_.size());
  for (const StateSpec& state : state_specs_) {
    state_values_.push_back(
        {state.shape, std::vector<float>(ElementCount(state.shape), 0.0f)});
  }
}

bool StreamingModel::Run(const float* input, std::size_t input_size,
                         std::vector<float>* output, std::string* error) {
  if (input == nullptr || output == nullptr ||
      input_size != ElementCount(input_shape_)) {
    if (error != nullptr) {
      *error = "streaming model received an input with the wrong size";
    }
    return false;
  }

  std::vector<std::string> input_names{input_name_};
  std::vector<OpenCvModel::Tensor> inputs{
      {input_shape_, std::vector<float>(input, input + input_size)}};
  std::vector<std::string> output_names{output_name_};
  input_names.reserve(state_specs_.size() + 1);
  inputs.reserve(state_specs_.size() + 1);
  output_names.reserve(state_specs_.size() + 1);
  for (std::size_t i = 0; i < state_specs_.size(); ++i) {
    input_names.push_back(state_specs_[i].input_name);
    inputs.push_back(state_values_[i]);
    output_names.push_back(state_specs_[i].output_name);
  }

  std::vector<OpenCvModel::Tensor> results;
  if (!model_->Run(input_names, inputs, output_names, &results, error) ||
      results.size() != output_names.size()) {
    if (results.size() != output_names.size() && error != nullptr) {
      *error = "streaming model returned an unexpected number of tensors";
    }
    return false;
  }
  *output = std::move(results[0].values);
  for (std::size_t i = 0; i < state_values_.size(); ++i) {
    state_values_[i] = std::move(results[i + 1]);
  }
  return true;
}

}  // namespace lyra
