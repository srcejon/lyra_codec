// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/opencv_model.h"
#include "lyra_codec/streaming_model.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: opencv_model_test <model.onnx>\n";
    return 2;
  }
  std::string error;
  auto model = lyra::OpenCvModel::Create(argv[1], &error);
  if (model == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto outputs = model->OutputNames();
  std::cout << "loaded model with " << outputs.size() << " outputs\n";
  if (outputs.empty()) {
    return 1;
  }

  const std::string path = argv[1];
  const bool encoder = path.find("encoder") != std::string::npos;
  auto streaming = encoder
                       ? lyra::StreamingModel::CreateEncoder(
                             path, &error)
                       : lyra::StreamingModel::CreateDecoder(
                             path, &error);
  if (streaming == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  std::vector<float> input(encoder ? 320 : 64);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = 0.1f * std::sin(static_cast<float>(i) * 0.03f);
  }
  std::vector<float> output;
  for (int frame = 0; frame < 3; ++frame) {
    if (!streaming->Run(input.data(), input.size(), &output, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
  }
  const std::size_t expected_size = encoder ? 64 : 320;
  if (output.size() != expected_size ||
      !std::all_of(output.begin(), output.end(),
                   [](float value) { return std::isfinite(value); })) {
    std::cerr << "model returned invalid output\n";
    return 1;
  }
  std::cout << "ran 3 stateful frames\n";
  return 0;
}
