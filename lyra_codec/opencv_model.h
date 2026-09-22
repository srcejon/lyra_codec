// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef LYRA_CODEC_OPENCV_MODEL_H_
#define LYRA_CODEC_OPENCV_MODEL_H_

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lyra {

// PImpl keeps OpenCV types out of Lyra's public headers.
class OpenCvModel {
 public:
  struct Tensor {
    std::vector<int> shape;
    std::vector<float> values;
  };

  static std::unique_ptr<OpenCvModel> Create(
      const std::filesystem::path& model_path, std::string* error);

  ~OpenCvModel();
  OpenCvModel(OpenCvModel&&) noexcept;
  OpenCvModel& operator=(OpenCvModel&&) noexcept;
  OpenCvModel(const OpenCvModel&) = delete;
  OpenCvModel& operator=(const OpenCvModel&) = delete;

  std::vector<std::string> OutputNames() const;

  bool Run(const std::vector<std::string>& input_names,
           const std::vector<Tensor>& inputs,
           const std::vector<std::string>& output_names,
           std::vector<Tensor>* outputs, std::string* error);

 private:
  class Impl;
  explicit OpenCvModel(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace lyra

#endif  // LYRA_CODEC_OPENCV_MODEL_H_
