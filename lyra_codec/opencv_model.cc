// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/opencv_model.h"

#include <exception>
#include <numeric>
#include <utility>

#include <opencv2/dnn.hpp>

namespace lyra {

class OpenCvModel::Impl {
 public:
  explicit Impl(cv::dnn::Net network) : network(std::move(network)) {}
  cv::dnn::Net network;
};

std::unique_ptr<OpenCvModel> OpenCvModel::Create(
    const std::filesystem::path& model_path, std::string* error) {
  try {
    cv::dnn::Net network = cv::dnn::readNetFromONNX(model_path.string());
    if (network.empty()) {
      if (error != nullptr) {
        *error = "OpenCV returned an empty network for " + model_path.string();
      }
      return nullptr;
    }
    network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return std::unique_ptr<OpenCvModel>(
        new OpenCvModel(std::make_unique<Impl>(std::move(network))));
  } catch (const cv::Exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return nullptr;
  }
}

OpenCvModel::OpenCvModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

OpenCvModel::~OpenCvModel() = default;
OpenCvModel::OpenCvModel(OpenCvModel&&) noexcept = default;
OpenCvModel& OpenCvModel::operator=(OpenCvModel&&) noexcept = default;

std::vector<std::string> OpenCvModel::OutputNames() const {
  return impl_->network.getUnconnectedOutLayersNames();
}

bool OpenCvModel::Run(const std::vector<std::string>& input_names,
                      const std::vector<Tensor>& inputs,
                      const std::vector<std::string>& output_names,
                      std::vector<Tensor>* outputs, std::string* error) {
  if (inputs.size() != input_names.size() || outputs == nullptr) {
    if (error != nullptr) {
      *error = "OpenCV model received inconsistent input or output storage";
    }
    return false;
  }
  try {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const Tensor& tensor = inputs[i];
      const std::size_t expected_size = std::accumulate(
          tensor.shape.begin(), tensor.shape.end(), std::size_t{1},
          [](std::size_t product, int dimension) {
            return product * static_cast<std::size_t>(dimension);
          });
      if (tensor.shape.empty() || expected_size != tensor.values.size()) {
        if (error != nullptr) {
          *error = "OpenCV model input has an invalid shape";
        }
        return false;
      }
      cv::Mat blob(static_cast<int>(tensor.shape.size()), tensor.shape.data(),
                   CV_32F, const_cast<float*>(tensor.values.data()));
      impl_->network.setInput(blob, input_names[i]);
    }

    std::vector<cv::Mat> result;
    impl_->network.forward(result, output_names);
    outputs->clear();
    outputs->reserve(result.size());
    for (const cv::Mat& blob : result) {
      Tensor tensor;
      tensor.shape.assign(blob.size.p, blob.size.p + blob.dims);
      const cv::Mat contiguous = blob.isContinuous() ? blob : blob.clone();
      const float* begin = contiguous.ptr<float>();
      tensor.values.assign(begin, begin + contiguous.total());
      outputs->push_back(std::move(tensor));
    }
    return true;
  } catch (const cv::Exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return false;
  }
}

}  // namespace lyra
