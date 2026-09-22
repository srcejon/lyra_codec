// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef LYRA_CODEC_LYRA_CODEC_H_
#define LYRA_CODEC_LYRA_CODEC_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lyra {

class NativeQuantizer;
class StreamingModel;
class ComfortNoise;

inline constexpr int kSampleRateHz = 16000;
inline constexpr int kFrameRate = 50;
inline constexpr std::size_t kSamplesPerFrame = 320;

class Encoder {
 public:
  static std::unique_ptr<Encoder> Create(
      const std::filesystem::path& model_directory, int bitrate,
      std::string* error);

  ~Encoder();

  bool Encode(const std::int16_t* audio, std::size_t sample_count,
              std::vector<std::uint8_t>* packet, std::string* error);
  bool SetBitrate(int bitrate, std::string* error);
  void Reset();

  int bitrate() const;

 private:
  Encoder(std::unique_ptr<StreamingModel> model,
          std::unique_ptr<NativeQuantizer> quantizer,
          std::size_t quantizer_stages);

  std::unique_ptr<StreamingModel> model_;
  std::unique_ptr<NativeQuantizer> quantizer_;
  std::size_t quantizer_stages_;
};

class Decoder {
 public:
  static std::unique_ptr<Decoder> Create(
      const std::filesystem::path& model_directory, std::string* error);

  ~Decoder();

  bool Decode(const std::uint8_t* packet, std::size_t packet_size,
              std::vector<std::int16_t>* audio, std::string* error);

  // Conceals one missing 20 ms packet while advancing the decoder state.
  // Call this exactly once for each absent packet in the stream.
  bool DecodeLostFrame(std::vector<std::int16_t>* audio, std::string* error);

  // True after concealment has fully transitioned to comfort noise.
  bool is_comfort_noise() const;
  void Reset();

 private:
  Decoder(std::unique_ptr<StreamingModel> model,
          std::unique_ptr<NativeQuantizer> quantizer,
          std::unique_ptr<ComfortNoise> comfort_noise);

  bool DecodeFeatures(const float* features, std::size_t feature_count,
                      std::vector<std::int16_t>* audio, std::string* error);

  std::unique_ptr<StreamingModel> model_;
  std::unique_ptr<NativeQuantizer> quantizer_;
  std::unique_ptr<ComfortNoise> comfort_noise_;
  std::size_t consecutive_lost_frames_ = 0;
  float comfort_noise_mix_ = 0.0f;
};

}  // namespace lyra

#endif  // LYRA_CODEC_LYRA_CODEC_H_
