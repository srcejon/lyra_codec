// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/lyra_codec.h"

#include "lyra_codec/comfort_noise.h"
#include "lyra_codec/native_quantizer.h"
#include "lyra_codec/streaming_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace lyra {
namespace {

bool BitrateToStages(int bitrate, std::size_t* stages) {
  switch (bitrate) {
    case 3200:
      *stages = 16;
      return true;
    case 6000:
      *stages = 30;
      return true;
    case 9200:
      *stages = 46;
      return true;
    default:
      return false;
  }
}

bool PacketSizeToStages(std::size_t packet_size, std::size_t* stages) {
  switch (packet_size) {
    case 8:
      *stages = 16;
      return true;
    case 15:
      *stages = 30;
      return true;
    case 23:
      *stages = 46;
      return true;
    default:
      return false;
  }
}

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

std::vector<std::uint8_t> PackIndices(
    const std::vector<std::uint8_t>& indices) {
  std::vector<std::uint8_t> packet(indices.size() / 2);
  for (std::size_t i = 0; i < packet.size(); ++i) {
    packet[i] = static_cast<std::uint8_t>((indices[2 * i] << 4) |
                                          indices[2 * i + 1]);
  }
  return packet;
}

std::vector<std::uint8_t> UnpackIndices(const std::uint8_t* packet,
                                        std::size_t packet_size) {
  std::vector<std::uint8_t> indices(packet_size * 2);
  for (std::size_t i = 0; i < packet_size; ++i) {
    indices[2 * i] = static_cast<std::uint8_t>(packet[i] >> 4);
    indices[2 * i + 1] = static_cast<std::uint8_t>(packet[i] & 0x0f);
  }
  return indices;
}

std::int16_t UnitToInt16(float value) {
  const float scaled = value * 32768.0f;
  const float clipped = std::clamp(
      scaled, static_cast<float>(std::numeric_limits<std::int16_t>::min()),
      static_cast<float>(std::numeric_limits<std::int16_t>::max()));
  return static_cast<std::int16_t>(clipped);
}

void MixComfortNoise(const std::vector<std::int16_t>& decoded,
                     const std::vector<std::int16_t>& comfort_noise,
                     float start_mix, float end_mix,
                     std::vector<std::int16_t>* output) {
  output->resize(decoded.size());
  for (std::size_t i = 0; i < decoded.size(); ++i) {
    const float progress =
        static_cast<float>(i + 1) / static_cast<float>(decoded.size());
    const float eased =
        0.5f - 0.5f * std::cos(progress * 3.14159265358979323846f);
    const float noise_mix = start_mix + (end_mix - start_mix) * eased;
    const float sample = decoded[i] * (1.0f - noise_mix) +
                         comfort_noise[i] * noise_mix;
    (*output)[i] = static_cast<std::int16_t>(std::clamp(
        sample, static_cast<float>(std::numeric_limits<std::int16_t>::min()),
        static_cast<float>(std::numeric_limits<std::int16_t>::max())));
  }
}

}  // namespace

std::unique_ptr<Encoder> Encoder::Create(
    const std::filesystem::path& model_directory, int bitrate,
    std::string* error) {
  std::size_t stages = 0;
  if (!BitrateToStages(bitrate, &stages)) {
    SetError(error, "bitrate must be 3200, 6000, or 9200 bits/s");
    return nullptr;
  }
  auto model = StreamingModel::CreateEncoder(
      model_directory / "soundstream_encoder.onnx", error);
  if (model == nullptr) {
    return nullptr;
  }
  auto quantizer = NativeQuantizer::Create(
      model_directory / "quantizer_codebook.bin", error);
  if (quantizer == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<Encoder>(
      new Encoder(std::move(model), std::move(quantizer), stages));
}

Encoder::Encoder(std::unique_ptr<StreamingModel> model,
                 std::unique_ptr<NativeQuantizer> quantizer,
                 std::size_t quantizer_stages)
    : model_(std::move(model)),
      quantizer_(std::move(quantizer)),
      quantizer_stages_(quantizer_stages) {}

Encoder::~Encoder() = default;

bool Encoder::Encode(const std::int16_t* audio, std::size_t sample_count,
                     std::vector<std::uint8_t>* packet,
                     std::string* error) {
  if (audio == nullptr || packet == nullptr ||
      sample_count != kSamplesPerFrame) {
    SetError(error, "encoder requires exactly 320 mono PCM samples");
    return false;
  }
  std::vector<float> normalized(sample_count);
  std::transform(audio, audio + sample_count, normalized.begin(),
                 [](std::int16_t sample) {
                   return static_cast<float>(sample) / 32768.0f;
                 });
  std::vector<float> features;
  if (!model_->Run(normalized.data(), normalized.size(), &features, error)) {
    return false;
  }
  const auto indices = quantizer_->Quantize(
      features.data(), features.size(), quantizer_stages_, error);
  if (indices.size() != quantizer_stages_) {
    return false;
  }
  *packet = PackIndices(indices);
  return true;
}

bool Encoder::SetBitrate(int bitrate, std::string* error) {
  std::size_t stages = 0;
  if (!BitrateToStages(bitrate, &stages)) {
    SetError(error, "bitrate must be 3200, 6000, or 9200 bits/s");
    return false;
  }
  quantizer_stages_ = stages;
  return true;
}

void Encoder::Reset() { model_->Reset(); }

int Encoder::bitrate() const {
  return static_cast<int>(quantizer_stages_ * NativeQuantizer::kBitsPerStage *
                          kFrameRate);
}

std::unique_ptr<Decoder> Decoder::Create(
    const std::filesystem::path& model_directory, std::string* error) {
  auto model =
      StreamingModel::CreateDecoder(model_directory / "lyragan.onnx", error);
  if (model == nullptr) {
    return nullptr;
  }
  auto quantizer = NativeQuantizer::Create(
      model_directory / "quantizer_codebook.bin", error);
  if (quantizer == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<Decoder>(new Decoder(
      std::move(model), std::move(quantizer), std::make_unique<ComfortNoise>()));
}

Decoder::Decoder(std::unique_ptr<StreamingModel> model,
                 std::unique_ptr<NativeQuantizer> quantizer,
                 std::unique_ptr<ComfortNoise> comfort_noise)
    : model_(std::move(model)),
      quantizer_(std::move(quantizer)),
      comfort_noise_(std::move(comfort_noise)) {}

Decoder::~Decoder() = default;

bool Decoder::Decode(const std::uint8_t* packet, std::size_t packet_size,
                     std::vector<std::int16_t>* audio, std::string* error) {
  std::size_t stages = 0;
  if (packet == nullptr || audio == nullptr ||
      !PacketSizeToStages(packet_size, &stages)) {
    SetError(error, "decoder packet size must be 8, 15, or 23 bytes");
    return false;
  }
  const auto indices = UnpackIndices(packet, packet_size);
  const auto features = quantizer_->Decode(indices.data(), stages, error);
  if (features.size() != NativeQuantizer::kNumFeatures) {
    return false;
  }
  std::vector<std::int16_t> decoded;
  if (!DecodeFeatures(features.data(), features.size(), &decoded, error)) {
    return false;
  }
  comfort_noise_->Observe(decoded);
  if (comfort_noise_mix_ > 0.0f) {
    const float next_mix = std::max(0.0f, comfort_noise_mix_ - 0.5f);
    const auto noise = comfort_noise_->Generate(kSamplesPerFrame);
    MixComfortNoise(decoded, noise, comfort_noise_mix_, next_mix, audio);
    comfort_noise_mix_ = next_mix;
  } else {
    *audio = std::move(decoded);
  }
  consecutive_lost_frames_ = 0;
  return true;
}

bool Decoder::DecodeLostFrame(std::vector<std::int16_t>* audio,
                              std::string* error) {
  if (audio == nullptr) {
    SetError(error, "decoder audio output storage is null");
    return false;
  }
  static const std::vector<float> estimated_features(
      NativeQuantizer::kNumFeatures, 0.0f);

  const bool start_comfort_noise = consecutive_lost_frames_ >= 4;
  const float next_mix =
      start_comfort_noise || comfort_noise_mix_ > 0.0f
          ? std::min(1.0f, comfort_noise_mix_ + 0.5f)
          : 0.0f;

  if (comfort_noise_mix_ == 1.0f && next_mix == 1.0f) {
    *audio = comfort_noise_->Generate(kSamplesPerFrame);
  } else {
    std::vector<std::int16_t> concealed;
    if (!DecodeFeatures(estimated_features.data(), estimated_features.size(),
                        &concealed, error)) {
      return false;
    }
    if (next_mix > 0.0f) {
      const auto noise = comfort_noise_->Generate(kSamplesPerFrame);
      MixComfortNoise(concealed, noise, comfort_noise_mix_, next_mix, audio);
    } else {
      *audio = std::move(concealed);
    }
  }
  comfort_noise_mix_ = next_mix;
  ++consecutive_lost_frames_;
  return true;
}

bool Decoder::DecodeFeatures(const float* features,
                             std::size_t feature_count,
                             std::vector<std::int16_t>* audio,
                             std::string* error) {
  if (features == nullptr || audio == nullptr ||
      feature_count != NativeQuantizer::kNumFeatures) {
    SetError(error, "decoder requires exactly 64 conditioning features");
    return false;
  }
  std::vector<float> unit_audio;
  if (!model_->Run(features, feature_count, &unit_audio, error) ||
      unit_audio.size() != kSamplesPerFrame) {
    if (unit_audio.size() != kSamplesPerFrame) {
      SetError(error, "decoder model returned an unexpected audio frame size");
    }
    return false;
  }
  audio->resize(unit_audio.size());
  std::transform(unit_audio.begin(), unit_audio.end(), audio->begin(),
                 UnitToInt16);
  return true;
}

bool Decoder::is_comfort_noise() const { return comfort_noise_mix_ == 1.0f; }

void Decoder::Reset() {
  model_->Reset();
  comfort_noise_->Reset();
  consecutive_lost_frames_ = 0;
  comfort_noise_mix_ = 0.0f;
}

}  // namespace lyra
