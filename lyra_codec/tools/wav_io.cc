// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "lyra_codec/tools/wav_io.h"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>

namespace lyra::tools {
namespace {

std::uint16_t ReadU16(const unsigned char* bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8);
}

std::uint32_t ReadU32(const unsigned char* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

void WriteU16(std::ostream& output, std::uint16_t value) {
  const std::array<char, 2> bytes = {
      static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff)};
  output.write(bytes.data(), bytes.size());
}

void WriteU32(std::ostream& output, std::uint32_t value) {
  const std::array<char, 4> bytes = {
      static_cast<char>(value & 0xff), static_cast<char>((value >> 8) & 0xff),
      static_cast<char>((value >> 16) & 0xff),
      static_cast<char>((value >> 24) & 0xff)};
  output.write(bytes.data(), bytes.size());
}

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

}  // namespace

bool ReadMono16KhzWav(const std::filesystem::path& path,
                      std::vector<std::int16_t>* samples,
                      std::string* error) {
  if (samples == nullptr) {
    SetError(error, "WAV output storage is null");
    return false;
  }
  std::ifstream input(path, std::ios::binary);
  std::array<unsigned char, 12> header{};
  if (!input.read(reinterpret_cast<char*>(header.data()), header.size()) ||
      std::memcmp(header.data(), "RIFF", 4) != 0 ||
      std::memcmp(header.data() + 8, "WAVE", 4) != 0) {
    SetError(error, "input is not a RIFF/WAVE file: " + path.string());
    return false;
  }

  bool found_format = false;
  bool found_data = false;
  std::uint16_t format = 0;
  std::uint16_t channels = 0;
  std::uint16_t bits_per_sample = 0;
  std::uint32_t sample_rate = 0;
  std::vector<unsigned char> data;
  while (input && (!found_format || !found_data)) {
    std::array<unsigned char, 8> chunk{};
    if (!input.read(reinterpret_cast<char*>(chunk.data()), chunk.size())) {
      break;
    }
    const std::uint32_t size = ReadU32(chunk.data() + 4);
    if (std::memcmp(chunk.data(), "fmt ", 4) == 0) {
      if (size < 16) {
        SetError(error, "WAV format chunk is too short");
        return false;
      }
      std::vector<unsigned char> bytes(size);
      if (!input.read(reinterpret_cast<char*>(bytes.data()), size)) {
        SetError(error, "could not read WAV format chunk");
        return false;
      }
      format = ReadU16(bytes.data());
      channels = ReadU16(bytes.data() + 2);
      sample_rate = ReadU32(bytes.data() + 4);
      bits_per_sample = ReadU16(bytes.data() + 14);
      found_format = true;
    } else if (std::memcmp(chunk.data(), "data", 4) == 0) {
      data.resize(size);
      if (!input.read(reinterpret_cast<char*>(data.data()), size)) {
        SetError(error, "could not read WAV sample data");
        return false;
      }
      found_data = true;
    } else {
      input.seekg(size, std::ios::cur);
    }
    if ((size & 1u) != 0u) {
      input.seekg(1, std::ios::cur);
    }
  }
  if (!found_format || !found_data || format != 1 || channels != 1 ||
      sample_rate != 16000 || bits_per_sample != 16 ||
      (data.size() % 2) != 0) {
    SetError(error, "WAV must be 16-bit PCM, mono, and 16000 Hz");
    return false;
  }
  samples->resize(data.size() / 2);
  for (std::size_t i = 0; i < samples->size(); ++i) {
    (*samples)[i] = static_cast<std::int16_t>(ReadU16(data.data() + 2 * i));
  }
  return true;
}

bool WriteMono16KhzWav(const std::filesystem::path& path,
                       const std::vector<std::int16_t>& samples,
                       std::string* error) {
  const std::size_t byte_count = samples.size() * sizeof(std::int16_t);
  if (byte_count > std::numeric_limits<std::uint32_t>::max() - 36u) {
    SetError(error, "WAV file is too large");
    return false;
  }
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    SetError(error, "could not create WAV file: " + path.string());
    return false;
  }
  output.write("RIFF", 4);
  WriteU32(output, static_cast<std::uint32_t>(36 + byte_count));
  output.write("WAVEfmt ", 8);
  WriteU32(output, 16);
  WriteU16(output, 1);
  WriteU16(output, 1);
  WriteU32(output, 16000);
  WriteU32(output, 32000);
  WriteU16(output, 2);
  WriteU16(output, 16);
  output.write("data", 4);
  WriteU32(output, static_cast<std::uint32_t>(byte_count));
  for (std::int16_t sample : samples) {
    WriteU16(output, static_cast<std::uint16_t>(sample));
  }
  if (!output) {
    SetError(error, "could not write WAV file: " + path.string());
    return false;
  }
  return true;
}

}  // namespace lyra::tools
