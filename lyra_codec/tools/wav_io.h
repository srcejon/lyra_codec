// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef LYRA_CODEC_TOOLS_WAV_IO_H_
#define LYRA_CODEC_TOOLS_WAV_IO_H_

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lyra::tools {

bool ReadMono16KhzWav(const std::filesystem::path& path,
                      std::vector<std::int16_t>* samples,
                      std::string* error);

bool WriteMono16KhzWav(const std::filesystem::path& path,
                       const std::vector<std::int16_t>& samples,
                       std::string* error);

}  // namespace lyra::tools

#endif  // LYRA_CODEC_TOOLS_WAV_IO_H_
