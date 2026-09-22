// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef LYRA_CODEC_C_API_H_
#define LYRA_CODEC_C_API_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(LYRA_CODEC_RUNTIME_BUILD)
#define LYRACODEC_API __declspec(dllexport)
#else
#define LYRACODEC_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define LYRACODEC_API __attribute__((visibility("default")))
#else
#define LYRACODEC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LYRACODEC_ABI_VERSION 2
#define LYRACODEC_SAMPLE_RATE 16000
#define LYRACODEC_SAMPLES_PER_FRAME 320
#define LYRACODEC_MAX_PACKET_BYTES 23

typedef struct lyracodec_encoder lyracodec_encoder;
typedef struct lyracodec_decoder lyracodec_decoder;

LYRACODEC_API int lyracodec_abi_version(void);
LYRACODEC_API const char* lyracodec_version(void);
LYRACODEC_API const char* lyracodec_built_model_directory(void);
LYRACODEC_API const char* lyracodec_last_error(void);

LYRACODEC_API int lyracodec_packet_bytes(int bitrate);

LYRACODEC_API lyracodec_encoder* lyracodec_encoder_create(
    const char* model_directory, int bitrate);
LYRACODEC_API int lyracodec_encode(lyracodec_encoder* encoder,
                                   const int16_t* pcm, uint8_t* packet,
                                   int packet_capacity);
LYRACODEC_API void lyracodec_encoder_reset(lyracodec_encoder* encoder);
LYRACODEC_API void lyracodec_encoder_destroy(lyracodec_encoder* encoder);

LYRACODEC_API lyracodec_decoder* lyracodec_decoder_create(
    const char* model_directory);
LYRACODEC_API int lyracodec_decode(lyracodec_decoder* decoder,
                                   const uint8_t* packet, int packet_bytes,
                                   int16_t* pcm, int pcm_capacity);
LYRACODEC_API int lyracodec_decode_lost(lyracodec_decoder* decoder,
                                        int16_t* pcm, int pcm_capacity);
LYRACODEC_API int lyracodec_decoder_is_comfort_noise(
    lyracodec_decoder* decoder);
LYRACODEC_API void lyracodec_decoder_reset(lyracodec_decoder* decoder);
LYRACODEC_API void lyracodec_decoder_destroy(lyracodec_decoder* decoder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LYRA_CODEC_C_API_H_
