// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: runtime_load_test RUNTIME_LIBRARY\n";
    return 2;
  }

#if defined(_WIN32)
  HMODULE library = LoadLibraryA(argv[1]);
  if (!library) {
    std::cerr << "LoadLibrary failed with error " << GetLastError() << '\n';
    return 1;
  }
  auto abi_version = reinterpret_cast<int (*)(void)>(
      GetProcAddress(library, "lyracodec_abi_version"));
#else
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    std::cerr << "dlopen failed: " << dlerror() << '\n';
    return 1;
  }
  auto abi_version =
      reinterpret_cast<int (*)(void)>(dlsym(library, "lyracodec_abi_version"));
#endif

  if (!abi_version || abi_version() != 2) {
    std::cerr << "runtime ABI symbol is missing or incompatible\n";
#if defined(_WIN32)
    FreeLibrary(library);
#else
    dlclose(library);
#endif
    return 1;
  }

#if defined(_WIN32)
  FreeLibrary(library);
#else
  dlclose(library);
#endif
  return 0;
}
