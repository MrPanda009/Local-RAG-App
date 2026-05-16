cmake_minimum_required(VERSION 3.22.1)
project(graphrag)

# ---------------------------------------------------------------------------
# llama.cpp build flags — trim what we don't need for Android
# ---------------------------------------------------------------------------
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

# Enable ARM NEON SIMD intrinsics (arm64-v8a has these)
set(GGML_NEON ON CACHE BOOL "" FORCE)

# Vulkan backend — comment out until you want to test GPU inference
# set(GGML_VULKAN ON CACHE BOOL "" FORCE)

# Pull in the submodule
add_subdirectory(llama.cpp)

# ---------------------------------------------------------------------------
# Our JNI shared library
# ---------------------------------------------------------------------------
add_library(
        graphrag-jni
        SHARED
        llama-bridge.cpp
)

target_include_directories(graphrag-jni PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

target_link_libraries(graphrag-jni
        llama       # llama.cpp core
        ggml        # ggml tensor backend
        android     # Android NDK
        log         # __android_log_print
)

# C++17 for llama.cpp internals
target_compile_features(graphrag-jni PRIVATE cxx_std_17)