# Optimized build for Xeon E5-2660 v2 (Ivy Bridge) + Tesla P40
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    git cmake build-essential python3 python3-pip curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ivy Bridge ISA flags (no AVX2, no FMA, no BMI1/BMI2, no AVX512)
ENV IVY_CFLAGS="-march=x86-64 -msse4.2 -mavx -mno-avx2 -mno-fma -mno-avx512f -mno-bmi -mno-bmi2"

# --- Build llama.cpp ---
WORKDIR /build/llama.cpp
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git .
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_NO_VMM=ON \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DGGML_AVX=ON \
    -DGGML_AVX2=OFF \
    -DGGML_AVX512=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DGGML_BMI2=OFF \
    -DCMAKE_C_FLAGS="$IVY_CFLAGS" \
    -DCMAKE_CXX_FLAGS="$IVY_CFLAGS" \
    -DLLAMA_SERVER=ON \
    -DLLAMA_FFMPEG=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j$(nproc)

# --- Build whisper.cpp ---
WORKDIR /build/whisper.cpp
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git .
# CUDA driver stubs needed for linking in Docker (no real driver at build time)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=ON \
    -DGGML_AVX2=OFF \
    -DGGML_AVX512=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DGGML_BMI2=OFF \
    -DCMAKE_C_FLAGS="$IVY_CFLAGS" \
    -DCMAKE_CXX_FLAGS="$IVY_CFLAGS" \
    -DWHISPER_BUILD_EXAMPLES=ON \
    -DWHISPER_BUILD_SERVER=ON \
    -DWHISPER_BUILD_TESTS=OFF \
    && cmake --build build --config Release --target whisper-server -j$(nproc)

# Collect all shared libs into a staging directory
RUN mkdir -p /build/libs && find /build -name "*.so*" -exec cp -a {} /build/libs/ \;

# --- Final image: copy binaries + wrapper ---
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/libs/ /usr/local/lib/
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/whisper.cpp/build/bin/whisper-server /usr/local/bin/
RUN ldconfig
COPY gpu_wrapper.py /usr/local/bin/gpu-wrapper
RUN chmod +x /usr/local/bin/gpu-wrapper
WORKDIR /models
EXPOSE 9080 9081
