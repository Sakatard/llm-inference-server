# Optimized build for Xeon E5-2660 v2 (Ivy Bridge) + Tesla P40
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Australia/Perth

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential python3 python3-pip curl ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

# Ivy Bridge ISA flags (no AVX2, no FMA, no BMI1/BMI2, no AVX512)
ENV IVY_CFLAGS="-march=x86-64 -msse4.2 -mavx -mno-avx2 -mno-fma -mno-avx512f -mno-bmi -mno-bmi2"

# --- Build llama.cpp (upstream + local patch series) ---
# Strategy: clone upstream ggml-org/llama.cpp at the pinned SHA, then apply
# the patch series in patches/llama-cpp/. Patch 0001 is TurboQuant-only
# (MTP/Qwen35 dropped — upstream merged native MTP via PR #22673 + fixes
# #23198, #23237). 0002–0006 + 0008–0010 are Phase 0g/0h (dflash compile/link
# + bridge + Pascal CUDA fix). 0007 dropped (was lucebox-hub LFS symlink
# fixup — no-op against fresh vendor clone at pinned SHA). Lucebox-hub vendor
# tree is fetched separately at its pinned SHA — Phase 0g/0h patches reference
# vendor/lucebox-hub/dflash/* paths so the clone must happen before 0002+.
ARG LLAMA_UPSTREAM_URL=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_UPSTREAM_SHA=a135ec0baa1bcf7eb0437c9fd04920f87cf33ace
ARG LUCEBOX_URL=https://github.com/Luce-Org/lucebox-hub.git
ARG LUCEBOX_SHA=6fe0d9a0a
WORKDIR /build/llama.cpp
RUN git init -q . && \
    git remote add origin "$LLAMA_UPSTREAM_URL" && \
    git -c protocol.version=2 fetch --depth 1 origin "$LLAMA_UPSTREAM_SHA" && \
    git checkout -q FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "$LLAMA_UPSTREAM_SHA"

COPY patches/llama-cpp/ /build/patches/llama-cpp/
RUN git apply --whitespace=nowarn /build/patches/llama-cpp/0001-turboquant-base.patch
RUN git clone "$LUCEBOX_URL" vendor/lucebox-hub && \
    git -C vendor/lucebox-hub checkout -q "$LUCEBOX_SHA" && \
    rm -rf vendor/lucebox-hub/.git
RUN for p in /build/patches/llama-cpp/0002-*.patch \
              /build/patches/llama-cpp/0003-*.patch \
              /build/patches/llama-cpp/0004-*.patch \
              /build/patches/llama-cpp/0005-*.patch \
              /build/patches/llama-cpp/0006-*.patch \
              /build/patches/llama-cpp/0008-*.patch \
              /build/patches/llama-cpp/0009-*.patch \
              /build/patches/llama-cpp/0010-*.patch; do \
      echo "applying $(basename $p)"; \
      git apply --whitespace=nowarn "$p"; \
    done

# Build for P40 — dflash decode engine in-tree (patches 0002-0010) + upstream
# native MTP+speculative now baked in (PR #22673 + #23198 + #23237).
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="61" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_SERVER=ON \
    -DLLAMA_FFMPEG=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_DFLASH=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets"

RUN cmake --build build --config Release -j$(nproc)

# --- Build whisper.cpp ---
WORKDIR /build/whisper.cpp
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git .
# CUDA driver stubs needed for linking in Docker (no real driver at build time)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="61" \
    -DCMAKE_BUILD_TYPE=Release \
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
    -DWHISPER_BUILD_SERVER=ON \
    -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets"

RUN cmake --build build --config Release --target whisper-server -j$(nproc)

# Collect all shared libs into a staging directory
RUN mkdir -p /build/libs && find /build -name "*.so*" -exec cp -a {} /build/libs/ \;

# --- Final image: copy binaries + wrapper ---
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/libs/ /usr/local/lib/
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-cli /usr/local/bin/
COPY --from=builder /build/whisper.cpp/build/bin/whisper-server /usr/local/bin/
RUN ldconfig
COPY gpu_wrapper.py /usr/local/bin/gpu-wrapper
RUN chmod +x /usr/local/bin/gpu-wrapper
WORKDIR /models
EXPOSE 9080 9081
