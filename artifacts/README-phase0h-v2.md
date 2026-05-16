# Phase 0h v2 deployable binary (2026-05-16)

## File

`llama-server-phase0h-v2` — 65,493,976 bytes

## Built from

`Sakatard/llama-cpp-turboquant` fork @ branch `phase-0g-dflash`,
HEAD `5d7aae248` after history rewrite (PNG/GIF/JPG/SVG/WEBP/MP4 dropped from all 9768 commits to satisfy GitHub LFS hook):

```
5d7aae248  phase0h v2: revert lucebox CUDA kernel wholesale-copy — fixes Pascal regression
720ba87ce  phase0h v1: --decode-engine dflash dispatch + project_hidden stub
539aa3813  phase0h v1: real llama_model_embed_input_tokens API + wire bridge
17d5aff1a  phase0h: strip lucebox LFS pointer assets (now dropped entirely via filter-repo)
6aa9fd487  phase0h: fix ssm_conv signature + shim sources/includes
…  phase0h: skeleton LlamaToDFlashTarget bridge + server CMake hook
…  phase0g: add LLAMA_DFLASH build option + --decode-engine CLI flag
…  phase0g: tree-op ggml extensions + integration shim
…  phase0g: register TURBO[234]_0 as pre-rotated in dflash qwen35 graph
…  Squashed lucebox-hub @ 6fe0d9a0 (binary media stripped)
```

Pre-rewrite SHAs (now orphaned): d7ae4f39d / cb0bafb56 / 4e854590e / 4361635c3 / 25e9739aa / 67a4f232d / b508d2ac1 / 7c9358754 / d07efcee9 / 91b1a7396.

## Build flags

```
nvidia/cuda:12.8.0-devel-ubuntu22.04   (glibc 2.35 for prod container compat)
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_DFLASH=ON \
    -DLLAMA_OPENSSL=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets"
```

## P40 bench result (2026-05-16)

```
Qwen3.6-27B-IQ4_XS.gguf, --cache-type-k turbo4 --cache-type-v turbo4
--spec-type draft-mtp --spec-draft-n-max 4 --decode-engine dflash
--ctx-size 16384 -ngl 99

prompt_n:          4075
prompt_per_sec:    174.12 tok/s
predicted_n:       256
predicted_per_sec: 19.69 tok/s   (Phase 0c baseline: 18.48, +6.5%)
draft_accept:      178/307 = 58.0%
VRAM peak:         18,979 MiB
```

## Features wired

| Layer | Status |
|---|---|
| Sakatard fork: turbo3/turbo4 KV + MTP self-spec (PR #22673) | ✓ |
| GGML_TYPE_TQ3_0 enum stub (lucebox compat) | ✓ |
| Tree-op ggml.c wrappers (ggml_ssm_conv_tree, gated_delta_net_tree[_persist]) | ✓ wrappers compile; CUDA kernels reverted to stock Sakatard (tree-mode disabled at runtime) |
| dflash27b_iface static lib (Pascal sm_61) | ✓ linked |
| `--decode-engine dflash` CLI flag | ✓ recognized |
| `LLAMA_HAS_DFLASH=1` compile def | ✓ |
| Phase 0h startup logs | ✓ fire on `--decode-engine dflash` |
| LlamaToDFlashTarget bridge (8 vtable methods + real embed_input_tokens) | ✓ vtable present in binary |
| Server request flow dispatch to dflash | TODO v3 — current path = standard llama_decode |
| project_hidden_to_tokens real impl | TODO v3 — stub returns mask_token |
| Intermediate feature capture for dflash drafter | TODO v3 |
| Tree-mode CUDA kernels (surgical port from lucebox) | TODO v3 |

## Deploy procedure

```bash
# Drop binary into running container (single-shot test):
docker cp artifacts/llama-server-phase0h-v2 llm-inference-server:/usr/local/bin/llama-server-dflash

# To make it the default llama-server (production swap):
# 1. Stop existing llama-server processes inside container
# 2. mv /usr/local/bin/llama-server /usr/local/bin/llama-server.bak
# 3. mv /usr/local/bin/llama-server-dflash /usr/local/bin/llama-server
# 4. Add --decode-engine dflash to server.py's qwen route flag list
# 5. docker compose restart llm-inference-server
```

## Branch state

Pushed to `Sakatard/llama-cpp-turboquant` `phase-0g-dflash` @ `5d7aae248` (2026-05-16).
LFS pointer rejection resolved by `git-filter-repo --invert-paths` dropping
all binary media globs (`*.png *.gif *.jpg *.jpeg *.svg *.webp *.mp4`)
from every commit. Local working copy at `/tmp/mtp-scratch/llama`.

## v3 priorities

1. Surgical port of lucebox tree-mode CODE additions to Sakatard kernels
   (NOT wholesale file overwrite — that broke Pascal mmq.cuh shared-mem)
2. Real `project_hidden_to_tokens` ggml graph (output_norm + lm_head + argmax)
3. Real intermediate-layer feature capture (modify llama_decode or parallel
   graph that writes to dflash feature ring)
4. server-context.cpp request flow dispatch:
   when `params.decode_engine == "dflash"`, build `LlamaToDFlashTarget`,
   spawn dflash spec-decode loop via `run_dflash_spec_decode()`, stream
   tokens back through SSE pipe
5. v3 P40 bench target: ≥28 tok/s decode (1.5× Phase 0c baseline)
