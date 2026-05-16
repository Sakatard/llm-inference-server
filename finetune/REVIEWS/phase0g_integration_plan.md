# Phase 0g — Integration Plan (dFlash decode engine into llama-server)

**Goal:** single-binary integration. llama-server gains `--decode-engine dflash`
flag that swaps the decode loop to lucebox-hub's dFlash kernels while keeping
turbo4 KV cache + MTP self-spec-decode.

## Status (2026-05-16)

| Item | State |
|---|---|
| Sakatard fork: clone at `/tmp/mtp-scratch/llama` | ✓ existing |
| New branch `phase-0g-dflash` | ✓ created |
| Vendor lucebox-hub @ SHA `6fe0d9a0a9b79855cc56967a60f6d35a5532cdd7` via git subtree at `vendor/lucebox-hub/` | ✓ commit `91b1a7396` |
| `TURBO[234]_0` pre-rotated flag patch in `dflash/src/qwen35/qwen35_target_graph.cpp:130` | ✓ commit `d07efcee9` |
| `vendor/lucebox-hub/dflash/SAKATARD_INTEGRATION.cmake` (Pascal-only source subset) | ✓ written |
| Symlink `vendor/lucebox-hub/dflash/deps/llama.cpp` → relative `../../../../` | ✓ (untracked, not committed yet) |
| Standalone dflash27b build proof on Pascal sm_61 | ⏳ in progress (~30min) |
| Root CMakeLists.txt `LLAMA_DFLASH` option + include hook | pending |
| `tools/server/CMakeLists.txt` link dflash27b_iface when present | pending |
| Server CLI flag `--decode-engine dflash` | pending |
| Server decode loop dispatch on engine type | pending |
| Dockerfile bump (LLAMA_DFLASH=ON) | pending |
| P40 deploy + bench vs 18.48 tok/s baseline | pending |

## File modifications planned (Sakatard fork)

### Root `CMakeLists.txt`

Insert near the other `option()` calls (around line 35-40):
```cmake
option(LLAMA_DFLASH "Build dflash decode engine into llama-server" OFF)
```

Near the end (before final `install()` calls):
```cmake
if(LLAMA_DFLASH)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/vendor/lucebox-hub/dflash/SAKATARD_INTEGRATION.cmake")
        message(FATAL_ERROR "LLAMA_DFLASH=ON but vendor/lucebox-hub/dflash/SAKATARD_INTEGRATION.cmake missing — run git subtree pull")
    endif()
    include(vendor/lucebox-hub/dflash/SAKATARD_INTEGRATION.cmake)
endif()
```

### `tools/server/CMakeLists.txt`

After the existing `target_link_libraries(${TARGET} ...)` (line ~164):
```cmake
if(TARGET dflash27b_iface)
    target_link_libraries(${TARGET} PRIVATE dflash27b_iface)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_HAS_DFLASH=1)
endif()
```

### Server CLI flag — `common/arg.cpp` or `tools/server/server.cpp`

Add new flag parsing. Pattern matches existing `--cache-type-k`:
```cpp
add_opt(common_arg(
    {"--decode-engine"}, "ENGINE",
    "decode loop engine: legacy (default) or dflash (requires LLAMA_DFLASH=ON build)",
    [](common_params & params, const std::string & value) {
        if (value != "legacy" && value != "dflash") {
            throw std::invalid_argument("decode-engine must be 'legacy' or 'dflash'");
        }
        params.decode_engine = value;
    }
).set_env("LLAMA_DECODE_ENGINE"));
```

Add `std::string decode_engine = "legacy";` to `common_params` struct.

### Decode loop dispatch — `tools/server/server-context.cpp`

Find the existing decode call (likely `llama_decode(ctx, batch)`). Branch:
```cpp
#if LLAMA_HAS_DFLASH
if (params.decode_engine == "dflash") {
    // Initialize dflash context once on first decode
    static std::unique_ptr<dflash::DFlashContext> g_dflash_ctx;
    if (!g_dflash_ctx) {
        g_dflash_ctx = dflash::DFlashContext::create_from_llama_ctx(ctx);
    }
    g_dflash_ctx->decode(batch);
    return;
}
#endif
llama_decode(ctx, batch);
```

NOTE: real wiring depends on lucebox's actual API. Open-question: does dflash expose
a clean `decode(batch)` entry point or does it own the whole inference loop?
Inspecting `vendor/lucebox-hub/dflash/src/common/dflash_spec_decode.cpp` will answer.

### Dockerfile (`Dockerfile` line ~28-36 cmake block)

Add `-DLLAMA_DFLASH=ON` to the cmake configure step. Bump SHA arg.

## Build commands

Local dev:
```bash
cd /tmp/mtp-scratch/llama
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_DFLASH=ON \
    -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets"
cmake --build build -j$(nproc) --target llama-server
```

Container rebuild (production):
```bash
cd /home/xel/containers/llm-inference-server
docker compose down
docker compose up -d --force-recreate --build
```

After production rebuild, `server.py` would need `--decode-engine dflash` injected
into the qwen-trader route's argv list. Stays as `qwen` baseline default for safety.

## Smoke test on P40 (after first successful build)

Same prompt + bench harness as Phase 0c. Compare:
```
prompt:     ~4K filler tokens
decode:     1024 tokens max
metrics:    predicted_per_second, draft_n_accepted/draft_n
gates:      decode ≥ 28 tok/s (vs Phase 0c's 18.48 baseline = ~1.5x target)
            VRAM peak unchanged (~22 GB)
            MTP acceptance unchanged (~61%)
```

If decode ≥ 28 tok/s → Phase 0g PASS. Lock SPEC v4. Move to Phase 4.

If decode regresses or unstable → debug or fall back to legacy engine, defer dflash to v1.

## Known integration risks

1. **lucebox's `dflash_spec_decode.cpp` may assume separate drafter model + GGUF**.
   Their `--spec-type draft-model` path. Our MTP head provides drafts inline.
   Will need to disable their drafter loading OR build a thin adapter that exposes
   our MTP head's draft logits through their drafter interface.

2. **flashprefill_scalar.cu may have sm_70+ intrinsics slipped in**.
   First Pascal compile attempt (in progress) will reveal.

3. **`ggml` version mismatch**. lucebox vendored ggml at a different version than
   our Sakatard fork. Functions may have changed signatures. Our integration uses
   our ggml — should resolve at link time if dflash compiles cleanly against our
   ggml headers.

4. **DDTree verify integration with MTP draft.** dflash's DDTree expects multi-token
   draft trees. MTP produces N=1 future-token prediction. May only get single-chain
   accept, not tree benefit. Throughput gain reduced from "tree×scalar" to just "scalar".

## Next session priorities (in order)

1. Wait for standalone build complete. If FAIL → fix kernel issues. If PASS → proceed.
2. Apply root CMakeLists + tools/server CMakeLists edits
3. First integration build attempt; capture errors
4. Iterate on missing headers / unresolved symbols
5. Inspect dflash API surface for decode entry point
6. Wire `--decode-engine dflash` flag
7. Build container, deploy to P40, bench
