# Phase 0g — Foundational Milestone (2026-05-16)

**Status:** Foundational compile/link infrastructure COMPLETE. Server runtime
wire-up deferred to Phase 0h (separate bridge work).

## What's done

| Item | Commit | Notes |
|---|---|---|
| Vendor lucebox-hub @ `6fe0d9a0` via git subtree | `91b1a7396` | `vendor/lucebox-hub/` |
| TURBO[234]_0 pre-rotated flag patch | `d07efcee9` | qwen35_target_graph.cpp:130 |
| GGML_TYPE_TQ3_0 enum stub | `7c9358754` | ggml/include/ggml.h:436 |
| ggml_turbo_wht 3-arg → 5-arg sed-patch | `7c9358754` | 3 sites in qwen35_target_graph.cpp |
| Tree-op extensions ported to Sakatard ggml | `7c9358754` | ssm_conv_tree, gated_delta_net_tree, gated_delta_net_tree_persist |
| Lucebox CUDA kernels (gated_delta_net + ssm-conv) wholesale-copied | `7c9358754` | tree-mode path activated by src[2/6/7] != NULL |
| SAKATARD_INTEGRATION.cmake shim | `7c9358754` | Pascal-only source subset, excludes qwen3 drafter / megakernel / pflash / BSA / kv_quant |
| Standalone build proof: all 16 shim files compile on sm_61 | docker `nvidia/cuda:12.8.0-devel-ubuntu24.04` | qwen35_target_graph.cpp.o = 50824 bytes, flashprefill_scalar.cu.o = 327920 bytes |
| LLAMA_DFLASH option + CMake include hook | `b508d2ac1` | Root CMakeLists.txt; default OFF |
| `--decode-engine ENGINE` CLI flag | `b508d2ac1` | common/arg.cpp; legacy\|dflash; LLAMA_ARG_DECODE_ENGINE env |
| `common_params::decode_engine` field | `b508d2ac1` | common/common.h |

## Branch

`phase-0g-dflash` on `/tmp/mtp-scratch/llama` (4 new commits ahead of `feature/mtp-turboquant-integration`):
```
b508d2ac1 phase0g: add LLAMA_DFLASH build option + --decode-engine CLI flag
7c9358754 phase0g: port lucebox tree-op ggml extensions + integration shim
d07efcee9 phase0g: register TURBO[234]_0 as pre-rotated in dflash qwen35 graph
91b1a7396 Squashed 'vendor/lucebox-hub/' content from commit 6fe0d9a0a
```

NOT pushed to Sakatard remote yet — user authorization required before push.

## Empirical evidence of compile health

Build inside `nvidia/cuda:12.8.0-devel-ubuntu24.04` with `-DCMAKE_CUDA_ARCHITECTURES=61`:

```
all dflash sources except qwen3/qwen3_graph.cpp: COMPILED on sm_61
flashprefill_scalar.cu (Pascal scalar FA path): COMPILED (with deprecated-intrinsic warnings, expected on sm_60-69)
qwen35_target_graph.cpp (uses tree ops): COMPILED
gated_delta_net.cu (Sakatard kernels + lucebox tree path): COMPILED
ssm-conv.cu (Sakatard kernels + lucebox tree path): COMPILED
```

The only file NOT compiled is `qwen3/qwen3_graph.cpp` — the Qwen3-0.6B external drafter, explicitly excluded from our integration shim (we use MTP head instead).

## Deferred to Phase 0h

| Item | Effort | Why deferred |
|---|---|---|
| Server runtime wire-up (`server-context.cpp` decode dispatch) | ~1-2 weeks | dflash's `run_dflash_spec_decode` is a STANDALONE inference loop, not a drop-in `llama_decode` replacement. Need bridge layer `LlamaToDFlashTarget` translating llama_model state → dflash::TargetWeights+TargetCache+StepGraph. |
| Server streaming mode for dflash | included in above | dflash writes tokens to stream_fd; server needs new SSE pipe that reads from this. |
| Draft model loading (Qwen3-0.6B-derived) | ~1 day | dflash needs separate DraftWeights. Either ship stock lucebox drafter or train trader-aligned 0.6B drafter (Phase 5c). |
| Container Dockerfile bump (LLAMA_DFLASH=ON) | ~10 min | Trivial once 0h done. |
| P40 deploy + bench vs Phase 0c (18.48 tok/s) | ~30 min | Final validation. |

## v0 ship plan unchanged

For v0 (Phase 4-7), continue with `--decode-engine legacy` (default). This uses our existing Sakatard fork's MTP+turbo4 path that Phase 0c proved at 18.48 tok/s. Phase 0g's `LLAMA_DFLASH=OFF` default keeps current container unchanged.

dflash mode = v1 throughput optimization, gated on Phase 0h bridge work.

## Files modified across Phase 0g

```
ggml/include/ggml.h               +27 lines (TQ3_0 stub + 3 tree-op decls)
ggml/src/ggml.c                   +73 lines (3 tree-op impl wrappers)
ggml/src/ggml-cuda/gated_delta_net.cu   wholesale lucebox copy (+121 lines tree path)
ggml/src/ggml-cuda/gated_delta_net.cuh  wholesale lucebox copy
ggml/src/ggml-cuda/ssm-conv.cu          wholesale lucebox copy (+102 lines tree path)
CMakeLists.txt                    +16 lines (LLAMA_DFLASH option + include hook)
common/common.h                   +4 lines (decode_engine field)
common/arg.cpp                    +14 lines (--decode-engine flag)
vendor/lucebox-hub/dflash/SAKATARD_INTEGRATION.cmake     NEW (147 lines)
vendor/lucebox-hub/dflash/src/qwen35/qwen35_target_graph.cpp   +6 lines (TURBO* pre-rotated, ggml_turbo_wht 5-arg)
vendor/lucebox-hub/...   (subtree squashed import, 8 MB)

Total: ~610 lines of code changes, ~8 MB vendored subtree.
```
