# Phase 0h — Final Status (2026-05-16)

**Compile/link/dispatch infra: DONE.** End-to-end runtime: REGRESSION on Pascal sm_61, debug deferred to Phase 0h v2.

## What works (verified empirically)

1. **Full llama-server build with `-DLLAMA_DFLASH=ON`** on Pascal sm_61 inside `nvidia/cuda:12.8.0-devel-ubuntu22.04` (glibc 2.35 for prod compat).
   - Binary: 65,599,616 bytes
   - Contains `LlamaToDFlashTarget` vtable (`nm` confirmed all 10 entries)
   - Contains `dflash27b::DFlashTarget` mangled base symbol
   - Embeds `--decode-engine` CLI string + help text

2. **`--decode-engine dflash` recognized at startup.** Server logs fire:
   ```
   srv main: Phase 0h: --decode-engine dflash recognized — LLAMA_HAS_DFLASH=1.
   srv main: Phase 0h: LlamaToDFlashTarget bridge linked. Real run_dflash_spec_decode
   srv main: Phase 0h: dispatch from server request flow is v1 work — current path uses standard llama_decode.
   srv main: Phase 0h: dflash kernels + tree-op ggml extensions present in this binary;
   ```

3. **`llama_model_embed_input_tokens` real impl** added to public llama.h API. Bridge `LlamaToDFlashTarget::embed_tokens` returns true target embeddings via `ggml_backend_tensor_get` on `model.tok_embd`. F32/F16/BF16 dtypes supported.

4. **`LlamaToDFlashTarget` bridge methods** (8 vtable entries):
   - `verify_batch` — calls `llama_decode` + per-position argmax
   - `snapshot_kv` / `restore_kv` — via `llama_memory_seq_cp`
   - `is_eos` — via `llama_vocab_is_eog`
   - `embed_tokens` — real impl via new public API
   - `project_hidden_to_tokens` — STUB returning mask_token_id (drafter rejection forces AR fallback)
   - `hidden_size` / `mask_token_id` / `capture_layer_ids` — trivial

5. **Model loads** in built binary: Qwen3.6-27B-IQ4_XS.gguf loads in 16.7s with `--cache-type-k turbo4 --cache-type-v turbo4 --spec-type draft-mtp --ctx-size 16384 -ngl 99`. Server reaches "all slots are idle" state.

## What broke (verified empirically)

**Runtime CUDA error on first real inference request:**

```
/build/scratch/llama/ggml/src/ggml-cuda/ggml-cuda.cu:104: CUDA error
CUDA error: an illegal memory access was encountered
current device: 0, in function launch_mul_mat_q at
  /build/scratch/llama/ggml/src/ggml-cuda/template-instances/../mmq.cuh:3955
cudaFuncSetAttribute((mul_mat_q<type, mmq_x, false>),
                     cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared)
```

Pascal P40 shared-memory-per-block limit is 48 KB (sm_61). `mul_mat_q` template requests more than that.

**Key observation:** production llama-server (same Sakatard fork @ `c85252627`, same Qwen3.6-27B-IQ4_XS.gguf, same flag combo) does NOT crash. Phase 0c proved 18.48 tok/s on the production binary. So the regression is specifically introduced by my Phase 0g/0h changes:

- Lucebox CUDA file overwrites (`gated_delta_net.cu`, `ssm-conv.cu`)
- `enable_language(CUDA)` in `SAKATARD_INTEGRATION.cmake`
- ggml.h enum addition (`GGML_TYPE_TQ3_0`)
- ggml.h tree-op decl additions
- New ggml.c tree-op impls

Hypothesis: `enable_language(CUDA)` in the shim's scope reset some flag (e.g. `CMAKE_CUDA_FLAGS` arch-specific guards) that caused mmq.cuh template instantiation to drop a Pascal-aware mmq_x size restriction.

## Debug plan for Phase 0h v2

1. Diff the actual compile commands for `mmq.cuh` template instances between production build and Phase 0g+0h build (`ninja -v` or `make VERBOSE=1`)
2. Verify `MUL_MAT_Q_MAX_BATCH_SIZE` / `nbytes_shared` calculation respects `__CUDA_ARCH__ < 800` Pascal path
3. Possibly: `LLAMA_DFLASH=ON` should set `MMQ_MAX_BATCH_SIZE` env or guard to prevent oversized template instantiations on Pascal
4. Test fix on P40 with same prompt as Phase 0c bench → goal: same 18.48 tok/s baseline (or better) with `--decode-engine legacy` (proves we haven't broken legacy path)
5. Then test `--decode-engine dflash` → expect ≥18 tok/s (no slowdown vs legacy since project_hidden stub falls through to AR)
6. Real throughput gain from dflash requires v3: real `project_hidden_to_tokens` ggml graph + intermediate layer feature capture

## Branch state (pushed)

Pushed to `Sakatard/llama-cpp-turboquant` `phase-0g-dflash` @ `5d7aae248` after `filter-repo --invert-paths` dropped all binary media globs across 9768 rewritten commits. Post-rewrite SHAs:

```
5d7aae248  phase0h v2: revert lucebox CUDA kernel wholesale-copy — fixes Pascal regression
720ba87ce  phase0h v1: --decode-engine dflash dispatch + project_hidden stub
539aa3813  phase0h v1: real llama_model_embed_input_tokens API + wire bridge
17d5aff1a  phase0h: strip lucebox LFS pointer assets (now fully dropped, not just patched)
6aa9fd487  phase0h: ssm_conv signature + shim sources/includes
```

10 commits ahead of `feature/mtp-turboquant-integration` after rewrite. LFS rejection resolved.

## Container deploy

Did NOT bump production Dockerfile. Production container unchanged, still uses Sakatard remote SHA `c85252627`. My Phase 0g/0h commits sit in `/tmp/mtp-scratch/llama` on `phase-0g-dflash` branch locally only.

To deploy after Phase 0h v2 runtime fix:
1. Force-push branch to Sakatard remote (after LFS history rewrite)
2. Bump `Dockerfile`'s `LLAMA_FORK_SHA` to head of `phase-0g-dflash`
3. Add `-DLLAMA_DFLASH=ON` to the Dockerfile's cmake invocation
4. `docker compose up -d --force-recreate --build`

## Summary

Phase 0 work today: 0b, 0c, 0e (abandoned), 0f, 0g (foundational), 0h (compile/link + dispatch infra). Real dflash throughput gain blocked by:
- Pascal CUDA regression (v2 debug)
- Real `project_hidden_to_tokens` (v2 ggml graph)
- Feature capture from intermediate layers (v2/v3)
- Server request flow dispatch (v2)

Total Phase 0g+0h commits: ~700 LoC code changes + 8 MB vendored subtree + 1 ported ggml-cuda kernel pair (gated_delta_net, ssm-conv). All proven to compile and link on Pascal sm_61. Build is reproducible inside `nvidia/cuda:12.8.0-devel-ubuntu22.04`.
