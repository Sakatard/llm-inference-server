# Phase 0b — Vast.ai 4090 Setup Instructions

Goal: smoke-test the Unsloth training-loader API for `unsloth/Qwen3.5-9B` end-to-end (load → LoRA → 1 step → merge → GGUF → quantize). Expected wall time ~25-45 min including setup. Cost ~$0.15-0.30.

## 1. Pick a Vast.ai instance

Filter:
- **GPU**: RTX 4090 (sm_89, 24 GiB) — closest cheap match to our 4090 target
- **CUDA**: 12.1+ (Unsloth/Triton compatibility)
- **Disk**: ≥80 GB allocation (model + adapter + GGUF round-trip)
- **Image**: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` (or any modern PyTorch+CUDA devel image)
- **Network**: usually unmetered on Vast

Bid type: on-demand (interruptible OK for this short run, but reliable preferred).

## 2. SSH access

Once instance is running, get the ssh command from the Vast UI (looks like `ssh -p <PORT> root@<HOST> -L 8080:localhost:8080`).

Share with me the full ssh command — I'll connect, run setup + smoke, capture report, then you can destroy the instance.

## 3. What the smoke-test does (see `phase0b_smoke.py`)

1. Install unsloth + transformers + trl + bitsandbytes + datasets
2. Clone llama.cpp, build llama-quantize + convert_hf_to_gguf
3. `pip install -r requirements-convert_hf_to_gguf.txt`
4. Try `from unsloth import FastModel` (multimodal path). Fallback to `FastLanguageModel`. Document which works.
5. `Loader.from_pretrained("unsloth/Qwen3.5-9B", load_in_4bit=True, max_seq_length=4096)`
6. `Loader.get_peft_model(rank=8, alpha=16, target_modules=all_linear)`
7. Build 5 dummy ChatML examples via `apply_chat_template(..., enable_thinking=False)`. Assert no `<think>` tag in rendered text.
8. SFTTrainer 1 step, batch=1
9. `model.save_pretrained_merged(...)` → fp16 merged dir
10. `python convert_hf_to_gguf.py merged_dir --outtype f16` → `model_fp16.gguf`
11. `llama-quantize model_fp16.gguf model_Q4_K_M.gguf Q4_K_M`
12. Emit `phase0b_report.json` with timings + VRAM peaks + final loader API name

## 4. Acceptance gate

`phase0b_report.json` must contain `"result": "PASS"` AND `loader_api` is one of `FastModel` or `FastLanguageModel` (NOT raw `AutoModelForImageTextToText`).

If FAIL:
- `FAIL_unsloth_import` → check unsloth/triton install
- `FAIL_load` → either base model isn't text-loadable, or loader API needs adjustment
- `FAIL_merge_api` → Unsloth doesn't expose `save_pretrained_merged` on the current model object — refactor needed
- `FAIL_llama_cpp_missing` → build llama.cpp first
- `FAIL_unhandled` → check log

## 5. Output files to copy back

- `phase0b_report.json` — primary, paste back here
- `phase0b_smoke.log` — full stdout/stderr in case of failure
- (Optional) `model_Q4_K_M.gguf` — verifies the merge+quantize round-trip; can be discarded if smoke is PASS

## 6. Destroy instance after run

Smoke-test is one-shot. Once `result: PASS` is captured, destroy. ~25-45 min total = ~$0.15-0.30.
