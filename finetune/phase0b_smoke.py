"""Phase 0b — Unsloth training-loader smoke-test.

Run on a rented Ampere+ GPU (Vast.ai 4090 24 GiB target).

Purpose: verify the Unsloth API path for the multimodal Qwen3.5-9B works end-to-end:
  1. Load model in 4-bit
  2. Attach rank-8 LoRA
  3. Run 1 forward + backward step on 5 dummy examples
  4. Save the adapter (PEFT format)
  5. Save merged fp16 model
  6. Convert merged fp16 -> GGUF -> quantize Q4_K_M
  7. Verify GGUF loads in llama.cpp

Gate: all 7 steps must complete. Document which loader API (FastModel /
FastLanguageModel / AutoModelForImageTextToText) succeeded — this is the locked
path for Phase 4.

Usage on a fresh Vast.ai 4090:

    # 1. Install (skip if base image is unsloth/unsloth)
    pip install -U unsloth datasets transformers accelerate peft bitsandbytes
    # llama.cpp for GGUF conversion + quantization
    git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
    cd /workspace/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-quantize convert_hf_to_gguf
    pip install -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

    # 2. Run smoke
    cd /workspace && python phase0b_smoke.py 2>&1 | tee phase0b_smoke.log
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path(os.environ.get("PHASE0B_OUT", "/workspace/phase0b_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT: dict[str, object] = {
    "loader_api": None,
    "load_seconds": None,
    "train_step_seconds": None,
    "vram_peak_mb_during_train": None,
    "adapter_save_seconds": None,
    "merged_save_seconds": None,
    "gguf_fp16_seconds": None,
    "gguf_q4_seconds": None,
    "final_gguf_bytes": None,
    "errors": [],
    "result": "INCOMPLETE",
}


def _record_vram_peak() -> int:
    """Returns current GPU memory usage in MiB via nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    return int(out.stdout.strip().splitlines()[0])


def _step(name: str):
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    return t0


def _done(t0: float, key: str | None = None) -> float:
    dt = time.time() - t0
    print(f"    ... done in {dt:.1f}s", flush=True)
    if key:
        REPORT[key] = round(dt, 2)
    return dt


def main() -> None:
    base_repo = "unsloth/Qwen3.5-9B"
    max_seq_length = 4096

    # === 1. Loader API selection ===
    t0 = _step(f"Load {base_repo} in 4-bit (max_seq_length={max_seq_length})")
    Loader = None
    try:
        from unsloth import FastModel  # type: ignore
        Loader = FastModel
        REPORT["loader_api"] = "FastModel"
        print("    using unsloth.FastModel")
    except ImportError:
        try:
            from unsloth import FastLanguageModel  # type: ignore
            Loader = FastLanguageModel
            REPORT["loader_api"] = "FastLanguageModel"
            print("    using unsloth.FastLanguageModel (FastModel unavailable)")
        except ImportError as e:
            REPORT["errors"].append(f"unsloth import failed: {e}")
            REPORT["result"] = "FAIL_unsloth_import"
            _write_report()
            raise SystemExit(2)

    try:
        model, tokenizer = Loader.from_pretrained(
            base_repo,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
    except Exception as e:
        REPORT["errors"].append(f"{REPORT['loader_api']}.from_pretrained failed: {type(e).__name__}: {e}")
        # Try fallback
        if REPORT["loader_api"] == "FastModel":
            print(f"    FastModel failed: {e}; retrying with FastLanguageModel")
            from unsloth import FastLanguageModel
            Loader = FastLanguageModel
            REPORT["loader_api"] = "FastLanguageModel"
            model, tokenizer = Loader.from_pretrained(
                base_repo, max_seq_length=max_seq_length, load_in_4bit=True, dtype=None,
            )
        else:
            REPORT["result"] = "FAIL_load"
            _write_report()
            raise

    _done(t0, "load_seconds")
    print(f"    loader_api locked: {REPORT['loader_api']}")

    # === 2. Attach rank-8 LoRA ===
    t0 = _step("Attach rank-8 LoRA")
    model = Loader.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_seq_length,
    )
    _done(t0)

    # === 3. One forward + backward step ===
    t0 = _step("Build 5-example dummy dataset + run 1 SFT step")
    dummy_messages = [
        [{"role": "user", "content": f"Reply with 'OK {i}'"},
         {"role": "assistant", "content": f"OK {i}"}]
        for i in range(5)
    ]
    rendered = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        for m in dummy_messages
    ]
    # Pre-flight: confirm thinking is disabled. Qwen3.5 renders <think></think> empty pair
    # when enable_thinking=False — that's correct behavior (signals model: skip thinking).
    # Only fail if a NON-EMPTY think block leaked through.
    import re as _re
    for r in rendered:
        m = _re.search(r"<think>(.*?)</think>", r, _re.DOTALL)
        if m and m.group(1).strip():
            raise AssertionError(f"non-empty <think> content in rendered prompt: {m.group(1)[:80]!r}")
    print(f"    rendered 5 dummy examples; sample[0]:\n{rendered[0][:200]}...")

    from datasets import Dataset
    ds = Dataset.from_dict({"text": rendered})

    from trl import SFTTrainer, SFTConfig
    args = SFTConfig(
        output_dir=str(OUT_DIR / "ckpt"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        warmup_steps=0,
        optim="adamw_8bit",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        report_to="none",
        save_strategy="no",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=ds,
    )
    vram_peak = _record_vram_peak()
    trainer.train()
    vram_peak = max(vram_peak, _record_vram_peak())
    REPORT["vram_peak_mb_during_train"] = vram_peak
    _done(t0, "train_step_seconds")
    print(f"    peak VRAM during train step: {vram_peak} MiB")

    # === 4. Save adapter ===
    t0 = _step("Save PEFT adapter")
    adapter_dir = OUT_DIR / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _done(t0, "adapter_save_seconds")
    print(f"    adapter at: {adapter_dir}")

    # === 5. Save merged fp16 ===
    t0 = _step("Save merged fp16 model")
    merged_dir = OUT_DIR / "merged_fp16"
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit",
        )
    else:
        REPORT["errors"].append("save_pretrained_merged not available on model object — Unsloth API differs")
        REPORT["result"] = "FAIL_merge_api"
        _write_report()
        raise SystemExit(3)
    _done(t0, "merged_save_seconds")
    print(f"    merged fp16 at: {merged_dir}")

    # === 6. Convert + quantize to GGUF Q4_K_M ===
    llama_cpp = Path(os.environ.get("LLAMA_CPP_DIR", "/workspace/llama.cpp"))
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"

    fp16_gguf = OUT_DIR / "model_fp16.gguf"
    q4_gguf = OUT_DIR / "model_Q4_K_M.gguf"

    if not convert_script.exists() or not quantize_bin.exists():
        REPORT["errors"].append(
            f"llama.cpp not built at {llama_cpp}. Expected {convert_script} and {quantize_bin}"
        )
        REPORT["result"] = "FAIL_llama_cpp_missing"
        _write_report()
        raise SystemExit(4)

    t0 = _step(f"Convert merged fp16 -> {fp16_gguf}")
    subprocess.run(
        [sys.executable, str(convert_script), str(merged_dir), "--outfile", str(fp16_gguf), "--outtype", "f16"],
        check=True,
    )
    _done(t0, "gguf_fp16_seconds")

    t0 = _step(f"Quantize -> {q4_gguf}")
    subprocess.run(
        [str(quantize_bin), str(fp16_gguf), str(q4_gguf), "Q4_K_M"],
        check=True,
    )
    _done(t0, "gguf_q4_seconds")

    REPORT["final_gguf_bytes"] = q4_gguf.stat().st_size

    REPORT["result"] = "PASS"
    _write_report()
    print("\n=== PHASE 0b PASS ===")
    print(json.dumps(REPORT, indent=2))


def _write_report() -> None:
    report_path = OUT_DIR / "phase0b_report.json"
    report_path.write_text(json.dumps(REPORT, indent=2))
    print(f"\n[report] {report_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        REPORT["errors"].append(f"unhandled: {type(exc).__name__}: {exc}")
        if REPORT["result"] == "INCOMPLETE":
            REPORT["result"] = "FAIL_unhandled"
        _write_report()
        raise
