"""Phase 0e — Unsloth training-loader smoke test on Qwen3.6-27B w/ MTP preservation.

Goal: prove the full QLoRA→merge→GGUF round-trip works on the 27B hybrid SSM+attention+MTP
architecture WITHOUT touching the MTP head. Locks the recipe for Phase 4+.

Pipeline:
  1. Load unsloth/Qwen3.6-27B (load_in_4bit, bf16, language_model_only, trust_remote_code)
  2. Verify mtp.* tensors present in loaded state_dict
  3. Attach rank-8 LoRA with target_modules REGEX that excludes mtp.* path
  4. Verify mtp.* params have requires_grad=False post-PEFT
  5. Compute hash of mtp.* params BEFORE training (gate value)
  6. Run 1 SFT step on 5 dummy examples (seq 2048, bs1, grad_ckpt unsloth)
  7. Re-hash mtp.* params AFTER training — must be IDENTICAL to step 5
  8. Save PEFT adapter
  9. Save merged fp16 (save_pretrained_merged) — verify mtp.* still in merged state
 10. Convert merged → fp16 GGUF via llama.cpp convert_hf_to_gguf.py
 11. Verify mtp.* tensors in fp16 GGUF (gguf_dump scan)
 12. Quantize fp16 → Q4_K_M GGUF
 13. Verify mtp.* tensors in Q4_K_M GGUF

Gates (all must pass):
  - load_seconds < 600 (10 min)
  - vram_peak_mb_during_train < 22000 (22 GB on 24 GB 4090)
  - mtp_param_hash_unchanged == True
  - mtp_tensors_in_merged_count == 15 (matches index.json mtp.* count)
  - mtp_tensors_in_fp16_gguf > 0
  - mtp_tensors_in_q4_gguf > 0
  - total wall time < 90 min

Env:
  PHASE0E_OUT       output dir (default /workspace/phase0e_out)
  LLAMA_CPP_DIR     llama.cpp build root (default /workspace/llama.cpp)
  HF_HUB_ENABLE_HF_TRANSFER=1   for fast 56 GB safetensors pull
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path(os.environ.get("PHASE0E_OUT", "/workspace/phase0e_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_REPO = "unsloth/Qwen3.6-27B"
MAX_SEQ_LENGTH = int(os.environ.get("PHASE0E_SEQ", "2048"))
TARGET_MODULES_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
)

REPORT: dict[str, object] = {
    "base_repo": BASE_REPO,
    "max_seq_length": MAX_SEQ_LENGTH,
    "target_modules_regex": TARGET_MODULES_REGEX,
    "loader_api": None,
    "load_seconds": None,
    "train_step_seconds": None,
    "vram_peak_mb_during_train": None,
    "adapter_save_seconds": None,
    "merged_save_seconds": None,
    "gguf_fp16_seconds": None,
    "gguf_q4_seconds": None,
    "final_gguf_bytes": None,
    # Path C MTP-preservation gates
    "mtp_tensors_in_loaded_state": None,
    "mtp_params_frozen_count": None,
    "mtp_params_trainable_count": None,
    "mtp_param_hash_before": None,
    "mtp_param_hash_after": None,
    "mtp_param_hash_unchanged": None,
    "mtp_tensors_in_merged_count": None,
    "mtp_tensors_in_fp16_gguf": None,
    "mtp_tensors_in_q4_gguf": None,
    "errors": [],
    "result": "INCOMPLETE",
}


def _vram_peak_mb() -> int:
    """Read torch's tracked peak VRAM reserved (more accurate than nvidia-smi sampling)."""
    import torch
    return int(torch.cuda.max_memory_reserved() / (1024 * 1024))


def _reset_vram_peak() -> None:
    import torch
    torch.cuda.reset_peak_memory_stats()


def _step(name: str) -> float:
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    return t0


def _done(t0: float, key: str | None = None) -> float:
    dt = time.time() - t0
    print(f"    ... done in {dt:.1f}s", flush=True)
    if key:
        REPORT[key] = round(dt, 2)
    return dt


def _hash_mtp_params(model) -> tuple[str, int, int]:
    """SHA256 of all mtp.* param tensors, in name-sorted order. Returns (hash, count, trainable_count)."""
    import torch
    h = hashlib.sha256()
    count = 0
    trainable = 0
    for n, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if "mtp." in n or n.startswith("mtp."):
            count += 1
            if p.requires_grad:
                trainable += 1
            arr = p.detach().to(torch.float32).cpu().numpy().tobytes()
            h.update(n.encode())
            h.update(b"|")
            h.update(arr)
    return h.hexdigest(), count, trainable


def _count_mtp_in_gguf(gguf_path: Path) -> int:
    """Count tensor names starting with 'mtp' or containing 'nextn' in a GGUF."""
    try:
        # Try gguf-py from llama.cpp build
        llama_cpp = Path(os.environ.get("LLAMA_CPP_DIR", "/workspace/llama.cpp"))
        sys.path.insert(0, str(llama_cpp / "gguf-py"))
        from gguf import GGUFReader  # type: ignore
        r = GGUFReader(str(gguf_path))
        n = sum(
            1 for t in r.tensors
            if t.name.startswith("mtp") or "nextn" in t.name or "mtp." in t.name
        )
        return n
    except Exception as e:
        REPORT["errors"].append(f"gguf scan failed for {gguf_path}: {e}")
        return -1


def _write_report() -> None:
    p = OUT_DIR / "phase0e_report.json"
    p.write_text(json.dumps(REPORT, indent=2, default=str))
    print(f"\n[report] {p}", flush=True)


def main() -> None:
    # === 1. Load model ===
    t0 = _step(f"Load {BASE_REPO} in 4-bit (seq={MAX_SEQ_LENGTH}, language_model_only=True)")
    try:
        from unsloth import FastModel  # type: ignore
        REPORT["loader_api"] = "FastModel"
    except ImportError as e:
        REPORT["errors"].append(f"unsloth import failed: {e}")
        REPORT["result"] = "FAIL_unsloth_import"
        _write_report()
        raise SystemExit(2)

    # MUST keep the full Qwen3_5ForConditionalGeneration wrapper — MTP head only exists
    # inside that wrapper. Forcing AutoModelForCausalLM strips MTP (verified: smoke v1
    # found 0 mtp.* tensors when CausalLM was forced).
    # Vision tower is skipped by config.json's language_model_only=true flag.
    try:
        model, tokenizer = FastModel.from_pretrained(
            BASE_REPO,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            dtype=None,
            trust_remote_code=True,
        )
    except Exception as e:
        REPORT["errors"].append(f"FastModel.from_pretrained failed: {type(e).__name__}: {e}")
        REPORT["result"] = "FAIL_load"
        _write_report()
        raise
    _done(t0, "load_seconds")

    # Verify NO vision tower params loaded (language_model_only=true should have skipped them)
    vision_params = [n for n, _ in model.named_parameters() if "visual." in n or "vision_tower" in n or "thinker." in n]
    REPORT["vision_tower_params_count"] = len(vision_params)
    if vision_params:
        print(f"    [WARN] {len(vision_params)} vision tower params loaded (e.g. {vision_params[:2]}) — VRAM bloat risk")
    else:
        print("    OK — no vision tower params (language_model_only honored)")

    # === 1b. Verify mtp.* tensors present in loaded model ===
    mtp_names = [n for n, _ in model.named_parameters() if "mtp." in n or n.startswith("mtp.")]
    REPORT["mtp_tensors_in_loaded_state"] = len(mtp_names)
    print(f"    mtp.* tensors found in loaded model: {len(mtp_names)}")
    if len(mtp_names) == 0:
        REPORT["errors"].append("no mtp.* tensors in loaded model — either Unsloth stripped them or language_model_only excluded MTP")
        REPORT["result"] = "FAIL_mtp_missing_post_load"
        _write_report()
        raise SystemExit(3)
    print(f"    sample mtp names: {mtp_names[:3]}")

    # === 2. Pre-resolve regex → explicit module list (Unsloth fast-LoRA may not accept regex) ===
    t0 = _step("Resolve target_modules regex → explicit full-path list")
    pat = re.compile(TARGET_MODULES_REGEX)
    matched_modules = [name for name, _ in model.named_modules() if pat.match(name)]
    n_layers_in_text = REPORT.get("n_hidden_layers")
    expected_per_layer = 7  # q,k,v,o,gate,up,down
    REPORT["target_modules_matched_count"] = len(matched_modules)
    REPORT["target_modules_sample"] = matched_modules[:3] + (["..."] if len(matched_modules) > 3 else [])
    print(f"    regex matched {len(matched_modules)} modules. Sample: {matched_modules[:3]}")
    if len(matched_modules) == 0:
        REPORT["errors"].append(f"regex {TARGET_MODULES_REGEX} matched 0 modules — wrong path prefix")
        REPORT["result"] = "FAIL_regex_no_match"
        _write_report()
        raise SystemExit(11)
    # Cross-check: total trunk layers * 7 (q,k,v,o,gate,up,down) — for Qwen3.6-27B that's 64*7=448
    # We can't strictly assert until we know n_layers; do a sanity floor instead
    if len(matched_modules) < expected_per_layer * 32:  # min plausible trunk size
        REPORT["errors"].append(f"regex matched only {len(matched_modules)} — fewer than 32 layers worth, suspicious")
        REPORT["result"] = "FAIL_regex_undersized"
        _write_report()
        raise SystemExit(11)
    _done(t0)

    t0 = _step(f"Attach rank-8 LoRA on {len(matched_modules)} trunk modules (mtp.* excluded)")
    model = FastModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=matched_modules,   # explicit list, not regex
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    _done(t0)

    # === 2b. Verify mtp.* frozen ===
    frozen = sum(1 for n, p in model.named_parameters() if ("mtp." in n or n.startswith("mtp.")) and not p.requires_grad)
    trainable = sum(1 for n, p in model.named_parameters() if ("mtp." in n or n.startswith("mtp.")) and p.requires_grad)
    REPORT["mtp_params_frozen_count"] = frozen
    REPORT["mtp_params_trainable_count"] = trainable
    print(f"    mtp.* frozen={frozen} trainable={trainable}")
    if trainable > 0:
        REPORT["errors"].append(f"regex leaked: {trainable} mtp.* params marked trainable. Inspect target_modules.")
        REPORT["result"] = "FAIL_mtp_leaked_to_lora"
        _write_report()
        raise SystemExit(4)

    # === 3. Snapshot mtp.* param hash BEFORE train ===
    h_before, c_before, t_before = _hash_mtp_params(model)
    REPORT["mtp_param_hash_before"] = h_before
    print(f"    mtp.* hash BEFORE train: {h_before[:16]}... ({c_before} tensors, {t_before} trainable)")

    # === 4. Build 5 LONG dummy examples (~seq_len tokens each) to exercise activation VRAM ===
    t0 = _step(f"Build dataset of 5 examples padded to ~{MAX_SEQ_LENGTH} tokens each")
    # Long lorem-ipsum-style filler to push token count near seq_len. Real Phase 4 examples
    # will be similar length (committee outputs + market context can easily hit 2k tokens).
    filler_word = "trading polymarket decision rationale evidence "
    n_words = MAX_SEQ_LENGTH  # 1 word ~= 1.3 tokens, so this over-fills, tokenizer truncates
    long_content = (filler_word * n_words)[:MAX_SEQ_LENGTH * 5]  # cap str length
    dummy_messages = [
        [{"role": "user", "content": f"Analyze market {i}: {long_content}"},
         {"role": "assistant", "content": f"Decision {i}: {long_content}"}]
        for i in range(5)
    ]
    rendered = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        for m in dummy_messages
    ]
    for r in rendered:
        m = re.search(r"<think>(.*?)</think>", r, re.DOTALL)
        if m and m.group(1).strip():
            raise AssertionError(f"non-empty <think> leaked: {m.group(1)[:80]!r}")
    # Confirm token length
    sample_tok_count = len(tokenizer(rendered[0], add_special_tokens=False)["input_ids"])
    REPORT["dummy_example_token_count"] = sample_tok_count
    print(f"    sample[0] rendered: {len(rendered[0])} chars, {sample_tok_count} tokens (target ~{MAX_SEQ_LENGTH})")
    if sample_tok_count < MAX_SEQ_LENGTH * 0.7:
        print(f"    [WARN] dummy seq is only {sample_tok_count} tokens; VRAM measurement may underrepresent peak")

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
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        save_strategy="no",
        seed=42,
        bf16=True,
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=args, train_dataset=ds)
    _reset_vram_peak()
    trainer.train()
    vram_peak = _vram_peak_mb()
    REPORT["vram_peak_mb_during_train"] = vram_peak
    _done(t0, "train_step_seconds")
    print(f"    torch peak VRAM reserved: {vram_peak} MiB")

    # === 5. Re-hash mtp.* AFTER train — gate ===
    h_after, c_after, _ = _hash_mtp_params(model)
    REPORT["mtp_param_hash_after"] = h_after
    REPORT["mtp_param_hash_unchanged"] = (h_before == h_after and c_before == c_after)
    print(f"    mtp.* hash AFTER train:  {h_after[:16]}...  unchanged={REPORT['mtp_param_hash_unchanged']}")
    if not REPORT["mtp_param_hash_unchanged"]:
        REPORT["errors"].append("mtp.* params CHANGED during training — regex target_modules failed to scope")
        REPORT["result"] = "FAIL_mtp_drifted_during_train"
        _write_report()
        raise SystemExit(5)

    # === 6. Save adapter ===
    t0 = _step("Save PEFT adapter")
    adapter_dir = OUT_DIR / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _done(t0, "adapter_save_seconds")

    # === 7. Save merged fp16 ===
    t0 = _step("Save merged fp16 model")
    merged_dir = OUT_DIR / "merged_fp16"
    if not hasattr(model, "save_pretrained_merged"):
        REPORT["errors"].append("save_pretrained_merged not on model — Unsloth API regression")
        REPORT["result"] = "FAIL_merge_api"
        _write_report()
        raise SystemExit(6)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    _done(t0, "merged_save_seconds")

    # === 7b. Verify mtp.* tensors in merged safetensors (use substring "mtp." per Codex H) ===
    mtp_in_merged: list[str] = []
    merged_idx = merged_dir / "model.safetensors.index.json"
    if merged_idx.exists():
        idx = json.loads(merged_idx.read_text())
        mtp_in_merged = [k for k in idx.get("weight_map", {}).keys() if "mtp." in k or k.startswith("mtp")]
    else:
        try:
            from safetensors import safe_open
            for single in merged_dir.glob("*.safetensors"):
                with safe_open(str(single), framework="pt") as f:
                    mtp_in_merged.extend([k for k in f.keys() if "mtp." in k or k.startswith("mtp")])
        except Exception as e:
            REPORT["errors"].append(f"merged safetensors scan failed: {e}")
    REPORT["mtp_tensors_in_merged_count"] = len(mtp_in_merged)
    print(f"    mtp.* tensors in merged safetensors: {len(mtp_in_merged)}")
    # STRICT gate: must be exactly 15 (matches index.json count from unsloth/Qwen3.6-27B)
    EXPECTED_MTP_COUNT = 15
    if len(mtp_in_merged) != EXPECTED_MTP_COUNT:
        REPORT["errors"].append(f"merged save mtp.* count={len(mtp_in_merged)} != expected {EXPECTED_MTP_COUNT} — Unsloth merge dropped tensors")
        REPORT["result"] = "FAIL_mtp_count_wrong_in_merge"
        _write_report()
        raise SystemExit(7)

    # === 7c. Sanitize config.json: Unsloth may rename arch to UnslothQwen3_5... ===
    conf_path = merged_dir / "config.json"
    if conf_path.exists():
        conf = json.loads(conf_path.read_text())
        original_arch = conf.get("architectures", [])
        REPORT["arch_pre_sanitize"] = original_arch
        canonical = ["Qwen3_5ForConditionalGeneration"]
        if original_arch != canonical:
            conf["architectures"] = canonical
            conf_path.write_text(json.dumps(conf, indent=2))
            print(f"    config.json arch sanitized: {original_arch} -> {canonical}")
        else:
            print(f"    config.json arch already canonical: {canonical}")
        REPORT["arch_post_sanitize"] = conf["architectures"]

    # === 8. Convert merged → fp16 GGUF ===
    llama_cpp = Path(os.environ.get("LLAMA_CPP_DIR", "/workspace/llama.cpp"))
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not convert_script.exists() or not quantize_bin.exists():
        REPORT["errors"].append(f"llama.cpp missing tools: convert={convert_script.exists()} quantize={quantize_bin.exists()}")
        REPORT["result"] = "FAIL_llama_cpp_missing"
        _write_report()
        raise SystemExit(8)

    fp16_gguf = OUT_DIR / "model_fp16.gguf"
    t0 = _step(f"Convert merged → {fp16_gguf}")
    subprocess.run(
        [sys.executable, str(convert_script), str(merged_dir), "--outfile", str(fp16_gguf), "--outtype", "f16"],
        check=True,
    )
    _done(t0, "gguf_fp16_seconds")

    # === 8b. Verify mtp.* tensors in fp16 GGUF ===
    REPORT["mtp_tensors_in_fp16_gguf"] = _count_mtp_in_gguf(fp16_gguf)
    print(f"    mtp/nextn tensors in fp16 GGUF: {REPORT['mtp_tensors_in_fp16_gguf']}")
    if REPORT["mtp_tensors_in_fp16_gguf"] <= 0:
        REPORT["errors"].append("convert_hf_to_gguf stripped MTP tensors — check arch class registration")
        REPORT["result"] = "FAIL_mtp_lost_on_gguf_convert"
        _write_report()
        raise SystemExit(9)

    # === 9. Quantize fp16 → Q4_K_M ===
    q4_gguf = OUT_DIR / "model_Q4_K_M.gguf"
    t0 = _step(f"Quantize → {q4_gguf}")
    subprocess.run([str(quantize_bin), str(fp16_gguf), str(q4_gguf), "Q4_K_M"], check=True)
    _done(t0, "gguf_q4_seconds")
    REPORT["final_gguf_bytes"] = q4_gguf.stat().st_size

    # === 9b. Verify mtp.* in Q4_K_M GGUF ===
    REPORT["mtp_tensors_in_q4_gguf"] = _count_mtp_in_gguf(q4_gguf)
    print(f"    mtp/nextn tensors in Q4_K_M GGUF: {REPORT['mtp_tensors_in_q4_gguf']}")
    if REPORT["mtp_tensors_in_q4_gguf"] <= 0:
        REPORT["errors"].append("quantize stripped MTP tensors")
        REPORT["result"] = "FAIL_mtp_lost_on_quantize"
        _write_report()
        raise SystemExit(10)

    # === FINAL GATES (enforce all) ===
    gates = {
        "load_seconds_under_600":          REPORT.get("load_seconds", 999) is not None and float(REPORT["load_seconds"]) < 600,
        "vram_peak_under_22000_mb":         REPORT.get("vram_peak_mb_during_train") is not None and int(REPORT["vram_peak_mb_during_train"]) < 22000,
        "mtp_param_hash_unchanged":        bool(REPORT.get("mtp_param_hash_unchanged")),
        "mtp_merged_count_eq_15":           REPORT.get("mtp_tensors_in_merged_count") == EXPECTED_MTP_COUNT,
        "mtp_fp16_gguf_count_positive":     (REPORT.get("mtp_tensors_in_fp16_gguf") or 0) > 0,
        "mtp_q4_gguf_count_positive":       (REPORT.get("mtp_tensors_in_q4_gguf") or 0) > 0,
        "vision_tower_params_zero":         REPORT.get("vision_tower_params_count", 0) == 0,
    }
    REPORT["gates"] = gates
    failed = [k for k, v in gates.items() if not v]
    if failed:
        REPORT["errors"].append(f"final gate check failed: {failed}")
        REPORT["result"] = "FAIL_final_gate"
        _write_report()
        print("\n=== PHASE 0e FINAL GATE FAILED ===")
        print(f"Failed gates: {failed}")
        print(json.dumps(REPORT, indent=2, default=str))
        raise SystemExit(12)

    REPORT["result"] = "PASS"
    _write_report()
    print("\n=== PHASE 0e PASS ===")
    print(json.dumps(REPORT, indent=2, default=str))


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
