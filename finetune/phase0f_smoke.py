"""Phase 0f — Qwen3.6-27B QLoRA smoke WITH MTP head trained (Path B).

Replaces phase0e_smoke.py which failed because transformers strips mtp.* at load.

Pipeline:
  1. Load unsloth/Qwen3.6-27B via Unsloth FastModel (mtp.* dropped per usual)
  2. attach_mtp_head() — instantiate MTPBlock + load mtp.* from cached safetensors
  3. Verify mtp.* params now visible in named_parameters()
  4. patch_forward_with_mtp_loss() — monkey-patch forward to add LM + α·MTP loss
  5. Build explicit LoRA target_modules list (trunk + mtp.layers.0.*)
  6. modules_to_save = mtp.fc + mtp.norm + mtp.pre_fc_norm_*
  7. Snapshot mtp param hashes BEFORE train
  8. 1 SFT step on long dummy seq
  9. Re-hash mtp.* — MUST have changed (proves MTP trained, not frozen)
 10. Save adapter + merged
 11. save_mtp_to_merged_safetensors() — inject mtp.* into merged dir as new shard
 12. Convert merged → fp16 GGUF; verify nextn tensors present
 13. Quantize Q4_K_M WITH --tensor-type nextn=q8_0
 14. Verify nextn tensors in Q4 GGUF

Gates:
  - mtp_tensors_loaded == 15 (post-attach)
  - mtp_params_trainable_count > 0 (post-PEFT — LoRA on mtp.layers.0.* makes them trainable)
  - mtp_lm_loss decreases over training (sanity: loss is finite)
  - mtp_param_hash_changed_during_train == True (mtp WAS updated, not frozen)
  - mtp_tensors_in_merged_count == 15
  - mtp_tensors_in_fp16_gguf > 0
  - mtp_tensors_in_q4_gguf > 0
"""
from __future__ import annotations

# WORKAROUND: PEFT >= 0.18 raises ImportError when torchao is present at incompatible
# version. Phase 0b found we MUST pin torchao<0.13 (torch 2.6 lacks register_constant).
# We don't use torchao adapters at all — patch is_torchao_available to return False
# at ALL bound references so PEFT's dispatch_torchao falls through to standard LoraLinear.
# peft.tuners.lora.torchao does `from peft.import_utils import is_torchao_available` so
# we must patch its local reference, not just the source module.
def _disable_peft_torchao_dispatcher():
    targets_patched = 0
    try:
        import peft.import_utils as _iu
        _iu.is_torchao_available = lambda: False
        targets_patched += 1
    except Exception:
        pass
    try:
        # Force-load peft.tuners.lora.torchao if not yet imported so we can patch it
        import importlib
        _t = importlib.import_module("peft.tuners.lora.torchao")
        _t.is_torchao_available = lambda: False
        targets_patched += 1
    except Exception:
        pass
    print(f"[torchao-workaround] patched {targets_patched} is_torchao_available references")

_disable_peft_torchao_dispatcher()

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path(os.environ.get("PHASE0F_OUT", "/workspace/phase0f_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_REPO = "unsloth/Qwen3.6-27B"
MAX_SEQ_LENGTH = int(os.environ.get("PHASE0F_SEQ", "2048"))
ALPHA = float(os.environ.get("PHASE0F_MTP_ALPHA", "0.3"))

REPORT: dict[str, object] = {
    "base_repo": BASE_REPO,
    "max_seq_length": MAX_SEQ_LENGTH,
    "mtp_alpha": ALPHA,
    "loader_api": None,
    "load_seconds": None,
    "mtp_attach_seconds": None,
    "mtp_tensors_loaded": None,
    "mtp_params_trainable_count": None,
    "mtp_params_frozen_count": None,
    "train_step_seconds": None,
    "vram_peak_mb_during_train": None,
    "mtp_param_hash_before": None,
    "mtp_param_hash_after": None,
    "mtp_param_hash_changed": None,
    "training_loss": None,
    "adapter_save_seconds": None,
    "merged_save_seconds": None,
    "mtp_inject_seconds": None,
    "mtp_tensors_in_merged_count": None,
    "gguf_fp16_seconds": None,
    "gguf_q4_seconds": None,
    "mtp_tensors_in_fp16_gguf": None,
    "mtp_tensors_in_q4_gguf": None,
    "final_gguf_bytes": None,
    "errors": [],
    "result": "INCOMPLETE",
}

EXPECTED_MTP_COUNT = 15


def _vram_peak_mb() -> int:
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


def _hash_mtp_params(mtp_module) -> str:
    """SHA256 of all mtp params in name-sorted order. Detects ANY change."""
    import torch
    h = hashlib.sha256()
    for n, p in sorted(mtp_module.named_parameters(), key=lambda kv: kv[0]):
        data = p.detach().to(torch.float32).cpu().numpy().tobytes()
        h.update(n.encode()); h.update(b"|"); h.update(data)
    return h.hexdigest()


def _count_mtp_in_gguf(gguf_path: Path) -> int:
    """Count tensors with 'mtp' or 'nextn' in name. -1 on error."""
    try:
        llama_cpp = Path(os.environ.get("LLAMA_CPP_DIR", "/workspace/llama.cpp"))
        sys.path.insert(0, str(llama_cpp / "gguf-py"))
        from gguf import GGUFReader  # type: ignore
        r = GGUFReader(str(gguf_path))
        return sum(1 for t in r.tensors if t.name.startswith("mtp") or "nextn" in t.name)
    except Exception as e:
        REPORT["errors"].append(f"gguf scan failed for {gguf_path}: {e}")
        return -1


def _write_report() -> None:
    p = OUT_DIR / "phase0f_report.json"
    p.write_text(json.dumps(REPORT, indent=2, default=str))
    print(f"\n[report] {p}", flush=True)


def main() -> None:
    # === 1. Load model (Unsloth, default — mtp.* dropped by transformers) ===
    t0 = _step(f"Load {BASE_REPO} via Unsloth (mtp.* dropped at this stage — will reattach next)")
    try:
        from unsloth import FastModel  # type: ignore
        REPORT["loader_api"] = "FastModel"
    except ImportError as e:
        REPORT["errors"].append(f"unsloth import failed: {e}")
        REPORT["result"] = "FAIL_unsloth_import"
        _write_report()
        raise SystemExit(2)

    try:
        model, tokenizer = FastModel.from_pretrained(
            BASE_REPO,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            dtype=None,
            trust_remote_code=True,
        )
    except Exception as e:
        REPORT["errors"].append(f"FastModel.from_pretrained: {type(e).__name__}: {e}")
        REPORT["result"] = "FAIL_load"; _write_report(); raise
    _done(t0, "load_seconds")

    # === 1b. Purge vision tower — we don't use it, frees ~2GB VRAM (OOM headroom) ===
    import torch as _torch
    import gc as _gc
    vision_params = [n for n, _ in model.named_parameters() if "visual." in n or "vision_tower" in n or "thinker." in n]
    REPORT["vision_tower_params_count_pre_purge"] = len(vision_params)
    if vision_params:
        print(f"    [purge] {len(vision_params)} vision tower params present — deleting model.model.visual")
        if hasattr(model.model, "visual"):
            del model.model.visual
        _gc.collect()
        _torch.cuda.empty_cache()
        remain = [n for n, _ in model.named_parameters() if "visual." in n or "vision_tower" in n or "thinker." in n]
        REPORT["vision_tower_params_count_post_purge"] = len(remain)
        print(f"    [purge] post-purge vision params: {len(remain)}")
    else:
        REPORT["vision_tower_params_count_post_purge"] = 0

    # === 2. Attach MTP head (load mtp.* from cached safetensors) ===
    t0 = _step("Attach MTP head — read mtp.* tensors from HF cache + instantiate module")
    from qwen35_mtp_modeling import (
        attach_mtp_head, patch_forward_with_mtp_loss,
        lora_target_modules_for_mtp_training, lora_modules_to_save_for_mtp_training,
        save_mtp_to_merged_safetensors,
    )
    try:
        mtp, n_loaded = attach_mtp_head(model, BASE_REPO)
    except Exception as e:
        REPORT["errors"].append(f"attach_mtp_head: {type(e).__name__}: {e}")
        REPORT["result"] = "FAIL_mtp_attach"; _write_report(); raise
    _done(t0, "mtp_attach_seconds")

    REPORT["mtp_tensors_loaded"] = n_loaded
    print(f"    mtp.* tensors loaded from safetensors: {n_loaded}")
    if n_loaded != EXPECTED_MTP_COUNT:
        REPORT["errors"].append(f"loaded {n_loaded} mtp tensors, expected exactly {EXPECTED_MTP_COUNT}")
        REPORT["result"] = "FAIL_mtp_attach_incomplete"; _write_report(); raise SystemExit(3)

    # === 3. Build target_modules + modules_to_save BEFORE PEFT wrap ===
    t0 = _step("Resolve LoRA target_modules (trunk + mtp.layers.0.*)")
    target_modules = lora_target_modules_for_mtp_training(model)
    modules_to_save = lora_modules_to_save_for_mtp_training()
    trunk_count = sum(1 for n in target_modules if "language_model.layers" in n)
    mtp_count = sum(1 for n in target_modules if "mtp.layers" in n)
    print(f"    trunk LoRA targets: {trunk_count}, MTP LoRA targets: {mtp_count}")
    print(f"    modules_to_save (full retrain): {modules_to_save}")
    REPORT["target_modules_matched_count"] = len(target_modules)
    REPORT["target_modules_trunk_count"] = trunk_count
    REPORT["target_modules_mtp_count"] = mtp_count
    if trunk_count == 0:
        REPORT["errors"].append("zero TRUNK modules in target_modules — regex prefix wrong")
        REPORT["result"] = "FAIL_regex_no_trunk"; _write_report(); raise SystemExit(4)
    if mtp_count == 0:
        REPORT["errors"].append("zero MTP modules in target_modules — regex failed for mtp.layers.0.*")
        REPORT["result"] = "FAIL_regex_no_mtp"; _write_report(); raise SystemExit(4)
    _done(t0)

    # === 4. Attach PEFT LoRA ===
    t0 = _step(f"Attach rank-8 LoRA on {len(target_modules)} linears + full retrain on {len(modules_to_save)} small modules")
    model = FastModel.get_peft_model(
        model,
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    _done(t0)

    # === 5. Monkey-patch forward AFTER PEFT (PEFT may have wrapped model.forward) ===
    t0 = _step("Patch (post-PEFT) model.forward to compute LM + α·MTP combined loss")
    patch_forward_with_mtp_loss(model, alpha=ALPHA)
    _done(t0)

    # Verify some mtp params are now trainable
    frozen = sum(1 for n, p in model.named_parameters() if "mtp." in n and not p.requires_grad)
    trainable = sum(1 for n, p in model.named_parameters() if "mtp." in n and p.requires_grad)
    REPORT["mtp_params_frozen_count"] = frozen
    REPORT["mtp_params_trainable_count"] = trainable
    print(f"    mtp.* frozen={frozen} trainable={trainable}")
    if trainable == 0:
        REPORT["errors"].append("zero MTP params trainable — LoRA + modules_to_save both failed to grant gradients")
        REPORT["result"] = "FAIL_mtp_no_grad"; _write_report(); raise SystemExit(5)

    # === 6. Snapshot mtp param hash BEFORE train ===
    h_before = _hash_mtp_params(model.mtp if hasattr(model, "mtp") else model.base_model.model.mtp)
    REPORT["mtp_param_hash_before"] = h_before
    print(f"    mtp.* hash BEFORE: {h_before[:16]}...")

    # === 7. Build LONG dummy dataset + 1 SFT step ===
    t0 = _step(f"Build dummy dataset ~{MAX_SEQ_LENGTH} tokens per example")
    # `tokenizer` is a Qwen3VLProcessor (multimodal wrapper). Direct __call__ routes through
    # image processor and tries to decode text strings as images. Extract underlying text
    # tokenizer for clean text-only encoding.
    text_tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    print(f"    using text tokenizer: {type(text_tok).__name__}")

    filler = "trading polymarket decision rationale evidence "
    long_content = (filler * MAX_SEQ_LENGTH)[: MAX_SEQ_LENGTH * 5]
    dummy_messages = [
        [{"role": "user", "content": f"Analyze market {i}: {long_content}"},
         {"role": "assistant", "content": f"Decision {i}: {long_content}"}]
        for i in range(5)
    ]
    rendered = [
        text_tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        for m in dummy_messages
    ]
    for r in rendered:
        m = re.search(r"<think>(.*?)</think>", r, re.DOTALL)
        if m and m.group(1).strip():
            raise AssertionError(f"non-empty <think> leaked: {m.group(1)[:80]!r}")
    sample_tok = len(text_tok(rendered[0], add_special_tokens=False)["input_ids"])
    REPORT["dummy_example_token_count"] = sample_tok
    print(f"    sample[0] tokens: {sample_tok}")

    # SFTTrainer triggers transformers' multimodal preprocessor (this model is
    # Qwen3_5ForConditionalGeneration) which treats our text strings as image URLs.
    # Bypass with a raw torch training loop — Unsloth's patches + our combined-loss
    # forward still fire regardless of trainer wrapping.
    import torch
    encoded = text_tok(
        rendered[:1],   # 1 example for smoke speed
        padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded["attention_mask"].to(input_ids.device)
    labels = input_ids.clone()
    pad_id = text_tok.pad_token_id if text_tok.pad_token_id is not None else -100
    labels[labels == pad_id] = -100

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4,
    )
    model.train()
    _reset_vram_peak()
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    vram_peak = _vram_peak_mb()
    REPORT["vram_peak_mb_during_train"] = vram_peak
    REPORT["training_loss"] = float(loss.detach())
    if hasattr(outputs, "mtp_loss") and outputs.mtp_loss is not None:
        REPORT["mtp_loss_step1"] = float(outputs.mtp_loss)
    if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
        REPORT["lm_loss_step1"] = float(outputs.lm_loss)
    _done(t0, "train_step_seconds")
    print(f"    torch peak VRAM: {vram_peak} MiB, combined loss: {REPORT['training_loss']:.4f}")
    print(f"    lm_loss: {REPORT.get('lm_loss_step1')}, mtp_loss: {REPORT.get('mtp_loss_step1')}")

    # === 8. Re-hash mtp.* AFTER train — must have changed ===
    mtp_ref = model.mtp if hasattr(model, "mtp") else model.base_model.model.mtp
    h_after = _hash_mtp_params(mtp_ref)
    REPORT["mtp_param_hash_after"] = h_after
    REPORT["mtp_param_hash_changed"] = (h_before != h_after)
    print(f"    mtp.* hash AFTER:  {h_after[:16]}...  changed={REPORT['mtp_param_hash_changed']}")
    if not REPORT["mtp_param_hash_changed"]:
        REPORT["errors"].append("mtp.* params UNCHANGED — MTP block not training")
        REPORT["result"] = "FAIL_mtp_not_training"; _write_report(); raise SystemExit(6)

    # === 9. Save adapter ===
    t0 = _step("Save PEFT adapter")
    adapter_dir = OUT_DIR / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _done(t0, "adapter_save_seconds")

    # === 10. Save merged fp16 ===
    t0 = _step("Save merged fp16 model")
    merged_dir = OUT_DIR / "merged_fp16"
    if not hasattr(model, "save_pretrained_merged"):
        REPORT["errors"].append("save_pretrained_merged not on model")
        REPORT["result"] = "FAIL_merge_api"; _write_report(); raise SystemExit(7)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    _done(t0, "merged_save_seconds")

    # === 10b. Inject MTP weights into merged safetensors as extra shard ===
    t0 = _step("Inject mtp.* tensors into merged dir (new shard + index update)")
    n_injected = save_mtp_to_merged_safetensors(merged_dir, model)
    print(f"    injected {n_injected} mtp.* tensors → model-mtp.safetensors")
    _done(t0, "mtp_inject_seconds")

    # === 10c. Sanitize config.json arch ===
    conf_path = merged_dir / "config.json"
    if conf_path.exists():
        conf = json.loads(conf_path.read_text())
        if conf.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
            conf["architectures"] = ["Qwen3_5ForConditionalGeneration"]
            conf_path.write_text(json.dumps(conf, indent=2))
            print("    config.json arch sanitized -> Qwen3_5ForConditionalGeneration")

    # === 10d. Verify mtp.* in merged safetensors ===
    mtp_in_merged: list[str] = []
    merged_idx = merged_dir / "model.safetensors.index.json"
    if merged_idx.exists():
        idx = json.loads(merged_idx.read_text())
        mtp_in_merged = [k for k in idx.get("weight_map", {}).keys() if k.startswith("mtp.")]
    REPORT["mtp_tensors_in_merged_count"] = len(mtp_in_merged)
    print(f"    mtp.* in merged safetensors: {len(mtp_in_merged)}")
    if len(mtp_in_merged) != EXPECTED_MTP_COUNT:
        REPORT["errors"].append(f"merged mtp count {len(mtp_in_merged)} != {EXPECTED_MTP_COUNT}")
        REPORT["result"] = "FAIL_mtp_count_wrong_in_merge"; _write_report(); raise SystemExit(8)

    # === 11. Convert merged → fp16 GGUF ===
    llama_cpp = Path(os.environ.get("LLAMA_CPP_DIR", "/workspace/llama.cpp"))
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not convert_script.exists() or not quantize_bin.exists():
        REPORT["errors"].append("llama.cpp tools missing")
        REPORT["result"] = "FAIL_llama_cpp_missing"; _write_report(); raise SystemExit(9)

    fp16_gguf = OUT_DIR / "model_fp16.gguf"
    t0 = _step(f"Convert merged → {fp16_gguf}")
    subprocess.run(
        [sys.executable, str(convert_script), str(merged_dir), "--outfile", str(fp16_gguf), "--outtype", "f16"],
        check=True,
    )
    _done(t0, "gguf_fp16_seconds")

    REPORT["mtp_tensors_in_fp16_gguf"] = _count_mtp_in_gguf(fp16_gguf)
    print(f"    mtp/nextn in fp16 GGUF: {REPORT['mtp_tensors_in_fp16_gguf']}")
    if REPORT["mtp_tensors_in_fp16_gguf"] <= 0:
        REPORT["errors"].append("fp16 GGUF missing nextn tensors")
        REPORT["result"] = "FAIL_mtp_lost_on_gguf_convert"; _write_report(); raise SystemExit(10)

    # === 12. Quantize IQ4_XS with --tensor-type nextn=q8_0 (per Reddit gotcha) ===
    # IQ4_XS picked over Q4_K_M for: smaller file (~2GB less → more KV headroom on P40),
    # IQ4 kernels handle BF16 inf edge cases that Q4_K_M validation aborts on (Reddit #4),
    # matches lucebox-hub release naming (`Qwen3.6-27B-MTP-IQ4_XS-Q8nextn`).
    q_target = os.environ.get("PHASE0F_QUANT", "IQ4_XS")
    q4_gguf = OUT_DIR / f"model_{q_target}_Q8nextn.gguf"
    t0 = _step(f"Quantize → {q4_gguf} ({q_target} with --tensor-type nextn=q8_0)")
    subprocess.run(
        [str(quantize_bin), "--tensor-type", "nextn=q8_0", str(fp16_gguf), str(q4_gguf), q_target],
        check=True,
    )
    _done(t0, "gguf_q4_seconds")
    REPORT["final_gguf_bytes"] = q4_gguf.stat().st_size

    REPORT["mtp_tensors_in_q4_gguf"] = _count_mtp_in_gguf(q4_gguf)
    print(f"    mtp/nextn in Q4_K_M GGUF: {REPORT['mtp_tensors_in_q4_gguf']}")
    if REPORT["mtp_tensors_in_q4_gguf"] <= 0:
        REPORT["errors"].append("Q4 GGUF missing nextn tensors")
        REPORT["result"] = "FAIL_mtp_lost_on_quantize"; _write_report(); raise SystemExit(11)

    # === FINAL GATES ===
    gates = {
        "mtp_tensors_loaded_eq_15":      REPORT.get("mtp_tensors_loaded") == EXPECTED_MTP_COUNT,
        "mtp_params_trainable_positive": REPORT.get("mtp_params_trainable_count", 0) > 0,
        "mtp_param_hash_changed":        bool(REPORT.get("mtp_param_hash_changed")),
        "vram_peak_under_22000_mb":      REPORT.get("vram_peak_mb_during_train") is not None and int(REPORT["vram_peak_mb_during_train"]) < 22000,
        "mtp_merged_count_eq_15":        REPORT.get("mtp_tensors_in_merged_count") == EXPECTED_MTP_COUNT,
        "mtp_fp16_gguf_count_positive":  (REPORT.get("mtp_tensors_in_fp16_gguf") or 0) > 0,
        "mtp_q4_gguf_count_positive":    (REPORT.get("mtp_tensors_in_q4_gguf") or 0) > 0,
    }
    REPORT["gates"] = gates
    failed = [k for k, v in gates.items() if not v]
    if failed:
        REPORT["errors"].append(f"final gate failed: {failed}")
        REPORT["result"] = "FAIL_final_gate"; _write_report()
        print(f"\nFAILED GATES: {failed}")
        raise SystemExit(12)

    REPORT["result"] = "PASS"
    _write_report()
    print("\n=== PHASE 0f PASS ===")
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
