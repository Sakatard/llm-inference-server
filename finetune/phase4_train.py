"""Phase 4 trainer — runs on Vast.ai 4090. QLoRA over the MTP-preserved
Qwen3.6-27B checkpoint via Unsloth (FastModel) for 27B-on-24GB memory
fit. MTP heads are explicitly frozen via target_modules regex so they
keep their pretrained values and survive merge unchanged.

Inputs (all in --work-dir):
  train.jsonl     — chat-template messages, one row per example
  holdout.jsonl   — holdout for eval

Outputs (under --work-dir/output):
  adapter/             PEFT weights (~200 MB)
  merged/              full bf16 merged checkpoint (~54 GB) — skip via --skip-merge
  train_metrics.json   loss + holdout direction acc + Brier
  holdout_preds.jsonl  per-row prediction with extracted p_yes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE_MODEL_ID = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _extract_p_yes(text: str) -> Optional[float]:
    m = re.search(r"p_yes\s*=\s*([0-9]*\.?[0-9]+)", text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


def _build_target_modules(model) -> List[str]:
    """Match the trunk attention + FFN linears under
    `(model.)?model.language_model.layers.<i>.{self_attn|mlp}.*` and EXCLUDE
    anything under `mtp.*`. The double `model.model.` prefix accounts for
    Qwen3.5's wrapped layout (CausalLM wraps base which wraps language_model)."""
    import torch.nn as nn
    pat = re.compile(
        r"^(model\.)?model\.language_model\.layers\.\d+\."
        r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
    )
    return [
        name for name, mod in model.named_modules()
        if "mtp." not in name and pat.match(name) and isinstance(mod, nn.Linear)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--base-model", default=_BASE_MODEL_ID)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--skip-merge", action="store_true")
    args = ap.parse_args()

    wd = args.work_dir
    train_path = wd / "train.jsonl"
    holdout_path = wd / "holdout.jsonl"
    out_dir = wd / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imports deferred so --help works without GPU stack.
    from unsloth import FastModel  # type: ignore
    import torch  # noqa: F401
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    import gc

    _log(f"FastModel.from_pretrained {args.base_model} (4-bit nf4)")
    model, tokenizer = FastModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
        trust_remote_code=True,
    )

    # Purge vision tower (vision params unused; frees ~2 GB headroom on 4090).
    vision_n = sum(1 for n, _ in model.named_parameters()
                   if "visual." in n or "vision_tower" in n or "thinker." in n)
    if vision_n:
        _log(f"purge {vision_n} vision-tower params")
        if hasattr(model.model, "visual"):
            del model.model.visual
        gc.collect()
        import torch as _t
        _t.cuda.empty_cache()

    _log("build target_modules list (attention + FFN under language_model.layers, EXCLUDING mtp.*)")
    target_modules = _build_target_modules(model)
    trunk_count = sum(1 for n in target_modules if "language_model.layers" in n)
    mtp_count = sum(1 for n in target_modules if "mtp." in n)
    _log(f"target_modules: {len(target_modules)} total, trunk={trunk_count}, mtp={mtp_count}")
    if trunk_count == 0:
        _log("[fail] zero trunk modules matched — name regex misses the actual module layout")
        return 2
    if mtp_count != 0:
        _log("[fail] mtp modules leaked into target_modules — would un-freeze MTP")
        return 2

    _log(f"FastModel.get_peft_model rank={args.lora_rank}")
    model = FastModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=args.max_seq_len,
    )
    model.print_trainable_parameters()

    _log(f"build datasets")
    train_rows = _load_jsonl(train_path)
    holdout_rows = _load_jsonl(holdout_path)
    texts = []
    for r in train_rows:
        text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    train_ds = Dataset.from_dict({"text": texts})
    _log(f"train={len(train_rows)} holdout={len(holdout_rows)}")

    sft_cfg = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
    )

    _log("train")
    t0 = time.time()
    train_out = trainer.train()
    train_elapsed_s = time.time() - t0
    _log(f"train done in {train_elapsed_s:.0f}s")

    _log("save adapter")
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # --- Holdout eval ---
    _log("eval on holdout")
    eval_results: List[Dict[str, Any]] = []
    model.eval()
    correct = 0
    import torch
    with torch.no_grad():
        for i, row in enumerate(holdout_rows):
            msgs = row["messages"]
            prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
            true_assistant = next(m for m in msgs if m["role"] == "assistant")["content"]
            prompt = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=args.max_seq_len - 256).to(model.device)
            out_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            pred_text = tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            true_p = _extract_p_yes(true_assistant)
            pred_p = _extract_p_yes(pred_text)
            row_out = {
                "market_id": row.get("market_id"),
                "true_p_yes": true_p,
                "pred_p_yes": pred_p,
                "pred_full":  pred_text[:800],
            }
            eval_results.append(row_out)
            if true_p is not None and pred_p is not None:
                if (true_p >= 0.5) == (pred_p >= 0.5):
                    correct += 1
            if (i + 1) % 5 == 0:
                _log(f"eval {i + 1}/{len(holdout_rows)}")

    n_valid = sum(1 for r in eval_results if r["true_p_yes"] is not None and r["pred_p_yes"] is not None)
    direction_acc = correct / max(1, n_valid)
    brier = None
    if n_valid:
        brier = sum(
            (r["pred_p_yes"] - r["true_p_yes"]) ** 2
            for r in eval_results
            if r["true_p_yes"] is not None and r["pred_p_yes"] is not None
        ) / n_valid

    metrics = {
        "train_elapsed_s": train_elapsed_s,
        "train_loss":      train_out.training_loss,
        "n_train":         len(train_rows),
        "n_holdout":       len(holdout_rows),
        "n_holdout_valid": n_valid,
        "direction_acc":   direction_acc,
        "brier":           brier,
        "n_target_modules": len(target_modules),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out_dir / "holdout_preds.jsonl", "w") as fh:
        for r in eval_results:
            fh.write(json.dumps(r) + "\n")
    _log(f"metrics: {metrics}")

    # --- Merge bf16 (optional) ---
    if not args.skip_merge:
        _log("merge LoRA + save bf16 (~54 GB)")
        merged_dir = out_dir / "merged"
        # Unsloth's save_pretrained_merged handles the merge in low-mem mode.
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit",
        )
        _log(f"merged → {merged_dir}")

    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
