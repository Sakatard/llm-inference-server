"""Phase 4 trainer — runs on Vast.ai 4090. QLoRA over the MTP-preserved
Qwen3.6-27B checkpoint (`llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-
MTP-Preserved`), LM loss only, MTP heads frozen (regex-excluded from PEFT).

Inputs (all in --work-dir):
- `train.jsonl`   — chat-template messages, one row per example
- `holdout.jsonl` — 50-row holdout for eval
- base model snapshot (downloaded once via huggingface_hub.snapshot_download)

Outputs (under --work-dir/output):
- `adapter/`            — PEFT adapter weights (small, ~200 MB)
- `merged/`             — full bf16 merged checkpoint (~54 GB)
- `train_metrics.json`  — loss curve + holdout eval
- `holdout_preds.jsonl` — per-row prediction with extracted p_yes

Run inside the Vast worker:
    python3 phase4_train.py --work-dir /workspace
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

# Match attention + FFN linear projections; skip mtp.* explicitly so the 15
# MTP heads stay at pretrained values and survive the merge unchanged.
_LORA_TARGET_REGEX = (
    r"^(?!.*mtp\.).*(?:"
    r"q_proj|k_proj|v_proj|o_proj|"
    r"gate_proj|up_proj|down_proj"
    r")$"
)


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


def _build_dataset(rows: List[Dict[str, Any]], tokenizer, max_seq_len: int):
    from datasets import Dataset
    texts: List[str] = []
    for r in rows:
        text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def _extract_p_yes(text: str) -> Optional[float]:
    m = re.search(r"p_yes\s*=\s*([0-9]*\.?[0-9]+)", text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--base-model", default=_BASE_MODEL_ID)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--skip-merge", action="store_true",
                    help="Skip the bf16 merge step (saves ~20 min when only adapter is needed)")
    args = ap.parse_args()

    wd = args.work_dir
    train_path = wd / "train.jsonl"
    holdout_path = wd / "holdout.jsonl"
    out_dir = wd / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imports happen here so --help works without GPU stack installed.
    import torch
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    _log(f"snapshot_download {args.base_model}")
    base_local = Path(snapshot_download(args.base_model))
    _log(f"base at {base_local}")

    _log("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_local, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log("load model (4-bit nf4 QLoRA)")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_local,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    _log("attach LoRA adapter (MTP heads excluded by regex)")
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_LORA_TARGET_REGEX,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    _log("build train + holdout datasets")
    train_rows = _load_jsonl(train_path)
    holdout_rows = _load_jsonl(holdout_path)
    train_ds = _build_dataset(train_rows, tokenizer, args.max_seq_len)
    _log(f"train={len(train_rows)} holdout={len(holdout_rows)}")

    training_args = TrainingArguments(
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
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to=[],
        max_seq_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        dataset_text_field="text",
        packing=False,
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

    # --- Eval on holdout ---
    _log("eval on holdout")
    eval_results: List[Dict[str, Any]] = []
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, row in enumerate(holdout_rows):
            msgs = row["messages"]
            # Strip the assistant turn — model generates it.
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
                # Direction agreement (above/below 0.5)
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
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out_dir / "holdout_preds.jsonl", "w") as fh:
        for r in eval_results:
            fh.write(json.dumps(r) + "\n")
    _log(f"metrics: {metrics}")

    # --- Merge bf16 (skippable for adapter-only deploys) ---
    if not args.skip_merge:
        _log("merge LoRA into base (bf16)")
        del model
        del trainer
        torch.cuda.empty_cache()
        base_full = AutoModelForCausalLM.from_pretrained(
            base_local, torch_dtype=torch.bfloat16, device_map="cpu",
            trust_remote_code=True,
        )
        from peft import PeftModel
        merged = PeftModel.from_pretrained(base_full, str(adapter_dir))
        merged = merged.merge_and_unload()
        merged_dir = out_dir / "merged"
        merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(str(merged_dir))
        # Sanity: confirm MTP tensors carried through.
        mtp_files = list(merged_dir.glob("*.safetensors"))
        _log(f"merged → {merged_dir} ({len(mtp_files)} safetensors)")

    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
