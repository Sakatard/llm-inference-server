"""Phase 4 step 3: join bundles.jsonl + labels.jsonl into training JSONL.

Output format: one object per line, `{"messages": [{"role":..., "content":...}, ...]}`,
ready for `tokenizer.apply_chat_template` inside the Vast trainer.

Stratification (eval gating per SPEC Phase 4):
- 50-market holdout set is the LAST 50 lines of input (after stable sort by
  market_id), to avoid leakage across the random split. Written to
  <out>.holdout.jsonl; training set goes to <out>.train.jsonl.

Run from the host (no special deps beyond stdlib + Python):

    python3 -m finetune.format_jsonl \\
        --bundles /tmp/bundles.jsonl --labels /tmp/labels.jsonl \\
        --out finetune/datasets/phase4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


_SYSTEM = (
    "You are a probabilistic forecaster reviewing prediction markets. For each "
    "market you receive (1) the question and rules, (2) recent news headlines "
    "from the 7-day window before the decision, and (3) microstructure context. "
    "Produce a brief, calibrated assessment ending with a single line "
    '`p_yes=<float between 0 and 1>`.'
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] skipping malformed line in {path}: {exc}", file=sys.stderr)
    return rows


def _format_user(bundle: Dict[str, Any]) -> str:
    m = bundle.get("market_payload", {})
    rules = bundle.get("parsed_rules", {})
    news = bundle.get("connector_summary_redacted", {}).get("news", []) or []
    lines: List[str] = []
    lines.append(f"Market: {m.get('question', '')}")
    end = m.get("endDate", "")
    if end:
        lines.append(f"Resolution date: {end}")
    desc = m.get("description") or rules.get("raw_text") or ""
    if desc:
        lines.append("")
        lines.append("Rules / description:")
        lines.append(desc.strip()[:1500])
    if news:
        lines.append("")
        lines.append("Recent news (last 7 days before decision):")
        for n in news[:15]:
            title = (n.get("title") or "").strip()
            src = (n.get("source") or "").strip()
            if title:
                lines.append(f"  - [{src}] {title}" if src else f"  - {title}")
    lines.append("")
    lines.append("Provide your assessment.")
    return "\n".join(lines)


def _format_assistant(label: Dict[str, Any]) -> Optional[str]:
    if label.get("status") != "ok":
        return None
    summary = (label.get("referee_summary") or "").strip()
    rationale = (label.get("referee_rationale") or "").strip()
    p_yes = label.get("referee_raw_p_yes")
    if p_yes is None:
        return None
    parts: List[str] = []
    if summary:
        parts.append(summary)
    if rationale:
        if parts:
            parts.append("")
        parts.append(rationale)
    if parts:
        parts.append("")
    parts.append(f"p_yes={float(p_yes):.3f}")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output prefix (writes <out>.train.jsonl + <out>.holdout.jsonl)")
    ap.add_argument("--holdout", type=int, default=50)
    args = ap.parse_args()

    bundles = {b["market_id"]: b for b in _load_jsonl(args.bundles)}
    labels = {l["market_id"]: l for l in _load_jsonl(args.labels)}
    joint_ids = sorted(set(bundles) & set(labels))
    print(f"[load] bundles={len(bundles)} labels={len(labels)} joint={len(joint_ids)}",
          file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for mid in joint_ids:
        user_content = _format_user(bundles[mid])
        asst_content = _format_assistant(labels[mid])
        if asst_content is None:
            continue
        rows.append({
            "market_id": mid,
            "messages": [
                {"role": "system",    "content": _SYSTEM},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": asst_content},
            ],
        })

    if not rows:
        print("[fail] no rows produced (all labels status!=ok or missing p_yes)",
              file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_holdout = min(args.holdout, len(rows) // 5)
    train = rows[:-n_holdout] if n_holdout else rows
    holdout = rows[-n_holdout:] if n_holdout else []

    with open(str(args.out) + ".train.jsonl", "w") as fh:
        for r in train:
            fh.write(json.dumps(r) + "\n")
    with open(str(args.out) + ".holdout.jsonl", "w") as fh:
        for r in holdout:
            fh.write(json.dumps(r) + "\n")

    print(f"[done] train={len(train)} holdout={len(holdout)} → {args.out}.{{train,holdout}}.jsonl",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
