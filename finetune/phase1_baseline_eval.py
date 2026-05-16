"""Phase 1 — Baseline-without-FT eval.

Pulls 50 resolved Polymarket markets via gamma-api.polymarket.com, builds a
context for each (event/rules + current market state), sends to local
Qwen3.5-9B-Q4_K_M (no LoRA), scores:

  - schema_validity: did the model emit valid JSON matching polymarket_decision_v0
  - outcome_accuracy: did argmax(probability) match the actual resolved outcome
  - latency / VRAM
  - Brier score across the 50

Gate (per SPEC v3 §1 Phase 1): if schema_validity >= 95% AND outcome_accuracy >=
committee_accuracy_on_same_set, then fine-tuning is NOT NEEDED for v0.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any

OUT_DIR = Path("/home/xel/containers/llm-inference-server/finetune/REVIEWS")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Endpoints
LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8088")  # router on host; or http://127.0.0.1:9190 inside container
# If running locally on host, llama-server side-spawn at 9190 isn't reachable from host network — use the router.
GAMMA_URL = "https://gamma-api.polymarket.com"

N_MARKETS = int(os.environ.get("N_MARKETS", "50"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2000"))
TEMP = float(os.environ.get("TEMP", "0.3"))

SCHEMA_PATH = Path(__file__).parent / "schemas" / "polymarket_decision.schema.json"

SYSTEM_PROMPT = """You are a Polymarket trading analyst. For each market, output ONE JSON object matching the polymarket_decision_v0 schema.

The JSON object MUST have these fields:
- schema_version: exactly "polymarket_decision_v0"
- summary: string 50-500 chars (1-3 sentence summary)
- global_confidence: number 0-1
- choice_assessments: array of objects, one per outcome. Each: {label, probability, confidence, evidence: {reliability, diversity, recency, overall_strength}, data_coverage, contradictions: [], key_points: []}
- uncertainty_flags: array (from controlled vocab: yes_thesis_failed, no_thesis_failed, microstructure_unavailable, stale_data, thin_evidence_yes, thin_evidence_no, rule_ambiguous, high_disagreement, starved_bear_evidence, starved_bull_evidence, unsupported_citations_yes, unsupported_citations_no)
- trade_blockers: array (controlled vocab: wide_spread, thin_book, exit_depth_high_risk, stale_quote, committee_partial, rule_unmet, data_coverage_low, orderbook_unavailable)
- rationale: string 100-4000 chars (step-by-step reasoning)

All probability values 0-1. Probabilities across choice_assessments must sum to 1.0 ± 0.01. Labels must match the market's outcome labels exactly. Output ONLY the JSON object, no preamble or trailing text."""


def fetch_markets(n: int) -> list[dict]:
    """Pull closed (resolved) Polymarket markets via Gamma API."""
    # Filter for closed=true, has outcomes, decent volume, recent
    params = {
        "closed": "true",
        "limit": str(n * 3),  # over-fetch so we can filter
        "ascending": "false",
        "order": "endDate",
    }
    url = f"{GAMMA_URL}/markets?" + urllib.parse.urlencode(params)
    print(f"[gamma] fetching {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "phase1-eval/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        markets = json.loads(r.read())
    print(f"[gamma] got {len(markets)} markets", flush=True)
    # Filter: must have outcomes + resolved outcome + volume
    good: list[dict] = []
    for m in markets:
        outcomes = m.get("outcomes")
        outcome_prices = m.get("outcomePrices")
        if not outcomes or not outcome_prices:
            continue
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except Exception:
                continue
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except Exception:
                continue
        if not outcomes or len(outcomes) < 2:
            continue
        # resolved outcome = the one with price == "1" (or 1.0)
        winner = None
        for i, p in enumerate(outcome_prices):
            try:
                if float(p) >= 0.999:
                    winner = outcomes[i]
                    break
            except Exception:
                pass
        if winner is None:
            continue
        volume = m.get("volumeNum") or m.get("volume") or 0
        try:
            volume = float(volume)
        except Exception:
            volume = 0
        if volume < 10000:
            continue
        good.append({
            "market_id": m.get("id") or m.get("conditionId"),
            "slug": m.get("slug"),
            "question": m.get("question"),
            "description": m.get("description"),
            "outcomes": outcomes,
            "outcome_prices": outcome_prices,
            "resolved_outcome": winner,
            "volume_usd": volume,
            "end_date": m.get("endDate"),
        })
        if len(good) >= n:
            break
    print(f"[gamma] filtered to {len(good)} usable markets", flush=True)
    return good


def build_context(m: dict) -> str:
    """Build user-message context for one market."""
    rules = (m.get("description") or "").strip()[:1500]
    return (
        f"Market question: {m['question']}\n\n"
        f"Resolution rules:\n{rules}\n\n"
        f"Outcomes: {m['outcomes']}\n"
        f"(No news/microstructure feed available for this baseline run — score based on rules + general knowledge.)\n\n"
        f"Output the polymarket_decision_v0 JSON object now."
    )


def call_model(system_prompt: str, user_msg: str) -> tuple[str, dict]:
    """Send chat completion request, return (content, timings)."""
    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=json.dumps({
            "model": "qwen35-9b-eval",  # ignored if router routes by path; only for OpenAI compat
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMP,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    return content, d.get("timings", {})


def score_one(market: dict, response_content: str, schema: dict) -> dict:
    """Validate response against schema + check argmax-outcome match."""
    out = {
        "market_id": market["market_id"],
        "question": market["question"],
        "resolved_outcome": market["resolved_outcome"],
        "raw_len": len(response_content),
        "json_valid": False,
        "schema_valid": False,
        "argmax_label": None,
        "argmax_probability": None,
        "outcome_match": False,
        "prob_sum_ok": False,
        "errors": [],
    }
    # Strip any markdown fencing
    s = response_content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    try:
        j = json.loads(s)
        out["json_valid"] = True
    except Exception as e:
        out["errors"].append(f"json_parse: {type(e).__name__}: {e}")
        return out

    # Loose schema check (the most important: required top-level keys + choices array)
    required_top = ["schema_version", "summary", "global_confidence",
                    "choice_assessments", "uncertainty_flags", "trade_blockers", "rationale"]
    missing = [k for k in required_top if k not in j]
    if missing:
        out["errors"].append(f"missing_keys: {missing}")
    elif j.get("schema_version") != "polymarket_decision_v0":
        out["errors"].append(f"wrong_schema_version: {j.get('schema_version')!r}")
    else:
        ca = j.get("choice_assessments") or []
        if not ca:
            out["errors"].append("empty_choice_assessments")
        else:
            try:
                probs = [float(c.get("probability") or 0) for c in ca]
                psum = sum(probs)
                out["prob_sum_ok"] = 0.99 <= psum <= 1.01
                if not out["prob_sum_ok"]:
                    out["errors"].append(f"probs_dont_sum: {psum}")
                else:
                    best = max(ca, key=lambda c: float(c.get("probability") or 0))
                    out["argmax_label"] = best.get("label")
                    out["argmax_probability"] = float(best.get("probability") or 0)
                    # outcome match: label == resolved_outcome (case-insensitive trim)
                    if out["argmax_label"] and market["resolved_outcome"]:
                        out["outcome_match"] = (out["argmax_label"].strip().lower()
                                                == market["resolved_outcome"].strip().lower())
                    out["schema_valid"] = (
                        len(out["errors"]) == 0
                        and isinstance(j.get("summary"), str)
                        and 50 <= len(j.get("summary", "")) <= 500
                        and 0 <= float(j.get("global_confidence") or -1) <= 1
                        and isinstance(j.get("rationale"), str)
                        and 100 <= len(j.get("rationale", "")) <= 4000
                    )
            except Exception as e:
                out["errors"].append(f"schema_eval: {e}")
    return out


def main() -> int:
    schema = json.loads(SCHEMA_PATH.read_text())
    print(f"[schema] loaded {SCHEMA_PATH}", flush=True)

    markets = fetch_markets(N_MARKETS)
    if len(markets) < N_MARKETS:
        print(f"[WARN] only got {len(markets)}/{N_MARKETS} usable markets — continuing", flush=True)

    results: list[dict] = []
    for i, m in enumerate(markets, 1):
        ctx = build_context(m)
        t0 = time.time()
        try:
            content, timings = call_model(SYSTEM_PROMPT, ctx)
        except Exception as e:
            print(f"[{i}/{len(markets)}] {m['question'][:60]} CALL_FAIL: {e}", flush=True)
            results.append({
                "market_id": m["market_id"],
                "question": m["question"],
                "errors": [f"call_fail: {e}"],
                "json_valid": False,
                "schema_valid": False,
                "outcome_match": False,
            })
            continue
        dt = time.time() - t0
        scored = score_one(m, content, schema)
        scored["latency_s"] = round(dt, 2)
        scored["tok_per_s"] = timings.get("predicted_per_second")
        scored["predicted_n"] = timings.get("predicted_n")
        results.append(scored)
        status = ("OK" if scored["schema_valid"] else "BAD")
        match = "✓" if scored["outcome_match"] else "✗"
        print(f"[{i}/{len(markets)}] {status} {match} ({scored['latency_s']}s) {m['question'][:70]}", flush=True)

    # Aggregate
    n = len(results)
    json_valid = sum(1 for r in results if r["json_valid"])
    schema_valid = sum(1 for r in results if r["schema_valid"])
    outcome_match = sum(1 for r in results if r["outcome_match"])
    prob_sum_ok = sum(1 for r in results if r.get("prob_sum_ok"))
    # Brier on schema-valid runs only
    brier_terms = []
    for r in results:
        if r["schema_valid"] and r.get("argmax_probability") is not None:
            actual = 1.0 if r["outcome_match"] else 0.0
            p = r["argmax_probability"]
            brier_terms.append((p - actual) ** 2)
    brier = sum(brier_terms) / len(brier_terms) if brier_terms else None

    summary = {
        "n_total": n,
        "json_valid": json_valid,
        "json_valid_rate": json_valid / n if n else 0,
        "schema_valid": schema_valid,
        "schema_valid_rate": schema_valid / n if n else 0,
        "outcome_match": outcome_match,
        "outcome_match_rate": outcome_match / n if n else 0,
        "outcome_match_among_schema_valid": (outcome_match / schema_valid) if schema_valid else 0,
        "prob_sum_ok_rate": prob_sum_ok / n if n else 0,
        "brier_score_argmax": brier,
        "gate_schema_validity_pass": (schema_valid / n if n else 0) >= 0.95,
    }
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    ts = int(time.time())
    out_path = OUT_DIR / f"phase1_baseline_report_{ts}.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\n[report] saved to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
