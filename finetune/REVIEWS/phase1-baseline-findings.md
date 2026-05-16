# Phase 1 — Baseline-without-FT Eval Findings

**Verdict: GATE FAILS. Proceed to Phase 2 → 3 → 4.**

## Setup

- Model: `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf` (5.4 GB) on P40
- Cache: **f16** (q8_0 produces garbage on Qwen3.5-9B — confirmed family-wide quirk, same as Qwen3.6-MTP-UD)
- Context: 16k
- 45 markets pulled from `gamma-api.polymarket.com/markets?closed=true` (filtered for resolved + volume ≥ $10k)
- System prompt: describes polymarket_decision_v0 schema inline; no grammar enforcement
- No news context provided (markets are FDV-after-launch crypto events)
- temp=0.3, max_tokens=2000

## Results

| Metric | Value | Gate target |
|---|---|---|
| JSON-parseable | 45/45 (100%) | — |
| Schema-valid | **39/45 (86.7%)** | ≥95% ✗ |
| Prob-sum OK | 45/45 (100%) | — |
| Outcome match (all) | 22/45 (48.9%) | ≥committee — n/a |
| Outcome match (schema-valid) | 22/39 (56.4%) | — |
| Brier (argmax) | 0.34 | — |
| Avg latency | ~17s/market | — |
| Decode tok/s | ~38-55 | — |

## Failure mode analysis

6 schema-invalid rows all JSON-parsed cleanly; failed on length constraints (summary <50 or >500 chars; rationale <100 or >4000). Not structural — formatting drift.

**3 of those 6 had correct argmax**: reasoning was right, JSON shape slightly off. With `response_format: json_schema` grammar at inference time, schema_valid would likely climb to ~100% without any fine-tune.

## Decisions for downstream phases

1. **Grammar enforcement** at inference is mandatory in Phase 7 deploy — eliminates this failure mode entirely.
2. **News context is mandatory** for outcome accuracy. The 56% rate is barely above random because the model has no information about token launches. Phase 4 dataset MUST include news.
3. **Better market selection**. Gamma's `closed=true` returned too many niche FDV-launch markets in one batch. Phase 4 needs category-stratified sampling (politics, sports, current events, mixed crypto).
4. **No need to chase 95% schema-valid via FT alone** — grammar handles that. FT's job is reasoning + calibration on real polymarket data with news.

## Q8_0 cache: BROKEN on Qwen3.5-9B

Same as Qwen3.6-MTP-UD earlier. Garbage output (`?????`). Always use f16 cache (or turbo4 if on TurboQuant fork). Documenting for the qwen-trader Service config in Phase 7.

## Reports

- `phase1_baseline_report.json` (full 45-row results, locally archived)
- This file (`phase1-baseline-findings.md`)
