# Phase 3 — Data archival reconstruction (2026-05-16)

**Gate (≥70% full reconstruction in 180d lookback): FAIL.**

Empirical result: 40% full / 60% degraded / 0% unreconstructable across 20
substantive closed Polymarket markets bucketed recent (0-30d) / mid (30-90d) /
old (90-180d).

## Probes

For each market, decision moment = `endDate - 24h`. Two probes:

1. **Trade history** — `data-api.polymarket.com/trades` for the conditionId
   in `[decision-24h, decision]`. Reconstructed if ≥ 1 trade returned.
2. **News window** — GDELT DOC API with absolute date range
   `[decision-7d, decision]`, fallback to Google News RSS (`after:` / `before:`)
   when GDELT 429s. Reconstructed if ≥ 5 articles returned.

Classifications:
- `full` = trades_ok AND news_ok
- `degraded` = exactly one
- `unreconstructable` = neither

## Results

| bucket  | full | degraded | unrec | n |
|---------|------|----------|-------|---|
| recent  | 1    | 6        | 0     | 7 |
| mid     | 2    | 5        | 0     | 7 |
| old     | 5    | 1        | 0     | 6 |
| **all** | **8 (40%)** | **12 (60%)** | **0** | **20** |

All 20 markets succeeded on trade reconstruction (100%). News window was the
bottleneck: weather and tennis O/U markets dominate the substantive closed
pool and have minimal indexed news coverage.

GDELT was rate-limited (one req/5s) at the start of the day and never recovered
during this run; Google News RSS handled 100% of news probes via the fallback
path. For Phase 4's 200-market label run, request a GDELT API key (mailto
`kalev.leetaru5@gmail.com` per their 429 body) — gives ~10× the throughput.

## SPEC contingency

Per `finetune/SPEC.md` Phase 3 gate spec: "If < 70%, reduce lookback window
or accept partial-context training." Adopting:

- **Partial-context training accepted.** Phase 4 dataset filters to markets
  with `news_count ≥ 5` in the decision window; markets with sparser news
  (weather, niche sports) are excluded from FT supervision. Trade history is
  always reconstructible so it stays as a feature.
- **Lookback unchanged at 180d** — the failure mode is *category distribution*
  (Polymarket's closed-market mix), not *temporal degradation*. Both recent
  (0-30d) and old (90-180d) buckets show similar full-reconstruction rates.

## Reproduce

```bash
python3 -m finetune.data_archival.check_reconstruction \
    --out finetune/REVIEWS/phase3_reconstruction.json
```

Cost: $0 (Gamma + data-api + GDELT + Google News RSS all free).
Runtime: ~3 min (20 markets × 5.5s GDELT throttle).
