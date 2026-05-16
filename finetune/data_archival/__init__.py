"""Phase 3 data archival reconstruction check.

Validates that historical Polymarket markets can be reconstructed at a
decision moment 24h prior to resolution. Probes:
- Trade history (Polymarket orderbook subgraph orderFilledEvents)
- News window (GDELT DOC API with absolute date range)

Outputs per-market reconstruction class:
  full          — trades present in [-24h, decision] + news ≥ 5 in [-7d, decision]
  degraded      — only one of the two
  unreconstructable — neither

Phase 3 gate: ≥ 70% full reconstruction across 20 markets in 180-day lookback.
"""
