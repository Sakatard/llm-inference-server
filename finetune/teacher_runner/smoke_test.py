"""Phase 2 smoke test: 5 fixtures → 5 schema-valid TeacherLabel outputs.

Gate (per finetune/SPEC.md Phase 2): 5/5 produce schema-valid labels.

Uses polymarket-agents test fixtures (tests/fixtures/committee/) to exercise
a spread of market categories (election, sports, crypto, geopolitical, weather).

Each fixture has shape:
    {
      "market_payload": {...},
      "parsed_rules": {...},
      "connector_summary": {...},
      ...
    }

We map this to TeacherContextBundle. `connector_summary_redacted` is
constructed from `connector_summary` minus any field run_committee strips
internally (article bodies, etc.); for the smoke test the fixtures are
already redacted so we pass through.

Run:
    python3 -m finetune.teacher_runner.smoke_test

Exit code 0 on 5/5 pass; non-zero on any failure.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from . import TeacherContextBundle, TeacherRunnerError, run_teacher


# Inside polymarket-trader, /app/tests is a bind mount of host
# /home/xel/containers/polymarket-agents/tests. The runner is designed to run
# inside the trader for plumbing reasons (pydantic + committee), so default
# to the container path; allow override via env.
_FIXTURES_DIR = Path(os.environ.get(
    "TEACHER_FIXTURES_DIR",
    "/app/tests/fixtures/committee",
))
_FIXTURES = [
    "01_election.json",
    "02_sports.json",
    "03_crypto.json",
    "04_geopolitical.json",
    "05_weather.json",
]


def _bundle_from_fixture(path: Path) -> TeacherContextBundle:
    fx = json.loads(path.read_text())
    return TeacherContextBundle(
        market_id=fx["market_payload"].get("market_id", path.stem),
        decision_ts_utc="2026-05-16T00:00:00Z",  # smoke-test placeholder
        market_payload=fx["market_payload"],
        parsed_rules=fx["parsed_rules"],
        connector_summary_redacted=fx.get("connector_summary", {}),
        intended_size_usd=fx.get("intended_size_usd", 500),
    )


def main() -> int:
    if not _FIXTURES_DIR.is_dir():
        sys.stderr.write(f"fixtures dir missing: {_FIXTURES_DIR}\n")
        return 2

    results = []
    for fname in _FIXTURES:
        path = _FIXTURES_DIR / fname
        if not path.is_file():
            results.append((fname, "MISSING", "fixture file not found"))
            continue
        bundle = _bundle_from_fixture(path)
        t0 = time.monotonic()
        try:
            label = run_teacher(bundle)
            elapsed_s = time.monotonic() - t0
            results.append((
                fname,
                "PASS",
                f"status={label.status} p_yes={label.referee_raw_p_yes} "
                f"sha={label.committee_sha[:8]} elapsed={elapsed_s:.1f}s",
            ))
        except TeacherRunnerError as exc:
            elapsed_s = time.monotonic() - t0
            results.append((fname, "FAIL", f"{exc} (after {elapsed_s:.1f}s)"))
        except Exception as exc:  # noqa: BLE001
            elapsed_s = time.monotonic() - t0
            results.append((fname, "FAIL", f"{type(exc).__name__}: {exc} (after {elapsed_s:.1f}s)"))

    passes = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n=== Phase 2 smoke test: {passes}/{len(results)} PASS ===\n")
    for fname, status, msg in results:
        print(f"  {status:4}  {fname:30}  {msg}")
    return 0 if passes == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
