"""Phase 4 step 2: run teacher_runner.run_teacher on every bundle in a
bundles.jsonl, write labels.jsonl.

Resumable: rows already present in --out (matched by market_id) are skipped.
On failure, logs to stderr and continues; the failed market is NOT written so
re-runs retry it. Use --max-fail-rate to abort the batch if too many fail
(likely indicates committee outage, not data issue).

Run inside polymarket-trader (committee + pydantic):

    docker exec -i -e PYTHONPATH=/tmp/teacher_runner -e COMMITTEE_MODE=COMMITTEE \\
        polymarket-trader python3 /tmp/label_bundles.py \\
            --in /tmp/bundles.jsonl --out /tmp/labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Set

from teacher_runner import TeacherContextBundle, TeacherRunnerError, run_teacher


def _already_labeled(out_path: Path) -> Set[str]:
    seen: Set[str] = set()
    if not out_path.is_file():
        return seen
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line).get("market_id", ""))
            except json.JSONDecodeError:
                pass
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--max-fail-rate", type=float, default=0.5,
                    help="abort if failures/attempts > this after the first 20 attempts")
    args = ap.parse_args()

    seen = _already_labeled(args.out_path)
    print(f"[resume] {len(seen)} markets already labeled in {args.out_path}", file=sys.stderr)

    out_fh = open(args.out_path, "a")
    attempts = 0
    fails = 0
    successes = 0
    t0 = time.monotonic()
    with open(args.in_path) as in_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            try:
                bundle = TeacherContextBundle.model_validate_json(line)
            except Exception as exc:
                print(f"[skip] bundle parse err: {exc}", file=sys.stderr)
                continue
            if bundle.market_id in seen:
                continue
            attempts += 1
            t_market = time.monotonic()
            try:
                label = run_teacher(bundle)
                out_fh.write(label.model_dump_json() + "\n")
                out_fh.flush()
                successes += 1
                elapsed = time.monotonic() - t_market
                print(
                    f"[{successes:>3}] {bundle.market_id[:10]}.. status={label.status} "
                    f"p_yes={label.referee_raw_p_yes} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                )
            except (TeacherRunnerError, Exception) as exc:
                fails += 1
                print(
                    f"[fail] {bundle.market_id[:10]}.. {type(exc).__name__}: {str(exc)[:200]}",
                    file=sys.stderr,
                )
                if attempts >= 20 and fails / attempts > args.max_fail_rate:
                    print(f"[abort] fail rate {fails}/{attempts} > {args.max_fail_rate}",
                          file=sys.stderr)
                    out_fh.close()
                    return 2
    out_fh.close()
    total = time.monotonic() - t0
    print(
        f"\n[done] {successes} labeled, {fails} failed, {len(seen)} already-present "
        f"in {total:.0f}s",
        file=sys.stderr,
    )
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
