"""Manual SCP-back + destroy for the Phase 4 rescue path.

Used when vast_run_phase4.py was killed mid-run and replaced by a hand-driven
resume script on Vast. Pulls the standard output artifacts back into
finetune/REVIEWS/phase4_<ts>/ and (optionally) destroys the instance.

Usage:
    VAST_API_KEY=... python3 finetune/phase4_pull.py \\
        --instance 36893062 --host 79.160.189.79 --port 13264 \\
        [--fetch-merged]   # opt-in 54 GB merged bf16 pull
        [--destroy]        # destroy instance after successful pull
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).parent
LOCAL_REPORT_DIR = REPO_DIR / "REVIEWS"


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--user", default="root")
    ap.add_argument("--fetch-merged", action="store_true")
    ap.add_argument("--destroy", action="store_true")
    args = ap.parse_args()

    LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_local = LOCAL_REPORT_DIR / f"phase4_{ts}"
    out_local.mkdir(parents=True, exist_ok=True)

    scp_opts = [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-P", str(args.port),
    ]
    target = f"{args.user}@{args.host}"

    print(f"[pull] -> {out_local}", flush=True)
    # Metrics + preds + log (always)
    for src in (
        "/workspace/output/train_metrics.json",
        "/workspace/output/holdout_preds.jsonl",
        "/workspace/resume.log",
        "/workspace/phase4_train.log",
    ):
        run(["scp", *scp_opts, f"{target}:{src}", str(out_local / Path(src).name)])

    # Adapter (always — small)
    run(["scp", *scp_opts, "-r",
         f"{target}:/workspace/output/adapter",
         str(out_local / "adapter")])

    # Merged bf16 (opt-in — large)
    if args.fetch_merged:
        print("[pull] fetching merged bf16 (~54 GB) — slow", flush=True)
        run(["scp", *scp_opts, "-r",
             f"{target}:/workspace/output/merged",
             str(out_local / "merged")])

    metrics_path = out_local / "train_metrics.json"
    if metrics_path.exists():
        print("\n=== train_metrics.json ===")
        print(metrics_path.read_text())
    else:
        print(f"\n[WARN] no train_metrics.json at {metrics_path}; eval may have failed")

    if args.destroy:
        api_key = os.environ.get("VAST_API_KEY")
        if not api_key:
            print("[WARN] VAST_API_KEY not set; cannot auto-destroy. "
                  "Manual: vastai destroy instance {0}".format(args.instance))
            return 1
        try:
            from vastai import VastAI
            v = VastAI(api_key=api_key)
            v.destroy_instance(id=args.instance)
            print(f"[destroyed] instance {args.instance}")
        except Exception as exc:
            print(f"[ERROR] destroy failed: {exc}")
            print(f"  Manual: vastai destroy instance {args.instance}")
            return 1
    else:
        print(f"\n[skip-destroy] instance {args.instance} left running. "
              f"To destroy: vastai destroy instance {args.instance}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
