"""Phase 2 teacher-runner entrypoint.

Designed to execute inside the polymarket-trader container (where the
committee + its env are available). When invoked from the host, dispatches
via `docker exec polymarket-trader` automatically.

The contract is frozen:
  - PINNED_SHA.txt declares the committee commit; the runner verifies the
    bind-mounted polymarket-agents on the host (== /app/agents in the
    trader) matches before each call.
  - Inputs are TeacherContextBundle (schemas.py).
  - Outputs are TeacherLabel (schemas.py).

CLI usage:
    # From host — auto-dispatches into trader:
    python3 -m finetune.teacher_runner.run < bundle.json > label.json

    # From inside trader (after mounting this repo to /teacher_runner):
    PYTHONPATH=/teacher_runner python3 -m teacher_runner.run < bundle.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .schemas import TeacherContextBundle, TeacherLabel


_PKG_DIR = Path(__file__).parent
_PINNED_SHA = (_PKG_DIR / "PINNED_SHA.txt").read_text().strip()

# Trader container + bind-mount host path. /app/agents in the container is
# a bind mount of this host path, so resolving SHA host-side gives the exact
# code the committee will execute. The trader image has no `git` binary.
_TRADER_CONTAINER = "polymarket-trader"
_AGENTS_REPO_HOST = Path("/home/xel/containers/polymarket-agents")
_INSIDE_TRADER = Path("/app/agents/research/committee/runner.py").exists()


class TeacherRunnerError(RuntimeError):
    """Raised when committee invocation fails or violates the pinned contract."""


def _resolve_host_sha() -> str:
    """HEAD of polymarket-agents on host. Identical to /app/agents in trader."""
    proc = subprocess.run(
        ["git", "-C", str(_AGENTS_REPO_HOST), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return proc.stdout.strip()


def _verify_sha(sha: str) -> None:
    if not sha.startswith(_PINNED_SHA):
        raise TeacherRunnerError(
            f"committee SHA drift: repo at {sha}, pinned {_PINNED_SHA}. "
            "Bump PINNED_SHA.txt (and re-label) or `git checkout` the pinned SHA."
        )


def _call_committee_locally(bundle: TeacherContextBundle) -> Dict[str, Any]:
    """Direct import + invocation. Only valid when running inside trader."""
    from agents.research.committee.runner import run_committee  # type: ignore[import-not-found]
    out = run_committee(
        market_payload=bundle.market_payload,
        parsed_rules=bundle.parsed_rules,
        connector_summary_redacted=bundle.connector_summary_redacted,
        intended_size_usd=bundle.intended_size_usd,
        cancellation_flag=None,
    )
    return out.model_dump(mode="json")


def _call_committee_via_docker(bundle: TeacherContextBundle) -> Dict[str, Any]:
    """Dispatch into trader. Streams the runner module via PYTHONPATH=/teacher_runner."""
    # Ensure the runner dir is reachable inside the trader. We use docker cp
    # to lay it down at /teacher_runner on every call — idempotent and avoids
    # docker-compose volume edits.
    subprocess.run(
        ["docker", "exec", _TRADER_CONTAINER, "mkdir", "-p", "/teacher_runner/teacher_runner"],
        check=True,
    )
    for fname in ("__init__.py", "schemas.py", "run.py", "PINNED_SHA.txt"):
        subprocess.run(
            ["docker", "cp", str(_PKG_DIR / fname),
             f"{_TRADER_CONTAINER}:/teacher_runner/teacher_runner/{fname}"],
            check=True,
        )
    proc = subprocess.run(
        ["docker", "exec", "-i", "-e", "PYTHONPATH=/teacher_runner", "-e",
         "COMMITTEE_MODE=COMMITTEE",  # ensure committee path is exercised
         _TRADER_CONTAINER, "python3", "-m", "teacher_runner.run"],
        input=bundle.model_dump_json(), capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise TeacherRunnerError(
            f"trader exec rc={proc.returncode}; stderr={proc.stderr[-800:]}"
        )
    # Inner runner outputs the TeacherLabel JSON itself; round-trip back through
    # the model to confirm schema validity before returning to caller.
    try:
        label = TeacherLabel.model_validate_json(proc.stdout)
    except Exception as exc:  # pydantic ValidationError lives in pydantic
        raise TeacherRunnerError(
            f"trader stdout failed TeacherLabel validation: {exc}; "
            f"raw={proc.stdout[:500]!r}"
        )
    # Re-package the committee_output-equivalent for the local path so the
    # outer caller's contract stays unified.
    return {"_already_label": label}


def run_teacher(bundle: TeacherContextBundle) -> TeacherLabel:
    """Frozen-contract committee invocation. Raises TeacherRunnerError on any
    contract violation (SHA drift, malformed output, container failure)."""
    if _INSIDE_TRADER:
        # SHA verified by host before dispatch; if we're already inside the
        # trader, the bind-mount tells us we're on the same code, but we
        # cannot resolve git SHA without git. Trust the host-side check.
        sha = os.environ.get("COMMITTEE_PINNED_SHA", _PINNED_SHA)
        out = _call_committee_locally(bundle)
        return _output_to_label(bundle, out, sha)

    sha = _resolve_host_sha()
    _verify_sha(sha)
    result = _call_committee_via_docker(bundle)
    if "_already_label" in result:
        return result["_already_label"]
    return _output_to_label(bundle, result, sha)


def _output_to_label(bundle: TeacherContextBundle, out: Dict[str, Any], sha: str) -> TeacherLabel:
    return TeacherLabel(
        market_id=bundle.market_id,
        decision_ts_utc=bundle.decision_ts_utc,
        status=out["status"],
        failure_reason=out.get("failure_reason"),
        referee_raw_p_yes=out.get("referee_raw_p_yes"),
        referee_global_confidence=out.get("referee_global_confidence"),
        referee_summary=out.get("referee_summary"),
        referee_rationale=out.get("referee_rationale"),
        referee_choice_assessments=out.get("referee_choice_assessments"),
        yes_thesis_output=out.get("yes_thesis_output"),
        no_thesis_output=out.get("no_thesis_output"),
        uncertainty_flags=out.get("uncertainty_flags", []),
        trade_blockers=out.get("trade_blockers", []),
        fallback_used=bool(out.get("fallback_used", False)),
        committee_sha=sha,
        models_used=out.get("models_used", {}),
    )


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.stderr.write("usage: python -m finetune.teacher_runner.run < bundle.json\n")
        return 2
    bundle = TeacherContextBundle.model_validate_json(raw)
    label = run_teacher(bundle)
    sys.stdout.write(label.model_dump_json(indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
