"""Pydantic schemas for the Phase 2 teacher-runner frozen contract.

Bump bundle_version / label_version on any field add/remove/rename so
downstream training code can refuse to consume mismatched samples.

bundle_version 1 — initial release, matches polymarket-agents @ 7d4fdf8
                   committee.runner.run_committee signature.
label_version  1 — initial release, captures CommitteeOutput-derived fields
                   sufficient for FT supervision (referee p_yes, recommended
                   side, rationale, blockers, uncertainty flags).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Input: bundle that travels from data archival → committee
# -----------------------------------------------------------------------------


class TeacherContextBundle(BaseModel):
    """Frozen input contract for committee invocation.

    Mirrors the four kwargs of polymarket-agents
    `agents.research.committee.runner.run_committee`:
      market_payload, parsed_rules, connector_summary_redacted, intended_size_usd

    Plus identifier + version metadata.
    """

    model_config = ConfigDict(extra="forbid")

    bundle_version: Literal[1] = 1

    # Identification — present on every bundle for cross-referencing data
    # archives and committee outputs.
    market_id: str = Field(..., description="Polymarket condition_id or crypto-event id")
    decision_ts_utc: str = Field(
        ...,
        description=(
            "ISO-8601 UTC timestamp of the decision moment this bundle reconstructs. "
            "Phase 3 validates orderbook/news reconstructibility at this timestamp."
        ),
    )

    # Committee inputs (exact shapes consumed by run_committee). Kept as
    # Dict[str, Any] because the committee internally validates via its own
    # Pydantic models — duplicating those validators here would couple us to
    # committee internals and defeat the contract.
    market_payload: Dict[str, Any]
    parsed_rules: Dict[str, Any]
    connector_summary_redacted: Dict[str, Any]
    intended_size_usd: int = 500


# -----------------------------------------------------------------------------
# Output: label produced by the committee, formatted for downstream training
# -----------------------------------------------------------------------------


class TeacherLabel(BaseModel):
    """Frozen output contract — supervision target for FT.

    Derived from `agents.research.committee.contracts.CommitteeOutput`.
    Captures the fields needed for FT loss + later auditing; drops timings
    + token counts (those go into a separate run_metrics stream).
    """

    model_config = ConfigDict(extra="forbid")

    label_version: Literal[1] = 1

    market_id: str
    decision_ts_utc: str

    # Top-level status from CommitteeOutput
    status: Literal["ok", "partial", "failed"]
    failure_reason: Optional[str] = None

    # Referee outputs — the supervision signal
    referee_raw_p_yes: Optional[float] = None
    referee_global_confidence: Optional[float] = None
    referee_summary: Optional[str] = None
    referee_rationale: Optional[str] = None
    # choice_assessments preserves the multi-choice probability mass.
    # Each entry mirrors committee.contracts.ChoiceAssessment.
    referee_choice_assessments: Optional[List[Dict[str, Any]]] = None

    # Thesis outputs — for auditing / hindsight teacher in v1
    yes_thesis_output: Optional[Dict[str, Any]] = None
    no_thesis_output: Optional[Dict[str, Any]] = None

    # Diagnostic flags
    uncertainty_flags: List[str] = Field(default_factory=list)
    trade_blockers: List[str] = Field(default_factory=list)
    fallback_used: bool = False

    # Provenance: which committee SHA + which models ran
    committee_sha: str
    models_used: Dict[str, str] = Field(default_factory=dict)
