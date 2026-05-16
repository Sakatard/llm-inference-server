"""Phase 2 teacher-runner: frozen contract for polymarket-agents committee inference.

The committee is pinned at PINNED_SHA.txt. Inputs/outputs are versioned via the
Pydantic models in schemas.py — bump bundle_version / label_version on any
schema change so old training data can be re-validated.
"""
from .schemas import TeacherContextBundle, TeacherLabel
from .run import TeacherRunnerError, run_teacher

__all__ = [
    "TeacherContextBundle",
    "TeacherLabel",
    "TeacherRunnerError",
    "run_teacher",
]
