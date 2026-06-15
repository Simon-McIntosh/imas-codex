"""Physics-correctness judge for SN composer benchmark.

Scores a composed standard name against the *physical* meaning of the DD
path it was derived from, rather than against the (gameable) grammar and
convention rubric. The judge asks whether the name is faithful to what the
quantity physically is and how it is measured — e.g. a Rogowski coil
measures enclosed current via induced voltage, so naming its signal
``current_of_rogowski_coil`` mis-states the measurement principle even
though it is grammatically valid.

This module provides the verdict schema (:class:`PhysicsVerdict`) and the
benchmark test-set loader (:func:`load_bench_paths`). Later tasks add the
LLM judge call, the calibration gate, and wire the judge into
``imas_codex/standard_names/benchmark.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class PhysicsVerdict(BaseModel):
    """Per-name physics-correctness verdict from the judge.

    Each boolean axis isolates one way a composed name can be physically
    wrong; ``faithful`` is the overall conjunctive verdict, and ``reason``
    carries the judge's free-text justification.
    """

    name: str = Field(description="The composed standard name being judged.")
    faithful: bool = Field(
        description="Overall verdict: the name is physically faithful to the "
        "quantity and its measurement on every axis below."
    )
    base_correct: bool = Field(
        description="The physical base (the underlying quantity) is correct."
    )
    measurement_principle_correct: bool = Field(
        description="The name does not mis-state how the quantity is measured "
        "(e.g. a Rogowski coil measures enclosed current via induced "
        "voltage, not the coil's own current)."
    )
    qualifiers_preserved: bool = Field(
        description="Physically meaningful qualifiers from the path (locus, "
        "source, extremum, ordering) are preserved in the name."
    )
    no_over_qualification: bool = Field(
        description="The name is not over-qualified with redundant or "
        "physically unwarranted modifiers."
    )
    valid: bool = Field(description="The name is a grammatically valid standard name.")
    reason: str = Field(
        description="Free-text justification for the verdict, citing the "
        "physics where relevant."
    )


def load_bench_paths(path: Path | None = None) -> list[dict]:
    """Load the physics benchmark test set.

    Returns a list of ``{"path", "category"}`` objects. When ``path`` is
    ``None`` the default ``research/physics_bench_paths.json`` shipped with
    the repo is used, resolved relative to this module.
    """
    if path is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "research"
            / "physics_bench_paths.json"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
