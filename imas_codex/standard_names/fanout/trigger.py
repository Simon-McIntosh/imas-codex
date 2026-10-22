"""Trigger predicate + arm assignment for refine-site fan-out.

The trigger gate is the orthogonality knob between fan-out and the
existing grammar-retry compose path: fan-out fires when an answered
retrieval check's failure probability reaches the calibrated threshold,
so the targeted DD context it buys is spent only where a named check
asks for retrieval.  Reviewer comment text is never read — the comments
a person writes stay on the node for people.

Public surface
--------------

- :func:`should_trigger_fanout` — return ``(fire, excerpt)`` for the
  refine call site.  ``excerpt`` is always empty: the check path reads
  no comments.
- :func:`should_trigger_fanout_from_checks` — fire on the maximum failure
  probability among answered checks that request retrieval.
- :func:`assign_arm` — deterministic 50/50 (or
  ``refine_fanout_arm_percent``) within-cohort A/B label, hashing
  ``(sn_id, cycle_index)`` via blake2b for stability across runs.

This module is intentionally pure-Python with no graph or LLM imports;
the refine worker calls it once per item before any LLM spend.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from imas_codex.standard_names.review.checks import ReviewCheck

__all__ = [
    "assign_arm",
    "should_trigger_fanout",
    "should_trigger_fanout_from_checks",
]


def should_trigger_fanout_from_checks(
    checks: Sequence[ReviewCheck],
    check_results: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float,
) -> bool:
    """Fire when a retrieval check's measured failure probability reaches the threshold."""
    probabilities = [
        float(check_results[check.id].get("failure_probability", 0.0))
        for check in checks
        if check.retrieval
        and check.id in check_results
        and check_results[check.id].get("answered", False)
    ]
    return bool(probabilities) and max(probabilities) >= threshold


def should_trigger_fanout(
    *,
    checks: Sequence[ReviewCheck] = (),
    check_results: Mapping[str, Mapping[str, Any]] | None = None,
    check_threshold: float = 0.5,
) -> tuple[bool, str]:
    """Return ``(fire, excerpt)`` for the refine-site fan-out trigger.

    The trigger fires when the maximum failure probability over the
    answered retrieval checks reaches ``check_threshold``.  Comment text
    is never read, so the returned ``excerpt`` is always the empty
    string; it is returned so the call site keeps one shape, and callers
    must not read it as review signal.
    """
    fire = should_trigger_fanout_from_checks(
        checks, check_results or {}, threshold=check_threshold
    )
    return fire, ""


def assign_arm(
    sn_id: str,
    cycle_index: int,
    *,
    arm_percent: int = 50,
) -> str:
    """Assign a stable ``"on"`` / ``"off"`` arm for the within-cohort A/B.

    Hashes ``(sn_id, cycle_index)`` via blake2b (8-byte digest) and
    maps the integer mod 100 against ``arm_percent``.  Stability:

    - Identical ``(sn_id, cycle_index)`` always lands in the same arm.
    - Different ``cycle_index`` values for the same ``sn_id`` are
      independently distributed (no correlation across cycles).
    - Default ``arm_percent=50`` yields a 50/50 split; values 0 and
      100 are honoured (all-off / all-on respectively).

    The split uses blake2b rather than Python's ``hash()`` because the latter is
    PYTHONHASHSEED-randomised between processes.  The functional
    contract (deterministic 50/50) is preserved.
    """
    if arm_percent <= 0:
        return "off"
    if arm_percent >= 100:
        return "on"
    payload = f"{sn_id}|{int(cycle_index)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="big") % 100
    return "on" if bucket < arm_percent else "off"
