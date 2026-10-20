"""Paid-panel audit sample kept beside the deciding Jev route.

When the calibrated cascade decides, the paid review panel no longer sets an
item's stage, score or route. A deterministic fraction of decided items still
runs the panel and its verdict is stored as a non-deciding audit row, so the
two judges' drift is observable without either of them changing the other's
outcome.

Three things keep the audit from deciding anything:

* the sample is a stable hash of the item id and the run id, so a replay picks
  the same items and the same rate selects the same set;
* the audit row is written as a non-canonical judge record
  (``judge_kind='jev'``, ``resolution_role='audit'``), which the review
  aggregate query already excludes the way it excludes a shadow row;
* the audit is attempted only after the route is stored, spends through the
  same budget pool, and a refused reservation or a failed panel run records a
  skipped audit instead of failing the decision.

The row is a judge record rather than an incumbent record because no accepting,
refining or exporting path may read it: it carries the panel's verdict for
comparison, not a score that mirrors onto the StandardName.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

_SAMPLE_DOMAIN = b"sn-review:panel-audit:v1"
_PANEL_MODEL = "panel-quorum"
_PANEL_ROUTE_KEY = "panel:route"
ACCEPT_ROUTE = "accept"
REFINE_ROUTE = "refine"
AUDIT_ROLE = "audit"
AUDIT_METHOD = "panel_audit"
AUDIT_SKIPPED_METHOD = "panel_audit_skipped"


def panel_audit_rate() -> float:
    """The configured audit rate from ``[sn-review].panel-audit-rate``."""
    from imas_codex.settings import get_sn_review_panel_audit_rate

    return get_sn_review_panel_audit_rate()


def audit_selected(axis: str, item_id: str, run_id: str, rate: float) -> bool:
    """Whether *axis, item* is sampled into the audit for this run.

    The draw is ``sha256(run_id, axis, item_id)`` over its first 64 bits, so it
    is stable for a given triple and independent across items. A rate at or
    below zero selects nothing and a rate at or above one selects everything.
    """
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    payload = _SAMPLE_DOMAIN + b"\x00" + "\x00".join((run_id, axis, item_id)).encode()
    draw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return draw / 2**64 < rate


def panel_route(score: float, *, min_score: float) -> str:
    """The panel's accept or refine route at its own cutoff."""
    return ACCEPT_ROUTE if float(score) >= min_score else REFINE_ROUTE


def _panel_payload(route: str | None, score: float | None) -> dict[str, dict]:
    return {
        _PANEL_ROUTE_KEY: {
            "answer": route,
            "probs": {"acceptance": score},
            "confidence": None,
        }
    }


def _audit_record(
    item: Mapping[str, Any],
    axis: str,
    *,
    panel_result: Mapping[str, Any] | None,
    jev_route: str,
    skipped_reason: str | None,
    min_score: float,
) -> dict[str, Any]:
    from imas_codex.standard_names.review.pipeline import _build_review_record

    score = None if panel_result is None else panel_result.get("winning_score")
    route = (
        None
        if panel_result is None or score is None
        else panel_route(float(score), min_score=min_score)
    )
    model = str((panel_result or {}).get("canonical_model") or _PANEL_MODEL)
    comments: dict[str, Any] = {"jev_route": jev_route, "panel_route": route}
    if score is not None:
        comments["panel_score"] = float(score)
        comments["panel_resolution_method"] = (panel_result or {}).get(
            "resolution_method"
        )
    if skipped_reason is not None:
        comments["audit_skipped"] = skipped_reason
    cost = float((panel_result or {}).get("total_cost") or 0.0)
    record = _build_review_record(
        dict(item),
        model=model,
        is_canonical=False,
        reviewed_at=datetime.now(UTC).isoformat(),
        judgment=_panel_payload(route, None if score is None else float(score)),
        judgment_model=model,
        comments=json.dumps(comments),
        cost_usd=cost,
    )
    record.update(
        review_axis=axis,
        cycle_index=-1,
        resolution_role=AUDIT_ROLE,
        resolution_method=(
            AUDIT_METHOD if skipped_reason is None else AUDIT_SKIPPED_METHOD
        ),
    )
    return record


async def store_panel_audit(
    item: Mapping[str, Any],
    axis: str,
    *,
    panel_result: Mapping[str, Any] | None,
    jev_route: str,
    skipped_reason: str | None = None,
    min_score: float,
) -> None:
    """Write the panel's verdict beside Jev's as a non-deciding audit row.

    *panel_result* is the incumbent's returned verdict (the same mapping the
    caller would have persisted from). ``None`` records a skipped audit: the
    row still lands, carrying the reason, so a gap in the sample is visible
    rather than absent.
    """
    from imas_codex.standard_names.graph_ops import write_reviews

    record = _audit_record(
        item,
        axis,
        panel_result=panel_result,
        jev_route=jev_route,
        skipped_reason=skipped_reason,
        min_score=min_score,
    )
    if not record:
        return
    await asyncio.to_thread(write_reviews, [record])


@dataclass(frozen=True)
class PanelAuditAxisReport:
    """Per-axis agreement between the panel's audit verdict and Jev's route."""

    axis: str
    sample_size: int = 0
    agreement_count: int = 0
    disagreement_counts: Mapping[str, int] = field(default_factory=dict)
    skipped_count: int = 0

    @property
    def agreement_rate(self) -> float | None:
        return self.agreement_count / self.sample_size if self.sample_size else None

    @property
    def disagreement_counts_total(self) -> int:
        return sum(self.disagreement_counts.values())


def panel_audit_report(
    *,
    since: datetime | str,
    axes: Sequence[str] = ("name", "docs"),
) -> dict[str, PanelAuditAxisReport]:
    """Read-only agreement between the panel's audit verdict and Jev's route.

    Pairs each audit row against the Jev route it was stored beside (recorded
    on the row itself), over the window ``since``. A skipped audit is counted
    in ``skipped_count`` and carries no pair, so a rate near zero shows up as
    an empty sample rather than as perfect agreement. Returns one report per
    axis requested; an axis with no audited items reports a zero sample.
    """
    from imas_codex.graph.client import GraphClient

    since_iso = since.isoformat() if isinstance(since, datetime) else str(since)
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (r:StandardNameReview)
            WHERE r.resolution_role = $role
              AND r.reviewed_at >= datetime($since)
              AND r.review_axis IN $axes
            RETURN r.review_axis AS axis, r.comments AS comments
            """,
            role=AUDIT_ROLE,
            since=since_iso,
            axes=list(axes),
        )

    tallies: dict[str, dict[str, Any]] = {
        axis: {"sample": 0, "agree": 0, "directions": {}, "skipped": 0} for axis in axes
    }
    for row in rows or []:
        axis = row.get("axis")
        bucket = tallies.get(axis)
        try:
            comments = json.loads(row.get("comments") or "{}")
        except ValueError:
            logger.warning("unreadable audit row comments for axis %r", axis)
            continue
        if bucket is None:
            continue
        if comments.get("audit_skipped"):
            bucket["skipped"] += 1
            continue
        panel = comments.get("panel_route")
        jev = comments.get("jev_route")
        if panel is None or jev is None:
            continue
        bucket["sample"] += 1
        if panel == jev:
            bucket["agree"] += 1
        else:
            key = f"jev_{jev}_panel_{panel}"
            bucket["directions"][key] = bucket["directions"].get(key, 0) + 1

    return {
        axis: PanelAuditAxisReport(
            axis=axis,
            sample_size=bucket["sample"],
            agreement_count=bucket["agree"],
            disagreement_counts=bucket["directions"],
            skipped_count=bucket["skipped"],
        )
        for axis, bucket in tallies.items()
    }
