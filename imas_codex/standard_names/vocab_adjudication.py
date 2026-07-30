"""Typed editorial decisions for stored vocabulary-gap observations."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import VocabGapDisposition


class VocabGapAdjudicationRow(BaseModel):
    """One reviewed action for a normalized vocabulary-gap identity."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    segment: str = Field(min_length=1)
    token: str
    disposition: VocabGapDisposition = Field(alias="decision")
    canonical_target: str | None = None
    reason: str = Field(alias="rationale", min_length=1)
    example_count: int | None = Field(default=None, ge=0)
    last_seen: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> VocabGapAdjudicationRow:
        """Require a target exactly where the editorial action uses one."""
        self.segment = self.segment.strip()
        self.token = self.token.strip()
        self.reason = self.reason.strip()
        if not self.segment or not self.reason:
            raise ValueError("segment and rationale must be nonempty")

        target = (self.canonical_target or "").strip() or None
        if self.disposition in (
            VocabGapDisposition.add,
            VocabGapDisposition.fold,
        ):
            if target is None:
                raise ValueError(
                    f"{self.disposition.value} disposition requires canonical_target"
                )
        elif target is not None:
            raise ValueError("reject disposition must not define canonical_target")
        self.canonical_target = target
        return self

    @property
    def gap_id(self) -> str:
        """Return the normalized graph identity for this decision."""
        return f"vocab_gap:{self.segment}:{self.token}"


class VocabGapAdjudicationFile(BaseModel):
    """Validated editorial batch loaded from an explicit JSON artifact."""

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    policy: str | None = None
    count: int | None = Field(default=None, ge=0)
    summary: dict[str, int] | None = None
    decisions: list[VocabGapAdjudicationRow] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_batch(self) -> VocabGapAdjudicationFile:
        """Pin unique identities and any declared count/summary metadata."""
        identities = [row.gap_id for row in self.decisions]
        duplicate_ids = sorted(
            gap_id for gap_id, n in Counter(identities).items() if n > 1
        )
        if duplicate_ids:
            raise ValueError(
                "exactly one decision is required per gap identity; duplicates: "
                + ", ".join(duplicate_ids)
            )
        if self.count is not None and self.count != len(self.decisions):
            raise ValueError(
                f"declared count {self.count} does not match "
                f"{len(self.decisions)} decisions"
            )
        if self.summary is not None:
            actual = Counter(row.disposition.value for row in self.decisions)
            declared = {str(key): int(value) for key, value in self.summary.items()}
            expected = {key: actual.get(key, 0) for key in ("add", "fold", "reject")}
            if declared != expected:
                raise ValueError(
                    f"declared summary {declared} does not match decisions {expected}"
                )
        return self

    def disposition_counts(self) -> dict[str, int]:
        """Return stable action counts for operator output."""
        counts = Counter(row.disposition.value for row in self.decisions)
        return {key: counts.get(key, 0) for key in ("add", "fold", "reject")}


def load_vocab_gap_adjudications(path: Path) -> VocabGapAdjudicationFile:
    """Load and validate a JSON editorial batch from an explicit path."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read adjudication file {path}: {exc}") from exc
    return VocabGapAdjudicationFile.model_validate(raw)


def _isn_version() -> str:
    """Return the installed grammar package version."""
    import imas_standard_names

    return str(imas_standard_names.__version__)


def _record_dict(record: Any) -> dict[str, Any]:
    """Normalize Neo4j records and test doubles to plain dictionaries."""
    if isinstance(record, dict):
        return record
    data = getattr(record, "data", None)
    if callable(data):
        return data()
    return dict(record)


def _decision_payload(
    batch: VocabGapAdjudicationFile,
    *,
    actor: str,
    grammar_signature: str,
    grammar_version: str,
    applied_at: str,
) -> list[dict[str, Any]]:
    """Project validated rows to graph properties."""
    return [
        {
            "id": row.gap_id,
            "disposition": row.disposition.value,
            "target": row.canonical_target,
            "reason": row.reason,
            "actor": actor,
            "grammar_signature": grammar_signature,
            "grammar_version": grammar_version,
            "applied_at": applied_at,
        }
        for row in batch.decisions
    ]


def _substantive_change(existing: dict[str, Any], desired: dict[str, Any]) -> bool:
    """Ignore audit time when deciding whether an apply is idempotent."""
    return (
        existing.get("editorial_disposition") != desired["disposition"]
        or existing.get("editorial_target") != desired["target"]
        or existing.get("editorial_reason") != desired["reason"]
        or existing.get("editorial_actor") != desired["actor"]
        or existing.get("editorial_grammar_signature") != desired["grammar_signature"]
        or existing.get("editorial_grammar_version") != desired["grammar_version"]
        or existing.get("editorial_active") is not True
    )


def apply_vocab_gap_adjudications(
    batch: VocabGapAdjudicationFile,
    *,
    actor: str,
    dry_run: bool = True,
    grammar_signature: str | None = None,
    grammar_version: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Validate existence and atomically apply a complete editorial batch.

    The transaction reads every requested node before any write. A missing
    identity aborts the entire operation. Reapplying the same substantive
    decisions changes nothing and preserves the first application timestamp.
    """
    actor = actor.strip()
    if not actor:
        raise ValueError("a nonempty editorial actor is required")

    if grammar_signature is None:
        from imas_codex.standard_names.graph_ops import isn_vocabulary_signature

        grammar_signature = isn_vocabulary_signature()
    if not grammar_signature:
        raise ValueError("ISN grammar vocabulary is unavailable")
    grammar_version = grammar_version or _isn_version()
    applied_at = datetime.now(UTC).isoformat()
    payload = _decision_payload(
        batch,
        actor=actor,
        grammar_signature=grammar_signature,
        grammar_version=grammar_version,
        applied_at=applied_at,
    )

    own = gc is None
    client = GraphClient() if own else gc
    try:
        with client.session() as session:
            tx = session.begin_transaction()
            try:
                records = tx.run(
                    """
                    UNWIND $ids AS gap_id
                    OPTIONAL MATCH (vg:VocabGap {id: gap_id})
                    RETURN gap_id AS requested_id, vg.id AS id,
                           vg.segment AS segment, vg.token AS token,
                           vg.editorial_disposition AS editorial_disposition,
                           vg.editorial_target AS editorial_target,
                           vg.editorial_reason AS editorial_reason,
                           vg.editorial_actor AS editorial_actor,
                           vg.editorial_grammar_signature
                               AS editorial_grammar_signature,
                           vg.editorial_grammar_version
                               AS editorial_grammar_version,
                           vg.editorial_active AS editorial_active
                    ORDER BY requested_id
                    """,
                    ids=[item["id"] for item in payload],
                )
                existing_rows = [_record_dict(record) for record in records]
                missing = sorted(
                    row["requested_id"]
                    for row in existing_rows
                    if row.get("id") is None
                )
                if missing:
                    raise ValueError(
                        "adjudication references missing VocabGap nodes: "
                        + ", ".join(missing)
                    )

                existing_by_id = {row["id"]: row for row in existing_rows}
                changed = [
                    item
                    for item in payload
                    if _substantive_change(existing_by_id[item["id"]], item)
                ]
                if dry_run:
                    tx.close()
                else:
                    if changed:
                        tx.run(
                            """
                            UNWIND $items AS item
                            MATCH (vg:VocabGap {id: item.id})
                            SET vg.editorial_disposition = item.disposition,
                                vg.editorial_target = item.target,
                                vg.editorial_reason = item.reason,
                                vg.editorial_actor = item.actor,
                                vg.editorial_at = datetime(item.applied_at),
                                vg.editorial_grammar_signature =
                                    item.grammar_signature,
                                vg.editorial_grammar_version =
                                    item.grammar_version,
                                vg.editorial_active = true
                            """,
                            items=changed,
                        )
                    tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    finally:
        if own:
            client.close()

    return {
        "rows": len(payload),
        "changed": len(changed),
        "unchanged": len(payload) - len(changed),
        "counts": batch.disposition_counts(),
        "grammar_signature": grammar_signature,
        "grammar_version": grammar_version,
        "dry_run": dry_run,
    }


def reset_vocab_gap_adjudications(
    grammar_signature: str,
    *,
    actor: str,
    reason: str,
    dry_run: bool = True,
    current_grammar_signature: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Explicitly deactivate decisions made against one grammar signature."""
    grammar_signature = grammar_signature.strip()
    actor = actor.strip()
    reason = reason.strip()
    if not grammar_signature or not actor or not reason:
        raise ValueError("signature, actor, and reset reason must be nonempty")
    if current_grammar_signature is None:
        from imas_codex.standard_names.graph_ops import isn_vocabulary_signature

        current_grammar_signature = isn_vocabulary_signature()
    if not current_grammar_signature:
        raise ValueError("ISN grammar vocabulary is unavailable")

    own = gc is None
    client = GraphClient() if own else gc
    try:
        rows = client.query(
            """
            MATCH (vg:VocabGap {
                editorial_grammar_signature: $grammar_signature,
                editorial_active: true
            })
            RETURN count(vg) AS eligible
            """,
            grammar_signature=grammar_signature,
        )
        eligible = rows[0]["eligible"] if rows else 0
        if not dry_run and eligible:
            client.query(
                """
                MATCH (vg:VocabGap {
                    editorial_grammar_signature: $grammar_signature,
                    editorial_active: true
                })
                SET vg.editorial_active = false,
                    vg.editorial_invalidated_at = datetime(),
                    vg.editorial_invalidation_actor = $actor,
                    vg.editorial_invalidation_reason = $reason,
                    vg.editorial_invalidation_grammar_signature =
                        $current_grammar_signature
                RETURN count(vg) AS reset
                """,
                grammar_signature=grammar_signature,
                actor=actor,
                reason=reason,
                current_grammar_signature=current_grammar_signature,
            )
    finally:
        if own:
            client.close()

    return {
        "eligible": eligible,
        "reset": 0 if dry_run else eligible,
        "grammar_signature": grammar_signature,
        "current_grammar_signature": current_grammar_signature,
        "dry_run": dry_run,
    }


def editorial_retry_guidance(
    rows: list[dict[str, Any]],
) -> tuple[str, bool]:
    """Render exact editorial guidance and whether it warrants recomposition."""
    lines: list[str] = []
    should_retry = False
    for row in sorted(rows, key=lambda item: (item["segment"], item["token"])):
        disposition = VocabGapDisposition(row["disposition"])
        token = row["token"]
        segment = row["segment"]
        reason = row["reason"]
        if disposition is VocabGapDisposition.fold:
            should_retry = True
            lines.append(
                f"Segment `{segment}` token `{token}` was reviewed as FOLD. "
                f"Re-compose with canonical `{row['target']}`. {reason}"
            )
        elif disposition is VocabGapDisposition.reject:
            should_retry = True
            lines.append(
                f"Segment `{segment}` token `{token}` was reviewed as REJECT. "
                "Do not request or add this token; express the quantity with "
                f"registered grammar or retain the detail as metadata. {reason}"
            )
        else:
            lines.append(
                f"Segment `{segment}` token `{token}` was reviewed as ADD "
                f"(`{row['target']}`), but remains unavailable until the "
                "installed grammar registers it. Preserve an honest vocabulary "
                f"gap until then. {reason}"
            )
    if not lines:
        return "", False
    return "Editorial vocabulary guidance:\n" + "\n".join(lines), should_retry
