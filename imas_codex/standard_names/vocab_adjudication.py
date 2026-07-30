"""Typed editorial decisions for stored vocabulary-gap observations."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
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
            "segment": row.segment,
            "token": row.token,
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


def _public_grammar_contract(
    grammar_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, frozenset[str]], frozenset[str], Mapping[str, Any]]:
    """Load exact segment vocabulary and optional aliases from ISN's public API."""
    if grammar_context is None:
        try:
            from imas_standard_names import get_grammar_context

            grammar_context = get_grammar_context()
        except Exception as exc:
            raise ValueError("ISN grammar vocabulary is unavailable") from exc
    if not isinstance(grammar_context, Mapping):
        raise ValueError("ISN grammar context is not a mapping")

    sections = grammar_context.get("vocabulary_sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("ISN grammar context has no vocabulary sections")
    tokens_by_segment: dict[str, frozenset[str]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            raise ValueError("ISN vocabulary section is not a mapping")
        segment = section.get("segment")
        tokens = section.get("tokens")
        if not isinstance(segment, str) or not segment:
            raise ValueError("ISN vocabulary section has no segment")
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise ValueError(f"ISN vocabulary section {segment!r} has invalid tokens")
        tokens_by_segment[segment] = frozenset(tokens)

    grammar = grammar_context.get("grammar")
    if not isinstance(grammar, Mapping):
        raise ValueError("ISN grammar context has no grammar registry")
    aliases: Any = grammar.get("advisory_aliases", {})
    all_tokens = {
        token
        for segment_tokens in tokens_by_segment.values()
        for token in segment_tokens
    }
    vocabularies = grammar.get("vocabularies")
    if not isinstance(vocabularies, Mapping):
        raise ValueError("ISN grammar context has no vocabulary registry")
    for vocabulary in vocabularies.values():
        if isinstance(vocabulary, Mapping):
            all_tokens.update(token for token in vocabulary if isinstance(token, str))
        elif isinstance(vocabulary, list):
            if not all(isinstance(token, str) for token in vocabulary):
                raise ValueError("ISN grammar vocabulary contains invalid tokens")
            all_tokens.update(vocabulary)
    if aliases is None:
        aliases = {}
    if not isinstance(aliases, Mapping):
        raise ValueError("ISN advisory aliases are not a mapping")
    return tokens_by_segment, frozenset(all_tokens), aliases


def _missing_graph_token_rows(
    tx: Any,
    missing_items: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Find any graph observation carrying each missing decision's token."""
    records = tx.run(
        """
        UNWIND $items AS item
        OPTIONAL MATCH (other:VocabGap {token: item.token})
        RETURN item.id AS requested_id,
               collect(other.id) AS graph_token_ids
        ORDER BY requested_id
        """,
        items=[{"id": item["id"], "token": item["token"]} for item in missing_items],
    )
    rows = {
        row["requested_id"]: list(row.get("graph_token_ids") or [])
        for record in records
        if (row := _record_dict(record)).get("requested_id") is not None
    }
    expected_ids = {item["id"] for item in missing_items}
    if set(rows) != expected_ids:
        unresolved = sorted(expected_ids - set(rows))
        unexpected = sorted(set(rows) - expected_ids)
        raise ValueError(
            "graph token audit did not return every missing identity: "
            f"unresolved={unresolved} unexpected={unexpected}"
        )
    return rows


def _missing_resolution(
    item: dict[str, Any],
    *,
    tokens_by_segment: Mapping[str, frozenset[str]],
    all_grammar_tokens: frozenset[str],
    aliases: Mapping[str, Any],
    graph_token_ids: list[str],
) -> str:
    """Classify one absent graph identity only when grammar proves it resolved."""
    segment = item["segment"]
    token = item["token"]
    target = item["target"]
    disposition = VocabGapDisposition(item["disposition"])
    segment_tokens = tokens_by_segment.get(segment, frozenset())

    if disposition is VocabGapDisposition.add:
        if target not in segment_tokens:
            raise ValueError(
                f"{item['id']}: add target is not registered in exact "
                f"segment {segment!r}: {target!r}"
            )
        return "satisfied_by_grammar"

    if disposition is VocabGapDisposition.reject:
        if token in all_grammar_tokens:
            raise ValueError(
                f"{item['id']}: reject token remains registered in grammar"
            )
        if graph_token_ids:
            raise ValueError(
                f"{item['id']}: reject token remains in the graph at "
                + ", ".join(sorted(graph_token_ids))
            )
        return "resolved_reject"

    segment_aliases = aliases.get(segment)
    definition = (
        segment_aliases.get(token) if isinstance(segment_aliases, Mapping) else None
    )
    if not isinstance(definition, Mapping):
        raise ValueError(
            f"{item['id']}: fold has no exact advisory alias in segment {segment!r}"
        )
    alias_target = definition.get("canonical")
    if alias_target != target:
        raise ValueError(
            f"{item['id']}: fold advisory alias targets {alias_target!r}, "
            f"not reviewed target {target!r}"
        )
    if target not in segment_tokens:
        raise ValueError(
            f"{item['id']}: fold target is not registered in exact "
            f"segment {segment!r}: {target!r}"
        )
    return "satisfied_by_grammar"


def _receipt(
    batch: VocabGapAdjudicationFile,
    *,
    payload: list[dict[str, Any]],
    existing_by_id: Mapping[str, dict[str, Any]],
    changed_ids: set[str],
    missing_resolutions: Mapping[str, str],
    actor: str,
    applied_at: str,
    grammar_signature: str,
    grammar_version: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Build one complete machine-readable record for every reviewed row."""
    decisions: list[dict[str, Any]] = []
    for row, item in zip(batch.decisions, payload, strict=True):
        resolution = missing_resolutions.get(item["id"], "applied")
        decisions.append(
            {
                "id": item["id"],
                "segment": row.segment,
                "token": row.token,
                "disposition": row.disposition.value,
                "canonical_target": row.canonical_target,
                "reason": row.reason,
                "resolution": resolution,
                "graph_node_exists": item["id"] in existing_by_id,
                "changed": item["id"] in changed_ids,
            }
        )
    resolution_counts = Counter(item["resolution"] for item in decisions)
    stable_counts = {
        key: resolution_counts.get(key, 0)
        for key in ("applied", "satisfied_by_grammar", "resolved_reject")
    }
    return {
        "format": "imas-codex-vocab-gap-adjudication-receipt",
        "format_version": 1,
        "source": batch.source,
        "actor": actor,
        "recorded_at": applied_at,
        "grammar_signature": grammar_signature,
        "grammar_version": grammar_version,
        "dry_run": dry_run,
        "rows": len(decisions),
        "disposition_counts": batch.disposition_counts(),
        "resolution_counts": stable_counts,
        "decisions": decisions,
    }


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically replace one explicit receipt path with formatted JSON."""
    output = path.expanduser()
    if not output.parent.is_dir():
        raise ValueError(f"receipt directory does not exist: {output.parent}")
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    except OSError as exc:
        raise ValueError(f"cannot write adjudication receipt {output}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def apply_vocab_gap_adjudications(
    batch: VocabGapAdjudicationFile,
    *,
    actor: str,
    dry_run: bool = True,
    grammar_signature: str | None = None,
    grammar_version: str | None = None,
    resolve_missing_from_grammar: bool = False,
    grammar_context: Mapping[str, Any] | None = None,
    receipt_path: Path | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Validate existence and atomically apply a complete editorial batch.

    The transaction reads every requested node before any write. A missing
    identity aborts the entire operation by default. The explicit grammar
    resolution mode accepts only mechanically proven historical resolutions.
    Reapplying the same substantive decisions changes nothing and preserves
    the first application timestamp.
    """
    actor = actor.strip()
    if not actor:
        raise ValueError("a nonempty editorial actor is required")
    if resolve_missing_from_grammar and receipt_path is None:
        raise ValueError(
            "an explicit receipt path is required when resolving missing history"
        )
    if receipt_path is not None and not receipt_path.expanduser().parent.is_dir():
        raise ValueError(
            f"receipt directory does not exist: {receipt_path.expanduser().parent}"
        )

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
    tokens_by_segment: dict[str, frozenset[str]] = {}
    all_grammar_tokens: frozenset[str] = frozenset()
    aliases: Mapping[str, Any] = {}
    if resolve_missing_from_grammar:
        tokens_by_segment, all_grammar_tokens, aliases = _public_grammar_contract(
            grammar_context
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
                expected_ids = {item["id"] for item in payload}
                returned_ids = {
                    row["requested_id"]
                    for row in existing_rows
                    if row.get("requested_id") is not None
                }
                if returned_ids != expected_ids:
                    unresolved = sorted(expected_ids - returned_ids)
                    unexpected = sorted(returned_ids - expected_ids)
                    raise ValueError(
                        "graph existence audit did not return every decision: "
                        f"unresolved={unresolved} unexpected={unexpected}"
                    )
                missing = sorted(
                    row["requested_id"]
                    for row in existing_rows
                    if row.get("id") is None
                )
                if missing and not resolve_missing_from_grammar:
                    raise ValueError(
                        "adjudication references missing VocabGap nodes: "
                        + ", ".join(missing)
                    )

                existing_by_id = {
                    row["id"]: row for row in existing_rows if row.get("id") is not None
                }
                payload_by_id = {item["id"]: item for item in payload}
                missing_resolutions: dict[str, str] = {}
                if missing:
                    missing_items = [payload_by_id[gap_id] for gap_id in missing]
                    graph_tokens_by_id = _missing_graph_token_rows(tx, missing_items)
                    resolution_errors: list[str] = []
                    for item in missing_items:
                        try:
                            missing_resolutions[item["id"]] = _missing_resolution(
                                item,
                                tokens_by_segment=tokens_by_segment,
                                all_grammar_tokens=all_grammar_tokens,
                                aliases=aliases,
                                graph_token_ids=graph_tokens_by_id.get(item["id"], []),
                            )
                        except ValueError as exc:
                            resolution_errors.append(str(exc))
                    if resolution_errors:
                        raise ValueError(
                            "missing adjudications are not resolved by grammar: "
                            + "; ".join(resolution_errors)
                        )

                changed = [
                    item
                    for item in payload
                    if item["id"] in existing_by_id
                    and _substantive_change(existing_by_id[item["id"]], item)
                ]
                changed_ids = {item["id"] for item in changed}
                receipt = _receipt(
                    batch,
                    payload=payload,
                    existing_by_id=existing_by_id,
                    changed_ids=changed_ids,
                    missing_resolutions=missing_resolutions,
                    actor=actor,
                    applied_at=applied_at,
                    grammar_signature=grammar_signature,
                    grammar_version=grammar_version,
                    dry_run=dry_run,
                )
                if dry_run:
                    tx.close()
                else:
                    if changed:
                        write_records = tx.run(
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
                            RETURN count(vg) AS applied
                            """,
                            items=changed,
                        )
                        write_rows = [_record_dict(record) for record in write_records]
                        applied = write_rows[0].get("applied", 0) if write_rows else 0
                        if applied != len(changed):
                            raise ValueError(
                                "transaction did not apply every extant "
                                f"adjudication: expected={len(changed)} "
                                f"applied={applied}"
                            )
                    tx.commit()
            except BaseException:
                if tx.closed() is False:
                    tx.close()
                raise
    finally:
        if own:
            client.close()

    if receipt_path is not None:
        _write_receipt(receipt_path, receipt)

    result = {
        "rows": len(payload),
        "changed": len(changed),
        "unchanged": len(existing_by_id) - len(changed),
        "counts": batch.disposition_counts(),
        "resolution_counts": receipt["resolution_counts"],
        "grammar_signature": grammar_signature,
        "grammar_version": grammar_version,
        "dry_run": dry_run,
        "receipt": receipt,
    }
    return result


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
