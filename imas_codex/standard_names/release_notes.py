"""Grounded PR descriptions for catalog review-batch releases.

The review PR is read by human physics experts, so its description must be a
concise, factual summary of what the batch actually changes — never boilerplate
and never invention. :func:`build_pr_notes` synthesizes the title/body from
three evidence sources:

1. the maintainer's release message (``-m``),
2. the frozen batch artifact (RC version, size, provenance),
3. the real catalog diff (per-domain added/changed/removed entry names,
   computed from git against the base branch).

The LLM is a summarizer over supplied evidence only; on any failure the static
fallback body is used so a release never blocks on the notes model.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from imas_codex.graph.models import DDGapStatus

logger = logging.getLogger(__name__)


class PrNotes(BaseModel):
    """Structured PR description returned by the notes model."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1,
        max_length=70,
        description="Required short human title naming the review batch in words",
    )
    body: str = Field(
        min_length=1,
        description="A few grounded prose sentences with no entry enumeration",
    )

    @field_validator("title", "body")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ApprovalNotes(BaseModel):
    """Grounded human summary of what review did to a batch, for the fold-back tag."""

    summary: str = Field(
        description=(
            "A concise, factual account of what the review changed in the batch, "
            "grounded strictly on the supplied evidence (PR description, "
            "conversation, commit messages, review-delta diff). GitHub-flavoured "
            "markdown; a few short sentences to a short paragraph. Never invent."
        )
    )


def build_approval_notes(
    *,
    pr_description: str,
    conversation: list[dict[str, Any]],
    commit_messages: list[str],
    review_delta: str,
) -> str:
    """Synthesize the grounded human summary for the fold-back tag message.

    Reuses the ``sn-release-notes`` seat over four evidence sources recovered
    from the merged PR: its description, the full conversation (review comments
    + threads), the commit messages, and the review-delta diff (what reviewers
    actually changed). Grounded summarisation only — never invention.

    Never raises: on any synthesis failure this logs and returns ``""`` so the
    fold-back tag is written with its deterministic contract block alone (a
    notes failure must never block or delay the fold-back).
    """
    try:
        from imas_codex.discovery.base.llm import call_llm_structured
        from imas_codex.llm.prompt_loader import render_prompt
        from imas_codex.settings import get_model

        system = render_prompt("sn/approval_notes_system", {})
        user = render_prompt(
            "sn/approval_notes_user",
            {
                "pr_description": pr_description,
                "conversation": conversation or [],
                "commit_messages": commit_messages or [],
                "review_delta": review_delta,
            },
        )
        notes, _cost, _tokens = call_llm_structured(
            model=get_model("sn-release-notes"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=ApprovalNotes,
            service="standard-names",
        )
        return notes.summary.strip()
    except Exception:
        logger.warning(
            "approval-notes synthesis failed — the fold-back tag keeps its "
            "deterministic block alone",
            exc_info=True,
        )
        return ""


def collect_catalog_changes(
    isnc_path: str | Path, base_ref: str = "main"
) -> list[dict[str, Any]]:
    """Summarise per-domain catalog changes between *base_ref* and the worktree.

    Returns ``[{domain, added: [name], changed: [name], removed: [name]}, …]``
    sorted by domain. Entry-level matching is by ``name`` (the graph id) using
    the same YAML reader as :mod:`promote`, so the evidence the notes model sees
    is exactly what ``sn approve`` will later act on.
    """
    from imas_codex.standard_names.promote import _git, _parse_entries

    isnc = Path(isnc_path)
    listing = _git(["diff", "--name-only", base_ref, "--", "standard_names"], isnc)
    out: list[dict[str, Any]] = []
    if not listing:
        return out
    files = sorted(
        line.strip()
        for line in listing.splitlines()
        if line.strip().endswith((".yml", ".yaml"))
    )
    for rel in files:
        base_entries = _parse_entries(_git(["show", f"{base_ref}:{rel}"], isnc))
        head_path = isnc / rel
        head_entries = _parse_entries(
            head_path.read_text() if head_path.exists() else None
        )
        added = sorted(n for n in head_entries if n not in base_entries)
        removed = sorted(n for n in base_entries if n not in head_entries)
        changed = sorted(
            n
            for n, entry in head_entries.items()
            if n in base_entries and entry != base_entries[n]
        )
        if added or removed or changed:
            out.append(
                {
                    "domain": Path(rel).stem,
                    "added": added,
                    "changed": changed,
                    "removed": removed,
                }
            )
    return out


def summarize_dd_gap_facts(facts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Build deterministic, warning-only release evidence from DD-gap facts.

    The input is the canonical read-only lifecycle projection supplied by
    :mod:`imas_codex.standard_names.dd_gaps`. This function deliberately has no
    graph access and no mutation path: a release report may expose authoritative
    DD defects, but it cannot turn observational evidence into an export gate.
    """
    from imas_codex.standard_names.dd_resolutions import (
        load_dd_resolution_manifest,
    )

    resolution_authority = load_dd_resolution_manifest()
    open_statuses = frozenset({DDGapStatus.flagged.value})
    triaged_statuses = frozenset(
        {
            DDGapStatus.triaged.value,
            DDGapStatus.registered_exception.value,
            DDGapStatus.upstream_issue.value,
        }
    )
    retired_statuses = frozenset(
        {DDGapStatus.rejected.value, DDGapStatus.resolved_upstream.value}
    )

    normalized: list[dict[str, Any]] = []
    for fact in facts:
        status = str(fact.get("status") or "")
        exact_paths = sorted(
            {
                str(path)
                for path in fact.get("source_paths", []) or []
                if str(path).strip()
            }
        )
        normalized.append(
            {
                "id": str(fact.get("id") or ""),
                "path": str(fact.get("path") or ""),
                "kind": str(fact.get("kind") or ""),
                "status": status,
                "exact_paths": exact_paths,
                "upstream_url": str(fact.get("upstream_url") or ""),
                "registry_backend": str(fact.get("registry_backend") or ""),
                "resolved_dd_version": str(fact.get("resolved_dd_version") or ""),
            }
        )
    normalized.sort(key=lambda item: (item["status"], item["kind"], item["id"]))

    by_status = Counter(item["status"] for item in normalized)
    by_kind = Counter(item["kind"] for item in normalized)
    unresolved = [
        item
        for item in normalized
        if item["status"] in open_statuses | triaged_statuses
    ]
    retired = [item for item in normalized if item["status"] in retired_statuses]
    stale_registry = [
        item
        for item in retired
        if item["status"] == DDGapStatus.resolved_upstream.value
        and item["registry_backend"]
    ]
    return {
        "available": True,
        "read_error": "",
        "total": len(normalized),
        "open_count": sum(by_status[status] for status in open_statuses),
        "triaged_count": sum(by_status[status] for status in triaged_statuses),
        "retired_count": len(retired),
        "unresolved_count": len(unresolved),
        "stale_registry_count": len(stale_registry),
        "by_status": dict(sorted(by_status.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "facts": normalized,
        "unresolved_facts": unresolved,
        "retired_facts": retired,
        "stale_registry_facts": stale_registry,
        "warning_only": True,
        "blocks_release": False,
        "dd_resolution_manifest_digest": resolution_authority.digest,
        "dd_resolution_record_count": len(resolution_authority.resolutions),
        "dd_resolution_bridges": [
            {
                "id": record.id,
                "path": record.path,
                "field": record.field.value,
                "published": record.observed.model_dump(mode="json"),
                "effective": record.effective.model_dump(mode="json"),
                "upstream_reference": record.upstream_reference,
                "retiring_release": record.retiring_release,
            }
            for record in sorted(
                resolution_authority.resolutions, key=lambda item: item.id
            )
        ],
    }


def unavailable_dd_gap_summary(error: str) -> dict[str, Any]:
    """Represent a read failure visibly without converting it into a gate."""
    summary = summarize_dd_gap_facts([])
    summary["available"] = False
    summary["read_error"] = error.strip() or "unknown read failure"
    return summary


def _dd_gap_sentence(summary: Mapping[str, Any]) -> str:
    """Render DD lifecycle evidence as one non-enumerating sentence."""
    if not bool(summary.get("available", True)):
        return (
            " Linked Data Dictionary caveat evidence could not be read, so this "
            "batch makes no claim that the caveat count is zero."
        )
    if not int(summary.get("total", 0) or 0):
        return ""
    return (
        " Linked Data Dictionary evidence reports "
        f"{summary.get('unresolved_count', 0)} unresolved and "
        f"{summary.get('retired_count', 0)} retired caveats; these observations "
        "are warning-only."
    )


def _change_counts(changes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        kind: sum(len(change.get(kind, []) or []) for change in changes)
        for kind in ("added", "changed", "removed")
    }


def _count_phrase(count: int, noun: str) -> str:
    return f"{count} {noun if count == 1 else noun + 's'}"


def _dominant_domain(changes: list[dict[str, Any]]) -> str:
    ranked = sorted(
        (
            (
                sum(
                    len(change.get(kind, []) or [])
                    for kind in ("added", "changed", "removed")
                ),
                str(change.get("domain") or ""),
            )
            for change in changes
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked or ranked[0][0] <= 0:
        return ""
    return ranked[0][1].replace("_", " ").replace("-", " ").strip()


def _facility_label(rc_version: str) -> str:
    local = rc_version.partition("+")[2]
    token = re.split(r"[-_.]", local, maxsplit=1)[0] if local else ""
    return token.upper() if token.isalpha() else ""


def review_pr_title(
    *, rc_version: str, changes: list[dict[str, Any]] | None = None
) -> str:
    """Derive a short human title from the batch label and dominant domain."""
    facility = _facility_label(rc_version)
    domain = _dominant_domain(changes or [])
    if facility and domain:
        return f"{facility} {domain} review batch"
    if facility:
        return f"{facility} standard names review batch"
    if domain:
        return f"{domain.capitalize()} standard names review batch"
    return "Standard names review batch"


_OUTPUT_ENUMERATION_RE = re.compile(r"(?m)^\s*(?:[-*+]\s|\d+[.)]\s|#+\s|\|)")
_VERSION_RE = re.compile(r"(?i)\bv?\d+\.\d+(?:\.\d+)?(?:rc\d+)?\b")


def _validate_pr_notes(
    notes: Any, *, required_title: str, allowed_numbers: set[int]
) -> tuple[str, str]:
    """Reject malformed model output before it can reach PR publication."""
    title = str(getattr(notes, "title", "") or "").strip()
    body = str(getattr(notes, "body", "") or "").strip()
    if not title:
        raise ValueError("release-notes response omitted the required title")
    if title != required_title:
        raise ValueError("release-notes response did not use the required batch title")
    if "\n" in title or _VERSION_RE.search(title):
        raise ValueError("release-notes title must name the batch in words")
    if not body or "\n" in body or _OUTPUT_ENUMERATION_RE.search(body):
        raise ValueError("release-notes body must be one paragraph of prose")
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", body))
    if sentence_count < 2 or sentence_count > 5:
        raise ValueError("release-notes body must contain two to five sentences")
    reported_numbers = {int(value) for value in re.findall(r"\b\d+\b", body)}
    if not reported_numbers <= allowed_numbers:
        raise ValueError("release-notes body introduced an unsupported count")
    return title, body


def _static_pr_body(
    *,
    rc_version: str,
    batch_size: int,
    minted_from: str,
    unmatched_count: int,
    changes: list[dict[str, Any]],
    dd_gaps: Mapping[str, Any],
) -> str:
    facility = _facility_label(rc_version)
    domain = _dominant_domain(changes)
    scope = " ".join(part for part in (facility, domain) if part) or "standard names"
    counts = _change_counts(changes)
    unmatched = (
        f", with {unmatched_count} source paths lacking a linked name"
        if unmatched_count
        else ""
    )
    return (
        f"This {scope} review batch contains {batch_size} standard names assembled "
        f"from {Path(minted_from).name}. "
        f"The catalog diff contains {_count_phrase(counts['added'], 'addition')}, "
        f"{_count_phrase(counts['changed'], 'change')}, and "
        f"{_count_phrase(counts['removed'], 'removal')}{unmatched}. "
        "Review the fixed batch view and check each entry's wording, units, and "
        "physics meaning before approving." + _dd_gap_sentence(dd_gaps)
    )


def static_pr_notes(
    *,
    message: str,
    rc_version: str,
    batch_size: int,
    minted_from: str,
    unmatched_count: int = 0,
    changes: list[dict[str, Any]] | None = None,
    dd_gaps: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """The deterministic fallback title/body (no LLM)."""
    del message
    changes = changes or []
    summary = dd_gaps or summarize_dd_gap_facts([])
    title = review_pr_title(rc_version=rc_version, changes=changes)
    body = _static_pr_body(
        rc_version=rc_version,
        batch_size=batch_size,
        minted_from=minted_from,
        unmatched_count=unmatched_count,
        changes=changes,
        dd_gaps=summary,
    )
    return title, body


def build_pr_notes(
    *,
    message: str,
    rc_version: str,
    batch_size: int,
    minted_from: str,
    unmatched_count: int = 0,
    changes: list[dict[str, Any]] | None = None,
    dd_gaps: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Synthesize a grounded PR title/body; fall back to the static form.

    Never raises — a notes-model failure logs and returns
    :func:`static_pr_notes` so the release proceeds.
    """
    try:
        from imas_codex.discovery.base.llm import call_llm_structured
        from imas_codex.llm.prompt_loader import render_prompt
        from imas_codex.settings import get_model

        changes = changes or []
        summary = dd_gaps or summarize_dd_gap_facts([])
        title = review_pr_title(rc_version=rc_version, changes=changes)
        counts = _change_counts(changes)
        system = render_prompt("sn/release_notes_system", {})
        user = render_prompt(
            "sn/release_notes_user",
            {
                "required_title": title,
                "message": message,
                "rc_version": rc_version,
                "batch_size": batch_size,
                "minted_from": minted_from,
                "unmatched_count": unmatched_count,
                "facility": _facility_label(rc_version),
                "dominant_domain": _dominant_domain(changes),
                "change_counts": counts,
                "dd_gaps": summary,
            },
        )
        notes, _cost, _tokens = call_llm_structured(
            model=get_model("sn-release-notes"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=PrNotes,
            service="standard-names",
        )
        allowed_numbers = {
            batch_size,
            unmatched_count,
            *counts.values(),
            int(summary.get("unresolved_count", 0) or 0),
            int(summary.get("retired_count", 0) or 0),
        }
        return _validate_pr_notes(
            notes, required_title=title, allowed_numbers=allowed_numbers
        )
    except Exception:
        logger.warning(
            "release-notes synthesis failed — using the static PR body",
            exc_info=True,
        )
        return static_pr_notes(
            message=message,
            rc_version=rc_version,
            batch_size=batch_size,
            minted_from=minted_from,
            unmatched_count=unmatched_count,
            changes=changes,
            dd_gaps=dd_gaps,
        )
