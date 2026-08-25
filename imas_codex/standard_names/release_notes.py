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

from pydantic import BaseModel, Field

from imas_codex.graph.models import DDGapStatus

logger = logging.getLogger(__name__)


class PrNotes(BaseModel):
    """Structured PR description returned by the notes model."""

    title: str = Field(description="PR title, <= 70 characters")
    body: str = Field(description="PR body, GitHub-flavoured markdown")


class MergeNotes(BaseModel):
    """Grounded human summary of what review did to a batch, for the fold-back tag."""

    summary: str = Field(
        description=(
            "A concise, factual account of what the review changed in the batch, "
            "grounded strictly on the supplied evidence (PR description, "
            "conversation, commit messages, review-delta diff). GitHub-flavoured "
            "markdown; a few short sentences to a short paragraph. Never invent."
        )
    )


def build_merge_notes(
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

        system = render_prompt("sn/merge_notes_system", {})
        user = render_prompt(
            "sn/merge_notes_user",
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
            response_model=MergeNotes,
            service="standard-names",
        )
        return notes.summary.strip()
    except Exception:
        logger.warning(
            "merge-notes synthesis failed — the fold-back tag keeps its "
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
    the same YAML reader as :mod:`merge`, so the evidence the notes model sees
    is exactly what ``sn merge`` will later act on.
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


def _static_dd_gap_caveat(summary: Mapping[str, Any]) -> str:
    """Render exact lifecycle evidence for the deterministic PR fallback."""
    if not bool(summary.get("available", True)):
        return (
            "\n\n## Data Dictionary caveats\n\n"
            "Warning only: linked DD-defect evidence could not be read "
            f"({summary.get('read_error', 'unknown read failure')}). The release "
            "continues, but this is not evidence that the batch has zero DD "
            "caveats."
        )
    total = int(summary.get("total", 0) or 0)
    if not total:
        return "\n\n## Data Dictionary caveats\n\nNo linked DD defects were reported."

    lines = [
        "\n\n## Data Dictionary caveats",
        "",
        (
            f"Warning only: {summary.get('unresolved_count', 0)} unresolved, "
            f"{summary.get('triaged_count', 0)} triaged, "
            f"{summary.get('retired_count', 0)} retired, and "
            f"{summary.get('stale_registry_count', 0)} stale registry fact(s). "
            "These observations do not suppress sources or block this release."
        ),
    ]
    for fact in summary.get("facts", []) or []:
        exact_paths = (
            ", ".join(f"`{path}`" for path in fact.get("exact_paths", []))
            or "(no linked exact path)"
        )
        upstream = fact.get("upstream_url")
        suffix = f" — [upstream]({upstream})" if upstream else ""
        lines.append(
            f"- `{fact.get('kind')}` / `{fact.get('status')}`: {exact_paths}{suffix}"
        )
    return "\n".join(lines)


_MODEL_DD_GAP_SECTION_RE = re.compile(
    r"(?ims)^##[ \t]+Data[ \t]+Dictionary[ \t]+caveats[ \t]*$"
    r".*?(?=^##[ \t]+|\Z)"
)


def _with_canonical_dd_gap_caveat(model_body: str, summary: Mapping[str, Any]) -> str:
    """Replace model-authored DD caveats with the deterministic rendering."""
    without_model_caveats = _MODEL_DD_GAP_SECTION_RE.sub("", model_body).strip()
    return without_model_caveats + _static_dd_gap_caveat(summary)


def static_pr_notes(
    *,
    message: str,
    rc_version: str,
    batch_size: int,
    minted_from: str,
    dd_gaps: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """The deterministic fallback title/body (no LLM)."""
    title = message or f"Standard-name review batch {rc_version}"
    body = (
        f"Review batch **{rc_version}** — {batch_size} standard name(s) for "
        f"first human review.\n\nMinted from `{minted_from}`."
    )
    body += _static_dd_gap_caveat(dd_gaps or summarize_dd_gap_facts([]))
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

        system = render_prompt("sn/release_notes_system", {})
        user = render_prompt(
            "sn/release_notes_user",
            {
                "message": message,
                "rc_version": rc_version,
                "batch_size": batch_size,
                "minted_from": minted_from,
                "unmatched_count": unmatched_count,
                "domains": changes or [],
                "dd_gaps": dd_gaps or summarize_dd_gap_facts([]),
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
        title = notes.title.strip() or message or rc_version
        summary = dd_gaps or summarize_dd_gap_facts([])
        return title, _with_canonical_dd_gap_caveat(notes.body, summary)
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
            dd_gaps=dd_gaps,
        )
