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
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PrNotes(BaseModel):
    """Structured PR description returned by the notes model."""

    title: str = Field(description="PR title, <= 70 characters")
    body: str = Field(description="PR body, GitHub-flavoured markdown")


def collect_catalog_changes(
    isnc_path: str | Path, base_ref: str = "main"
) -> list[dict[str, Any]]:
    """Summarise per-domain catalog changes between *base_ref* and the worktree.

    Returns ``[{domain, added: [name], changed: [name], removed: [name]}, …]``
    sorted by domain. Entry-level matching is by ``name`` (the graph id) using
    the same YAML reader as :mod:`merge`, so the evidence the notes model sees
    is exactly what ``sn merge`` will later act on.
    """
    from imas_codex.standard_names.merge import _git, _parse_entries

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


def static_pr_notes(
    *,
    message: str,
    rc_version: str,
    batch_size: int,
    minted_from: str,
) -> tuple[str, str]:
    """The deterministic fallback title/body (no LLM)."""
    title = message or f"Standard-name review batch {rc_version}"
    body = (
        f"Review batch **{rc_version}** — {batch_size} standard name(s) for "
        f"first human review.\n\nMinted from `{minted_from}`."
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
        return title, notes.body.strip()
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
        )
