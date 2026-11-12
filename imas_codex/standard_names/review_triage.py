"""Route external catalog-review comments through the standard-name pipeline.

Catalog reviewers normally leave a comment rather than editing a YAML entry.
This module reads both GitHub review-comment surfaces, resolves line comments
to their catalog entry, classifies the request conservatively, and reuses the
existing edit and contested mechanisms. It never treats an unknown comment as
approval and never rewrites a reviewer's requested wording.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.edit import EditPlan, apply_edit
from imas_codex.standard_names.promote import (
    contest as _contest,
    parse_pull_request_url,
)


class CommentClass(StrEnum):
    """The mutually exclusive actions a catalog reviewer can request."""

    NAME = "name"
    WORDING = "wording"
    PHYSICS = "physics"
    SCOPE = "scope"
    ACKNOWLEDGEMENT = "acknowledgement"
    UNKNOWN = "unknown"


class CommentDisposition(StrEnum):
    """Where a triaged review comment is sent."""

    PROPOSAL = "proposal"
    CONTESTED = "contested"
    BATCH_ADJUDICATION = "batch_adjudication"
    CLOSED = "closed"
    ADJUDICATION = "adjudication"


@dataclass(frozen=True, slots=True)
class ReviewComment:
    """A normalized line-anchored or request-level GitHub comment."""

    id: int
    body: str
    author: str | None = None
    url: str | None = None
    path: str | None = None
    line: int | None = None
    source: str = "request"

    @property
    def is_line_anchored(self) -> bool:
        return self.path is not None and self.line is not None


@dataclass(frozen=True, slots=True)
class ResolvedComment:
    """A comment plus the identity its line bears on, when one is knowable."""

    comment: ReviewComment
    standard_name_id: str | None
    resolution_error: str | None = None


@dataclass(frozen=True, slots=True)
class TriagedComment:
    """A classified comment and its fail-closed disposition."""

    resolved: ResolvedComment
    classification: CommentClass
    disposition: CommentDisposition
    proposed_value: str | None = None
    reason: str = ""


@dataclass(slots=True)
class TriageReport:
    """The complete comment census and each resulting disposition."""

    comments: list[TriagedComment] = field(default_factory=list)
    routed: list[dict[str, str]] = field(default_factory=list)

    @property
    def disposition_counts(self) -> dict[str, int]:
        return {
            disposition.value: sum(
                item.disposition is disposition for item in self.comments
            )
            for disposition in CommentDisposition
        }


GitHubCall = Callable[[str, str], tuple[int, Any]]

_ACKNOWLEDGEMENT = re.compile(
    r"\b(?:acknowledged|approved|lgtm|looks good|no action needed|thanks?)\b",
    re.IGNORECASE,
)
_SCOPE = re.compile(
    r"\b(?:batch|belongs? in (?:this|the) (?:batch|review)|include|exclude|"
    r"out of scope|should this (?:be )?reviewed)\b",
    re.IGNORECASE,
)
_PHYSICS = re.compile(
    r"\b(?:physics|physical(?:ly)?|quantity|dimension(?:al)?|unit mismatch|"
    r"does not measure|not the same quantity|wrong observable)\b",
    re.IGNORECASE,
)
_NAME_VALUE = re.compile(
    r"\b(?:rename\s+(?:[a-z][a-z0-9_]*\s+)?to|"
    r"(?:name|call)\s+(?:it\s+)?(?:as|to)|"
    r"(?:name|spelling)\s+should\s+be)\s*[`'\"]?"
    r"(?P<value>[a-z][a-z0-9_]*)",
    re.IGNORECASE,
)
_NAME_REQUEST = re.compile(r"\b(?:rename|name|spelling|call it)\b", re.IGNORECASE)
_WORDING_VALUE = re.compile(
    r"\b(?:description|documentation|wording|text)\b.{0,80}?"
    r"(?:should\s+read|replace(?:d)?\s+(?:with|by)|rewrite(?:d)?\s+as)\s*"
    r"(?P<quote>[`'\"])(?P<value>.+?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)
_WORDING_REQUEST = re.compile(
    r"\b(?:description|documentation|wording|clarify|reword|rewrite)\b",
    re.IGNORECASE,
)


def _comment_from_payload(payload: dict[str, Any], *, source: str) -> ReviewComment:
    """Normalize GitHub's two comment payload shapes without losing origin."""
    author = payload.get("user") or {}
    line = payload.get("line") or payload.get("original_line")
    return ReviewComment(
        id=int(payload.get("id") or 0),
        body=str(payload.get("body") or "").strip(),
        author=str(author.get("login")) if author.get("login") else None,
        url=str(payload.get("html_url")) if payload.get("html_url") else None,
        path=str(payload.get("path")) if payload.get("path") else None,
        line=int(line) if line is not None else None,
        source=source,
    )


def ingest_review_comments(
    pull_request_url: str,
    *,
    github_call: GitHubCall | None = None,
) -> list[ReviewComment]:
    """Read every line and request-level comment attached to one PR."""
    repo, number = parse_pull_request_url(pull_request_url)
    if github_call is None:
        from imas_codex.graph.ghcr import github_api_call

        github_call = github_api_call

    comments: list[ReviewComment] = []
    endpoints = (
        (f"/repos/{repo}/pulls/{number}/comments?per_page=100", "line"),
        (f"/repos/{repo}/issues/{number}/comments?per_page=100", "request"),
    )
    for endpoint, source in endpoints:
        status, payload = github_call("GET", endpoint)
        if status != 200 or not isinstance(payload, list):
            raise ValueError(
                f"could not ingest {source} comments for {repo}#{number}: HTTP {status}"
            )
        comments.extend(
            _comment_from_payload(row, source=source)
            for row in payload
            if isinstance(row, dict)
        )
    return comments


def _entry_spans(contents: str, *, source: Path) -> list[tuple[str, int, int]]:
    """Return one-based line spans for a top-level catalog sequence."""
    try:
        document = yaml.compose(contents)
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse catalog file {source}: {exc}") from exc
    if document is None or not isinstance(document, yaml.nodes.SequenceNode):
        raise ValueError(f"catalog file {source} is not a top-level sequence")

    entries: list[tuple[str, int]] = []
    for item in document.value:
        if not isinstance(item, yaml.nodes.MappingNode):
            raise ValueError(f"catalog file {source} has a non-mapping entry")
        name = next(
            (
                str(value.value)
                for key, value in item.value
                if isinstance(key, yaml.nodes.ScalarNode)
                and key.value == "name"
                and isinstance(value, yaml.nodes.ScalarNode)
            ),
            None,
        )
        if not name:
            raise ValueError(f"catalog file {source} has an entry without a name")
        entries.append((name, item.start_mark.line + 1))
    final_line = len(contents.splitlines()) + 1
    return [
        (
            name,
            start,
            entries[index + 1][1] - 1 if index + 1 < len(entries) else final_line,
        )
        for index, (name, start) in enumerate(entries)
    ]


def resolve_comment(
    comment: ReviewComment, *, catalog_root: str | Path
) -> ResolvedComment:
    """Resolve a line annotation to its entry; request comments bear on batch."""
    if not comment.is_line_anchored:
        return ResolvedComment(comment=comment, standard_name_id=None)

    root = Path(catalog_root).resolve()
    candidate = (root / str(comment.path)).resolve()
    if root not in candidate.parents or candidate.suffix not in {".yaml", ".yml"}:
        return ResolvedComment(
            comment=comment,
            standard_name_id=None,
            resolution_error="comment path is outside the catalog YAML tree",
        )
    if not candidate.is_file():
        return ResolvedComment(
            comment=comment,
            standard_name_id=None,
            resolution_error="annotated catalog file is not present",
        )
    try:
        for name, start, end in _entry_spans(candidate.read_text(), source=candidate):
            if start <= int(comment.line or 0) <= end:
                return ResolvedComment(comment=comment, standard_name_id=name)
    except ValueError as exc:
        return ResolvedComment(
            comment=comment, standard_name_id=None, resolution_error=str(exc)
        )
    return ResolvedComment(
        comment=comment,
        standard_name_id=None,
        resolution_error="annotated line does not belong to a catalog entry",
    )


def classify_comment(comment: ReviewComment) -> tuple[CommentClass, str | None]:
    """Classify the request, returning a proposal only when exact text is clear."""
    body = comment.body.strip()
    if _ACKNOWLEDGEMENT.search(body):
        return CommentClass.ACKNOWLEDGEMENT, None
    if _SCOPE.search(body):
        return CommentClass.SCOPE, None
    if _PHYSICS.search(body):
        return CommentClass.PHYSICS, None
    if match := _NAME_VALUE.search(body):
        return CommentClass.NAME, match.group("value").lower()
    if _NAME_REQUEST.search(body):
        return CommentClass.NAME, None
    if match := _WORDING_VALUE.search(body):
        return CommentClass.WORDING, match.group("value").strip()
    if _WORDING_REQUEST.search(body):
        return CommentClass.WORDING, None
    return CommentClass.UNKNOWN, None


def _provenance_reason(comment: ReviewComment, classification: CommentClass) -> str:
    """Keep a reviewer-comment address in the existing edit-reason provenance."""
    reviewer = f" by @{comment.author}" if comment.author else ""
    address = comment.url or f"GitHub comment {comment.id}"
    return (
        f"catalog review comment {comment.id}{reviewer} requested a "
        f"{classification.value} change: {comment.body.strip()} ({address})"
    )


def triage_comments(
    comments: list[ReviewComment], *, catalog_root: str | Path
) -> TriageReport:
    """Resolve and classify every comment without dropping any ambiguous row."""
    report = TriageReport()
    for comment in comments:
        resolved = resolve_comment(comment, catalog_root=catalog_root)
        classification, proposed_value = classify_comment(comment)
        reason = _provenance_reason(comment, classification)
        if classification is CommentClass.ACKNOWLEDGEMENT:
            disposition = CommentDisposition.CLOSED
        elif classification is CommentClass.SCOPE:
            disposition = CommentDisposition.BATCH_ADJUDICATION
        elif classification is CommentClass.PHYSICS and resolved.standard_name_id:
            disposition = CommentDisposition.CONTESTED
        elif (
            classification in {CommentClass.NAME, CommentClass.WORDING}
            and resolved.standard_name_id
            and proposed_value
        ):
            disposition = CommentDisposition.PROPOSAL
        else:
            disposition = CommentDisposition.ADJUDICATION
        report.comments.append(
            TriagedComment(
                resolved=resolved,
                classification=classification,
                disposition=disposition,
                proposed_value=proposed_value,
                reason=reason,
            )
        )
    return report


def route_triage(
    report: TriageReport,
    *,
    gc: GraphClient | None = None,
    threshold: float = 1.0,
) -> TriageReport:
    """Attach proposals and contest physics objections through existing paths."""
    owns_graph = gc is None
    if gc is None:
        gc = GraphClient()
    try:
        for item in report.comments:
            target = item.resolved.standard_name_id
            if item.disposition is CommentDisposition.PROPOSAL:
                assert target is not None
                if item.classification is CommentClass.NAME:
                    plan: EditPlan = apply_edit(
                        target=target,
                        rename=item.proposed_value,
                        reason=item.reason,
                        origin="human",
                        refine=False,
                        gc=gc,
                    )
                else:
                    plan = apply_edit(
                        target=target,
                        docs=item.proposed_value,
                        reason=item.reason,
                        origin="human",
                        refine=False,
                        gc=gc,
                    )
                outcome = "blocked" if plan.blocked else "staged"
                report.routed.append(
                    {
                        "comment_id": str(item.resolved.comment.id),
                        "target": target,
                        "disposition": outcome,
                        "reason": plan.blocked or item.reason,
                    }
                )
            elif item.disposition is CommentDisposition.CONTESTED:
                assert target is not None
                _contest(
                    target,
                    axis="name",
                    score=0.0,
                    threshold=threshold,
                    reason=item.reason,
                    catalog_pr_number=None,
                    catalog_pr_url=item.resolved.comment.url,
                    catalog_merge_commit_sha=None,
                    catalog_reviewer_actor=item.resolved.comment.author,
                    gc=gc,
                )
                report.routed.append(
                    {
                        "comment_id": str(item.resolved.comment.id),
                        "target": target,
                        "disposition": CommentDisposition.CONTESTED.value,
                        "reason": item.reason,
                    }
                )
    finally:
        if owns_graph:
            gc.close()
    return report


def ingest_and_triage(
    pull_request_url: str,
    *,
    catalog_root: str | Path,
    github_call: GitHubCall | None = None,
) -> TriageReport:
    """Fetch the two review surfaces and return the full disposition census."""
    return triage_comments(
        ingest_review_comments(pull_request_url, github_call=github_call),
        catalog_root=catalog_root,
    )
