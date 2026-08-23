"""Parse Standard Name references from documentation text."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Literal

ReferenceSyntax = Literal["bare_bracket", "inline_name"]
KnownNameResolver = Callable[[tuple[str, ...]], Collection[str]]

_MATH_RE = re.compile(
    r"\$\$.+?\$\$|(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)|"
    r"\\\[.+?\\\]|\\\(.+?\\\)",
    re.DOTALL,
)
_CODE_RE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)
_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)]+\)")
_BARE_BRACKET_RE = re.compile(r"(?<![!\\A-Za-z0-9_])\[([a-z][a-z0-9_]+)\](?!\()")
_INLINE_NAME_RE = re.compile(r"(?<![A-Za-z0-9_])name:([a-z][a-z0-9_]+)\b")


@dataclass(frozen=True)
class NameReference:
    """One reference occurrence and the syntax that expressed it."""

    name: str
    start: int
    end: int
    syntax: ReferenceSyntax


def _masked_spans(text: str) -> list[tuple[int, int]]:
    """Return spans whose bracket content is not documentation prose."""

    spans = [match.span() for match in _MATH_RE.finditer(text)]
    spans.extend(match.span() for match in _CODE_RE.finditer(text))
    spans.extend(match.span() for match in _MARKDOWN_LINK_RE.finditer(text))
    return sorted(spans)


def _inside_any_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(
        span_start <= start and end <= span_end for span_start, span_end in spans
    )


def find_name_references(
    text: str,
    *,
    known: Collection[str] | KnownNameResolver | None = None,
) -> tuple[NameReference, ...]:
    """Return explicit references and bare-bracket reference candidates.

    Mathematical notation, code, images, and the label portion of an already
    formed Markdown link are not bare references. Explicit ``name:identity``
    references remain references wherever they occur, including as Markdown
    link targets.

    When *known* is a collection or resolver, bare candidates are retained only
    when they name a known identity. A resolver is called once with the unique
    candidate names; its exceptions propagate so an unavailable authority
    cannot be mistaken for proof that a candidate is unresolved. With
    ``known=None``, candidates are returned without resolution.
    """

    masked = _masked_spans(text)
    references = [
        NameReference(match.group(1), *match.span(), "inline_name")
        for match in _INLINE_NAME_RE.finditer(text)
    ]
    bare_candidates = [
        NameReference(match.group(1), *match.span(), "bare_bracket")
        for match in _BARE_BRACKET_RE.finditer(text)
        if not _inside_any_span(*match.span(), masked)
    ]

    if known is not None and bare_candidates:
        candidate_names = tuple(dict.fromkeys(ref.name for ref in bare_candidates))
        resolved_names = set(known(candidate_names) if callable(known) else known)
        bare_candidates = [
            reference
            for reference in bare_candidates
            if reference.name in resolved_names
        ]

    references.extend(bare_candidates)
    return tuple(sorted(references, key=lambda reference: reference.start))
