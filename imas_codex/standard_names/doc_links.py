"""Parse Standard Name references from documentation text."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Literal

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
_NAME_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(name:([a-z][a-z0-9_]+)\)")
_BRACKET_CONTENT_RE = re.compile(r"\[([^\[\]\n]+)\]")
_NUMERIC_INTERVAL_RE = re.compile(r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?")
_UNIT_FACTOR_RE = re.compile(r"([A-Za-zµμ]+)(?:\^-?\d+)?")
_UNIT_SYMBOLS = frozenset(
    {
        "m",
        "s",
        "g",
        "kg",
        "A",
        "K",
        "mol",
        "cd",
        "Hz",
        "N",
        "Pa",
        "J",
        "W",
        "C",
        "V",
        "F",
        "T",
        "Wb",
        "H",
        "eV",
        "rad",
        "sr",
        "Bq",
        "Gy",
        "Sv",
        "L",
    }
)
_UNIT_PREFIXES = frozenset("YZEPTGMkhdcmunpf") | {"µ", "μ"}


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


def _is_prose_bracket(content: str) -> bool:
    """Return whether a bracket's content is documentation prose, not notation.

    A bracket holding a unit (``m^-3``, ``K``, ``m^2 s^-1``) or a numeric
    interval (``0, 1``) carries notation a reader reads directly, so it names
    nothing and makes no claim about a standard name. Anything else that holds
    words is prose: it is either a named identity the gate's reference rule
    already governs, or a phrase that should have been written as a link.
    """

    stripped = content.strip()
    if _NUMERIC_INTERVAL_RE.fullmatch(stripped):
        return False
    if not re.search(r"[A-Za-z]", stripped):
        return False
    tokens = stripped.split()
    if not tokens:
        return False
    factors = [factor for token in tokens for factor in re.split(r"[*/]", token)]
    if factors and all(
        (match := _UNIT_FACTOR_RE.fullmatch(factor))
        and (
            match.group(1) in _UNIT_SYMBOLS
            or (
                len(match.group(1)) > 1
                and match.group(1)[0] in _UNIT_PREFIXES
                and match.group(1)[1:] in _UNIT_SYMBOLS
            )
        )
        for factor in factors
    ):
        return False
    return True


def find_prose_bracket_spans(text: str) -> tuple[tuple[str, int, int], ...]:
    """Return prose phrase text and its complete bracket span in *text*.

    A bracketed phrase is prose when it is not inside mathematics or code and
    not the label of an already formed Markdown link — those brackets are
    notation and link text, not documentation prose — and its content is not a
    unit or a numeric interval.
    """

    masked = _masked_spans(text)
    return tuple(
        (match.group(1), *match.span())
        for match in _BRACKET_CONTENT_RE.finditer(text)
        if not _inside_any_span(*match.span(), masked)
        and _is_prose_bracket(match.group(1))
    )


def find_prose_bracket_phrases(text: str) -> tuple[str, ...]:
    """Return the bracketed prose phrases in *text*."""

    return tuple(phrase for phrase, _, _ in find_prose_bracket_spans(text))


@dataclass(frozen=True)
class NameLink:
    """One documentation link whose target names a standard name.

    ``label`` is the visible link text and ``target`` is the identity in the
    link's ``name:`` target, without the ``name:`` prefix.
    """

    label: str
    target: str
    start: int
    end: int


def find_name_links(text: str) -> tuple[NameLink, ...]:
    """Return every Markdown link whose target names a standard name.

    A link to a standard name is written ``[label](name:identity)``; images and
    links to any other target (a URL, an anchor) are not name links. The
    reference parser reads the *targets* of these links as name references, so
    this returns the label-and-target pairs the reference parser deliberately
    discards.
    """

    return tuple(
        NameLink(match.group(1), match.group(2), *match.span())
        for match in _NAME_LINK_RE.finditer(text)
    )


def find_self_links(text: str, documenting_name: str) -> tuple[NameLink, ...]:
    """Return the name links in *text* whose target is *documenting_name* itself.

    A name's documentation that links ``name:`` back to the documenting name
    points at itself where it should point at a neighbour, so this is the
    deterministic half of link-label analysis.
    """

    return tuple(
        link for link in find_name_links(text) if link.target == documenting_name
    )


LINK_LABEL_QUESTION_NAME = "label_describes_target"


def link_label_question() -> dict[str, Any]:
    """The one Noul asked of one documentation link: does the label fit the target.

    The question is shared by every link in a run, so it is built once and the
    per-link values ride in the state (:func:`link_judgement_state`).
    """

    return {
        LINK_LABEL_QUESTION_NAME: {
            "type": "noul",
            "instructions": (
                "The state describes one link inside a standard name's "
                "documentation. `link_text` is the link's visible text, "
                "`documenting_name` is the name whose documentation holds the "
                "link, `target_name` is the standard name the link points to, "
                "and `target_description` is that target's own description. "
                "Does the link text describe the target's description?"
            ),
            "criteria": {
                "true": (
                    "The link text names the quantity or concept the target's "
                    "description describes."
                ),
                "false": (
                    "The link text names a different quantity than the target "
                    "describes, including the documenting name's own quantity."
                ),
            },
        }
    }


def link_judgement_state(
    link: NameLink,
    *,
    documenting_name: str,
    target_description: str,
) -> dict[str, Any]:
    """The judgement context for one link: its text, target, and the target's description."""

    return {
        "documenting_name": documenting_name,
        "link_text": link.label,
        "target_name": link.target,
        "target_description": target_description,
    }
