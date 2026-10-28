"""Every status exclusion in the repository's Cypher states what null means.

In Cypher, ``x.status <> 'v'`` evaluates to null — not true — when
``x.status`` is null, and a null filter drops the row. A guard written that
way therefore excludes two classes of row it never names: the ones carrying a
different status, and the ones carrying no status at all. The second class is
silent, and the reading it should take is a judgement the query text never
records, so a later reader cannot tell whether the drop was intended.

This module reads every Cypher string under ``imas_codex/`` through the
project's own readers and refuses a ``.status`` comparison against ``<>``
whose null handling is implicit. A comparison is explicit when either

* the property is read inside ``coalesce(...)``, which names the value a null
  status takes; or
* the same ``alias.property`` carries an ``IS NULL`` or ``IS NOT NULL`` test
  in the same statement, which names the null case beside the exclusion.

Statements are read with ``python_cypher_strings``, so a string the repository
does not recognise as Cypher is not scanned, and literals and comments are
blanked before the operator is parsed, so a commented-out copy of an exclusion
is not a finding while a real one beside it still is.
"""

from __future__ import annotations

import re
from pathlib import Path

import imas_codex
from imas_codex.cli.sn import CYPHER_LITERAL_OR_COMMENT
from imas_codex.graph.cypher_property_check import python_cypher_strings

PACKAGE_ROOT = Path(imas_codex.__file__).resolve().parent

_STATUS_SUFFIX = "status"
_COALESCE_RE = re.compile(r"\bcoalesce\s*\(", re.IGNORECASE)
_EXCLUSION_RE = re.compile(
    r"(?P<alias>[A-Za-z_]\w*)\s*\.\s*`?(?P<property>[A-Za-z_]\w*)`?\s*<>\s*\S"
)
_NULL_TEST_TEMPLATE = r"\b{0}\s*\.\s*`?{1}`?\s+IS\s+(?:NOT\s+)?NULL"
_STATUS_REFERENCE_RE = re.compile(r"\.\s*`?[A-Za-z_]\w*status`?", re.IGNORECASE)


def _blank_literals_and_comments(cypher: str) -> str:
    """Blank comments and the contents of quoted runs, keeping quotes.

    The delimiters stay so a value operand still reads as an operand: the
    comparison in ``coalesce(x.status, '') <> 'stale'`` must survive as
    ``coalesce(x.status, '') <> ''``. A comment becomes whitespace, so a
    commented-out exclusion leaves no operator behind.
    """

    def _replace(match: re.Match[str]) -> str:
        span = match.group(0)
        if span.startswith(("/*", "//")):
            return " "
        return span[0] + span[-1] if len(span) >= 2 else " "

    return CYPHER_LITERAL_OR_COMMENT.sub(_replace, cypher)


def _coalesce_spans(cypher: str) -> list[tuple[int, int]]:
    """Return the character spans enclosed by each ``coalesce(...)`` call."""
    spans: list[tuple[int, int]] = []
    for match in _COALESCE_RE.finditer(cypher):
        depth = 0
        opening = match.end() - 1
        for index in range(opening, len(cypher)):
            character = cypher[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    spans.append((opening, index))
                    break
    return spans


def implicit_null_status_exclusions(cypher: str) -> list[tuple[str, str]]:
    """The status exclusions in *cypher* that leave null handling implicit.

    Each finding is ``(alias, property)``. Only a property named ``status`` or
    ending ``_status`` is a status exclusion: a lifecycle, validation or edit
    status drops a row exactly as a plain status does, and the null reading is
    equally the query author's decision.
    """
    executable = _blank_literals_and_comments(cypher)
    coalesced = _coalesce_spans(executable)
    findings: list[tuple[str, str]] = []
    for match in _EXCLUSION_RE.finditer(executable):
        alias = match.group("alias")
        prop = match.group("property")
        if not prop.endswith(_STATUS_SUFFIX):
            continue
        if any(start <= match.start() < end for start, end in coalesced):
            continue
        null_test = re.compile(
            _NULL_TEST_TEMPLATE.format(re.escape(alias), re.escape(prop)),
            re.IGNORECASE,
        )
        if null_test.search(executable):
            continue
        findings.append((alias, prop))
    return findings


def _package_modules() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if not any(
            imas_codex_part.startswith(".")
            for imas_codex_part in path.relative_to(PACKAGE_ROOT).parts
        )
    )


def _scan() -> tuple[list[str], int]:
    """Scan the package, returning findings as ``path:line alias.property``.

    The second figure counts the status property references the scan read,
    which is the positive control: a zero finding count from a scan that read
    no Cypher at all would report safety it never measured.
    """
    findings: list[str] = []
    references = 0
    for path in _package_modules():
        for node, text in python_cypher_strings(path, require_property=False):
            executable = _blank_literals_and_comments(text)
            references += len(_STATUS_REFERENCE_RE.findall(executable))
            for alias, prop in implicit_null_status_exclusions(text):
                relative = path.relative_to(PACKAGE_ROOT.parent)
                findings.append(f"{relative}:{node.lineno} {alias}.{prop}")
    return findings, references


def test_every_status_exclusion_states_its_null_handling() -> None:
    findings, references = _scan()
    assert references > 0, (
        "the scan read no status property reference at all, so its silence "
        "measures nothing; the reader or the package root is wrong"
    )
    assert not findings, (
        "a status exclusion leaves its null handling implicit: wrap the "
        "property in coalesce(...) to name the value a null status takes, or "
        "test the same property IS NULL beside the exclusion. In Cypher the "
        "comparison is null when the property is, so a status-less row drops "
        "out of the filter without the query saying so:\n  " + "\n  ".join(findings)
    )


def test_bare_exclusion_is_refused() -> None:
    assert implicit_null_status_exclusions(
        "MATCH (n:StandardName) WHERE n.status <> 'retired' RETURN n"
    ) == [("n", "status")]


def test_coalesced_exclusion_is_allowed() -> None:
    assert (
        implicit_null_status_exclusions(
            "MATCH (n:StandardName) WHERE coalesce(n.status, '') <> 'retired' RETURN n"
        )
        == []
    )


def test_exclusion_paired_with_null_test_is_allowed() -> None:
    assert (
        implicit_null_status_exclusions(
            "MATCH (n:StandardName) "
            "WHERE n.status IS NULL OR n.status <> 'retired' RETURN n"
        )
        == []
    )
    assert (
        implicit_null_status_exclusions(
            "MATCH (n:StandardName) "
            "WHERE n.status IS NOT NULL AND n.status <> 'retired' RETURN n"
        )
        == []
    )


def test_commented_exclusion_is_not_a_finding() -> None:
    assert (
        implicit_null_status_exclusions(
            "MATCH (n:StandardName) // n.status <> 'retired'\n"
            "/* n.status <> 'draft' */\n"
            "RETURN n"
        )
        == []
    )


def test_parameter_comparison_is_not_a_status_exclusion() -> None:
    assert (
        implicit_null_status_exclusions(
            "MATCH (gap:DDGap) WITH gap WHERE $new_status <> 'resolved' RETURN gap"
        )
        == []
    )
