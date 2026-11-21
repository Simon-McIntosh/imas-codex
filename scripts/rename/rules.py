"""Replacement rules for the project rename.

The old name is held as data here rather than spelled out in prose throughout
the tree: this module is the one place that knows what the old name looks like,
and :mod:`scripts.rename.check` uses the same pattern to search for it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.rename.tree import KeptContext

# One pattern covers the full name in any capitalization and either separator
# (`imas_codex`, `imas-codex`, `IMAS_CODEX`, `IMAS Codex`, `IMAS-Codex`) and the
# server key spelling `imas-cx`, without matching the kept
# `imas_standard_names`, `imas_dd` or `imas-dd` spellings.
OLD_NAME_PATTERN = re.compile(r"imas([-_ ])(codex|cx)", re.IGNORECASE)

# A stored artifact identity - the value of a `schema` key - is data, not a
# product name: signed authorities and receipts carry it verbatim and the
# evidence that holds them keeps its bytes.  The identities the rename must
# protect are read out of the kept paths themselves (see
# scripts.rename.tree.kept_identifiers), so the set cannot go stale and no
# section name or module path is swept in with it.
SCHEMA_VALUE_PATTERN = re.compile(r"""["']schema["']\s*:\s*["']([^"']+)["']""")


def placeholder(index: int) -> str:
    """A stand-in for a protected literal; contains no part of the old name."""
    return f"\x01rename-kept-{index}\x01"


# Basenames that carry the project name and therefore move with the rename.
# `imas_dd` and `imas_standard_names` are deliberately absent: they name the
# IMAS data dictionary and the standard-names grammar, which keep their names.
PATH_TOKENS: tuple[str, ...] = ("imas_codex", "imas-codex")

# Directories never walked or rewritten.  `.git` is version control, the rest
# are local build and environment outputs that are either regenerated or not
# part of the tracked tree.
SKIP_DIRS: frozenset[str] = frozenset(
    {".git", ".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__"}
)


def rename_match(match: re.Match[str]) -> str:
    """Rewrite one match, keeping the user-agent's capitalization and separator.

    A case-sensitive table was tried first and missed `IMAS-Codex/1.0` in a
    user-agent string; preserving the matched case means a new capitalization
    cannot slip past the rename while `imas_standard_names` and `imas-dd`
    stay untouched.
    """
    matched = match.group(0)
    prefix = "ITER" if matched[0].isupper() else "iter"
    return f"{prefix}{match.group(1)}{match.group(2)}"


def rename_text(text: str) -> tuple[str, Counter[str]]:
    """Return the replacement text and the spellings that were changed."""
    spellings: Counter[str] = Counter(
        match.group(0) for match in OLD_NAME_PATTERN.finditer(text)
    )
    return OLD_NAME_PATTERN.sub(rename_match, text), spellings


def count_hits(text: str) -> int:
    return len(OLD_NAME_PATTERN.findall(text))


def _kept_spans(
    text: str, identifiers: list[str], contexts: tuple[KeptContext, ...]
) -> list[tuple[int, int, str, str | None]]:
    """Every kept spelling's span, with the context id that claimed it.

    Identifiers are claimed before contexts so a stored artifact identity
    cannot be counted twice, and the first pattern to claim a span wins.
    """
    spans: list[tuple[int, int, str, str | None]] = []
    for identifier in identifiers:
        start = text.find(identifier)
        while start != -1:
            spans.append((start, start + len(identifier), identifier, None))
            start = text.find(identifier, start + len(identifier))
    for context in contexts:
        for match in re.finditer(context.pattern, text):
            if any(
                match.start() < end and start < match.end()
                for start, end, _, _ in spans
            ):
                continue
            spans.append((match.start(), match.end(), match.group(0), context.id))
    return spans


def count_context_occurrences(
    text: str, identifiers: list[str], contexts: tuple[KeptContext, ...]
) -> dict[str, int]:
    """Old-name occurrences each kept context holds, outside kept identifiers."""
    counts: Counter[str] = Counter()
    for _, _, original, context_id in _kept_spans(text, identifiers, contexts):
        if context_id is not None:
            counts[context_id] += len(OLD_NAME_PATTERN.findall(original))
    return dict(counts)


def protect(
    text: str,
    identifiers: list[str],
    contexts: tuple[KeptContext, ...] = (),
) -> tuple[str, list[tuple[str, str]], dict[str, int]]:
    """Replace every kept spelling with a stand-in before the rename runs."""
    spans = sorted(_kept_spans(text, identifiers, contexts))
    used: list[tuple[str, str]] = []
    for index in range(len(spans) - 1, -1, -1):
        start, end, _, _ = spans[index]
        stand_in = placeholder(index)
        text = text[:start] + stand_in + text[end:]
        used.append((stand_in, spans[index][2]))
    counts: Counter[str] = Counter()
    for _, _, original, context_id in spans:
        if context_id is not None:
            counts[context_id] += len(OLD_NAME_PATTERN.findall(original))
    return text, used, dict(counts)


def restore(text: str, used: list[tuple[str, str]]) -> str:
    for stand_in, identifier in used:
        text = text.replace(stand_in, identifier)
    return text
