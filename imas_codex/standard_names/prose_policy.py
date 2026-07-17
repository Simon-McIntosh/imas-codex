"""Banned-prose policy vocabulary for accepted-name documentation.

Accepted-name documentation must state *what* a quantity is, not narrate
typical magnitudes, estimator recipes, or procedural padding.  This module
owns the grep-audit vocabulary that flags such prose: the compiled patterns
and the counting helper.  It carries no benchmark or campaign logic, so both
the docs-seat benchmark and the production campaign selector depend on the
policy definition rather than on each other.
"""

from __future__ import annotations

import re

# Banned-prose classes for accepted-name documentation: typical values,
# estimator recipes, and procedural padding.  These are heuristic flags for a
# grep-audit, deliberately conservative; each group is reported separately so a
# reviewer can calibrate.
BANNED_PROSE_PATTERNS: dict[str, list[re.Pattern]] = {
    "typical_values": [
        re.compile(r"\btypical(?:ly)?\b", re.I),
        re.compile(r"\bon the order of\b", re.I),
        re.compile(r"\bof order\s+\d", re.I),
        re.compile(r"\branges?\s+from\b.*\bto\b", re.I),
        re.compile(r"~\s*\d"),
    ],
    "estimator_recipe": [
        re.compile(
            r"\bis (?:computed|calculated|estimated|obtained|derived) (?:as|by|from)\b",
            re.I,
        ),
        re.compile(
            r"\bcan be (?:computed|calculated|estimated|obtained|derived)\b", re.I
        ),
        re.compile(r"\bto (?:compute|calculate|estimate)\b", re.I),
    ],
    "procedural_padding": [
        re.compile(r"\bit should be noted\b", re.I),
        re.compile(r"\bnote that\b", re.I),
        re.compile(r"\bin practice\b", re.I),
        re.compile(r"\bfor example,", re.I),
    ],
}


def banned_prose_findings(text: str) -> dict[str, int]:
    """Count banned-prose matches per class in *text*.

    Returns a dict ``{class: match_count}`` including zero-count classes so the
    caller can aggregate uniformly.
    """
    text = text or ""
    return {
        cls: sum(len(p.findall(text)) for p in pats)
        for cls, pats in BANNED_PROSE_PATTERNS.items()
    }
