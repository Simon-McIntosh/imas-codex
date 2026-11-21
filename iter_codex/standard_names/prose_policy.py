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
        # A compute recipe narrates *how* a value is produced: "X is
        # computed/obtained/calculated/estimated as|by|from <procedure>".
        #
        # EXCEPTION — a compute verb that introduces a defining equation is a
        # mathematical DEFINITION, not a procedure: "the current is obtained by
        # integrating the current density over the cross-section: $$I=\int j
        # dA$$" or "the dimensional counterpart is obtained by omitting the
        # $1/(nT)$ factor".  Those are the definitional forms the refine seat
        # writes for derived/integral quantities and must not flag.  Exempt only
        # when a display ($$) or inline ($...$) equation follows within ~200
        # characters; a compute verb with no nearby equation ("is obtained from
        # Thomson scattering") stays a recipe and keeps flagging.
        re.compile(
            r"\bis (?:computed|calculated|estimated|obtained) (?:as|by|from)\b"
            r"(?![\s\S]{0,200}?(?:\$\$|\$[^$]+\$))",
            re.I,
        ),
        re.compile(
            r"\bcan be (?:computed|calculated|estimated|obtained)\b"
            r"(?![\s\S]{0,200}?(?:\$\$|\$[^$]+\$))",
            re.I,
        ),
        # "derived as|by <procedure>" is always a recipe.
        re.compile(r"\b(?:is|can be) derived (?:as|by)\b", re.I),
        # "derived from Y" is a recipe when Y is a procedure/measurement, but
        # NOT when Y is a catalogued quantity linked as [label](name:id) within
        # the same sentence — that is the parent/child provenance form the
        # refine seat writes for a derived quantity, e.g.
        # "derived from the local [hydrogen density](name:hydrogen_density)".
        # Exempt only that linked-quantity provenance; keep flagging
        # "derived from equilibrium reconstruction by fitting ...".
        re.compile(r"\b(?:is|can be) derived from\b(?![^.]*\[[^\]]+\]\(name:)", re.I),
        re.compile(r"\bto (?:compute|calculate|estimate)\b", re.I),
    ],
    "procedural_padding": [
        re.compile(r"\bit should be noted\b", re.I),
        re.compile(r"\bnote that\b", re.I),
        re.compile(r"\bin practice\b", re.I),
        # "for example," is padding EXCEPT when it introduces a catalogued child
        # quantity as a [label](name:id) link — the parent/child taxonomy form a
        # parent-concept doc legitimately uses to point at its specialisations,
        # e.g. "child quantities define those restrictions explicitly; for
        # example, [thermal power of breeder blanket](name:thermal_power_of_
        # breeder_blanket) denotes a thermal-load specialisation".  Exempt only
        # when a name link follows within ~120 characters; illustrative
        # "for example, $10^{20}\,m^{-3}$" padding keeps flagging.
        re.compile(r"\bfor example,(?![\s\S]{0,120}?\[[^\]]+\]\(name:)", re.I),
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
