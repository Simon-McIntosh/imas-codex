"""Signal source qualifier: determines which facility signals should receive standard names.

Minimal initial implementation — signals are generally higher quality than
raw DD paths (they've already been curated by facility teams). The main
qualification checks are:

- Description presence
- Unit viability
- Already-mapped check (has existing StandardName)

Future work may add diagnostic-specific rules or signal-quality gates.
"""

from __future__ import annotations

from imas_codex.standard_names.sources.base import (
    ELIGIBLE,
    Qualification,
    SourceCandidate,
    skip,
)


def qualify_signal(candidate: SourceCandidate) -> Qualification:
    """Qualify a facility signal for standard name generation.

    Args:
        candidate: Normalized signal source candidate.

    Returns:
        ``Qualification`` with eligible=True if the signal should proceed
        to LLM composition, or a skip result with reason codes.
    """
    # Must have a description.
    if not candidate.description or not candidate.description.strip():
        return skip(
            "empty_description",
            "Signal has no description — cannot generate a meaningful name.",
        )

    # Skip signals that already have standard names (redundant composition).
    if candidate.metadata.get("existing_standard_name"):
        return skip(
            "already_mapped",
            "Signal already has a standard name assigned.",
        )

    return ELIGIBLE
