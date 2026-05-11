"""Base contracts for SN extraction sources and source qualification.

Defines the universal data contracts that all SN source types (DD, signals,
future sources) implement. The key abstractions:

- **SourceCandidate** — normalized representation of a candidate item from
  any source, carrying the fields that qualification checks need.
- **Qualification** — result of evaluating whether a candidate should
  receive a standard name, with machine-readable reason codes.
- **SourceQualifier** — protocol for per-source qualifier implementations.
- **ExtractionBatch** — grouped candidates ready for LLM composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Source qualification contracts
# ---------------------------------------------------------------------------


class QualificationStatus(StrEnum):
    """Outcome of a source qualification check."""

    eligible = "eligible"
    skipped = "skipped"
    not_physical_quantity = "not_physical_quantity"


@dataclass(frozen=True)
class Qualification:
    """Result of qualifying a single source candidate.

    Attributes:
        status: Whether the candidate should proceed to composition.
        reason_code: Machine-readable label for audit (e.g.,
            ``"generic_cross_section_geometry"``). Empty when eligible.
        reason_detail: Human-readable explanation. Empty when eligible.
    """

    status: QualificationStatus = QualificationStatus.eligible
    reason_code: str = ""
    reason_detail: str = ""

    @property
    def eligible(self) -> bool:
        return self.status == QualificationStatus.eligible


# Convenience constructors
ELIGIBLE = Qualification()


def skip(reason_code: str, reason_detail: str = "") -> Qualification:
    """Create a skip qualification."""
    return Qualification(
        status=QualificationStatus.skipped,
        reason_code=reason_code,
        reason_detail=reason_detail,
    )


def not_physical(reason_code: str, reason_detail: str = "") -> Qualification:
    """Create a not-physical-quantity qualification."""
    return Qualification(
        status=QualificationStatus.not_physical_quantity,
        reason_code=reason_code,
        reason_detail=reason_detail,
    )


@dataclass
class SourceCandidate:
    """Normalized representation of a candidate from any SN source.

    Source-specific extractors populate this from their native format.
    Qualification checks operate on this contract, not on raw dicts.

    Attributes:
        source_id: Unique identifier (DD path or signal ID).
        source_kind: Source type discriminator (``"dd"`` or ``"signals"``).
        description: Human-readable description of the quantity.
        unit: Physical unit string (may be empty for dimensionless).
        value_type: Data type tag (e.g., ``"FLT_1D"``, ``"INT_0D"``).
            Source-specific but used by universal checks.
        hierarchy: Path segments as a tuple, enabling structural checks
            without string manipulation. For DD: ``("equilibrium",
            "time_slice", "profiles_1d", "psi")``. For signals:
            ``("tcv", "magnetics", "ip_measured")``.
        documentation: Full documentation text from the source (DD docs,
            signal metadata). Used by attribute-predicate checks.
        metadata: Source-specific metadata that doesn't fit the universal
            fields. Qualification checks declare which keys they need.
        raw: The original source dict, preserved for downstream use.
    """

    source_id: str
    source_kind: str
    description: str = ""
    unit: str = ""
    value_type: str = ""
    hierarchy: tuple[str, ...] = ()
    documentation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dd_row(cls, row: dict) -> SourceCandidate:
        """Create a SourceCandidate from a DD extraction query row."""
        path = row.get("path") or ""
        return cls(
            source_id=path,
            source_kind="dd",
            description=row.get("description") or "",
            unit=row.get("unit") or row.get("unit_from_rel") or "",
            value_type=row.get("data_type") or "",
            hierarchy=tuple(path.split("/")) if path else (),
            documentation=row.get("documentation") or "",
            metadata={
                "node_category": row.get("node_category") or "",
                "ids_name": path.split("/")[0] if "/" in path else path,
                "physics_domain": row.get("physics_domain") or "",
            },
            raw=row,
        )

    @classmethod
    def from_signal_row(cls, row: dict) -> SourceCandidate:
        """Create a SourceCandidate from a signal query row."""
        signal_id = row.get("signal_id") or ""
        return cls(
            source_id=signal_id,
            source_kind="signals",
            description=row.get("description") or "",
            unit=row.get("units") or "",
            value_type="",
            hierarchy=tuple(signal_id.split("/")) if signal_id else (),
            documentation="",
            metadata={
                "physics_domain": row.get("physics_domain") or "",
                "diagnostic": row.get("diagnostic") or "",
                "facility": row.get("facility") or "",
            },
            raw=row,
        )


class SourceQualifier(Protocol):
    """Protocol for per-source qualification implementations.

    Each source type provides its own qualifier that encodes domain-specific
    rules. The pipeline calls ``qualify()`` on each candidate and routes
    the result to either composition (eligible) or audit recording (skip).
    """

    def qualify(self, candidate: SourceCandidate) -> Qualification:
        """Decide whether *candidate* should receive a standard name."""
        ...


# ---------------------------------------------------------------------------
# Extraction batch (unchanged — used downstream by compose workers)
# ---------------------------------------------------------------------------


@dataclass
class ExtractionBatch:
    """A batch of candidates extracted from a source for LLM composition.

    Each batch groups related items (e.g., same IDS, same cluster, same diagnostic)
    to give the LLM coherent context for generating standard names.

    Attributes:
        mode: Grouping strategy. ``"default"`` uses the rich
            (cluster × unit) grouping with per-item sibling / COCOS /
            cross-IDS context. ``"names"`` uses coarse
            (physics_domain × unit) grouping in larger bins and pairs
            with a leaner user prompt that defers deep enrichment to
            a follow-up review pass.
    """

    source: str  # "dd" or "signals"
    group_key: str  # e.g., IDS name or diagnostic name
    items: list[dict]  # Source-specific extraction data
    context: str  # Human-readable context for the LLM prompt
    existing_names: set[str] = field(
        default_factory=set
    )  # Known standard names for dedup
    dd_version: str | None = None  # DD version whose conventions apply
    cocos_version: int | None = None  # COCOS convention from that DD version
    cocos_params: dict | None = None  # Full COCOS node properties
    mode: str = "default"  # "default" or "names"


class ExtractionSource(Protocol):
    """Protocol for source extraction plugins."""

    def extract(
        self,
        *,
        ids_filter: str | None = None,
        domain_filter: str | None = None,
        facility: str | None = None,
        limit: int = 500,
    ) -> list[ExtractionBatch]:
        """Extract candidate batches from this source."""
        ...
