"""DD unit override engine.

The IMAS Data Dictionary emits several invalid or context-dependent unit
strings that break the standard-name pipeline (prose names like
``"Elementary Charge Unit"``, unresolved Jinja templates like
``"m^dimension"``, and sentinel ``"1"`` values under
``pulse_schedule/*/reference``).  The DD team has confirmed these are
by-design; we handle them entirely here.

The defect catalog and the overrides are declared together in
``config/unit_overrides.yaml``. Each rule is
a (path_pattern, dd_unit) → action mapping with one of two strategies:

- ``override``: replace the reported dd_unit with ``override_unit``; the
  candidate then flows through the pipeline normally.
- ``skip``: mark the DD path as a skipped ``StandardNameSource`` with a
  documented ``skip_reason``; the candidate is removed from extraction.

Path patterns use glob-style syntax where ``*`` matches a single path
segment and ``**`` matches zero or more segments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from imas_codex.standard_names.dd_resolutions import DDResolutionManifest

_CONFIG = Path(__file__).parent / "config" / "unit_overrides.yaml"


@dataclass(frozen=True)
class OverrideRule:
    """A single DD unit override/skip rule."""

    path_pattern: str
    dd_unit: str
    strategy: str  # 'override' | 'skip'
    override_unit: str | None
    skip_reason: str | None
    reason: str

    def matches(self, path: str, unit: str) -> bool:
        if self.dd_unit != unit:
            return False
        return _glob_match(self.path_pattern, path)


def _glob_match(pattern: str, path: str) -> bool:
    """Glob-match a DD path against a pattern.

    - ``*``  matches a single path segment (no ``/``)
    - ``**`` matches zero or more path segments (may contain ``/``)
    - All other segment characters are compared literally.
    """
    parts = pattern.split("/")
    rx_segments: list[str] = []
    for p in parts:
        if p == "**":
            rx_segments.append("__DOUBLESTAR__")
        elif p == "*":
            rx_segments.append("[^/]+")
        elif "*" in p:
            # Single-segment wildcard mixed with literals (rare, but supported).
            escaped = re.escape(p).replace(r"\*", "[^/]*")
            rx_segments.append(escaped)
        else:
            rx_segments.append(re.escape(p))

    # Join with literal "/" — then rewrite "__DOUBLESTAR__" boundaries so
    # that a "**" can absorb its flanking slashes ("" zero-segment match).
    joined = "/".join(rx_segments)
    # "/__DOUBLESTAR__/" → "(?:/.*)?/" (match zero or more segments)
    joined = joined.replace("__DOUBLESTAR__/", "(?:.*/)?")
    joined = joined.replace("/__DOUBLESTAR__", "(?:/.*)?")
    # Standalone "**" (whole-path pattern)
    joined = joined.replace("__DOUBLESTAR__", ".*")

    regex = f"^{joined}$"
    return re.match(regex, path) is not None


@lru_cache(maxsize=1)
def _load_rules() -> tuple[OverrideRule, ...]:
    """Load and cache the override rule set from YAML."""
    if not _CONFIG.exists():
        return ()
    doc = yaml.safe_load(_CONFIG.read_text()) or {}
    rules: list[OverrideRule] = []
    for r in doc.get("overrides", []):
        strategy = r.get("strategy")
        if strategy not in ("override", "skip"):
            raise ValueError(
                f"unit_overrides.yaml: invalid strategy '{strategy}' in rule {r}"
            )
        if strategy == "override" and not r.get("override_unit"):
            raise ValueError(
                f"unit_overrides.yaml: override rule missing override_unit: {r}"
            )
        if strategy == "skip" and not r.get("skip_reason"):
            raise ValueError(f"unit_overrides.yaml: skip rule missing skip_reason: {r}")
        rules.append(
            OverrideRule(
                path_pattern=r["path_pattern"],
                dd_unit=r["dd_unit"],
                strategy=strategy,
                override_unit=r.get("override_unit"),
                skip_reason=r.get("skip_reason"),
                reason=r["reason"],
            )
        )
    return tuple(rules)


# Public export so sibling modules use a public name for the rule set.
load_rules = _load_rules


def _active_resolution_retires_rule(
    path: str,
    rule: OverrideRule,
    *,
    dd_version: str | None,
    manifest: DDResolutionManifest | None,
) -> bool:
    from imas_codex.settings import get_dd_version
    from imas_codex.standard_names.dd_resolutions import (
        DDResolutionField,
        active_dd_resolution,
    )

    record = active_dd_resolution(
        path=path,
        dd_version=dd_version or get_dd_version(),
        field=DDResolutionField.unit,
        manifest=manifest,
    )
    expected = rule.override_unit if rule.strategy == "override" else None
    return bool(
        record
        and record.observed.value == rule.dd_unit
        and record.effective.value == expected
    )


def resolve_unit(
    path: str,
    dd_unit: str | None,
    *,
    dd_version: str | None = None,
    manifest: DDResolutionManifest | None = None,
) -> tuple[str | None, dict | None]:
    """Return ``(effective_unit, metadata)`` for a (path, unit) pair.

    - If no rule matches: returns ``(dd_unit, None)`` — pass-through.
    - Override rule: returns ``(override_unit, {"rule": "override", ...})``.
    - Skip rule: returns ``(None, {"rule": "skip", "skip_reason": ..., ...})``.

    ``metadata`` is safe to persist on a ``StandardNameSource`` record when
    the path is skipped (keys ``skip_reason`` and ``skip_reason_detail``).
    """
    if dd_unit is None:
        return None, None
    for rule in _load_rules():
        if rule.matches(path, dd_unit):
            if _active_resolution_retires_rule(
                path, rule, dd_version=dd_version, manifest=manifest
            ):
                continue
            if rule.strategy == "override":
                return rule.override_unit, {
                    "rule": "override",
                    "original_unit": dd_unit,
                    "reason": rule.reason,
                    "path_pattern": rule.path_pattern,
                }
            if rule.strategy == "skip":
                return None, {
                    "rule": "skip",
                    "skip_reason": rule.skip_reason,
                    "skip_reason_detail": f"{rule.reason} (dd_unit={dd_unit!r})",
                    "original_unit": dd_unit,
                    "path_pattern": rule.path_pattern,
                }
    return resolve_quantity_spelling(path, dd_unit)


def _reload_rules_for_tests() -> None:
    """Clear the rule cache. Only for use in tests."""
    _load_rules.cache_clear()


# ---------------------------------------------------------------------------
# Recorded unit spelling per quantity
# ---------------------------------------------------------------------------

_SPELLINGS = _CONFIG


@dataclass(frozen=True)
class QuantitySpelling:
    """One quantity's recorded canonical unit spelling."""

    quantity: str
    canonical: str
    dd_path_patterns: tuple[str, ...]
    reason: str

    def matches(self, path: str) -> bool:
        return any(_glob_match(pattern, path) for pattern in self.dd_path_patterns)


@dataclass(frozen=True)
class SpellingException:
    """A dimensional difference that carries meaning and therefore stands."""

    quantity: str
    spellings: tuple[str, ...]
    reason: str


@lru_cache(maxsize=1)
def _load_spelling_record() -> tuple[
    tuple[QuantitySpelling, ...], tuple[SpellingException, ...]
]:
    """Load and validate the per-quantity spelling record."""
    if not _CONFIG.exists():
        return (), ()
    from imas_codex.units import unit_registry

    doc = yaml.safe_load(_SPELLINGS.read_text()) or {}
    spellings: list[QuantitySpelling] = []
    for entry in doc.get("spellings", []):
        try:
            unit_registry.Unit(entry["canonical"])
        except Exception as exc:
            raise ValueError(
                "unit_overrides.yaml: canonical unit "
                f"{entry.get('canonical')!r} for quantity {entry.get('quantity')!r} "
                f"does not parse: {exc}"
            ) from exc
        patterns = tuple(entry.get("dd_path_patterns") or ())
        if not patterns:
            raise ValueError(
                "unit_overrides.yaml: quantity "
                f"{entry.get('quantity')!r} declares no dd_path_patterns"
            )
        for pattern in patterns:
            # Compiling here refuses a malformed pattern at load rather than
            # silently never matching it.
            _glob_match(pattern, "")
        spellings.append(
            QuantitySpelling(
                quantity=entry["quantity"],
                canonical=entry["canonical"],
                dd_path_patterns=patterns,
                reason=entry["reason"],
            )
        )
    exceptions = tuple(
        SpellingException(
            quantity=entry["quantity"],
            spellings=tuple(entry["spellings"]),
            reason=entry["reason"],
        )
        for entry in doc.get("exceptions", [])
    )
    return tuple(spellings), exceptions


def quantity_spelling(quantity: str) -> str | None:
    """Return the recorded canonical spelling for *quantity*, if any."""
    spellings, _ = _load_spelling_record()
    for spelling in spellings:
        if spelling.quantity == quantity:
            return spelling.canonical
    return None


def spelling_exception(quantity: str) -> SpellingException | None:
    """Return the recorded exception for *quantity*, if any."""
    _, exceptions = _load_spelling_record()
    for exception in exceptions:
        if exception.quantity == quantity:
            return exception
    return None


def resolve_quantity_spelling(path: str, dd_unit: str) -> tuple[str, dict | None]:
    """Return ``(unit, metadata)`` rewritten to the quantity's recorded spelling.

    A rewrite happens only when the Data Dictionary path belongs to a recorded
    quantity and the two spellings carry one dimension. A spelling the record
    does not name, or one it cannot compare, passes through unchanged.
    """
    from imas_codex.standard_names.unit_audit import units_are_dimensionally_equal
    from imas_codex.units.dd_unit_exceptions import canonical_or_none

    spelled = canonical_or_none(dd_unit)
    if spelled is None:
        return dd_unit, None
    for spelling in _load_spelling_record()[0]:
        if not spelling.matches(path):
            continue
        if spelled == canonical_or_none(spelling.canonical):
            return dd_unit, None
        if not units_are_dimensionally_equal(dd_unit, spelling.canonical):
            # A different dimension on a quantity's path is not a spelling
            # choice; it is left for the dimension check to report.
            continue
        return spelling.canonical, {
            "rule": "override",
            "spelling_quantity": spelling.quantity,
            "original_unit": dd_unit,
            "reason": spelling.reason,
            "path_pattern": ",".join(spelling.dd_path_patterns),
        }
    return dd_unit, None


def _reload_spellings_for_tests() -> None:
    """Clear the spelling record cache. Only for use in tests."""
    _load_spelling_record.cache_clear()
