"""SN ↔ DD unit-mismatch axis exceptions.

The mismatch axis compares a standard name's declared unit against the unit its
Data-Dictionary source path declares, normalising both through the single
canonical authority (:func:`imas_standard_names.canonical_unit`). Ordering-only
and spelling-only differences collapse there. The residual disagreements are
either genuine defects or one of two curated exception classes recorded in
``dd_unit_exceptions.yaml``:

* **DD-side unit bugs** — the DD path declares a physically-wrong unit and the
  SN correctly overrides it (e.g. a charge NUMBER tagged with elementary-charge
  ``e``, a direction unit-vector component tagged with metre ``m``).
* **Unit equivalences** — two canonical forms that are the same physical unit
  but which the formatter keeps distinct (``Hz`` vs ``s^-1``; ``N.m`` vs
  ``kg.m^2.s^-2``).

``units_agree()`` is the single entry point the mismatch axis calls: it returns
``True`` when an SN unit and a DD-path unit should be treated as agreeing.
"""

from __future__ import annotations

import fnmatch
import importlib.resources
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import yaml
from imas_standard_names import canonical_unit

if TYPE_CHECKING:
    from imas_codex.standard_names.dd_resolutions import DDResolutionManifest

__all__ = [
    "canonical_or_none",
    "units_agree",
    "load_exceptions",
    "dd_unit_bug_globs",
]


def canonical_or_none(unit: str | None) -> str | None:
    """Canonicalise a unit string, or ``None`` if empty/unparseable.

    Wraps :func:`imas_standard_names.canonical_unit` so callers on either side
    of the comparison get a lenient normaliser: unset or DD-error units (which
    the canonical parser rejects) collapse to ``None`` rather than raising.
    """
    if not unit or not unit.strip():
        return None
    try:
        return canonical_unit(unit.strip())
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_exceptions() -> dict[str, Any]:
    """Load and cache the parsed exceptions YAML."""
    resource = importlib.resources.files("imas_codex.units").joinpath(
        "dd_unit_exceptions.yaml"
    )
    with importlib.resources.as_file(resource) as path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    return {
        "dd_unit_bugs": list(data.get("dd_unit_bugs") or []),
        "unit_equivalences": [
            frozenset(canonical_or_none(u) for u in pair)
            for pair in (data.get("unit_equivalences") or [])
        ],
    }


def dd_unit_bug_globs() -> list[str]:
    """Return the DD-path globs of every DD-side unit-bug entry (for tests)."""
    return [str(e["path"]) for e in load_exceptions()["dd_unit_bugs"]]


def _is_equivalent(a: str, b: str) -> bool:
    """True if two canonical unit forms are a recorded physical equivalence."""
    pair = {a, b}
    return any(pair <= eq for eq in load_exceptions()["unit_equivalences"])


def _active_resolution_retires_pair(
    dd_path: str,
    dd_canon: str | None,
    correct_canon: str | None,
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
        path=dd_path,
        dd_version=dd_version or get_dd_version(),
        field=DDResolutionField.unit,
        manifest=manifest,
    )
    return bool(
        record
        and canonical_or_none(record.observed.value) == dd_canon
        and canonical_or_none(record.effective.value) == correct_canon
    )


def _is_known_dd_bug(
    dd_path: str,
    sn_canon: str | None,
    dd_canon: str | None,
    *,
    dd_version: str | None,
    manifest: DDResolutionManifest | None,
) -> bool:
    """True if (dd_path, dd_canon, sn_canon) matches a recorded DD-side bug.

    The DD path must match the entry's glob, the DD unit must canonicalise to
    the entry's ``dd_unit`` and the SN unit to the entry's ``correct_unit`` —
    all three so a broad glob only ever suppresses the exact unit pair it
    documents, never an unrelated disagreement on the same path.
    """
    for entry in load_exceptions()["dd_unit_bugs"]:
        if not fnmatch.fnmatchcase(dd_path, str(entry["path"])):
            continue
        if canonical_or_none(str(entry["dd_unit"])) != dd_canon:
            continue
        if canonical_or_none(str(entry["correct_unit"])) != sn_canon:
            continue
        if _active_resolution_retires_pair(
            dd_path,
            dd_canon,
            sn_canon,
            dd_version=dd_version,
            manifest=manifest,
        ):
            continue
        return True
    return False


def graph_unit_correction(
    dd_path: str,
    dd_unit: str | None,
    *,
    dd_version: str | None = None,
    manifest: DDResolutionManifest | None = None,
) -> str | None:
    """Return the unit to STORE for *dd_path*, when the DD declaration is wrong.

    Most recorded DD unit bugs are only *suppressed* on the mismatch axis: the
    DD unit is left as declared so the axis can keep reporting it while the
    standard name carries the correct one. A small subset is flagged
    ``correct_in_graph`` because the DD contradicts **itself** on that quantity
    — the same quantity declared with two different dimensionalities — so there
    is no single DD answer to mirror, and storing the wrong one would feed a
    wrong dimensionality into standard-name generation. For those, the
    dimensionally-correct unit is returned so the build stores it instead.

    Returns None when the path/unit pair is not a flagged correction, i.e. the
    DD declaration should be stored as-is.
    """
    dd_canon = canonical_or_none(dd_unit)
    if dd_canon is None:
        return None
    for entry in load_exceptions()["dd_unit_bugs"]:
        if not entry.get("correct_in_graph"):
            continue
        if not fnmatch.fnmatchcase(dd_path, str(entry["path"])):
            continue
        if canonical_or_none(str(entry["dd_unit"])) != dd_canon:
            continue
        correct_canon = canonical_or_none(str(entry["correct_unit"]))
        if _active_resolution_retires_pair(
            dd_path,
            dd_canon,
            correct_canon,
            dd_version=dd_version,
            manifest=manifest,
        ):
            continue
        return str(entry["correct_unit"])
    return None


def units_agree(
    sn_unit: str | None,
    dd_unit: str | None,
    dd_path: str,
    *,
    dd_version: str | None = None,
    manifest: DDResolutionManifest | None = None,
) -> bool:
    """Return True if an SN unit and a DD-path unit should be treated as agreeing.

    Agreement holds when, after canonicalisation, the two forms are equal, are a
    recorded physical equivalence, or the disagreement is a recorded DD-side
    unit bug on ``dd_path``. A DD unit the canonical parser cannot resolve
    (``None``) never agrees — it is a DD-data problem the axis should surface.
    """
    sn_c = canonical_or_none(sn_unit)
    dd_c = canonical_or_none(dd_unit)
    if dd_c is None or sn_c is None:
        return False
    if sn_c == dd_c:
        return True
    if _is_equivalent(sn_c, dd_c):
        return True
    return _is_known_dd_bug(
        dd_path,
        sn_c,
        dd_c,
        dd_version=dd_version,
        manifest=manifest,
    )
