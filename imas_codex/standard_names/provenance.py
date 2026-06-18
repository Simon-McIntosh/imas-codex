"""Value-provenance facet detection for the standard-name pipeline.

A *value-provenance facet* is a DD leaf whose terminal segment marks how a
value of the parent quantity was obtained — ``.../measured``,
``.../reconstructed`` (reconstruction-constraint facets), or ``.../reference``
(a control setpoint). Per the locked ``name-multiplicity-rule`` /
``provenance-controlled-vocab`` decisions, these facets do NOT get their own
standard name: they COLLAPSE onto the base quantity's name (the measured,
reconstructed, and reference plasma current all map to ``plasma_current``),
with the estimator recorded on the ``StandardNameSource.provenance`` link —
never emitted in the name.

The controlled vocabulary is owned by ISN
(:func:`imas_standard_names.value_provenance_terms`); this module only detects
the facet on a DD path and strips it to the base-quantity path so the composer
grounds on the quantity, not the facet.

Contrast a *physical-locus* difference (``_of_plasma_boundary`` vs
``_of_flux_surface``), which DOES de-conflate into distinct names — handled by
the shape-parameter surface injection, not here. Collapse provenance,
de-conflate physics.
"""

from __future__ import annotations

try:
    from imas_standard_names import provenance_for_dd_facet
except Exception:  # pragma: no cover - ISN unavailable at build time
    provenance_for_dd_facet = None  # type: ignore[assignment]


def detect_value_provenance(dd_path: str | None) -> tuple[str | None, str]:
    """Detect a value-provenance facet on *dd_path*.

    Returns ``(provenance_term, base_path)``:

    - ``provenance_term`` — the ISN value-provenance token
      (``measured`` | ``reconstructed`` | ``reference``) when the terminal
      path segment is a value-provenance facet, else ``None``.
    - ``base_path`` — the path with the facet segment stripped (the base
      quantity) when a facet was detected, else the original path unchanged.

    Examples
    --------
    ``equilibrium/time_slice/constraints/ip/measured``
        -> (``"measured"``, ``equilibrium/time_slice/constraints/ip``)
    ``pulse_schedule/position_control/.../ip/reference``
        -> (``"reference"``, ``pulse_schedule/position_control/.../ip``)
    ``core_profiles/profiles_1d/electrons/temperature``
        -> (``None``, ``core_profiles/profiles_1d/electrons/temperature``)
    """
    path = dd_path or ""
    if not path or provenance_for_dd_facet is None or "/" not in path:
        return None, path
    base, _, terminal = path.rpartition("/")
    term = provenance_for_dd_facet(terminal)
    if term is None or not base:
        return None, path
    return term, base
