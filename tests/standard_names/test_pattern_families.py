"""Pattern-family resolution over a fixture graph.

A pattern family is the set of live standard names whose grammar parses
identically except along the variation axes. The invariant segments are the
base quantity, its transformation and process, plus the entity locus
(``object``): two names describing different entities are never family-mates.
The variation axes are ``subject``, ``component``, ``coordinate``,
``position``, ``geometry``, ``region`` and ``device``; the empty value of an
axis is one of its values, so ``electron_temperature`` and
``electron_temperature_at_magnetic_axis`` share a family.

The fixture graph answers the three queries the resolver issues:
:func:`~imas_codex.standard_names.families.fetch_pattern_family` (members),
:func:`~imas_codex.standard_names.families.fetch_family_base` (base quantity)
and :func:`~imas_codex.standard_names.families.count_pattern_families`.
"""

from __future__ import annotations

import re
from typing import Any

from imas_standard_names.grammar.vocab_loaders import load_locus_registry

from imas_codex.standard_names.context import fetch_sibling_family
from imas_codex.standard_names.families import (
    PATTERN_INVARIANT_SEGMENTS,
    PATTERN_SEGMENTS,
    PATTERN_VARIATION_AXES,
    count_pattern_families,
    fetch_family_base,
    fetch_pattern_family,
    pattern_signature,
)

# The species axis of the reduced-density cohort: 17 tokens.
_FLUX_SPECIES = (
    "electron",
    "ion",
    "total_ion",
    "bulk_ion",
    "deuterium",
    "tritium",
    "hydrogen",
    "helium",
    "helium_3",
    "beryllium",
    "carbon",
    "nitrogen",
    "oxygen",
    "neon",
    "argon",
    "krypton",
    "xenon",
)

_BOUNDARY = "plasma_boundary"


# The family queries compare the invariant columns through ``coalesce(m.COL,
# '') = COL`` (members) and project them the same way in the census. The
# fixture reads those columns back out of the query text so that dropping one
# from the query changes what the fixture does — a test that recomputed the
# signature in Python would stay green while the Cypher lost a column.
_MEMBER_COMPARE_RE = re.compile(r"coalesce\(m\.(\w+), ''\)\s*=\s*\w+")
_MEMBER_PROJECT_RE = re.compile(r"coalesce\(m\.(\w+), ''\)\s+AS\s+\w+")


def _compared_columns(cypher: str) -> tuple[str, ...]:
    """The invariant columns a family is keyed on, read from the Cypher text.

    The members query states them as equality predicates against the seed's
    projection; the census projects them from each member and groups. Either
    spelling is the query's own statement of what makes a family.
    """
    return tuple(_MEMBER_COMPARE_RE.findall(cypher)) or tuple(
        _MEMBER_PROJECT_RE.findall(cypher)
    )


class _PatternGraph:
    """In-memory StandardName graph answering the pattern-family queries.

    The fixture carries StandardName nodes and no ``HAS_PARENT`` edges, so a
    lookup that walks operator ancestry returns nothing — which is what makes
    the negative control bite: reverting the family lookup to the HAS_PARENT
    route leaves the 17-species cohort unresolved.

    Membership is decided by the columns the issued Cypher compares, not by a
    Python re-derivation of the signature: the fixture is the query's behaviour
    under test, so a query that stops comparing the entity locus stops
    separating entity owners here too.
    """

    def __init__(self, nodes: list[dict[str, Any]]):
        self.nodes = {n["id"]: dict(n) for n in nodes}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def _live(node: dict[str, Any], superseded: str) -> bool:
        return (node.get("name_stage") or "") != superseded

    @staticmethod
    def _key(node: dict[str, Any], columns: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(str(node.get(col) or "") for col in columns)

    def query(self, cypher: str, **params: Any):
        self.calls.append((cypher, params))
        superseded = params.get("superseded", "superseded")
        if "HAS_PARENT" in cypher:
            return []  # no operator edges in the fixture
        columns = _compared_columns(cypher)
        if "MATCH (seed:StandardName {id: $sn_id})" in cypher:
            seed = self.nodes.get(params["sn_id"])
            if seed is None or not self._live(seed, superseded):
                return []
            key = self._key(seed, columns)
            return [
                dict(n)
                for n in self.nodes.values()
                if self._live(n, superseded) and self._key(n, columns) == key
            ]
        if "MATCH (b:StandardName {id: $base_id})" in cypher:
            base = self.nodes.get(params["base_id"])
            if base is None or not self._live(base, superseded):
                return []
            return [dict(base)]
        if "count(m) AS n" in cypher:
            sizes: dict[tuple[str, ...], int] = {}
            for node in self.nodes.values():
                if not self._live(node, superseded):
                    continue
                key = self._key(node, columns)
                sizes[key] = sizes.get(key, 0) + 1
            kept = [n for n in sizes.values() if n >= params["min_members"]]
            return [{"families": len(kept), "members": sum(kept)}]
        return []


def _name(
    node_id: str,
    *,
    physical_base: str,
    transformation: str = "",
    geometric_base: str = "",
    process: str = "",
    subject: str = "",
    component: str = "",
    coordinate: str = "",
    position: str = "",
    geometry: str = "",
    region: str = "",
    device: str = "",
    object: str = "",
    name_stage: str = "drafted",
    description: str | None = None,
    documentation: str | None = None,
    docs_stage: str = "",
    reviewer_score_docs: float | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "physical_base": physical_base,
        "geometric_base": geometric_base,
        "transformation": transformation,
        "process": process,
        "subject": subject,
        "component": component,
        "coordinate": coordinate,
        "position": position,
        "geometry": geometry,
        "region": region,
        "device": device,
        "object": object,
        "name_stage": name_stage,
        "description": f"Docs for {node_id}." if description is None else description,
        "documentation": (
            f"Documentation for {node_id}." if documentation is None else documentation
        ),
        "docs_stage": docs_stage,
        "reviewer_score_docs": reviewer_score_docs,
        "has_dd_source": True,
    }


def _flux_node(species: str, **overrides: Any) -> dict[str, Any]:
    node_id = f"flux_surface_averaged_{species}_density_at_{_BOUNDARY}"
    fields: dict[str, Any] = {
        "physical_base": "density",
        "transformation": "flux_surface_averaged",
        "subject": species,
        "position": _BOUNDARY,
    }
    fields.update(overrides)
    return _name(node_id, **fields)


def _seed_id() -> str:
    return f"flux_surface_averaged_{_FLUX_SPECIES[0]}_density_at_{_BOUNDARY}"


def _flux_fixture() -> _PatternGraph:
    """The 17-species reduced-density cohort plus its unreduced base quantity."""
    nodes = [_flux_node(species) for species in _FLUX_SPECIES]
    nodes.append(
        _name(
            "density",
            physical_base="density",
            description="The density.",
            documentation="The density at a point.",
            docs_stage="accepted",
        )
    )
    return _PatternGraph(nodes)


# A slice of the ISN entity-locus vocabulary, read from the registry so the
# fixture is the registry's own notion of an entity, not a hand-picked list.
_ENTITY_TOKENS = tuple(
    sorted(
        token
        for token, locus in load_locus_registry().loci.items()
        if locus.type == "entity"
    )
)[:34]

_BASE_OF_ENTITY = "cross_section"


def _entity_owner_id(entity: str) -> str:
    return f"{_BASE_OF_ENTITY}_of_{entity}"


def _entity_owner_fixture() -> _PatternGraph:
    nodes = [
        _name(
            _entity_owner_id(entity),
            physical_base=_BASE_OF_ENTITY,
            object=entity,
        )
        for entity in _ENTITY_TOKENS
    ]
    return _PatternGraph(nodes)


# ---------------------------------------------------------------------------
# Segment definition
# ---------------------------------------------------------------------------


def test_variation_axes_are_the_non_locus_owner_segments():
    assert PATTERN_VARIATION_AXES == (
        "subject",
        "component",
        "coordinate",
        "position",
        "geometry",
        "region",
        "device",
    )
    assert "transformation" in PATTERN_INVARIANT_SEGMENTS
    assert "physical_base" in PATTERN_INVARIANT_SEGMENTS
    # the entity locus is invariant: entity owners are never family-mates
    assert "object" in PATTERN_INVARIANT_SEGMENTS


def test_variation_axes_and_invariant_segments_partition_every_segment():
    assert set(PATTERN_VARIATION_AXES) | set(PATTERN_INVARIANT_SEGMENTS) == set(
        PATTERN_SEGMENTS
    )
    assert not set(PATTERN_VARIATION_AXES) & set(PATTERN_INVARIANT_SEGMENTS)


def test_pattern_signature_is_the_invariant_tuple():
    sig = pattern_signature(
        {
            "physical_base": "density",
            "transformation": "flux_surface_averaged",
            "subject": "electron",
        }
    )
    assert sig == ("density", "", "flux_surface_averaged", "", "")
    # the subject axis does not enter the signature
    other = pattern_signature(
        {
            "physical_base": "density",
            "transformation": "flux_surface_averaged",
            "subject": "ion",
        }
    )
    assert other == sig


def test_pattern_signature_carries_the_entity_locus():
    """Two names describing different entities are not family-mates."""
    copper = pattern_signature({"physical_base": "cross_section", "object": "copper"})
    tungsten = pattern_signature(
        {"physical_base": "cross_section", "object": "tungsten"}
    )
    assert copper != tungsten


def test_members_query_compares_every_invariant_segment():
    """The members query keys a family on exactly the invariant segments.

    The expected list is written out here, independent of the constant the
    query is generated from, so a segment dropped from the generated Cypher
    fails this test instead of moving both sides of the comparison together.
    """
    from imas_codex.standard_names.families import _PATTERN_FAMILY_MEMBERS_QUERY

    assert set(_compared_columns(_PATTERN_FAMILY_MEMBERS_QUERY)) == {
        "physical_base",
        "geometric_base",
        "transformation",
        "process",
        "object",
    }


# ---------------------------------------------------------------------------
# Family resolution
# ---------------------------------------------------------------------------


def test_species_cohort_resolves_to_one_family_of_seventeen():
    gc = _flux_fixture()
    family = fetch_pattern_family(_seed_id(), gc)
    ids = sorted(row["id"] for row in family)
    assert len(ids) == 17
    assert ids == sorted(
        f"flux_surface_averaged_{s}_density_at_{_BOUNDARY}" for s in _FLUX_SPECIES
    )


def test_every_cohort_member_sees_the_same_family():
    gc = _flux_fixture()
    for species in _FLUX_SPECIES:
        family = fetch_pattern_family(
            f"flux_surface_averaged_{species}_density_at_{_BOUNDARY}", gc
        )
        assert len(family) == 17


def test_lone_name_is_a_family_of_one_not_an_error():
    gc = _PatternGraph([_name("plasma_current", physical_base="plasma_current")])
    family = fetch_pattern_family("plasma_current", gc)
    assert [row["id"] for row in family] == ["plasma_current"]


def test_superseded_names_leave_the_family():
    gc = _flux_fixture()
    gc.nodes[_seed_id()]["name_stage"] = "superseded"
    fam_seed = f"flux_surface_averaged_{_FLUX_SPECIES[1]}_density_at_{_BOUNDARY}"
    family = fetch_pattern_family(fam_seed, gc)
    assert len(family) == 16
    assert _seed_id() not in {row["id"] for row in family}


def test_superseded_seed_resolves_to_nothing():
    gc = _flux_fixture()
    gc.nodes[_seed_id()]["name_stage"] = "superseded"
    assert fetch_pattern_family(_seed_id(), gc) == []


def test_component_variant_joins_the_family():
    gc = _flux_fixture()
    gc.nodes["flux_surface_averaged_electron_density_toroidal_at_plasma_boundary"] = (
        _name(
            "flux_surface_averaged_electron_density_toroidal_at_plasma_boundary",
            physical_base="density",
            transformation="flux_surface_averaged",
            subject="electron",
            component="toroidal",
            position=_BOUNDARY,
        )
    )
    family = fetch_pattern_family(_seed_id(), gc)
    assert len(family) == 18


def test_different_transformation_is_a_separate_family():
    """The unreduced base quantity is not a sibling of the reduced cohort."""
    gc = _flux_fixture()
    assert "density" not in {row["id"] for row in fetch_pattern_family(_seed_id(), gc)}
    base_family = fetch_pattern_family("density", gc)
    assert [row["id"] for row in base_family] == ["density"]


def test_census_counts_cohorts_at_or_above_the_threshold():
    gc = _flux_fixture()
    census = count_pattern_families(gc, min_members=3)
    # the 17-species cohort qualifies; the lone base quantity does not
    assert census == {"families": 1, "members": 17}


def test_family_base_lookup_returns_the_base_row():
    gc = _flux_fixture()
    base = fetch_family_base("density", gc)
    assert base["id"] == "density"
    assert base["docs_stage"] == "accepted"
    assert fetch_family_base("no_such_name", gc) == {}


# ---------------------------------------------------------------------------
# Entity locus (invariant)
# ---------------------------------------------------------------------------


def test_entity_owners_are_not_a_cohort():
    """Names for different entities share no family, whatever else they share.

    The 34 registered entity tokens all fill the same base quantity and no
    variation axis, so they differ only in the entity locus. A family keyed on
    anything but that locus collapses them into one 34-member cohort.
    """
    gc = _entity_owner_fixture()
    for entity in _ENTITY_TOKENS:
        family = fetch_pattern_family(_entity_owner_id(entity), gc)
        assert [row["id"] for row in family] == [_entity_owner_id(entity)]


def test_entity_owner_has_no_sibling_family():
    gc = _entity_owner_fixture()
    assert fetch_sibling_family(_entity_owner_id(_ENTITY_TOKENS[0]), gc=gc) is None


# ---------------------------------------------------------------------------
# Prompt-grounding context
# ---------------------------------------------------------------------------


def test_sibling_family_default_caps_the_sibling_list():
    gc = _flux_fixture()
    family = fetch_sibling_family(_seed_id(), gc=gc)
    assert len(family["siblings"]) == 12


def test_sibling_family_names_the_base_quantity_as_parent():
    gc = _flux_fixture()
    family = fetch_sibling_family(_seed_id(), gc=gc, max_siblings=20)
    assert family is not None
    assert family["parent"]["name"] == "density"
    assert family["parent"]["docs_accepted"] is True
    # 16 siblings: the other cohort members, no seed, no base quantity
    assert len(family["siblings"]) == 16
    assert _seed_id() not in {s["name"] for s in family["siblings"]}
    assert "density" not in {s["name"] for s in family["siblings"]}
    # the accepted base quantity is the template anchor
    assert family["anchor"]["name"] == "density"
    assert family["anchor"]["is_parent"] is True


def test_sibling_family_records_the_varying_axes_per_member():
    gc = _flux_fixture()
    family = fetch_sibling_family(_seed_id(), gc=gc, max_siblings=20)
    ion = f"flux_surface_averaged_ion_density_at_{_BOUNDARY}"
    sibling = next(s for s in family["siblings"] if s["name"] == ion)
    # both axes the member fills, relative to the empty-variation base quantity
    assert sibling["operator_kind"] == "subject+position"
    assert sibling["axis"] == f"ion+{_BOUNDARY}"


def test_sibling_family_defers_when_no_member_is_accepted():
    gc = _flux_fixture()
    gc.nodes["density"]["docs_stage"] = "pending"
    family = fetch_sibling_family(_seed_id(), gc=gc, max_siblings=20)
    assert family["parent"]["docs_accepted"] is False
    assert family["anchor"] is None
    assert len(family["siblings"]) == 16


def test_sibling_family_is_none_for_a_lone_name():
    gc = _PatternGraph([_name("plasma_current", physical_base="plasma_current")])
    assert fetch_sibling_family("plasma_current", gc=gc) is None
