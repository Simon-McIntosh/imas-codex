"""Unit tests for automatic family-harmonization bookkeeping.

Covers the three always-on pieces that replaced the standalone harmonize
command:

* ``_doc_link_mismatches`` — the accept-path link label/target gate helper.
* ``restamp_harmonized_families`` — the post-drain signature reconcile.
* ``mark_families_for_regen`` — parent-id resolution for ``sn run --families``.

All graph access is mocked (conftest blocks live GraphClient by default).
"""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names.graph_ops import _doc_link_mismatches
from imas_codex.standard_names.harmonize import (
    assemble_family,
    group_signature,
    mark_families_for_regen,
    restamp_harmonized_families,
)

# ---------------------------------------------------------------------------
# _doc_link_mismatches
# ---------------------------------------------------------------------------


class _IdsGC:
    """Fake GraphClient answering only the existing-ids lookup."""

    def __init__(self, existing):
        self._existing = set(existing)

    def query(self, cypher, **params):
        ids = params.get("ids") or []
        return [{"id": i} for i in ids if i in self._existing]


def test_link_mismatch_flags_existing_label_different_target():
    gc = _IdsGC({"a_name", "b_name", "c_name"})
    doc = "See [b_name](name:c_name) for detail."
    assert _doc_link_mismatches(gc, doc) == ["[b_name] -> name:c_name"]


def test_link_mismatch_ignores_human_labels_and_self_links():
    gc = _IdsGC({"a_name", "b_name"})
    doc = (
        "See [electron temperature profile](name:b_name) and "
        "[b_name](name:b_name) and [B Name](name:b_name)."
    )
    assert _doc_link_mismatches(gc, doc) == []


def test_link_mismatch_empty_doc():
    gc = _IdsGC({"a_name"})
    assert _doc_link_mismatches(gc, None) == []
    assert _doc_link_mismatches(gc, "no links at all") == []


# ---------------------------------------------------------------------------
# restamp_harmonized_families
# ---------------------------------------------------------------------------


def _member(mid, desc="d", doc="x", stage="accepted"):
    return {
        "id": mid,
        "description": desc,
        "documentation": doc,
        "docs_stage": stage,
        "operator_kind": "projection",
    }


class _FamiliesGC:
    def __init__(self, rows):
        self._rows = rows

    def query(self, cypher, **params):
        return self._rows


def test_restamp_skips_unchanged_and_not_ready():
    fresh = [_member("a"), _member("b")]
    stale = [_member("c"), _member("d")]
    waiting = [_member("e"), _member("f", stage="pending")]
    rows = [
        {
            "parent_id": "p_fresh",
            "stored_signature": group_signature(fresh),
            "members": fresh,
        },
        {"parent_id": "p_stale", "stored_signature": "old-sig", "members": stale},
        {"parent_id": "p_waiting", "stored_signature": None, "members": waiting},
    ]
    stamped_calls = []

    def _fake_stamp(families):
        stamped_calls.extend(families)
        return len(families)

    with patch(
        "imas_codex.standard_names.graph_ops.stamp_harmonized_families",
        side_effect=_fake_stamp,
    ):
        out = restamp_harmonized_families(gc=_FamiliesGC(rows))

    assert out == {"restamped": 1, "unchanged": 1, "not_ready": 1}
    assert [f["parent"] for f in stamped_calls] == ["p_stale"]
    assert stamped_calls[0]["signature"] == group_signature(stale)


def test_restamp_noop_on_empty_graph():
    with patch(
        "imas_codex.standard_names.graph_ops.stamp_harmonized_families",
        side_effect=AssertionError("must not be called"),
    ):
        out = restamp_harmonized_families(gc=_FamiliesGC([]))
    assert out == {"restamped": 0, "unchanged": 0, "not_ready": 0}


def test_restamp_omits_locus_ancestry_without_documentation_write():
    members = [
        {
            **_member(f"radial_coordinate_of_distinct_owner_{index}"),
            "operator_kind": "locus",
        }
        for index in range(35)
    ]
    rows = [
        {
            "parent_id": "radial_coordinate",
            "stored_signature": None,
            "members": members,
        }
    ]

    with patch(
        "imas_codex.standard_names.graph_ops.stamp_harmonized_families",
        side_effect=AssertionError("locus ancestry must not be restamped"),
    ):
        out = restamp_harmonized_families(gc=_FamiliesGC(rows))

    assert out == {"restamped": 0, "unchanged": 0, "not_ready": 0}


def test_restamp_keeps_safe_family_and_carries_no_source_migration_manifest():
    members = [
        {**_member("poloidal_mode_number"), "operator_kind": "projection"},
        {**_member("toroidal_mode_number"), "operator_kind": "projection"},
    ]
    rows = [
        {
            "parent_id": "mode_number",
            "stored_signature": None,
            "members": members,
        }
    ]
    stamped_calls = []

    def _fake_stamp(families):
        stamped_calls.extend(families)
        return len(families)

    with patch(
        "imas_codex.standard_names.graph_ops.stamp_harmonized_families",
        side_effect=_fake_stamp,
    ):
        out = restamp_harmonized_families(gc=_FamiliesGC(rows))

    assert out == {"restamped": 1, "unchanged": 0, "not_ready": 0}
    assert stamped_calls == [
        {
            "parent": "mode_number",
            "members": ["poloidal_mode_number", "toroidal_mode_number"],
            "signature": group_signature(members),
        }
    ]
    assert "source_ids" not in stamped_calls[0]
    assert "successor" not in stamped_calls[0]


class _LocusSeedGC:
    """Expose a locus parent and reject any physical-base fallback query."""

    def __init__(self):
        self.calls = []

    def query(self, cypher, **params):
        self.calls.append(cypher)
        if "RETURN c.id AS id, c.physical_base AS physical_base" in cypher:
            return [
                {
                    "id": "radial_coordinate_of_control_surface",
                    "physical_base": "coordinate",
                }
            ]
        if "RETURN p.id AS parent_id" in cypher:
            return [
                {
                    "parent_id": "radial_coordinate",
                    "operator_kind": "locus",
                    "operator": "control_surface",
                }
            ]
        raise AssertionError("locus seed must not fall back to a base-wide cohort")


def test_locus_seed_is_not_regrouped_by_physical_base():
    gc = _LocusSeedGC()

    assert assemble_family("radial_coordinate_of_control_surface", gc=gc) is None
    assert len(gc.calls) == 2


# ---------------------------------------------------------------------------
# mark_families_for_regen
# ---------------------------------------------------------------------------


class _ParentGC:
    def __init__(self, families):
        self._families = families  # parent -> kids

    def query(self, cypher, **params):
        pid = params.get("pid")
        if pid not in self._families:
            return []
        return [{"parent_id": pid, "kids": self._families[pid]}]


def test_mark_families_resolves_members_and_flags_unknown():
    gc = _ParentGC({"mode_number": ["poloidal_mode_number", "toroidal_mode_number"]})
    captured = {}

    def _fake_mark(member_ids, *, dry_run=False):
        captured["ids"] = member_ids
        captured["dry_run"] = dry_run
        return {"run_id": "rid", "eligible": len(member_ids), "reset": 0}

    with patch(
        "imas_codex.standard_names.harmonize.mark_members_for_regen",
        side_effect=_fake_mark,
    ):
        out = mark_families_for_regen(
            ["mode_number", "no_such_parent"], gc=gc, dry_run=True
        )

    assert captured["ids"] == [
        "mode_number",
        "poloidal_mode_number",
        "toroidal_mode_number",
    ]
    assert captured["dry_run"] is True
    assert out["unknown_parents"] == ["no_such_parent"]
    assert out["eligible"] == 3
