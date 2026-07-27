"""The campaign manifest is contract-checked against its committed JSON Schema.

The manifest is the approve-before-you-spend object for a docs-refinement
campaign, so it gets the same treatment as its ``sn_sources`` / ``sn_names``
siblings: a schema beside theirs, and a test that validates REAL
:func:`build_manifest` output against it so emitter and contract cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_codex.standard_names.campaign import (
    CAMPAIGN_MANIFEST_KIND,
    CAMPAIGN_MANIFEST_SCHEMA_VERSION,
    CampaignSelection,
    CampaignSpec,
    CampaignTarget,
    build_manifest,
    default_campaign_manifest_path,
    write_manifest,
)

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "imas_codex"
    / "standard_names"
    / "config"
    / "campaign_manifest.schema.json"
)


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _targets(n: int, *, domain: str = "equilibrium", quarantined: bool = False):
    return [
        CampaignTarget(
            id=f"n{i}",
            name=f"n{i}",
            matched_predicates={"prose:typical_values": ["1 match(es)"]},
            quarantined=quarantined,
            physics_domain=domain,
        )
        for i in range(n)
    ]


def _selection(n: int = 12, spec: str = "prose", **kw) -> CampaignSelection:
    return CampaignSelection(spec=CampaignSpec.parse(spec), targets=_targets(n, **kw))


class TestSchemaItself:
    def test_schema_is_valid_json_schema(self, schema):
        jsonschema.Draft202012Validator.check_schema(schema)

    def test_schema_sits_beside_its_siblings(self):
        siblings = {p.name for p in SCHEMA_PATH.parent.glob("*.schema.json")}
        assert {
            "sn_sources.schema.json",
            "sn_names.schema.json",
            "campaign_manifest.schema.json",
        } <= siblings


class TestRealOutputValidates:
    """Every shape build_manifest actually emits must satisfy the schema."""

    def test_full_selection_manifest(self, schema):
        jsonschema.validate(build_manifest(_selection(30), batch_size=10), schema)

    def test_pilot_manifest_carries_the_marker(self, schema):
        m = build_manifest(_selection(4), sample_size=4, batch_size=10, pilot_from=2332)
        jsonschema.validate(m, schema)
        assert m["pilot"] == {"n": 4, "from_total": 2332}

    def test_empty_selection_manifest(self, schema):
        """A spec that selects nothing still emits a schema-valid manifest."""
        m = build_manifest(_selection(0), batch_size=10)
        jsonschema.validate(m, schema)
        assert m["total"] == 0
        assert m["sample"] == []

    def test_quarantine_and_missing_domain_shapes(self, schema):
        targets = [
            CampaignTarget(
                id="q1",
                name="q1",
                matched_predicates={"quarantined": ["banned prose"]},
                quarantined=True,
                physics_domain="",
            )
        ]
        sel = CampaignSelection(spec=CampaignSpec.parse("all"), targets=targets)
        m = build_manifest(sel, sample_size=1)
        jsonschema.validate(m, schema)
        # An empty physics_domain is grouped, not dropped.
        assert m["per_domain"] == {"(none)": 1}
        assert m["sample"][0]["physics_domain"] == ""

    def test_multi_predicate_evidence_shape(self, schema):
        targets = [
            CampaignTarget(
                id="m1",
                name="m1",
                matched_predicates={
                    "prose:typical_values": ["2 match(es)"],
                    "audit:latex": ["latex_def_check", "latex_math_check"],
                },
                physics_domain="transport",
            )
        ]
        sel = CampaignSelection(spec=CampaignSpec.parse("all"), targets=targets)
        jsonschema.validate(build_manifest(sel, sample_size=1), schema)

    def test_written_file_round_trips_and_validates(self, schema, tmp_path):
        path = write_manifest(
            build_manifest(_selection(3), sample_size=3), tmp_path / "m.json"
        )
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), schema)


class TestSelfIdentifying:
    """kind + schema_version, matching the sibling manifests."""

    def test_manifest_declares_kind_and_version(self):
        m = build_manifest(_selection(2))
        assert m["kind"] == CAMPAIGN_MANIFEST_KIND == "campaign_manifest"
        assert m["schema_version"] == CAMPAIGN_MANIFEST_SCHEMA_VERSION == 1

    def test_discriminator_leads_the_document(self):
        """kind/schema_version first so a human reading the raw JSON sees them."""
        assert list(build_manifest(_selection(2)))[:2] == ["kind", "schema_version"]


class TestSchemaCatchesDrift:
    """The schema must reject malformed manifests, not rubber-stamp anything."""

    def test_unknown_top_level_key_rejected(self, schema):
        m = build_manifest(_selection(2))
        m["unexpected_key"] = 1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(m, schema)

    def test_missing_required_key_rejected(self, schema):
        m = build_manifest(_selection(2))
        del m["per_domain"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(m, schema)

    def test_wrong_kind_rejected(self, schema):
        m = build_manifest(_selection(2))
        m["kind"] = "sn_names"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(m, schema)

    def test_sample_entry_missing_evidence_rejected(self, schema):
        m = build_manifest(_selection(2), sample_size=2)
        m["sample"][0]["matched_predicates"] = {}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(m, schema)

    def test_negative_count_rejected(self, schema):
        m = build_manifest(_selection(2))
        m["total"] = -1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(m, schema)


class TestDefaultPath:
    """The manifest has a real home; it never defaults into the working dir."""

    def test_default_is_outside_the_repo_tree(self):
        default = default_campaign_manifest_path()
        repo_root = Path(__file__).resolve().parents[2]
        assert not default.is_relative_to(repo_root)
        assert default.is_absolute()

    def test_default_lives_under_the_user_data_dir(self):
        default = default_campaign_manifest_path()
        assert default.is_relative_to(Path.home() / ".local" / "share" / "imas-codex")
        assert default.suffix == ".json"

    def test_default_is_not_the_bare_working_directory_name(self):
        assert default_campaign_manifest_path() != Path("campaign-manifest.json")
