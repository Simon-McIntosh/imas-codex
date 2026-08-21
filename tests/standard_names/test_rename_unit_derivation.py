"""Rename successors derive unit authority from their DD source cohort."""

from unittest.mock import MagicMock, patch

from imas_codex.standard_names import edit


def test_rename_successor_uses_unanimous_dd_source_unit() -> None:
    gc = MagicMock()

    def query(statement: str, **_params: object) -> list[dict[str, object]]:
        if "EDIT_CHECK_COLLISION" in statement:
            return [{"n": 0}]
        if "EDIT_DERIVE_RENAME_UNIT" in statement:
            return [
                {
                    "source_id": "dd:core_sources/source/electrons/explicit_part",
                    "dd_path": "core_sources/source/electrons/explicit_part",
                    "dd_unit": "m^-3.s^-1",
                    "dd_relationship_units": ["m^-3.s^-1"],
                },
                {
                    "source_id": "dd:edge_sources/source/electrons/values",
                    "dd_path": "edge_sources/source/electrons/values",
                    "dd_unit": "m^-3.s^-1",
                    "dd_relationship_units": ["m^-3.s^-1"],
                },
            ]
        return []

    gc.query.side_effect = query
    predecessor = {
        "name_stage": "reviewed",
        "description": "Volume-integrated electron source rate.",
        "kind": "scalar",
        "unit": "s^-1",
        "physics_domain": "transport",
        "origin": "pipeline",
        "tags": [],
        "chain_length": 0,
        "has_successor": False,
        "has_children": False,
        "has_live_source": True,
    }

    with (
        patch.object(edit, "_isn_round_trip_ok", return_value=(True, "")),
        patch.object(edit, "_base_token", return_value="source_rate"),
        patch.object(edit, "_grammar_segment_props", return_value={}),
        patch.object(edit, "_new_run_id", return_value="sn-edit-unit-derivation"),
        patch.object(
            edit,
            "persist_refined_name",
            return_value={"new_name": "electron_volumetric_source_rate"},
        ) as persist,
        patch.object(edit, "_stamp_successor_validation") as validate,
    ):
        plan = edit._apply_rename(
            gc,
            target="electron_source_rate",
            target_row=predecessor,
            new_name="electron_volumetric_source_rate",
            reason="match the DD volumetric source quantity",
            origin="user",
            scope="only_self",
            is_parent=False,
            override_edits=False,
            include_accepted=False,
            dry_run=False,
        )

    assert plan.applied is True
    assert persist.call_args.kwargs["unit"] == "m^-3.s^-1"
    assert validate.call_args.args[2]["unit"] == "m^-3.s^-1"
