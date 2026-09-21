"""Tests for the archive reconstruction adapter's committed entry point.

``sn restore compose`` signs one archive extraction into the authority the
adapter accepts, and ``sn restore apply`` is the committed caller that reaches
``apply_signed_manifest`` with the adapter's closed mutation and guard set. Both
live in ``imas_codex.cli.sn`` rather than in the out-of-tree driver that
composed and applied the input before, so these tests hold the reachability
that was previously uncommitted.

The composed artifact is checked against the adapter's own loader rather than
against a copy of its rules, and the CLI is driven with the graph factory
patched, because the autouse ``_block_live_graph`` fixture in ``conftest.py``
would raise if a real ``GraphClient`` were touched.
"""

from __future__ import annotations

import hashlib
import json

import click
import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import compose_archive_reconstruction_authority, sn
from imas_codex.standard_names.signed_manifest import (
    _load_archive_reconstruction_authority,
    signed_payload_sha256,
)


class AdapterReached(Exception):
    """Raised where the transaction would open, to show the CLI got that far."""


class _SentinelGraph:
    def session(self):
        raise AdapterReached


def _extraction(*, review_properties: bool = True, origin: str | None = None) -> dict:
    review = {
        "relationship_type": "HAS_REVIEW",
        "direction": "outgoing",
        "counterpart_id": "review-1",
        "counterpart_labels": ["StandardNameReview"],
        "properties": {"decision": "accepted"},
    }
    if review_properties:
        review["counterpart_properties"] = {"id": "review-1", "score": 0.9}
    properties = {"id": "archived_temperature", "description": "archived"}
    if origin is not None:
        properties["origin"] = origin
    return {
        "properties": properties,
        "edges": [
            review,
            {
                "relationship_type": "HAS_UNIT",
                "direction": "outgoing",
                "counterpart_id": "unit:eV",
                "counterpart_labels": ["Unit"],
                "properties": {},
            },
            {
                "relationship_type": "HAS_PARENT",
                "direction": "outgoing",
                "counterpart_id": "archived_temperature_parent",
                "counterpart_labels": ["StandardName"],
                "properties": {},
            },
            {
                "relationship_type": "EVIDENCED_BY",
                "direction": "incoming",
                "counterpart_id": "candidate-9",
                "counterpart_labels": ["PromotionCandidate"],
                "properties": {},
            },
        ],
    }


def _digests(path) -> tuple[str, str]:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), signed_payload_sha256(json.loads(raw))


def _load(path):
    file_sha256, payload_sha256 = _digests(path)
    return _load_archive_reconstruction_authority(
        path,
        expected_file_sha256=file_sha256,
        expected_payload_sha256=payload_sha256,
    )


@pytest.fixture
def authority_file(tmp_path):
    def write(record: dict, **kwargs) -> tuple:
        kwargs.setdefault("origin", "pipeline")
        authority = compose_archive_reconstruction_authority(record, **kwargs)
        path = tmp_path / "authority.json"
        path.write_text(json.dumps(authority, sort_keys=True))
        return path, authority

    return write


def test_the_restore_group_is_registered():
    """The adapter's entry point is a committed command, not a test helper."""
    assert "restore" in sn.commands
    assert sorted(sn.commands["restore"].commands) == ["apply", "compose"]


def test_composed_authority_loads_through_the_adapter_loader(authority_file):
    """The composer produces the closed artifact the adapter's loader accepts."""
    path, authority = authority_file(_extraction())

    loaded = _load(path)

    assert [node["id"] for node in loaded.nodes] == ["archived_temperature"]
    assert loaded.nodes[0]["properties"]["origin"] == "pipeline"
    assert loaded.operation_id == "reconstruct-archived-standard-name"
    asserted = {
        (edge["relationship_type"], edge["counterpart_id"]) for edge in loaded.edges
    }
    assert asserted
    assert ("HAS_REVIEW", "review-1") in asserted
    assert ("HAS_PARENT", "archived_temperature_parent") in asserted
    review = next(e for e in loaded.edges if e["relationship_type"] == "HAS_REVIEW")
    assert review["counterpart_label"] == "StandardNameReview"
    assert [c["label"] for c in loaded.counterparts] == ["StandardNameReview"]
    assert authority["identities"] == ["archived_temperature"]


def test_composed_authority_counts_a_role_with_no_reconstruction_route(authority_file):
    """A role no edge can carry reaches the receipt as a count, not a drop."""
    path, _ = authority_file(_extraction())

    loaded = _load(path)

    roles = loaded.archive_roles["archived_temperature"]
    assert roles["EVIDENCED_BY"] == 1
    assert roles["HAS_REVIEW"] == 1
    assert all(edge["relationship_type"] != "EVIDENCED_BY" for edge in loaded.edges)


def test_composition_refuses_a_counterpart_the_restore_must_create(authority_file):
    with pytest.raises(Exception) as caught:
        compose_archive_reconstruction_authority(
            _extraction(review_properties=False), origin="pipeline"
        )
    assert "no properties for the StandardNameReview counterpart" in str(caught.value)


def test_composition_refuses_a_rekey_that_names_no_counterpart():
    with pytest.raises(Exception) as caught:
        compose_archive_reconstruction_authority(
            _extraction(),
            origin="pipeline",
            counterpart_renames={
                "etendue_of_spectrometer_channel": "etendue_of_detector"
            },
        )
    assert "names no archived StandardName counterpart" in str(caught.value)


def test_composition_rekeys_an_archived_counterpart(authority_file):
    """The rekey the driver hardcoded is now stated at the entry point."""
    path, _ = authority_file(
        _extraction(),
        origin="pipeline",
        counterpart_renames={
            "archived_temperature_parent": "temperature_parent",
        },
    )

    loaded = _load(path)

    parents = [
        edge["counterpart_id"]
        for edge in loaded.edges
        if edge["relationship_type"] == "HAS_PARENT"
    ]
    assert parents == ["temperature_parent"]


def test_composition_replaces_archived_origin_with_the_stated_origin():
    """The archive cannot supply the delete-permission field."""
    record = _extraction(origin="catalog_edit")

    authority = compose_archive_reconstruction_authority(record, origin="pipeline")

    assert record["properties"]["origin"] == "catalog_edit"
    assert authority["nodes"][0]["properties"]["origin"] == "pipeline"


def test_composition_requires_an_explicit_current_origin():
    with pytest.raises(click.UsageError, match="origin must be stated"):
        compose_archive_reconstruction_authority(_extraction(), origin=None)


def test_composition_refuses_an_unknown_origin():
    with pytest.raises(click.UsageError, match="unknown origin"):
        compose_archive_reconstruction_authority(_extraction(), origin="restored")


def test_composition_rekeys_every_edge_to_the_same_archived_counterpart():
    record = _extraction()
    record["edges"].append(
        {
            "relationship_type": "REFERENCES",
            "direction": "outgoing",
            "counterpart_id": "archived_temperature_parent",
            "counterpart_labels": ["StandardName"],
            "properties": {},
        }
    )

    authority = compose_archive_reconstruction_authority(
        record,
        origin="pipeline",
        counterpart_renames={
            "archived_temperature_parent": "temperature_parent",
        },
    )

    renamed = [
        edge["counterpart_id"]
        for edge in authority["edges"]
        if edge["relationship_type"] in {"HAS_PARENT", "REFERENCES"}
    ]
    assert renamed == ["temperature_parent", "temperature_parent"]


def test_composition_refuses_two_archived_counterparts_rekeyed_to_one_target():
    record = _extraction()
    record["edges"].append(
        {
            "relationship_type": "REFERENCES",
            "direction": "outgoing",
            "counterpart_id": "other_archived_parent",
            "counterpart_labels": ["StandardName"],
            "properties": {},
        }
    )

    with pytest.raises(click.UsageError, match="counterpart rekeys collide"):
        compose_archive_reconstruction_authority(
            record,
            origin="pipeline",
            counterpart_renames={
                "archived_temperature_parent": "temperature_parent",
                "other_archived_parent": "temperature_parent",
            },
        )


def test_composition_refuses_a_registered_role_in_an_unregistered_direction():
    record = _extraction()
    record["edges"][0]["direction"] = "incoming"

    with pytest.raises(click.UsageError, match="HAS_REVIEW/incoming"):
        compose_archive_reconstruction_authority(record, origin="pipeline")


# ---------------------------------------------------------------------------
# CLI: the committed entry point
# ---------------------------------------------------------------------------


def test_cli_compose_writes_an_authority_the_loader_accepts(tmp_path, authority_file):
    extraction = tmp_path / "extraction.json"
    extraction.write_text(json.dumps(_extraction()))
    output = tmp_path / "composed.json"

    result = CliRunner().invoke(
        sn,
        [
            "restore",
            "compose",
            str(extraction),
            "--output",
            str(output),
            "--origin",
            "pipeline",
            "--identity",
            "temperature",
            "--rename-counterpart",
            "archived_temperature_parent=temperature_parent",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "composed temperature" in result.output
    loaded = _load(output)
    assert [node["id"] for node in loaded.nodes] == ["temperature"]
    parents = [
        edge["counterpart_id"]
        for edge in loaded.edges
        if edge["relationship_type"] == "HAS_PARENT"
    ]
    assert parents == ["temperature_parent"]


def test_cli_apply_reaches_the_adapter_through_the_committed_command(
    monkeypatch, authority_file
):
    """Removing the command fails this test: nothing else raises AdapterReached."""
    path, _ = authority_file(_extraction())
    monkeypatch.setattr(
        "imas_codex.cli.sn._archive_reconstruction_graph_client",
        lambda: _SentinelGraph(),
    )

    result = CliRunner().invoke(
        sn,
        ["restore", "apply", str(path), "--reason", "restore the archived identity"],
    )

    assert isinstance(result.exception, AdapterReached)


def test_cli_apply_requires_a_manifest_digest(authority_file):
    path, _ = authority_file(_extraction())

    result = CliRunner().invoke(
        sn, ["restore", "apply", str(path), "--reason", "x", "--apply"]
    )

    assert result.exit_code != 0
    assert "--manifest-sha256" in result.output


def test_cli_apply_refuses_a_digest_without_apply(authority_file):
    path, _ = authority_file(_extraction())

    result = CliRunner().invoke(
        sn,
        [
            "restore",
            "apply",
            str(path),
            "--reason",
            "restore it",
            "--manifest-sha256",
            "0" * 64,
        ],
    )

    assert result.exit_code != 0


def test_cli_apply_refuses_a_payload_its_signature_does_not_cover(authority_file):
    """The digest path is live: an edited payload no longer matches its signature."""
    path, authority = authority_file(_extraction())
    authority["operation_id"] = "something-else"
    path.write_text(json.dumps(authority, sort_keys=True))

    result = CliRunner().invoke(
        sn, ["restore", "apply", str(path), "--reason", "restore it"]
    )

    assert result.exit_code == 2
    assert "signature does not match canonical signed payload" in result.output


def test_cli_apply_refuses_a_blank_reason(authority_file):
    path, _ = authority_file(_extraction())

    result = CliRunner().invoke(sn, ["restore", "apply", str(path), "--reason", "   "])

    assert result.exit_code != 0
    assert "--reason" in result.output


def test_cli_compose_refuses_more_than_one_extraction(tmp_path):
    extraction = tmp_path / "extraction.json"
    extraction.write_text(json.dumps([_extraction(), _extraction()]))

    result = CliRunner().invoke(
        sn,
        [
            "restore",
            "compose",
            str(extraction),
            "--output",
            str(tmp_path / "out.json"),
            "--origin",
            "pipeline",
        ],
    )

    assert result.exit_code != 0
    assert "holds 2" in result.output


def test_cli_compose_requires_an_origin(tmp_path):
    extraction = tmp_path / "extraction.json"
    extraction.write_text(json.dumps(_extraction(origin="catalog_edit")))

    result = CliRunner().invoke(
        sn,
        [
            "restore",
            "compose",
            str(extraction),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )

    assert result.exit_code == 2
    assert "Missing option '--origin'" in result.output


def test_cli_compose_refuses_an_unknown_origin(tmp_path):
    extraction = tmp_path / "extraction.json"
    extraction.write_text(json.dumps(_extraction()))

    result = CliRunner().invoke(
        sn,
        [
            "restore",
            "compose",
            str(extraction),
            "--output",
            str(tmp_path / "out.json"),
            "--origin",
            "restored",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--origin'" in result.output
