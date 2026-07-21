"""Tests for the sn-sources manifest loader used by ``sn run --focus <file>``."""

from __future__ import annotations

import textwrap

import pytest

from imas_codex.standard_names.sources_manifest import (
    SourcesManifestError,
    expand_focus_tokens,
    is_sources_file,
    load_sources_file,
)

VALID = textwrap.dedent(
    """
    schema_version: 1
    name: test-batch
    sources:
      magnetics:
        - ip
        - flux_loop/flux
      equilibrium:
        - time_slice/global_quantities/ip
    """
)


def _write(tmp_path, text, name="m.yaml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_flattens_to_ids_paths(tmp_path):
    p = _write(tmp_path, VALID)
    assert load_sources_file(p) == [
        "magnetics/ip",
        "magnetics/flux_loop/flux",
        "equilibrium/time_slice/global_quantities/ip",
    ]


def test_missing_file_raises(tmp_path):
    with pytest.raises(SourcesManifestError, match="not found"):
        load_sources_file(tmp_path / "nope.yaml")


def test_bad_schema_version_raises(tmp_path):
    p = _write(tmp_path, VALID.replace("schema_version: 1", "schema_version: 2"))
    with pytest.raises(SourcesManifestError):
        load_sources_file(p)


def test_missing_sources_raises(tmp_path):
    p = _write(tmp_path, "schema_version: 1\nname: x\n")
    with pytest.raises(SourcesManifestError):
        load_sources_file(p)


def test_sources_wrong_shape_raises(tmp_path):
    p = _write(tmp_path, "schema_version: 1\nname: x\nsources: [a, b]\n")
    with pytest.raises(SourcesManifestError):
        load_sources_file(p)


def test_not_yaml_mapping_raises(tmp_path):
    p = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(SourcesManifestError):
        load_sources_file(p)


def test_is_sources_file(tmp_path):
    p = _write(tmp_path, VALID)
    assert is_sources_file(str(p)) is True
    assert is_sources_file("magnetics/ip") is False


def test_expand_mixes_files_and_bare_paths(tmp_path):
    p = _write(tmp_path, VALID)
    got = expand_focus_tokens(["core_profiles/profiles_1d/zeff", str(p), "magnetics/ip"])
    assert got == [
        "core_profiles/profiles_1d/zeff",
        "magnetics/ip",
        "magnetics/flux_loop/flux",
        "equilibrium/time_slice/global_quantities/ip",
    ]


def test_expand_bare_paths_unchanged():
    toks = ["magnetics/ip", "equilibrium/time_slice/boundary/elongation"]
    assert expand_focus_tokens(toks) == toks


def test_expand_propagates_manifest_error(tmp_path):
    bad = _write(tmp_path, "schema_version: 9\nsources: {}\n", name="bad.yaml")
    with pytest.raises(SourcesManifestError):
        expand_focus_tokens([str(bad)])
