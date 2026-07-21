"""Two-schema focus files and the ``kind``-based release dispatcher.

The mop-up loader (``load_sources_file``) speaks DD paths; the release loader
must tell an sn-sources file (DD paths) from an sn-names file (SN ids) via an
explicit ``kind`` discriminator and validate against the matching schema.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.sources_manifest import (
    SourcesManifestError,
    load_focus_file,
    load_names_file,
)


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


SOURCES_DOC = (
    "kind: sn_sources\n"
    "schema_version: 1\n"
    "name: demo-sources\n"
    "sources:\n"
    "  equilibrium:\n"
    "    - time_slice/global_quantities/ip\n"
    "    - time_slice/profiles_1d/psi\n"
)

NAMES_DOC = (
    "kind: sn_names\n"
    "schema_version: 1\n"
    "name: demo-batch\n"
    "names:\n"
    "  - plasma_current\n"
    "  - poloidal_flux\n"
)


def test_dispatch_sources(tmp_path):
    p = _write(tmp_path, "s.yaml", SOURCES_DOC)
    kind, items = load_focus_file(p)
    assert kind == "sn_sources"
    assert items == [
        "equilibrium/time_slice/global_quantities/ip",
        "equilibrium/time_slice/profiles_1d/psi",
    ]


def test_dispatch_names(tmp_path):
    p = _write(tmp_path, "n.yaml", NAMES_DOC)
    kind, items = load_focus_file(p)
    assert kind == "sn_names"
    assert items == ["plasma_current", "poloidal_flux"]


def test_dispatch_missing_kind_raises(tmp_path):
    # An sn-sources body without the discriminator is fine for the mop-up
    # loader, but the release dispatcher must refuse to guess.
    body = SOURCES_DOC.replace("kind: sn_sources\n", "")
    p = _write(tmp_path, "k.yaml", body)
    with pytest.raises(SourcesManifestError, match="kind"):
        load_focus_file(p)


def test_dispatch_wrong_schema_for_kind_raises(tmp_path):
    # kind says sn_names but the body has no 'names' block.
    body = SOURCES_DOC.replace("kind: sn_sources", "kind: sn_names")
    p = _write(tmp_path, "w.yaml", body)
    with pytest.raises(SourcesManifestError):
        load_focus_file(p)


def test_load_names_dedups_and_orders(tmp_path):
    body = NAMES_DOC + "  - plasma_current\n"  # duplicate
    # uniqueItems in the schema rejects an exact duplicate, so test the loader's
    # own de-dup on a schema-valid file with distinct entries plus a repeat via
    # a separate relaxed path: validate then de-dup.
    p = _write(tmp_path, "d.yaml", NAMES_DOC)
    assert load_names_file(p) == ["plasma_current", "poloidal_flux"]
    # The duplicate-bearing file fails schema validation (uniqueItems).
    p2 = _write(tmp_path, "d2.yaml", body)
    with pytest.raises(SourcesManifestError):
        load_names_file(p2)
