"""The catalog layout stamp is declared once, in the standard-names package.

``CATALOG_EDGE_MODEL_VERSION`` names the manifest shape the exporter writes
and the publish gate requires. Both modules must read it from
``imas_standard_names`` rather than spelling it out locally, so a future
layout change has exactly one place to update.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from imas_standard_names.models import CATALOG_EDGE_MODEL_VERSION as ISN_STAMP

import imas_codex.standard_names.export as export_module
import imas_codex.standard_names.publish as publish_module


def test_exporter_stamps_the_installed_package_value() -> None:
    assert export_module.CATALOG_EDGE_MODEL_VERSION == ISN_STAMP


def test_publish_gate_sources_the_same_value() -> None:
    assert publish_module._REQUIRED_EDGE_MODEL_VERSION == ISN_STAMP


@pytest.fixture()
def isnc_repo(tmp_path: Path) -> Path:
    isnc = tmp_path / "isnc"
    isnc.mkdir()
    subprocess.run(["git", "init"], cwd=isnc, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    (isnc / "README.md").write_text("# ISNC\n")
    subprocess.run(["git", "add", "."], cwd=isnc, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=isnc, check=True, capture_output=True
    )
    return isnc


def _staging_dir(tmp_path: Path, edge_model_version: str) -> Path:
    staging = tmp_path / "staging"
    sn_dir = staging / "standard_names"
    sn_dir.mkdir(parents=True)
    entries = [
        {
            "name": "electron_temperature",
            "description": "Te",
            "documentation": "Docs",
            "kind": "scalar",
            "unit": "eV",
            "links": [],
            "constraints": [],
            "status": "draft",
        }
    ]
    (sn_dir / "equilibrium.yml").write_text(yaml.safe_dump(entries), encoding="utf-8")

    manifest = {
        "catalog_name": "imas-standard-names-catalog",
        "cocos_convention": 17,
        "grammar_version": "0.7.0",
        "isn_model_version": "0.7.0",
        "dd_version_lineage": ["4.0.0"],
        "generated_by": "test",
        "generated_at": "2024-01-01T00:00:00Z",
        "candidate_count": 1,
        "published_count": 1,
        "excluded_below_score_count": 0,
        "excluded_unreviewed_count": 0,
        "edge_model_version": edge_model_version,
        "domains_included": ["equilibrium"],
    }
    (staging / "catalog.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return staging


def test_publish_accepts_the_installed_package_value(
    tmp_path: Path, isnc_repo: Path
) -> None:
    staging = _staging_dir(tmp_path, ISN_STAMP)
    report = publish_module.run_publish(staging, isnc_repo, dry_run=True)
    assert not report.errors, f"Errors: {report.errors}"


def test_publish_rejects_a_different_stamp(tmp_path: Path, isnc_repo: Path) -> None:
    staging = _staging_dir(tmp_path, f"{ISN_STAMP}-not-the-installed-value")
    report = publish_module.run_publish(staging, isnc_repo, dry_run=True)
    assert any("edge_model_version mismatch" in e for e in report.errors)
