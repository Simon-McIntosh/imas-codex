"""Schema parity for frozen review artifacts consumed by approval."""

from pathlib import Path

import pytest
import yaml

from imas_codex.standard_names.catalog_release import _freeze_review_artifact
from imas_codex.standard_names.sources_manifest import (
    SourcesManifestError,
    load_names_file,
)


def test_frozen_artifact_with_source_accounting_loads_for_approval(
    tmp_path: Path,
) -> None:
    manifest_sources = [
        {
            "source_path": "summary/global_quantities/ip/value",
            "source_status": "attached",
            "standard_name_id": "plasma_current",
            "terminal_stage": "accepted",
            "non_nameable_reason": "",
        },
        {
            "source_path": "summary/plasma_duration/value",
            "source_status": "failed",
            "standard_name_id": None,
            "terminal_stage": None,
            "non_nameable_reason": "",
        },
    ]

    artifact = _freeze_review_artifact(
        tmp_path,
        rc_version="v0.4.0rc1+west-review",
        names=["plasma_current"],
        minted_from="west_sources.yaml",
        unmatched=["summary/plasma_duration/value"],
        manifest_sources=manifest_sources,
        batch_label="west-review",
    )

    assert load_names_file(artifact) == ["plasma_current"]
    document = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    assert document["kind"] == "sn_names"
    assert document["schema_version"] == 1
    assert document["name"] == "review-v0-4-0rc1-west-review"
    assert document["names"] == ["plasma_current"]
    assert document["manifest_sources"] == manifest_sources


def test_frozen_artifact_rejects_incomplete_source_accounting(tmp_path: Path) -> None:
    incomplete_source = {
        "source_path": "summary/global_quantities/ip/value",
        "source_status": "attached",
        "standard_name_id": "plasma_current",
        "terminal_stage": "accepted",
    }

    with pytest.raises(SourcesManifestError, match="non_nameable_reason"):
        _freeze_review_artifact(
            tmp_path,
            rc_version="v0.4.0rc1+west-review",
            names=["plasma_current"],
            minted_from="west_sources.yaml",
            unmatched=[],
            manifest_sources=[incomplete_source],
            batch_label="west-review",
        )
