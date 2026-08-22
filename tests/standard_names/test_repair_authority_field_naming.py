"""Regression: the generated models must not shadow a pydantic base attribute.

``RepairAuthorityArtifact`` originally declared a ``schema`` slot, which
collides with ``ConfiguredBaseModel``'s inherited (deprecated) ``schema``
attribute and made pydantic emit a ``UserWarning`` at class-definition time
(that is, at import). The slot is renamed to ``schema_id``; the wire format
(committed JSON files, loader dicts) still uses the key ``schema``, adapted
at the loader/builder boundary via ``authority_artifact_wire_projection``.
"""

from __future__ import annotations

import subprocess
import sys

_SHADOW_WARNING_TEXT = "shadows an attribute in parent"


def test_generated_models_import_without_field_shadow_warning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import warnings; warnings.simplefilter('always'); "
            "import imas_codex.graph.models",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert _SHADOW_WARNING_TEXT not in result.stderr
    assert _SHADOW_WARNING_TEXT not in result.stdout


def test_repair_authority_artifact_schema_field_is_renamed() -> None:
    from imas_codex.graph.models import RepairAuthorityArtifact

    assert "schema_id" in RepairAuthorityArtifact.model_fields
    assert "schema" not in RepairAuthorityArtifact.model_fields


def test_repair_authority_artifact_schema_field_is_not_an_identifier() -> None:
    from imas_codex.graph.models import RepairAuthorityArtifact

    field = RepairAuthorityArtifact.model_fields["schema_id"]
    extra = field.json_schema_extra or {}
    linkml_meta = extra.get("linkml_meta", {}) if isinstance(extra, dict) else {}
    assert not linkml_meta.get("identifier")
