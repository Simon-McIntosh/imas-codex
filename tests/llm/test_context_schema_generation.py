"""Generation contract for provider-independent prompt context models."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_models import _harden_prompt_context_models

_PROJECT_ROOT = Path(__file__).parents[2]


def test_context_schema_is_dedicated_and_not_imported_by_graph_schemas() -> None:
    schema_path = _PROJECT_ROOT / "imas_codex" / "schemas" / "llm_context.yaml"
    assert schema_path.exists()
    assert "PromptEnvelope:" in schema_path.read_text()
    assert "PromptAttachment:" in schema_path.read_text()
    assert "BatchComparatorReceipt:" in schema_path.read_text()
    assert "TelemetryState:" in schema_path.read_text()
    assert "credential_source_identity:" in schema_path.read_text()
    assert "route_id:" in schema_path.read_text()
    assert "pricing_contract_digest:" in schema_path.read_text()
    assert "pricing_provider_identity:" in schema_path.read_text()
    assert "pricing_provider_selector:" in schema_path.read_text()
    assert "attempt_count_state:" in schema_path.read_text()
    assert "response_count_state:" in schema_path.read_text()
    assert "billability_state:" in schema_path.read_text()

    for name in ("facility.yaml", "imas_dd.yaml", "standard_name.yaml"):
        assert (
            "llm_context"
            not in (_PROJECT_ROOT / "imas_codex" / "schemas" / name).read_text()
        )


def test_generated_context_output_is_ignored() -> None:
    ignore_text = (_PROJECT_ROOT / ".gitignore").read_text()
    assert "imas_codex/llm/context_models.py" in ignore_text.splitlines()


def test_build_hook_requires_context_schema_and_generated_output() -> None:
    # The build hook needs hatchling, which only the build/test extras install;
    # importing it at module scope would fail collection for the whole repository.
    hatch_build_hooks = pytest.importorskip("hatch_build_hooks")

    source = Path(hatch_build_hooks.__file__).read_text()
    assert 'schemas_dir / "llm_context.yaml"' in source
    assert 'package_root / "imas_codex" / "llm" / "context_models.py"' in source


def test_context_model_postprocessing_is_frozen_and_strict() -> None:
    generated = """class ConfiguredBaseModel:\n    model_config = ConfigDict(\n        validate_assignment = True,\n        strict = False,\n    )\n"""

    hardened = _harden_prompt_context_models(generated)

    assert "frozen = True" in hardened
    assert "strict = True" in hardened
    assert "strict = False" not in hardened
