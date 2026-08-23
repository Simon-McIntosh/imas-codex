"""Keep generated Pydantic model fields aligned with their LinkML schemas."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest
from linkml_runtime.utils.schemaview import SchemaView
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_TARGETS = (
    ("imas_codex/schemas/facility.yaml", "imas_codex.graph.models"),
    ("imas_codex/schemas/imas_dd.yaml", "imas_codex.graph.dd_models"),
    ("imas_codex/schemas/facility_config.yaml", "imas_codex.config.models"),
)

FieldOverride = Mapping[tuple[str, str], set[str]]


def _generated_model_fields(module: ModuleType) -> dict[str, set[str]]:
    return {
        name: set(value.model_fields)
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and issubclass(value, BaseModel)
        and value.__module__ == module.__name__
        and name not in {"ConfiguredBaseModel", "LinkMLMeta"}
    }


def _expected_model_fields(schema_path: Path) -> dict[str, set[str]]:
    schema = SchemaView(str(schema_path))
    return {
        class_name: {slot.name for slot in schema.class_induced_slots(class_name)}
        for class_name, class_definition in schema.all_classes(imports=True).items()
        if class_definition.class_uri != "linkml:Any"
    }


def _currency_mismatches(
    field_overrides: FieldOverride | None = None,
) -> list[str]:
    overrides = field_overrides or {}
    mismatches: list[str] = []

    for schema_relative_path, module_name in MODEL_TARGETS:
        expected = _expected_model_fields(PROJECT_ROOT / schema_relative_path)
        actual = _generated_model_fields(importlib.import_module(module_name))

        missing_models = sorted(expected.keys() - actual.keys())
        extra_models = sorted(actual.keys() - expected.keys())
        if missing_models:
            mismatches.append(
                f"{module_name} missing models: {', '.join(missing_models)}"
            )
        if extra_models:
            mismatches.append(f"{module_name} extra models: {', '.join(extra_models)}")

        for class_name in sorted(expected.keys() & actual.keys()):
            model_fields = overrides.get((module_name, class_name), actual[class_name])
            missing_attributes = sorted(expected[class_name] - model_fields)
            extra_attributes = sorted(model_fields - expected[class_name])
            if missing_attributes:
                mismatches.append(
                    f"{module_name}.{class_name} missing attributes: "
                    f"{', '.join(missing_attributes)}"
                )
            if extra_attributes:
                mismatches.append(
                    f"{module_name}.{class_name} extra attributes: "
                    f"{', '.join(extra_attributes)}"
                )

    return mismatches


def _assert_generated_models_are_current(
    field_overrides: FieldOverride | None = None,
) -> None:
    mismatches = _currency_mismatches(field_overrides)
    if mismatches:
        details = "; ".join(mismatches)
        raise AssertionError(
            "Generated model/schema mismatch: "
            f"{details}. A detached worktree may be shadowing the current editable "
            "install with stale generated files; run `uv run build-models --force`."
        )


def test_generated_models_match_linkml_schemas() -> None:
    _assert_generated_models_are_current()


def test_stale_generated_model_reports_actionable_mismatch() -> None:
    module_name = "imas_codex.graph.models"
    model_name = "RepairAuthorityRow"
    model = getattr(importlib.import_module(module_name), model_name)
    stale_fields = set(model.model_fields) - {"signatures"}

    with pytest.raises(AssertionError) as raised:
        _assert_generated_models_are_current({(module_name, model_name): stale_fields})

    message = str(raised.value)
    assert (
        "imas_codex.graph.models.RepairAuthorityRow missing attributes: signatures"
        in message
    )
    assert "detached worktree" in message
    assert "uv run build-models --force" in message
