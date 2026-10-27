"""The documentation writers persist the selected registry form."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names import workers
from imas_codex.standard_names.models import GeneratedDocs, RefinedDocs
from imas_codex.standard_names.worker_tasks import (
    worker_generate_docs as _owning_worker_generate_docs,
    worker_refine_docs as _owning_worker_refine_docs,
)

_AVERAGED = "flux_surface_averaged_argon_density_at_plasma_boundary"
_CANONICAL_AVERAGE = (
    r"$$\langle f\rangle = \oint f\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|$$"
)
_SUBSCRIPTED_AVERAGE = (
    r"$$\langle n\rangle_S = \oint n\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|$$"
)
_LINE_AVERAGE = r"$$\langle n\rangle = \oint n\,dl/B \,/\, \oint dl/B$$"


def _persisted_documentation(name: str, documentation: str, writer: str) -> str:
    """Run one writer with model, budget and graph effects stubbed."""
    graph = MagicMock()
    graph.__enter__.return_value.query.return_value = []
    manager = MagicMock()
    manager.run_id = "test-run"
    manager.reserve.return_value = MagicMock()
    item: dict[str, Any] = {
        "id": name,
        "claim_token": "claim",
        "description": "A physical quantity described for testing.",
    }
    result = (
        GeneratedDocs(description=item["description"], documentation=documentation)
        if writer == "generate"
        else RefinedDocs(
            description=item["description"], documentation=documentation, links=[]
        )
    )
    persist = MagicMock()
    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            return_value=(result, 0.0, {}),
        ),
        patch("imas_codex.graph.client.GraphClient", return_value=graph),
        patch("imas_codex.llm.prompt_loader.render_prompt", return_value="prompt"),
        patch("imas_codex.settings.get_model", return_value="test/model"),
        patch(
            "imas_codex.standard_names.example_loader.load_compose_examples",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.graph_store.graph_docs_persistence.persist_generated_docs",
            persist,
        ),
        patch(
            "imas_codex.standard_names.graph_store.graph_docs_persistence.persist_refined_docs",
            persist,
        ),
        patch.object(_owning_worker_generate_docs, "_enrich_for_docs_gen"),
        patch.object(
            _owning_worker_generate_docs, "_nearby_names_for_docs_gen", return_value=[]
        ),
        patch.object(workers, "model_provider_exposure", return_value=0.0),
    ):
        operation = (
            _owning_worker_generate_docs.process_generate_docs_batch
            if writer == "generate"
            else _owning_worker_refine_docs.process_refine_docs_batch
        )
        assert asyncio.run(operation([item], manager, asyncio.Event())) == 1
    persist.assert_called_once()
    return persist.call_args.kwargs["documentation"]


@pytest.mark.parametrize("writer", ["generate", "refine"])
def test_wrong_opening_is_replaced_with_subject_template(writer: str) -> None:
    document = "A model-written opening about argon.\n\nFurther documentation remains."

    persisted = _persisted_documentation(_AVERAGED, document, writer)

    assert persisted.splitlines()[0] == "Number density of argon"
    assert "Further documentation remains." in persisted
    assert "A model-written opening" not in persisted


@pytest.mark.parametrize("writer", ["generate", "refine"])
def test_required_registry_equation_is_written(writer: str) -> None:
    persisted = _persisted_documentation(
        "elongation_of_plasma_boundary",
        "The model describes the vertical extent of the plasma boundary.",
        writer,
    )

    assert r"$$\kappa = (Z_{\max} - Z_{\min})/(R_{\max} - R_{\min})$$" in persisted


@pytest.mark.parametrize("equation", [_SUBSCRIPTED_AVERAGE, _LINE_AVERAGE])
def test_noncanonical_optional_equation_is_replaced(equation: str) -> None:
    persisted = _persisted_documentation(
        _AVERAGED,
        f"The model describes the argon average.\n\n{equation}",
        "generate",
    )

    assert _CANONICAL_AVERAGE in persisted
    assert equation not in persisted


def test_optional_equation_is_not_added_when_absent() -> None:
    persisted = _persisted_documentation(
        _AVERAGED,
        "The model describes the argon average without an equation.",
        "generate",
    )

    assert "$$" not in persisted


@pytest.mark.parametrize("writer", ["generate", "refine"])
def test_unfillable_template_keeps_model_opening(writer: str) -> None:
    document = "The model describes electron temperature in this plasma."

    persisted = _persisted_documentation("electron_temperature", document, writer)

    assert persisted == document
    assert "{" not in persisted


@pytest.mark.parametrize("writer", ["generate", "refine"])
def test_name_without_selected_template_keeps_model_opening(writer: str) -> None:
    document = "The model describes the magnetic field in this plasma."

    persisted = _persisted_documentation("magnetic_field", document, writer)

    assert persisted == document


_NOTE_NAME = "total_thermal_electron_density"
_NOTE_OPENING = "Number density of electron"


def _selected_template_notes(name: str) -> list[str]:
    """The note sentences the grammar selects for *name*, in order."""
    from imas_standard_names.grammar.terms import (
        select_conventions,
        template_sentences,
    )

    openings = {
        template.documentation for template in select_conventions(name).templates
    }
    return [
        sentence for sentence in template_sentences(name) if sentence not in openings
    ]


@pytest.mark.parametrize("writer", ["generate", "refine"])
def test_template_note_is_written_once_after_the_opening(writer: str) -> None:
    notes = _selected_template_notes(_NOTE_NAME)
    assert notes, "the selected template must carry a note for this check"
    note = notes[0]
    document = (
        "A model-written opening about the total electron density."
        "\n\nFurther documentation remains."
    )

    persisted = _persisted_documentation(_NOTE_NAME, document, writer)

    assert persisted.count(note) == 1
    assert persisted.index(note) > persisted.index(_NOTE_OPENING)
    assert "Further documentation remains." in persisted
