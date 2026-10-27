"""The documentation hold needs DD source text before it can compare claims.

`_documentation_contradiction_note` compares authored documentation claims
against the DD text a name is bound to. A name whose sources are all derived
or manual has no DD text to read, so there is nothing to compare and nothing
to hold for. A name that does name DD text — through a ``dd`` source binding
or a ``dd:``-prefixed path — is still held when that text is missing or empty,
because the comparison then genuinely could not run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names import workers
from imas_codex.standard_names.review import pipeline
from imas_codex.standard_names.worker_tasks import (
    worker_review_docs as _owning_worker_review_docs,
)

DD_PATH = "equilibrium/time_slice/profiles_1d/psi"


def _item(**overrides) -> dict:
    item = {
        "id": "poloidal_flux_at_axis",
        "source_paths": [],
        "source_bindings": [],
    }
    item.update(overrides)
    return item


@pytest.mark.asyncio
async def test_derived_only_binding_is_not_held():
    item = _item(
        id="power_of_divertor_due_to_recombination",
        source_bindings=[
            {
                "id": "src-derived-1",
                "source_type": "derived",
                "source_id": "derived:power_of_divertor_due_to_recombination",
                "dd_path": None,
            }
        ],
    )
    assert (
        await _owning_worker_review_docs._documentation_contradiction_note(
            item, MagicMock()
        )
        is None
    )


@pytest.mark.asyncio
async def test_manual_only_binding_is_not_held():
    item = _item(
        source_bindings=[
            {
                "id": "src-manual-1",
                "source_type": "manual",
                "source_id": "MANUAL-17",
                "dd_path": None,
            }
        ],
    )
    assert (
        await _owning_worker_review_docs._documentation_contradiction_note(
            item, MagicMock()
        )
        is None
    )


@pytest.mark.asyncio
async def test_dd_binding_without_fetched_text_is_held():
    item = _item(
        source_bindings=[
            {"id": "src-dd-1", "source_type": "dd", "dd_path": f"dd:{DD_PATH}"}
        ],
    )
    note = await _owning_worker_review_docs._documentation_contradiction_note(
        item, MagicMock()
    )
    assert note == f"DD source text unavailable at {DD_PATH}"


@pytest.mark.asyncio
async def test_dd_source_path_without_binding_is_held():
    item = _item(source_paths=[f"dd:{DD_PATH}"])
    note = await _owning_worker_review_docs._documentation_contradiction_note(
        item, MagicMock()
    )
    assert note == f"DD source text unavailable at {DD_PATH}"


@pytest.mark.asyncio
async def test_facility_only_source_path_is_not_held():
    item = _item(source_paths=["tcv:ip/measured"])
    assert (
        await _owning_worker_review_docs._documentation_contradiction_note(
            item, MagicMock()
        )
        is None
    )


@pytest.mark.asyncio
async def test_normalised_dd_source_path_still_holds():
    """A prepared item keeps its DD hold through the source-path rewrite."""
    item = _item(source_paths=["dd:equilibrium/time_slice/profiles_1d/j_tor"])
    _owning_worker_review_docs._set_docs_review_dd_source_paths(item)
    note = await _owning_worker_review_docs._documentation_contradiction_note(
        item, MagicMock()
    )
    assert (
        note == "DD source text unavailable at equilibrium/time_slice/profiles_1d/j_tor"
    )


@pytest.mark.asyncio
async def test_normalised_facility_source_path_is_not_held():
    item = _item(source_paths=["tcv:ip/measured"])
    _owning_worker_review_docs._set_docs_review_dd_source_paths(item)
    assert (
        await _owning_worker_review_docs._documentation_contradiction_note(
            item, MagicMock()
        )
        is None
    )


@pytest.mark.asyncio
async def test_facility_path_with_dd_binding_holds_only_the_dd_path():
    item = _item(
        source_paths=["tcv:ip/measured"],
        source_bindings=[
            {
                "id": "src-dd",
                "source_type": "dd",
                "dd_path": "dd:equilibrium/time_slice/profiles_1d/j_tor",
            }
        ],
    )
    note = await _owning_worker_review_docs._documentation_contradiction_note(
        item, MagicMock()
    )
    assert (
        note == "DD source text unavailable at equilibrium/time_slice/profiles_1d/j_tor"
    )


@pytest.mark.asyncio
async def test_dd_prefixed_binding_source_id_is_held():
    item = _item(
        source_bindings=[
            {"id": "src-x", "source_type": "derived", "source_id": f"dd:{DD_PATH}"}
        ],
    )
    note = await _owning_worker_review_docs._documentation_contradiction_note(
        item, MagicMock()
    )
    assert note == "DD source text unavailable; claim comparison could not run"


@pytest.mark.asyncio
async def test_veto_request_holds_a_dd_source_path_with_missing_text(monkeypatch):
    """The hold survives the request builder and the re-check it feeds.

    ``prepare_docs_review_request`` normalises the veto item's source paths to
    bare DD paths and the claimed worker re-runs the note on that item, so the
    item the request hands on must still name the DD path it held for. A
    ``dd`` binding alone cannot show this: it short-circuits the source-path
    normalisation, so the name bound only by a ``dd:``-prefixed source path is
    the case that exercises it.
    """
    monkeypatch.setattr(pipeline, "_fetch_review_dd_context", lambda names: None)
    item = _item(
        id="poloidal_flux_at_axis",
        source_paths=["dd:equilibrium/time_slice/profiles_1d/j_tor"],
    )
    request = await _owning_worker_review_docs.prepare_docs_review_request(item)
    note = await _owning_worker_review_docs._documentation_contradiction_note(
        request["veto_item"], MagicMock()
    )
    assert (
        note == "DD source text unavailable at equilibrium/time_slice/profiles_1d/j_tor"
    )


@pytest.mark.asyncio
async def test_veto_request_does_not_hold_a_facility_only_item(monkeypatch):
    monkeypatch.setattr(pipeline, "_fetch_review_dd_context", lambda names: None)
    item = _item(source_paths=["tcv:ip/measured"])
    request = await _owning_worker_review_docs.prepare_docs_review_request(item)
    assert (
        await _owning_worker_review_docs._documentation_contradiction_note(
            request["veto_item"], MagicMock()
        )
        is None
    )


def test_hold_predicate_reads_only_dd_sources():
    assert not _owning_worker_review_docs._has_dd_source_reference(
        _item(source_bindings=[{"source_type": "derived", "source_id": "derived:x"}])
    )
    assert not _owning_worker_review_docs._has_dd_source_reference(
        _item(source_bindings=[{"source_type": "manual", "source_id": "MANUAL-17"}])
    )
    assert _owning_worker_review_docs._has_dd_source_reference(
        _item(source_bindings=[{"source_type": "dd", "dd_path": f"dd:{DD_PATH}"}])
    )
    assert _owning_worker_review_docs._has_dd_source_reference(
        _item(source_paths=[f"dd:{DD_PATH}"])
    )
    assert not _owning_worker_review_docs._has_dd_source_reference(_item())
    assert not _owning_worker_review_docs._has_dd_source_reference(
        _item(source_paths=["tcv:ip/measured"])
    )
