"""The domain seeder pins new DD sources without re-pinning existing ones.

A ``StandardNameSource``'s ``dd_version`` is an immutable snapshot pin: the
source is read against the metadata of the version it was captured at, so a pin
behind the currently-configured DD is ordinary state, not staleness. Only a
genuinely-new source takes the current version, and it does so through the
batch-level ``default_dd_version`` — stamping a version onto every source dict
instead asserts one for re-seeds too, and ``_pin_dd_source_snapshots`` then
raises on the disagreement, aborting the whole run before any pool starts.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names import loop as loop_mod


class _Batch:
    def __init__(self, group_key: str, paths: list[str], dd_version: str | None):
        self.group_key = group_key
        self.dd_version = dd_version
        self.items = [{"path": p, "description": f"doc for {p}"} for p in paths]


@pytest.fixture
def seed_capture(monkeypatch):
    """Stub the extract + merge legs and capture what the seeder passes down."""
    calls: list[dict] = []

    def _merge(sources, *, force=False, default_dd_version=None):
        calls.append(
            {
                "sources": sources,
                "force": force,
                "default_dd_version": default_dd_version,
            }
        )
        return len(sources)

    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.merge_standard_name_sources", _merge
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.get_existing_standard_names",
        lambda *a, **k: set(),
    )
    return calls


def _stub_extract(monkeypatch, batches):
    monkeypatch.setattr(
        "imas_codex.standard_names.sources.dd.extract_dd_candidates",
        lambda **kwargs: batches,
    )


@pytest.mark.asyncio
async def test_current_version_travels_as_the_batch_default(monkeypatch, seed_capture):
    _stub_extract(monkeypatch, [_Batch("equilibrium", ["eq/a", "eq/b"], "4.1.1")])

    written = await loop_mod._seed_domain_sources("equilibrium", source="dd")

    assert written == 2
    assert seed_capture[0]["default_dd_version"] == "4.1.1"


@pytest.mark.asyncio
async def test_no_source_carries_a_per_source_version(monkeypatch, seed_capture):
    """The re-seed contract: a per-source version would collide with a stored pin.

    This is the regression. Stamping the current version onto every source made
    an existing source pinned at an earlier DD raise ``is pinned to ... not
    ...``, which killed the extract leg before any pool ran.
    """
    _stub_extract(monkeypatch, [_Batch("equilibrium", ["eq/a", "eq/b"], "4.1.1")])

    await loop_mod._seed_domain_sources("equilibrium", source="dd")

    for src in seed_capture[0]["sources"]:
        assert "dd_version" not in src, (
            f"seeded source {src.get('id')!r} must not assert a version — "
            "an existing source keeps its own immutable pin"
        )


@pytest.mark.asyncio
async def test_seeded_sources_keep_their_identity_and_status(monkeypatch, seed_capture):
    _stub_extract(monkeypatch, [_Batch("equilibrium", ["eq/a"], "4.1.1")])

    await loop_mod._seed_domain_sources("equilibrium", source="dd")

    (src,) = seed_capture[0]["sources"]
    assert src["id"] == "dd:eq/a"
    assert src["source_type"] == "dd"
    assert src["dd_path"] == "eq/a"
    assert src["status"] == "extracted"


@pytest.mark.asyncio
async def test_version_is_taken_from_the_batches_not_a_setting(
    monkeypatch, seed_capture
):
    """The pin records the version actually extracted at, whatever that is."""
    _stub_extract(monkeypatch, [_Batch("equilibrium", ["eq/a"], "4.0.0")])

    await loop_mod._seed_domain_sources("equilibrium", source="dd")

    assert seed_capture[0]["default_dd_version"] == "4.0.0"


@pytest.mark.asyncio
async def test_missing_batch_version_is_passed_as_none(monkeypatch, seed_capture):
    """No version to declare → None, so the never-infer-latest guard still fires.

    The seeder must not substitute a guess; refusing downstream is correct.
    """
    _stub_extract(monkeypatch, [_Batch("equilibrium", ["eq/a"], None)])

    await loop_mod._seed_domain_sources("equilibrium", source="dd")

    assert seed_capture[0]["default_dd_version"] is None


@pytest.mark.asyncio
async def test_max_sources_cap_still_passes_the_default(monkeypatch, seed_capture):
    _stub_extract(
        monkeypatch, [_Batch("equilibrium", ["eq/c", "eq/a", "eq/b"], "4.1.1")]
    )

    written = await loop_mod._seed_domain_sources(
        "equilibrium", source="dd", max_sources=2
    )

    assert written == 2
    assert seed_capture[0]["default_dd_version"] == "4.1.1"
    # Capped deterministically by path, so a repeat run seeds the same subset.
    assert [s["source_id"] for s in seed_capture[0]["sources"]] == ["eq/a", "eq/b"]


@pytest.mark.asyncio
async def test_non_dd_source_does_not_seed(monkeypatch, seed_capture):
    _stub_extract(monkeypatch, [_Batch("equilibrium", ["eq/a"], "4.1.1")])

    assert await loop_mod._seed_domain_sources("equilibrium", source="signals") == 0
    assert seed_capture == []
