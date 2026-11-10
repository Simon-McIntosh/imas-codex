"""Segment-order refusals are adopted only when the expression is unchanged."""

import pytest

from imas_codex.standard_names.grammar_adapter import (
    adopt_canonical_segment_order,
    is_canonical_name,
    parse_canonical_name,
)
from tests.standard_names.test_compose_batch_isolation import (
    _FakeBatchResult,
    _FakeCandidate,
    _FakeLLMOut,
    _make_batch_item,
    _run_batch,
    compose_deps,  # noqa: F401 — imported pytest fixture
)

# Cold fixture setup imports the grammar and provider modules before assertions.
pytestmark = pytest.mark.timeout(180)

# A flat spelling the strict parser reorders without moving any token between
# semantic roles, recorded when the composer emitted it for a DD source.
SAME_IR_FLAT_SPELLING = "radiated_energy_accumulated_due_to_impurity_radiation"
SAME_IR_CANONICAL = "accumulated_radiated_energy_due_to_impurity_radiation"

# A flat spelling whose canonical order assigns "poloidal" to a different
# segment, so the reorder denotes another quantity.
IR_CHANGING_FLAT_SPELLING = (
    "poloidal_momentum_neutral_internal_state_flux_limiter_coefficient"
)


def test_segment_order_refusal_is_adopted_when_the_ir_is_unchanged() -> None:
    assert not is_canonical_name(SAME_IR_FLAT_SPELLING)

    adopted = adopt_canonical_segment_order(SAME_IR_FLAT_SPELLING)

    assert adopted.name == SAME_IR_CANONICAL
    assert adopted.name == parse_canonical_name(adopted.name).name


def test_reorder_that_changes_the_ir_is_refused() -> None:
    with pytest.raises(ValueError, match="changes the semantic IR"):
        adopt_canonical_segment_order(IR_CHANGING_FLAT_SPELLING)


def test_non_order_grammar_failure_is_refused() -> None:
    with pytest.raises(ValueError) as refusal:
        adopt_canonical_segment_order("quux")

    assert "does not match any physical_base" in str(refusal.value)
    assert "flat segment order" not in str(refusal.value)


def test_refusal_carries_the_refused_spelling_for_the_compose_gate() -> None:
    """The pre-validation gate recovers the candidate from the parser's error."""
    with pytest.raises(ValueError) as refusal:
        parse_canonical_name(SAME_IR_FLAT_SPELLING)

    assert refusal.value.name == SAME_IR_FLAT_SPELLING
    assert refusal.value.canonical_form == SAME_IR_CANONICAL


@pytest.mark.asyncio
async def test_compose_batch_persists_the_adopted_canonical_spelling(compose_deps):
    with pytest.raises(ValueError, match="flat segment order") as refusal:
        parse_canonical_name(SAME_IR_FLAT_SPELLING)

    source_path = "radiation/time_slice/energy"
    candidate = _FakeCandidate(SAME_IR_FLAT_SPELLING, source_path, error=refusal.value)
    candidate.operators = []
    candidate.description = "Accumulated energy radiated by impurities"
    item = _make_batch_item(source_path)
    item.update(unit="J", description=candidate.description)

    async def answering(*_args, **_kwargs):
        return _FakeLLMOut(_FakeBatchResult([candidate]))

    count = await _run_batch([item], answering)

    assert count == 1
    assert [c["id"] for c in compose_deps.persisted] == [SAME_IR_CANONICAL]
    assert compose_deps.persisted[0]["source_id"] == source_path
    assert compose_deps.refusals == []
