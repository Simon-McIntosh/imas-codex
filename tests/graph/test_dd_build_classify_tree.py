"""The DD build's classify phase decides paths through the two-level tree.

The DD build reaches the tree through
``classify_worker`` → ``classify_domains`` → ``classify_tier1_llm`` →
``_classify_batch_tree``. A tier-1 DD path is put to the decisions endpoint on
the ``sn-domain-tree`` seat; a path the tree returns below its configured
confidence bar is re-judged by the chat classifier. The bar comes from the
``[sn-domain-tree].threshold`` setting, which the benchmark calibrated, not
from a constant in the classifier.

Every endpoint is stubbed — the GraphClient, the decisions batch and the chat
call — so the wiring is measured without a paid call. Each test names the
mechanism it holds, so removing that mechanism turns it red rather than
leaving it green by accident.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.graph.dd_domain_classifier import (
    CATEGORY_QUESTION,
    DOMAIN_QUESTION,
    _category_options,
    _domain_options,
)
from imas_codex.graph.dd_workers import DDBuildState, classify_worker
from imas_codex.settings import get_model, get_sn_domain_tree_threshold

PATH = "core_profiles/profiles_1d/electrons/temperature"


def path_row() -> dict:
    """One ``_query_unclassified_paths`` row: a classifiable tier-1 candidate."""
    return {
        "id": PATH,
        "node_category": "quantity",
        "description": "Electron temperature",
        "units": "eV",
    }


def context_row() -> dict:
    """One ``gather_classification_context`` row: the state a decision reads."""
    return {
        "id": PATH,
        "description": "Electron temperature",
        "documentation": "Temperature of the thermal electron population.",
        "units": "eV",
        "parent_path": "core_profiles/profiles_1d/electrons",
        "parent_description": "Electron species quantities",
        "siblings": ["density", "pressure"],
        "ids_name": "core_profiles",
        "node_category": "quantity",
    }


class FakeGraphClient:
    """A GraphClient answering each query the classify phase makes.

    The classify phase issues four shapes of query: the tier-1 candidate query,
    the context gatherer, a count-returning write per tier, and the tier-1
    result write. Each is recognised by a marker in its Cypher so the phase runs
    to completion without a database, and the counters tiers 2 and 3 read come
    back as zero.
    """

    def __init__(self) -> None:
        self.queries: list[tuple[str, dict]] = []

    def __enter__(self) -> FakeGraphClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def query(self, cypher: str, **params: object) -> list[dict]:
        self.queries.append((" ".join(cypher.split()), params))
        if "node_category AS node_category" in self.g(cypher):
            return [path_row()]
        if "parent_description" in self.g(cypher):
            return [context_row()]
        if "AS cnt" in self.g(cypher) or "AS updated" in self.g(cypher):
            return [{"cnt": 0, "updated": 0}]
        return []

    @staticmethod
    def g(cypher: str) -> str:
        return " ".join(cypher.split())

    def written_items(self) -> list[dict]:
        """The tier-1 result rows handed to the graph write, in order."""
        for _cypher, params in self.queries:
            if "items" in params:
                return list(params["items"])
        return []


def tree_stub(
    *, category: str, domain: str, probability: float, cost_per_state: float = 0.005
) -> AsyncMock:
    """A decisions endpoint answering one path at a chosen two-level probability.

    The first level is answered with certainty and the second at *probability*,
    so the product the tree records — and routes on — is *probability*. The
    answers are keyed under their question name, the shape the endpoint uses.
    """
    stub = AsyncMock()
    stub.calls = []

    async def _call(states, questions, **kwargs):
        stub.calls.append(kwargs)
        if CATEGORY_QUESTION in questions:
            answer = {
                CATEGORY_QUESTION: {
                    "choice": category,
                    "probabilities": {category: 1.0},
                }
            }
        else:
            answer = {
                DOMAIN_QUESTION: {
                    "choice": domain,
                    "probabilities": {domain: probability},
                }
            }
        results = [(answer, cost_per_state) for _ in states]
        return results, cost_per_state * len(states)

    stub.side_effect = _call
    return stub


def chat_stub(reply: str = "transport", cost: float = 0.02) -> AsyncMock:
    """A chat classifier that answers every path handed to it, recording them."""
    stub = AsyncMock()
    stub.batches = []

    async def _call(batch, *, model, service, valid_domains):
        stub.batches.append([b["id"] for b in batch])
        results = [
            {
                "id": b["id"],
                "physics_domain": reply,
                "domain_source": "llm_classified",
                "domain_model": model,
                "cost": cost,
            }
            for b in batch
        ]
        return results, cost

    stub.side_effect = _call
    return stub


async def drive(decide: AsyncMock, chat: AsyncMock) -> FakeGraphClient:
    """Drive ``classify_worker`` with the GraphClient and both LLM endpoints stubbed."""
    fake = FakeGraphClient()
    state = DDBuildState(facility="dd")
    with patch("imas_codex.graph.client.GraphClient", return_value=fake):
        with patch("imas_codex.graph.dd_domain_classifier.decide_batch", new=decide):
            with patch(
                "imas_codex.graph.dd_domain_classifier._classify_batch", new=chat
            ):
                await classify_worker(state)
    return fake


def first_category_domain() -> tuple[str, str]:
    """A real (category, domain) pair from the vocabulary the tree offers."""
    category = sorted(_category_options())[0]
    domain = sorted(_domain_options(category))[0]
    return category, domain


def test_the_configured_threshold_is_read_from_the_seat_section() -> None:
    """The bar the classifier routes on is the seat's setting, not a code constant."""
    assert get_sn_domain_tree_threshold() == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_tier_one_paths_reach_the_tree_on_its_configured_seat() -> None:
    """A tier-1 DD path is put to the decisions endpoint on the tree's seat.

    The wiring is what this holds: the build's classify phase, not merely a
    direct call to the tree, reaches the decisions endpoint, and it names the
    ``sn-domain-tree`` seat there. A drop of the tree from the classify phase,
    or a seat swapped for the chat classifier's, parts here.
    """
    category, domain = first_category_domain()
    decide = tree_stub(category=category, domain=domain, probability=0.95)
    chat = chat_stub()

    await drive(decide, chat)

    assert decide.calls, "the tier-1 path never reached the decisions endpoint"
    seats = {call["model"] for call in decide.calls}
    assert seats == {get_model("sn-domain-tree")}, (
        f"the tree was not reached on its configured seat: {seats}"
    )


@pytest.mark.asyncio
async def test_a_path_below_the_configured_threshold_reaches_chat() -> None:
    """A path the tree scores below the bar is re-judged by the chat classifier.

    The configured bar is 0.9, so a two-level probability of 0.85 belongs to
    the chat classifier; the domain it replies carries into the written result.
    """
    category, domain = first_category_domain()
    decide = tree_stub(category=category, domain=domain, probability=0.85)
    chat = chat_stub(reply="transport")

    fake = await drive(decide, chat)

    assert chat.batches == [[PATH]], (
        "a path below the configured threshold did not reach the chat classifier"
    )
    items = fake.written_items()
    assert items and items[0]["physics_domain"] == "transport"


@pytest.mark.asyncio
async def test_a_path_above_the_configured_threshold_does_not_reach_chat() -> None:
    """A path the tree scores at or above the bar keeps the tree's decision.

    A two-level probability of 0.95 clears the 0.9 bar, so the path is written
    from the tree and the chat classifier is never called for it.
    """
    category, domain = first_category_domain()
    decide = tree_stub(category=category, domain=domain, probability=0.95)
    chat = chat_stub()

    fake = await drive(decide, chat)

    assert chat.batches == [], "a confident path was sent to chat anyway"
    items = fake.written_items()
    assert items and items[0]["domain_source"] == "jev_tree", (
        f"the tree's confident decision was not the one written: {items}"
    )
    assert items[0]["physics_domain"] == domain


@pytest.mark.asyncio
async def test_the_threshold_setting_decides_the_routing(monkeypatch) -> None:
    """Changing the setting alone moves a path across the routing boundary.

    The same 0.85 two-level probability is below the configured 0.9 and reaches
    chat; lowering the setting to 0.8 puts the same path on the tree. The
    routing follows the setting rather than any constant, which is what a
    threshold held in code could not do.
    """
    category, domain = first_category_domain()

    below = tree_stub(category=category, domain=domain, probability=0.85)
    chat_below = chat_stub()
    await drive(below, chat_below)
    assert chat_below.batches == [[PATH]], (
        "0.85 is below the configured 0.9 but did not reach chat"
    )

    monkeypatch.setattr("imas_codex.settings.get_sn_domain_tree_threshold", lambda: 0.8)
    above = tree_stub(category=category, domain=domain, probability=0.85)
    chat_above = chat_stub()
    fake = await drive(above, chat_above)
    assert chat_above.batches == [], (
        "lowering the setting to 0.8 did not keep the 0.85 path on the tree"
    )
    items = fake.written_items()
    assert items and items[0]["domain_source"] == "jev_tree"
