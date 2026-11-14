"""Guards the refine worker's own prompt context against a silent regression.

The refinement prompt can only show the settled shared wording if
``process_refine_docs_batch`` puts it in the context it hands the renderer.
A test that builds its own context, or that replaces ``render_prompt`` with a
constant, stays green when those lines are removed, so this one drives the
worker's own context-building path and inspects what reaches the renderer.

No LLM call and no graph read: the budget reserve returns no lease, so the
worker stops once it has rendered its prompts.
"""

from __future__ import annotations

import asyncio

_SPECIES_SENTENCE = (
    "The density counts ions summed over all charge states; neutral atoms are excluded."
)


class _Conventions:
    """Duck-typed conventions carrying no registry entry.

    A truthy object rather than a dict: ``fixed_shared_parts`` validates a dict
    into the conventions model, while the species population part needs only a
    conventions object to be present.
    """

    concepts: list = []
    notation: list = []
    templates: list = []


class _NoLeaseBudget:
    """A budget that always reports itself exhausted."""

    run_id = "test-run"

    def reserve(self, *_args, **_kwargs):
        return None


class _NoGraph:
    def __enter__(self):
        raise RuntimeError("no graph in this test")

    def __exit__(self, *_exc):
        return False


def _dependencies(name: str):
    """The entrypoint's dependency resolver, stubbed to pure functions."""

    if name == "_documentation_with_registry_form":
        return lambda _sn_id, documentation, _conventions: documentation
    if name == "_is_refine_docs_failure":
        return lambda _exc: False
    if name == "bind_attempt_exposure":
        return lambda *_a, **_k: None
    if name == "charge_billable_exception":
        return lambda *_a, **_k: None
    if name == "model_provider_exposure":
        return lambda *_a, **_k: 0.01
    if name == "normalize_description_text":
        return lambda text, *a, **k: text
    raise AssertionError(f"unexpected worker dependency: {name}")


def test_refine_worker_puts_settled_shared_wording_in_the_prompt(monkeypatch):
    """A species density refinement is shown the settled population part.

    Reverting the ``shared_parts`` assignment in ``worker_refine_docs`` leaves
    the rendered context without the key, so the prompt can only re-guess
    wording the persistence gate refuses.
    """
    import imas_codex.standard_names.context as sn_context
    from imas_codex.graph import client as graph_client
    from imas_codex.llm import prompt_loader
    from imas_codex.standard_names.worker_tasks import worker_refine_docs

    monkeypatch.setattr(
        sn_context, "conventions_context_for", lambda _sn_id: _Conventions()
    )
    monkeypatch.setattr(sn_context, "fetch_sibling_family", lambda _sn_id: None)
    monkeypatch.setattr(sn_context, "locus_context_for", lambda _sn_id: None)
    monkeypatch.setattr(graph_client, "GraphClient", _NoGraph)

    recorded: list[dict] = []
    real_render = prompt_loader.render_prompt

    def _record(name, context=None):
        recorded.append(dict(context or {}))
        return real_render(name, context)

    monkeypatch.setattr(prompt_loader, "render_prompt", _record)

    asyncio.run(
        worker_refine_docs.process_refine_docs_batch(
            _dependencies,
            batch=[
                {
                    "id": "volume_averaged_oxygen_density",
                    "claim_token": "t",
                    "description": "Number density of oxygen ions.",
                    "documentation": "The oxygen ion density.",
                    "kind": "scalar",
                    "unit": "m^-3",
                    "physics_domain": "core_profiles",
                    "docs_chain_length": 1,
                }
            ],
            mgr=_NoLeaseBudget(),
            stop_event=asyncio.Event(),
        )
    )

    assert recorded, "the worker rendered no prompt"
    user_context = recorded[-1]
    parts = user_context.get("shared_parts")
    assert parts, "the refine context carried no settled shared wording"
    assert [p["anchor"] for p in parts] == ["population:species"]
    assert parts[0]["sentence"] == _SPECIES_SENTENCE
