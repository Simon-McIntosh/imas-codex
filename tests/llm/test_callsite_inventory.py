"""Focused tests for the executable structured-call inventory."""

from pathlib import Path

import pytest

from imas_codex.llm.callsite_registry import (
    CALLSITE_REGISTRY,
    CallsiteInventoryError,
    CallsiteRegistration,
    CallsiteSourceSyntaxError,
    RouteBinding,
    SourceCallIdentity,
    assert_registry_current,
    assert_zero_legacy_dispatches,
    get_route_binding,
    scan_provider_bypasses,
    scan_structured_calls,
)


def _write_source(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def test_scanner_recognizes_direct_threaded_and_injected_dispatches(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/carrier.py",
        """
import asyncio

def direct():
    call_llm_structured(
        model=model,
        messages=[],
        response_model=DirectResponse,
        service="data-dictionary",
    )

async def threaded():
    await asyncio.to_thread(
        call_llm_structured,
        model=model,
        messages=[],
        response_model=ThreadedResponse,
        service="facility-discovery",
    )

async def injected(acall_fn):
    await acall_fn(
        model=model,
        messages=[],
        response_model=InjectedResponse,
        service="standard-names",
    )
""",
    )

    calls = scan_structured_calls(tmp_path)

    assert [call.dispatch_style for call in calls] == [
        "direct",
        "to-thread",
        "injected",
    ]
    assert [call.source.scope for call in calls] == ["direct", "threaded", "injected"]
    assert [call.service_argument for call in calls] == [
        "'data-dictionary'",
        "'facility-discovery'",
        "'standard-names'",
    ]
    assert [call.response_model_argument for call in calls] == [
        "DirectResponse",
        "ThreadedResponse",
        "InjectedResponse",
    ]


def test_scanner_uses_line_independent_source_identity(tmp_path):
    source = """
def dispatch():
    call_llm_structured(
        model=model,
        messages=[],
        response_model=Response,
        service="data-dictionary",
    )
"""
    _write_source(tmp_path, "imas_codex/carrier.py", source)
    first = scan_structured_calls(tmp_path)[0]
    _write_source(tmp_path, "imas_codex/carrier.py", "\n\n" + source)
    second = scan_structured_calls(tmp_path)[0]

    assert first.source == second.source
    assert first.line != second.line


def test_scanner_recognizes_typed_sync_and_async_dispatches(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/carrier.py",
        """
def sync_call(envelope):
    return dispatch_context(envelope, "example.sync", route_id="sync-route")

async def async_call(envelope):
    return await adispatch_context(
        envelope, callsite_id="example.async", route_id="async-route"
    )
""",
    )

    calls = scan_structured_calls(tmp_path)

    assert [call.dispatch_style for call in calls] == ["typed-sync", "typed-async"]
    assert [call.transition_kind for call in calls] == ["typed", "typed"]
    assert [call.callsite_id for call in calls] == ["example.sync", "example.async"]
    assert [call.route_id for call in calls] == ["sync-route", "async-route"]


def test_scanner_resolves_import_assignment_attribute_and_thread_aliases(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/carrier.py",
        """
from asyncio import to_thread as run_thread
from imas_codex.discovery.base.llm import call_llm_structured as invoke
import imas_codex.llm.context_dispatch as typed

assigned = invoke

async def threaded_alias():
    await run_thread(
        assigned,
        model=model,
        messages=[],
        response_model=Response,
        service="data-dictionary",
    )

async def typed_attribute(envelope):
    return await typed.adispatch_context(
        envelope, "example.typed", route_id="typed-route"
    )
""",
    )

    calls = scan_structured_calls(tmp_path)

    assert [call.dispatch_style for call in calls] == ["to-thread", "typed-async"]
    assert [call.transition_kind for call in calls] == ["legacy", "typed"]


def test_scanner_closes_attribute_thread_and_injected_typed_aliases(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/carrier.py",
        """
import asyncio
from imas_codex.llm.context_dispatch import adispatch_context

holder.invoke = adispatch_context

async def attributed(envelope):
    return await holder.invoke(
        envelope, "example.attributed", route_id="attributed-route"
    )

async def threaded(envelope):
    return await asyncio.to_thread(
        adispatch_context,
        envelope,
        "example.threaded",
        route_id="threaded-route",
    )

async def injected(invoke_async, envelope):
    return await invoke_async(
        envelope,
        callsite_id="example.injected",
        route_id="injected-route",
    )

class Carrier:
    def __init__(self):
        self.invoke = adispatch_context

    async def dispatch(self, envelope):
        return await self.invoke(
            envelope, "example.instance", route_id="instance-route"
        )
""",
    )
    injected_registration = (
        CallsiteRegistration(
            callsite_id="example.injected",
            source=SourceCallIdentity(
                "imas_codex/carrier.py", "injected", "invoke_async"
            ),
            dispatch_style="injected",
            service_argument="'data-dictionary'",
            response_model_symbol="Response",
            reachability="active",
            routes=(
                RouteBinding(
                    route_id="injected-route",
                    service="data-dictionary",
                    seat="language",
                    model_source="section:language",
                    templates=("example/system",),
                    asset_mode="legacy-template",
                ),
            ),
        ),
    )

    calls = scan_structured_calls(tmp_path, registry=injected_registration)

    assert [call.transition_kind for call in calls] == [
        "typed",
        "typed",
        "typed",
        "typed",
    ]
    assert [call.route_id for call in calls] == [
        "attributed-route",
        "threaded-route",
        "injected-route",
        "instance-route",
    ]


def test_scanner_only_accepts_explicitly_registered_injected_parameter(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/carrier.py",
        """
async def injected(invoke_async):
    return await invoke_async(
        model=model,
        messages=[],
        response_model=Response,
        service="data-dictionary",
    )
""",
    )
    registration = (
        CallsiteRegistration(
            callsite_id="example.injected",
            source=SourceCallIdentity(
                "imas_codex/carrier.py", "injected", "invoke_async"
            ),
            dispatch_style="injected",
            service_argument="'data-dictionary'",
            response_model_symbol="Response",
            reachability="active",
            routes=(
                RouteBinding(
                    route_id="language",
                    service="data-dictionary",
                    seat="language",
                    model_source="section:language",
                    templates=("example/system",),
                    asset_mode="legacy-template",
                ),
            ),
        ),
    )

    calls = scan_structured_calls(tmp_path, registry=registration)

    assert len(calls) == 1
    assert calls[0].dispatch_style == "injected"
    assert calls[0].source.dispatch_symbol == "invoke_async"


def test_unregistered_dispatch_fails_loudly(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/carrier.py",
        """
def dispatch():
    call_llm_structured(
        model=model,
        messages=[],
        response_model=Response,
        service="data-dictionary",
    )
""",
    )

    with pytest.raises(CallsiteInventoryError, match="unregistered dispatch"):
        assert_registry_current(tmp_path, registry=())


def test_renamed_public_raw_message_wrapper_fails_inventory(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/carrier.py",
        """
from imas_codex.discovery.base.llm import call_llm

def public_completion(messages):
    return call_llm(model="example", messages=messages)
""",
    )

    with pytest.raises(CallsiteInventoryError, match="unregistered dispatch"):
        assert_registry_current(tmp_path, registry=())


def test_syntax_errors_fail_the_inventory_scan(tmp_path):
    _write_source(tmp_path, "imas_codex/broken.py", "def broken(:\n    pass\n")

    with pytest.raises(CallsiteSourceSyntaxError, match="cannot parse"):
        scan_structured_calls(tmp_path)


def test_raw_provider_dispatch_is_reported_outside_transport(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/business.py",
        """
def dispatch():
    return litellm.completion(model="example", messages=[])
""",
    )

    bypasses = scan_provider_bypasses(tmp_path)

    assert len(bypasses) == 1
    assert bypasses[0].source_path == "imas_codex/business.py"
    assert bypasses[0].symbol == "litellm.completion"


def test_assigned_raw_provider_alias_is_reported(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/business.py",
        """
from litellm import acompletion

send = acompletion
holder.send = acompletion

async def dispatch():
    await send(model="example", messages=[])
    return await holder.send(model="example", messages=[])
""",
    )

    bypasses = scan_provider_bypasses(tmp_path)

    assert len(bypasses) == 2
    assert {bypass.symbol for bypass in bypasses} == {"litellm.acompletion"}


def test_raw_provider_transport_whitelist_is_scope_exact(tmp_path):
    _write_source(
        tmp_path,
        "imas_codex/discovery/base/llm.py",
        """
import litellm

def renamed_transport():
    return litellm.completion(model="example", messages=[])
""",
    )

    bypasses = scan_provider_bypasses(tmp_path)

    assert len(bypasses) == 1
    assert bypasses[0].scope == "renamed_transport"


def test_registry_binds_response_models_and_complete_routes():
    observed = {call.source: call for call in scan_structured_calls()}

    assert len(CALLSITE_REGISTRY) == len(observed) == 46
    assert sum(len(entry.routes) for entry in CALLSITE_REGISTRY) == 51
    for entry in CALLSITE_REGISTRY:
        call = observed[entry.source]
        if entry.response_model_symbol == "caller-supplied":
            assert call.response_model_argument is None
        else:
            assert call.response_model_argument == entry.response_model_symbol
        assert entry.reachability in {"active", "active-public"}
        for route in entry.routes:
            assert route.service
            assert route.seat
            assert route.route_id
            assert route.model_source
            assert all(template for template in route.templates)
            assert route.asset_mode in {"legacy-template", "legacy-inline"}


def test_legacy_route_matching_is_exact_and_inline_assets_are_explicit() -> None:
    route = get_route_binding(
        "dd.cluster-labeling",
        route_id="dd-enrichment",
    )
    assert route.asset_mode == "legacy-template"

    with pytest.raises(ValueError, match="does not identify one"):
        get_route_binding(
            "dd.cluster-labeling",
            route_id="missing-route",
        )
    assert any(
        route.asset_mode == "legacy-inline"
        for entry in CALLSITE_REGISTRY
        for route in entry.routes
    )


def test_registry_accepts_exact_legacy_to_typed_expression_transition(
    tmp_path, monkeypatch
) -> None:
    _write_source(
        tmp_path,
        "imas_codex/carrier.py",
        """
def dispatch(envelope):
    return dispatch_context(envelope, "example.typed", route_id="language")
""",
    )
    registry = (
        CallsiteRegistration(
            callsite_id="example.typed",
            source=SourceCallIdentity(
                "imas_codex/carrier.py", "dispatch", "call_llm_structured"
            ),
            dispatch_style="direct",
            service_argument="'data-dictionary'",
            response_model_symbol="Response",
            reachability="active",
            routes=(
                RouteBinding(
                    route_id="language",
                    service="data-dictionary",
                    seat="language",
                    model_source="section:language",
                    templates=("example/system",),
                    asset_mode="legacy-template",
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "imas_codex.llm.dispatch_policy_registry.policy_registry_closure",
        lambda observed, registry: (0, 1),
    )
    observed = assert_registry_current(
        tmp_path,
        registry,
        typed_policy_registry={("example.typed", "language"): object()},
    )

    assert len(observed) == 1
    assert observed[0].transition_kind == "typed"


def test_final_closure_rejects_current_legacy_expressions() -> None:
    with pytest.raises(CallsiteInventoryError, match="zero legacy expressions"):
        assert_zero_legacy_dispatches()


def test_final_closure_reports_all_public_raw_message_surfaces() -> None:
    with pytest.raises(
        CallsiteInventoryError, match="public raw-message wrapper"
    ) as caught:
        assert_zero_legacy_dispatches()

    message = str(caught.value)
    assert "call_llm" in message
    assert "acall_llm" in message
    assert "call_llm_structured" in message
    assert "acall_llm_structured" in message
