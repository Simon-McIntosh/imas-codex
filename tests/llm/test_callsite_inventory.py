"""Focused tests for the executable structured-call inventory."""

from pathlib import Path

import pytest

from imas_codex.llm.callsite_registry import (
    CALLSITE_REGISTRY,
    CallsiteInventoryError,
    CallsiteSourceSyntaxError,
    assert_registry_current,
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


def test_registry_binds_response_models_and_complete_routes():
    observed = {call.source: call for call in scan_structured_calls()}

    assert len(CALLSITE_REGISTRY) == len(observed) == 46
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
            assert all(template for template in route.templates)
