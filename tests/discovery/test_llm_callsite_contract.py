"""Project-wide contract for registered structured LLM dispatches."""

from collections import Counter

from imas_codex.discovery.base.llm import _VALID_SERVICES, LLM_SERVICE
from imas_codex.llm.callsite_registry import (
    CALLSITE_REGISTRY,
    assert_no_provider_bypasses,
    assert_registry_current,
)


class TestLLMCallSiteContract:
    """Every production structured dispatch has one complete registration."""

    def test_registry_exactly_matches_source_dispatches(self):
        observed = assert_registry_current()

        assert len(observed) == 46
        assert len(CALLSITE_REGISTRY) == 46
        assert len({call.source.source_path for call in observed}) == 23
        assert Counter(call.dispatch_style for call in observed) == {
            "direct": 39,
            "to-thread": 5,
            "injected": 2,
        }

    def test_registered_routes_use_valid_services(self):
        for entry in CALLSITE_REGISTRY:
            assert entry.routes
            for route in entry.routes:
                assert route.service in _VALID_SERVICES
                assert route.seat
                assert route.templates

    def test_no_business_module_bypasses_structured_dispatch(self):
        assert_no_provider_bypasses()

    def test_valid_services_matches_type(self):
        expected = set(LLM_SERVICE.__args__)
        assert _VALID_SERVICES == expected
