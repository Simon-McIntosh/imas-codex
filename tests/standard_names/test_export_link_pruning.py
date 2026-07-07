"""Item H: dangling internal doc links are pruned at export.

The published ISNC catalog carried hundreds of ``links`` entries
(``name:<target>`` scheme) whose targets are not in the published set —
renamed, dropped below score, unreviewed, or rejected by ISN validation
after gate time. Export now runs a link-resolution pass over the final
published set: internal links to unpublished targets are dropped and
counted; external http(s) links are never touched. arguments[]/
error_variants[] are expected to resolve fully and are surfaced loudly if
they don't (but left in place).
"""

from __future__ import annotations

from imas_codex.standard_names.export import (
    ExportReport,
    _internal_link_target,
    _prune_dangling_links,
    _unresolved_computed_refs,
)


class TestInternalLinkTarget:
    def test_http_is_external(self) -> None:
        assert _internal_link_target("http://example.com/x") is None
        assert _internal_link_target("https://example.com/x") is None

    def test_name_scheme_target(self) -> None:
        assert _internal_link_target("name:electron_temperature") == (
            "electron_temperature"
        )

    def test_bare_token_target(self) -> None:
        assert _internal_link_target("electron_temperature") == "electron_temperature"


class TestPruneDanglingLinks:
    def test_unpublished_internal_link_dropped(self) -> None:
        domain_entries = {
            "core_plasma_physics": [
                {
                    "name": "electron_temperature",
                    "links": [
                        "name:ion_temperature",  # published -> keep
                        "name:removed_name",  # unpublished -> prune
                        "https://imas.iter.org/doc",  # external -> keep
                    ],
                }
            ]
        }
        published = {"electron_temperature", "ion_temperature"}
        pruned, examples = _prune_dangling_links(domain_entries, published)

        assert pruned == 1
        entry_links = domain_entries["core_plasma_physics"][0]["links"]
        assert "name:ion_temperature" in entry_links
        assert "https://imas.iter.org/doc" in entry_links
        assert "name:removed_name" not in entry_links
        assert examples == ["electron_temperature -> name:removed_name"]

    def test_all_resolvable_links_are_untouched(self) -> None:
        domain_entries = {
            "d": [{"name": "a", "links": ["name:b", "https://x.example"]}]
        }
        pruned, _ = _prune_dangling_links(domain_entries, {"a", "b"})
        assert pruned == 0
        assert domain_entries["d"][0]["links"] == ["name:b", "https://x.example"]

    def test_entry_without_links_is_safe(self) -> None:
        domain_entries = {"d": [{"name": "a"}, {"name": "b", "links": []}]}
        pruned, examples = _prune_dangling_links(domain_entries, {"a", "b"})
        assert pruned == 0
        assert examples == []

    def test_example_cap_at_twenty(self) -> None:
        entries = [{"name": f"n{i}", "links": [f"name:missing{i}"]} for i in range(30)]
        domain_entries = {"d": entries}
        pruned, examples = _prune_dangling_links(domain_entries, set())
        assert pruned == 30
        assert len(examples) == 20


class TestUnresolvedComputedRefs:
    def test_all_resolve(self) -> None:
        domain_entries = {
            "d": [
                {
                    "name": "flux",
                    "arguments": [{"name": "radius"}],
                    "error_variants": {"absolute": "flux_error"},
                }
            ]
        }
        published = {"flux", "radius", "flux_error"}
        assert _unresolved_computed_refs(domain_entries, published) == []

    def test_unresolved_argument_and_error_variant_flagged(self) -> None:
        domain_entries = {
            "d": [
                {
                    "name": "flux",
                    "arguments": [{"name": "gone"}],
                    "error_variants": {"absolute": "also_gone"},
                }
            ]
        }
        unresolved = _unresolved_computed_refs(domain_entries, {"flux"})
        assert "flux: argument -> gone" in unresolved
        assert "flux: error_variant -> also_gone" in unresolved


class TestReportShape:
    def test_pruned_links_in_counts(self) -> None:
        counts = ExportReport().to_dict()["counts"]
        assert counts["pruned_links"] == 0
