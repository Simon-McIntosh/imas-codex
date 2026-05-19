"""Regression tests pinning order-preserving dedup + self-link drop
on the two write paths that feed `StandardName.links` in the graph.

Background: the ISNC SQLite catalog enforces UNIQUE(name, link).
`plasma_stored_energy` shipped to rc6 with THREE `name:plasma_stored_energy`
self-links because:

* `_sanitize_links` (enrich_workers) had a dedup check only on its
  bare-id branch, not on its `name:`-prefixed branch — so an LLM that
  echoed `["name:foo", "name:foo", "name:foo"]` passed straight through.
* `resolve_doc_links` (graph_ops) rebuilt the links property from a
  naive list comprehension over every regex hit in the documentation
  — so a prose with three inline `(name:foo)` references produced
  three entries.

Both write the resulting list onto `sn.links`, which `export.py`
reads verbatim. The fixes:

1. dedup the `name:` / URL branch of `_sanitize_links`
2. dedup the regex-rebuilt list in `resolve_doc_links`
3. accept an optional `self_name` so an entry's own id is dropped from
   the structured cross-reference index — a self-loop there is
   semantically meaningless and used to trip the UNIQUE constraint.
"""

from __future__ import annotations

from imas_codex.standard_names.enrich_workers import _sanitize_links
from imas_codex.standard_names.graph_ops import _extract_links_from_docs


class TestSanitizeLinksDedup:
    def test_duplicate_name_prefixed_collapse(self):
        # Regression: rc6 shipped `plasma_stored_energy` with three of
        # these because the prefix branch skipped the seen-check.
        out = _sanitize_links(
            [
                "name:plasma_stored_energy",
                "name:plasma_stored_energy",
                "name:plasma_stored_energy",
                "name:beta",
            ]
        )
        assert out == ["name:plasma_stored_energy", "name:beta"]

    def test_duplicate_url_collapse(self):
        out = _sanitize_links(
            ["https://example.com", "https://example.com", "http://other.test"]
        )
        assert out == ["https://example.com", "http://other.test"]

    def test_duplicate_bare_id_already_collapsed(self):
        # Pre-existing behaviour preserved.
        out = _sanitize_links(["foo_bar", "foo_bar", "baz"])
        assert out == ["name:foo_bar", "name:baz"]

    def test_mixed_prefix_and_bare_dedup(self):
        out = _sanitize_links(["name:foo", "foo", "name:foo", "bar"])
        assert out == ["name:foo", "name:bar"]

    def test_self_name_dropped(self):
        out = _sanitize_links(
            ["name:plasma_stored_energy", "name:beta"],
            self_name="plasma_stored_energy",
        )
        assert out == ["name:beta"]

    def test_self_name_dropped_from_bare(self):
        out = _sanitize_links(
            ["plasma_stored_energy", "beta"],
            self_name="plasma_stored_energy",
        )
        assert out == ["name:beta"]

    def test_valid_names_filter_still_applies(self):
        out = _sanitize_links(
            ["name:known", "name:unknown", "name:known"],
            valid_names={"known"},
        )
        assert out == ["name:known"]

    def test_empty_and_none(self):
        assert _sanitize_links(None) == []
        assert _sanitize_links([]) == []

    def test_order_preserved(self):
        out = _sanitize_links(["name:c", "name:a", "name:b", "name:a", "name:c"])
        assert out == ["name:c", "name:a", "name:b"]


class TestExtractLinksFromDocs:
    def test_self_link_dropped(self):
        # Direct reproduction of the rc6 plasma_stored_energy data: the
        # entry's documentation references itself three times across
        # three distinct paragraphs (the source documentation actually
        # uses three different anchor labels — "stored plasma energy",
        # "MHD plasma stored energy", "diamagnetic plasma stored energy"
        # — but all pointing at the same target).
        docs = (
            "[stored plasma energy](name:plasma_stored_energy) and "
            "[MHD plasma stored energy](name:plasma_stored_energy) "
            "differ from [diamagnetic plasma stored energy]"
            "(name:plasma_stored_energy). See also "
            "[beta](name:beta)."
        )
        links = _extract_links_from_docs(docs, self_name="plasma_stored_energy")
        assert links == ["name:beta"]

    def test_self_link_kept_without_self_name(self):
        # When `self_name` is not provided the function preserves the
        # pre-existing dedup-only behaviour: 3 self-references collapse
        # to 1 entry but are not removed.
        docs = (
            "[x](name:plasma_stored_energy) [y](name:plasma_stored_energy) "
            "[z](name:plasma_stored_energy)"
        )
        links = _extract_links_from_docs(docs)
        assert links == ["name:plasma_stored_energy"]

    def test_dedup_across_paragraphs(self):
        docs = (
            "para one mentions [t](name:triangularity).\n\n"
            "para two also mentions [t](name:triangularity)."
        )
        links = _extract_links_from_docs(docs, self_name="upper_triangularity")
        assert links == ["name:triangularity"]

    def test_mixed_name_and_dd_dedup(self):
        docs = (
            "see [t](name:foo) and [u](name:foo) and "
            "[v](dd:core_profiles/x) and [w](dd:core_profiles/x)"
        )
        links = _extract_links_from_docs(docs)
        assert links == ["name:foo", "dd:core_profiles/x"]

    def test_empty(self):
        assert _extract_links_from_docs(None) is None
        assert _extract_links_from_docs("") is None
        assert _extract_links_from_docs("plain prose no links") is None
