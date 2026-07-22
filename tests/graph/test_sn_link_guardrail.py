"""Guardrail: a DD horizon update never loses a StandardName provenance link.

Two layers:

- Pure logic (no graph): :func:`check_sn_links_safe` flags a decreased edge
  count or a non-zero dangling count, and passes an all-non-decreasing pair.
- Live graph (``-m graph``): the never-delete invariant holds on the current
  graph — zero dangling StandardName sources (a source that names a DD path
  but no longer resolves to an IMASNode). This is the standing acceptance gate
  the ``imas dd update`` command enforces before and after every build.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.sn_link_guardrail import (
    SNLinkCounts,
    capture_sn_link_counts,
    check_sn_links_safe,
)


def _counts(from_dd=100, has_name=40, gap=20, dangling=0) -> SNLinkCounts:
    return SNLinkCounts(
        from_dd_path=from_dd,
        has_standard_name=has_name,
        vocab_gap=gap,
        dangling=dangling,
    )


class TestCheckSNLinksSafe:
    """Pure before/after comparison — no graph required."""

    def test_unchanged_is_safe(self) -> None:
        before = _counts()
        assert check_sn_links_safe(before, before) == []

    def test_growth_is_safe(self) -> None:
        before = _counts(from_dd=100, has_name=40, gap=20)
        after = _counts(from_dd=120, has_name=45, gap=22)
        assert check_sn_links_safe(before, after) == []

    def test_dropped_from_dd_path_is_flagged(self) -> None:
        before = _counts(from_dd=100)
        after = _counts(from_dd=90)
        violations = check_sn_links_safe(before, after)
        assert len(violations) == 1
        assert "from_dd_path decreased" in violations[0]
        assert "10 links lost" in violations[0]

    def test_dropped_has_standard_name_is_flagged(self) -> None:
        before = _counts(has_name=40)
        after = _counts(has_name=39)
        violations = check_sn_links_safe(before, after)
        assert any("has_standard_name decreased" in v for v in violations)

    def test_dropped_vocab_gap_is_flagged(self) -> None:
        before = _counts(gap=20)
        after = _counts(gap=0)
        violations = check_sn_links_safe(before, after)
        assert any("vocab_gap decreased" in v for v in violations)

    def test_dangling_after_is_flagged_even_if_counts_grow(self) -> None:
        before = _counts(dangling=0)
        after = _counts(from_dd=200, has_name=80, gap=40, dangling=3)
        violations = check_sn_links_safe(before, after)
        assert any("dangling" in v for v in violations)

    def test_multiple_violations_all_reported(self) -> None:
        before = _counts(from_dd=100, has_name=40, gap=20, dangling=0)
        after = _counts(from_dd=90, has_name=30, gap=20, dangling=5)
        violations = check_sn_links_safe(before, after)
        # from_dd_path drop + has_standard_name drop + dangling
        assert len(violations) == 3


@pytest.mark.graph
class TestLiveInvariant:
    """The never-delete invariant on the current graph."""

    def test_no_dangling_sn_links(self, graph_client) -> None:
        """Every SN source that names a DD path resolves to an IMASNode.

        This is the invariant the removed delete-paths (``--reset-to
        extracted``, ``--force``, ``imas dd clear``) could break and that
        ``imas dd update`` now enforces. Once the delete-paths are gone no
        code path can strand a link, so this must always pass.
        """
        counts = capture_sn_link_counts(graph_client)
        if counts.from_dd_path == 0:
            pytest.skip("no StandardName provenance links in graph yet")
        assert counts.dangling == 0, (
            f"{counts.dangling} StandardName source(s) name a DD path but no "
            "longer resolve to an IMASNode — the never-delete invariant is broken"
        )
