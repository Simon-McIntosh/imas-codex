"""Item F: exclusion accounting closes in the export report.

candidate_count - published_count must reconcile against the sum of the
exclusion buckets. Previously grammar-parse failures and ISN validation
rejections were in no bucket, and the deterministic-parent placeholder
exclusions were mis-filed under excluded_below_score. Dedicated counters
(excluded_placeholder, parse_failures, validation_failures) now live in
``.export_report.json`` (ExportReport.to_dict).

The manifest (catalog.yml) intentionally carries only ISN-model fields:
StandardNameCatalogManifest is extra='forbid', so adding codex-only counters
there would break publish / the downstream ISNC catalog validation. The full
reconcilable accounting therefore lives in the report, not the manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.defaults import (  # noqa: E402
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
)
from imas_codex.standard_names.export import (  # noqa: E402
    ExportReport,
    _run_gate_c,
    _write_manifest,
)

# Codex-only accounting keys must NOT leak into the ISN-model manifest.
_CODEX_ONLY_KEYS = {
    "excluded_placeholder_count",
    "parse_failure_count",
    "validation_failure_count",
    "excluded_placeholder",
    "parse_failures",
}


class TestManifestStaysIsnClean:
    def test_manifest_has_no_codex_only_keys(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            cocos_convention=17,
            candidate_count=10,
            published_count=6,
            excluded_below_score_count=1,
            excluded_unreviewed_count=1,
            min_score_applied=0.65,
            min_description_score_applied=None,
            include_unreviewed=False,
            source_commit_sha="0123456789abcdef0123456789abcdef01234567",
            export_scope="full",
            domains_included=["equilibrium"],
        )
        m = yaml.safe_load((tmp_path / "catalog.yml").read_text())
        leaked = _CODEX_ONLY_KEYS & set(m)
        assert not leaked, f"codex-only accounting keys leaked into manifest: {leaked}"
        # ISN fields still present / normalised.
        assert m["catalog_name"] == "imas-standard-names-catalog"
        assert m["generated_at"]


class TestReportAccountingReconciles:
    def test_report_buckets_close(self) -> None:
        """candidate - published == sum of the report's exclusion buckets."""
        r = ExportReport()
        r.total_candidates = 10
        r.exported_count = 6
        r.excluded_below_score = 1
        r.excluded_unreviewed = 1
        r.excluded_placeholder = 1
        r.parse_failures = 1
        r.validation_failures = 0
        counts = r.to_dict()["counts"]
        buckets = (
            counts["excluded_below_score"]
            + counts["excluded_unreviewed"]
            + counts["excluded_placeholder"]
            + counts["parse_failures"]
            + counts["validation_failures"]
        )
        assert counts["total_candidates"] - counts["exported"] == buckets

    def test_report_counts_include_new_buckets(self) -> None:
        counts = ExportReport().to_dict()["counts"]
        # L1: domain filtering happens in the query, so this is always 0.
        assert counts["excluded_by_domain"] == 0
        assert counts["excluded_placeholder"] == 0
        assert counts["parse_failures"] == 0
        assert counts["validation_failures"] == 0


class TestGateCPlaceholderCounting:
    def test_placeholder_not_counted_as_below_score(self) -> None:
        candidates = [
            {
                "id": "pending_parent",
                "reviewer_score_name": 0.95,
                "description": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
            }
        ]
        gate, filtered, below, unrev = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=True,
            min_description_score=None,
        )
        assert filtered == [], "placeholder candidate must be excluded"
        assert below == 0, "placeholder is not a score exclusion"
        assert unrev == 0
        assert any(
            i["type"] == "deterministic_parent_description_placeholder"
            for i in gate.issues
        )
