"""Assert every code claim and has-work predicate honours path prefixes.

A scoped code run names a set of source trees with ``--path-prefix``. Only the
scan claim honoured that list, so the triage, enrichment, scoring and
ingestion stages still claimed files anywhere in the facility and the has-work
predicates reported pending work for files outside the named trees. These
tests seed one claimable CodeFile inside a prefix and one outside for each
stage and assert the claim takes only the inside file, and that each has-work
predicate ignores work outside the prefix.

The stub GraphClient evaluates the seeded rows against the predicates the
query text actually carries, so a claim that drops the prefix clause from its
Cypher returns the outside file and fails.
"""

from unittest.mock import patch

INSIDE = "/analysis/src/SAeqread"
OUTSIDE = "/analysis/src/other"
GRAPH_OPS = "imas_codex.discovery.code.graph_ops.GraphClient"
GRAPH_GRAPH = "imas_codex.graph.GraphClient"
FACILITY = "jt-60sa"


def _code_file(path, status, **extra):
    row = {
        "id": path,
        "facility_id": FACILITY,
        "status": status,
        "language": "fortran",
    }
    row.update(extra)
    row["path"] = path
    return row


def _relevance(row):
    """The file's relevance: the largest of the four scope probabilities."""
    return max(
        float(row.get(key) or 0.0)
        for key in (
            "relevance_loads",
            "relevance_processes",
            "relevance_describes",
            "relevance_imas",
        )
    )


class StubGraphClient:
    """Evaluate seeded rows against the predicates the query itself carries."""

    def __init__(self, code_files):
        self.code_files = code_files

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        if "SET sf.claimed_at" in cypher:
            for row in self._match(cypher, kwargs):
                row["claim_token"] = kwargs["token"]
            return []
        if "{claim_token: $token}" in cypher:
            token = kwargs["token"]
            return [r for r in self.code_files if r.get("claim_token") == token]
        if "count(sf) > 0 AS has_work" in cypher:
            return [{"has_work": bool(self.code_files and self._match(cypher, kwargs))}]
        return []

    def _match(self, cypher, kwargs):
        rows = [r for r in self.code_files if r["facility_id"] == kwargs["facility"]]
        if "sf.relevance_stage IS NULL" in cypher:
            rows = [
                r
                for r in rows
                if r["status"] == "discovered" and r.get("relevance_stage") is None
            ]
        if "coalesce(sf.is_enriched, false) = false" in cypher:
            rows = [
                r
                for r in rows
                if r["status"] == "triaged"
                and _relevance(r) >= kwargs["min_relevance"]
                and not r.get("is_enriched", False)
            ]
        if "sf.is_enriched = true" in cypher:
            rows = [
                r for r in rows if r["status"] == "triaged" and r.get("is_enriched")
            ]
        if "sf.status = 'scored'" in cypher:
            rows = [
                r
                for r in rows
                if r["status"] == "scored"
                and _relevance(r) >= kwargs["min_relevance"]
                and r.get("line_count", 0) <= kwargs["max_line_count"]
            ]
        if "sf.status = 'ingested'" in cypher:
            rows = [
                r
                for r in rows
                if r["status"] == "ingested" and not r.get("evidence_linked", False)
            ]
        if "STARTS WITH prefix" in cypher:
            prefixes = kwargs["prefixes"]
            rows = [
                r for r in rows if any(str(r["path"]).startswith(p) for p in prefixes)
            ]
        return rows


def _call(fn, target, rows, **kwargs):
    with patch(target, return_value=StubGraphClient(rows)):
        return fn(FACILITY, **kwargs)


class TestClaimsHonourPrefix:
    def test_triage_claim_ignores_files_outside_the_prefix(self):
        from imas_codex.discovery.code.graph_ops import claim_files_for_triage

        rows = [
            _code_file(INSIDE + "/a.f", "discovered"),
            _code_file(OUTSIDE + "/b.f", "discovered"),
        ]
        files = _call(
            claim_files_for_triage, GRAPH_OPS, rows, limit=10, path_prefixes=[INSIDE]
        )
        assert [f["path"] for f in files] == [INSIDE + "/a.f"]

    def test_triage_claim_unscoped_takes_every_file(self):
        from imas_codex.discovery.code.graph_ops import claim_files_for_triage

        rows = [
            _code_file(INSIDE + "/a.f", "discovered"),
            _code_file(OUTSIDE + "/b.f", "discovered"),
        ]
        files = _call(claim_files_for_triage, GRAPH_OPS, rows, limit=10)
        assert {f["path"] for f in files} == {INSIDE + "/a.f", OUTSIDE + "/b.f"}

    def test_enrich_claim_ignores_files_outside_the_prefix(self):
        from imas_codex.discovery.code.graph_ops import claim_files_for_enrichment

        rows = [
            _code_file(INSIDE + "/a.f", "triaged", relevance_loads=0.8),
            _code_file(OUTSIDE + "/b.f", "triaged", relevance_loads=0.8),
        ]
        files = _call(
            claim_files_for_enrichment,
            GRAPH_OPS,
            rows,
            limit=10,
            min_relevance=0.5,
            path_prefixes=[INSIDE],
        )
        assert [f["path"] for f in files] == [INSIDE + "/a.f"]

    def test_score_claim_ignores_files_outside_the_prefix(self):
        from imas_codex.discovery.code.graph_ops import claim_files_for_scoring

        rows = [
            _code_file(INSIDE + "/a.f", "triaged", is_enriched=True),
            _code_file(OUTSIDE + "/b.f", "triaged", is_enriched=True),
        ]
        files = _call(
            claim_files_for_scoring, GRAPH_OPS, rows, limit=10, path_prefixes=[INSIDE]
        )
        assert [f["path"] for f in files] == [INSIDE + "/a.f"]

    def test_ingest_claim_ignores_files_outside_the_prefix(self):
        from imas_codex.discovery.code.workers import _claim_code_files_for_ingestion

        rows = [
            _code_file(INSIDE + "/a.f", "scored", relevance_loads=0.9, line_count=5),
            _code_file(OUTSIDE + "/b.f", "scored", relevance_loads=0.9, line_count=5),
        ]
        files = _call(
            _claim_code_files_for_ingestion,
            GRAPH_GRAPH,
            rows,
            limit=10,
            min_relevance=0.5,
            path_prefixes=[INSIDE],
        )
        assert [f["path"] for f in files] == [INSIDE + "/a.f"]


class TestHasWorkPredicatesHonourPrefix:
    def test_triage_predicate_ignores_outside_work(self):
        from imas_codex.discovery.code.graph_ops import has_pending_triage_work

        rows = [_code_file(OUTSIDE + "/b.f", "discovered")]
        assert (
            _call(has_pending_triage_work, GRAPH_OPS, rows, path_prefixes=[INSIDE])
            is False
        )
        assert _call(has_pending_triage_work, GRAPH_OPS, rows) is True

    def test_enrich_predicate_ignores_outside_work(self):
        from imas_codex.discovery.code.graph_ops import has_pending_enrich_work

        rows = [_code_file(OUTSIDE + "/b.f", "triaged", relevance_loads=0.8)]
        assert (
            _call(
                has_pending_enrich_work,
                GRAPH_OPS,
                rows,
                min_relevance=0.5,
                path_prefixes=[INSIDE],
            )
            is False
        )
        assert (
            _call(has_pending_enrich_work, GRAPH_OPS, rows, min_relevance=0.5)
            is True
        )

    def test_score_predicate_ignores_outside_work(self):
        from imas_codex.discovery.code.graph_ops import has_pending_score_work

        rows = [_code_file(OUTSIDE + "/b.f", "triaged", is_enriched=True)]
        assert (
            _call(has_pending_score_work, GRAPH_OPS, rows, path_prefixes=[INSIDE])
            is False
        )
        assert _call(has_pending_score_work, GRAPH_OPS, rows) is True

    def test_code_predicate_ignores_outside_work(self):
        from imas_codex.discovery.code.graph_ops import has_pending_code_work

        rows = [_code_file(OUTSIDE + "/b.f", "scored", relevance_loads=0.95)]
        assert (
            _call(
                has_pending_code_work,
                GRAPH_OPS,
                rows,
                min_relevance=0.5,
                path_prefixes=[INSIDE],
            )
            is False
        )
        assert _call(has_pending_code_work, GRAPH_OPS, rows, min_relevance=0.5) is True

    def test_link_predicate_ignores_outside_work(self):
        from imas_codex.discovery.code.graph_ops import has_pending_link_work

        rows = [_code_file(OUTSIDE + "/b.f", "ingested")]
        assert (
            _call(has_pending_link_work, GRAPH_OPS, rows, path_prefixes=[INSIDE])
            is False
        )
        assert _call(has_pending_link_work, GRAPH_OPS, rows) is True
