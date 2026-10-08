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


def _composite(row):
    """The file's stored relevance, written by both decision arms."""
    return float(row.get("score_composite") or 0.0)


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
        if (
            "{claim_token: $token}" in cypher
            or "WHERE sf.claim_token = $token" in cypher
        ):
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
                and _composite(r) >= kwargs["min_relevance"]
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
                and _composite(r) >= kwargs["min_relevance"]
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


class StaleJudgmentGraph(StubGraphClient):
    """Evaluate model freshness and claim priority as the query states them."""

    def query(self, cypher, **kwargs):
        if "SET sf.claimed_at" in cypher:
            rows = self._match(cypher, kwargs)
            if "CASE WHEN" in cypher:
                first_status = (
                    "triaged"
                    if "sf.relevance_stage = 'content'" in cypher
                    else "discovered"
                )
                rows.sort(key=lambda row: row["status"] != first_status)
            for row in rows[: kwargs.get("batch_size", kwargs.get("limit"))]:
                row["claim_token"] = kwargs["token"]
            return []
        return super().query(cypher, **kwargs)

    def _match(self, cypher, kwargs):
        rows = [
            row for row in self.code_files if row["facility_id"] == kwargs["facility"]
        ]
        if "sf.relevance_stage = 'content'" in cypher:
            fresh = [
                row
                for row in rows
                if row["status"] == "triaged" and row.get("is_enriched")
            ]
            stale = [
                row
                for row in rows
                if row.get("relevance_stage") == "content"
                and row["status"] in {"scored", "skipped", "ingested"}
                and row.get("preview_text")
            ]
        else:
            fresh = [
                row
                for row in rows
                if row["status"] == "discovered" and row.get("relevance_stage") is None
            ]
            stale = (
                [
                    row
                    for row in rows
                    if row.get("relevance_stage") == "name"
                    and row["status"] in {"triaged", "skipped"}
                ]
                if "sf.relevance_stage = 'name'" in cypher
                else []
            )
        if "coalesce(sf.relevance_model, '') <> $judgment_model" in cypher:
            stale = [
                row
                for row in stale
                if row.get("relevance_model", "") != kwargs["judgment_model"]
            ]
        if "STARTS WITH prefix" in cypher:
            prefixes = kwargs["prefixes"]
            fresh = [
                row for row in fresh if any(row["path"].startswith(p) for p in prefixes)
            ]
            stale = [
                row for row in stale if any(row["path"].startswith(p) for p in prefixes)
            ]
        return fresh + stale


def test_stale_name_claim_checks_model_and_claims_unjudged_first(monkeypatch):
    """A current judgment cannot consume a slot ahead of an old one."""
    from imas_codex.discovery.code.graph_ops import claim_files_for_triage

    current = _code_file(
        INSIDE + "/current.f", "skipped", relevance_stage="name", relevance_model="seat"
    )
    stale = _code_file(
        INSIDE + "/stale.f", "skipped", relevance_stage="name", relevance_model="older"
    )
    fresh = _code_file(INSIDE + "/fresh.f", "discovered")
    outside = _code_file(OUTSIDE + "/outside.f", "discovered")
    graph = StaleJudgmentGraph([current, stale, fresh, outside])
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.GraphClient", lambda: graph
    )
    monkeypatch.setattr("imas_codex.settings.get_model", lambda _: "seat")

    claimed = claim_files_for_triage(FACILITY, limit=2, path_prefixes=[INSIDE])

    assert {row["path"] for row in claimed} == {fresh["path"], stale["path"]}
    assert current.get("claim_token") is None
    assert outside.get("claim_token") is None


def test_stale_content_claim_and_pending_check_share_model_selection(monkeypatch):
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_scoring,
        has_pending_score_work,
    )

    current = _code_file(
        INSIDE + "/current.f",
        "scored",
        relevance_stage="content",
        relevance_model="seat",
        preview_text="stored",
    )
    stale = _code_file(
        INSIDE + "/stale.f",
        "ingested",
        relevance_stage="content",
        relevance_model="older",
        preview_text="stored",
    )
    fresh = _code_file(INSIDE + "/fresh.f", "triaged", is_enriched=True)
    graph = StaleJudgmentGraph([current, stale, fresh])
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.GraphClient", lambda: graph
    )
    monkeypatch.setattr("imas_codex.settings.get_model", lambda _: "seat")

    assert has_pending_score_work(FACILITY, path_prefixes=[INSIDE]) is True
    claimed = claim_files_for_scoring(FACILITY, limit=2, path_prefixes=[INSIDE])
    assert {row["path"] for row in claimed} == {fresh["path"], stale["path"]}
    assert current.get("claim_token") is None
    fresh["status"] = "scored"
    fresh["relevance_stage"] = "content"
    fresh["relevance_model"] = "seat"
    fresh["preview_text"] = "stored"
    stale["relevance_model"] = "seat"
    assert has_pending_score_work(FACILITY, path_prefixes=[INSIDE]) is False


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
            _code_file(INSIDE + "/a.f", "triaged", score_composite=0.8),
            _code_file(OUTSIDE + "/b.f", "triaged", score_composite=0.8),
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
            _code_file(INSIDE + "/a.f", "scored", score_composite=0.9, line_count=5),
            _code_file(OUTSIDE + "/b.f", "scored", score_composite=0.9, line_count=5),
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

        rows = [_code_file(OUTSIDE + "/b.f", "triaged", score_composite=0.8)]
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
            _call(has_pending_enrich_work, GRAPH_OPS, rows, min_relevance=0.5) is True
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

        rows = [_code_file(OUTSIDE + "/b.f", "scored", score_composite=0.95)]
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
