"""Assert every path/code claim and has-work predicate skips excluded prefixes.

A facility's exclusion prefixes name subtrees that must never be scanned or
ingested. Seeding enforces them at scan time, but rows already in the graph
under an excluded prefix were still claimed and still counted as pending work.
These tests seed one claimable row under an excluded prefix and one outside for
every claim and every has-work predicate, and assert only the outside row is
claimed or counted.

The stub GraphClient evaluates the seeded rows against the prefix predicates
the query itself carries, read from the named parameters the caller bound. A
claim that omits the exclusion clause binds no ``excluded_prefixes``, so the
excluded row passes through and the assertion fails.
"""

import importlib
from unittest.mock import patch

import pytest

from imas_codex.config.discovery_config import ExclusionConfig

FACILITY = "jt-60sa"
EXCLUDED_PREFIX = "/work/scratch/jt60sa"
OUTSIDE_PREFIX = "/analysis/src/SAeqread"
SCOPE_PREFIX = "/analysis/src"

DOC_CFG = "imas_codex.config.discovery_config.get_exclusion_config_for_facility"
GRAPH_OPS = "imas_codex.discovery.code.graph_ops.GraphClient"
GRAPH_GRAPH = "imas_codex.graph.GraphClient"

PATHS = "imas_codex.discovery.paths.parallel"
CODE_OPS = "imas_codex.discovery.code.graph_ops"
CODE_WORKERS = "imas_codex.discovery.code.workers"

CLAIM_CASES = [
    (PATHS, "claim_paths_for_scanning", GRAPH_GRAPH, {"limit": 10}),
    (PATHS, "claim_paths_for_expanding", GRAPH_GRAPH, {"limit": 10}),
    (PATHS, "claim_paths_for_triaging", GRAPH_GRAPH, {"limit": 10}),
    (PATHS, "claim_paths_for_enriching", GRAPH_GRAPH, {"limit": 10}),
    (PATHS, "claim_paths_for_scoring", GRAPH_GRAPH, {"limit": 10}),
    (CODE_OPS, "claim_paths_for_file_scan", GRAPH_OPS, {"limit": 10, "min_score": 0.5}),
    (CODE_OPS, "claim_files_for_triage", GRAPH_OPS, {"limit": 10}),
    (
        CODE_OPS,
        "claim_files_for_enrichment",
        GRAPH_OPS,
        {"limit": 10, "min_triage_composite": 0.5},
    ),
    (CODE_OPS, "claim_files_for_scoring", GRAPH_OPS, {"limit": 10}),
    (
        CODE_WORKERS,
        "_claim_code_files_for_ingestion",
        GRAPH_GRAPH,
        {"limit": 10, "min_score": 0.5},
    ),
]

PREDICATE_CASES = [
    (PATHS, "has_pending_work", GRAPH_GRAPH, {}),
    (PATHS, "_has_pending_scan_work", GRAPH_GRAPH, {}),
    (PATHS, "_has_pending_expand_work", GRAPH_GRAPH, {}),
    (PATHS, "_has_pending_triage_work", GRAPH_GRAPH, {}),
    (PATHS, "_has_pending_enrich_work", GRAPH_GRAPH, {}),
    (PATHS, "_has_pending_score_work", GRAPH_GRAPH, {}),
    (CODE_OPS, "has_pending_scan_work", GRAPH_OPS, {"min_score": 0.5}),
    (CODE_OPS, "has_pending_score_work", GRAPH_OPS, {}),
    (CODE_OPS, "has_pending_triage_work", GRAPH_OPS, {}),
    (CODE_OPS, "has_pending_enrich_work", GRAPH_OPS, {"min_triage_composite": 0.5}),
    (CODE_OPS, "has_pending_code_work", GRAPH_OPS, {"min_score": 0.5}),
    (CODE_OPS, "has_pending_link_work", GRAPH_OPS, {}),
]


class StubGraphClient:
    """Evaluate seeded rows against the prefix predicates the query carries."""

    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        text = " ".join(cypher.split())
        matched = self._match(kwargs)
        if "SET " in text and "claim_token" in text:
            for row in matched[: kwargs.get("limit", len(matched))]:
                row["claim_token"] = kwargs["token"]
            return []
        if "claim_token" in text and "$token" in text:
            token = kwargs["token"]
            return [r for r in self.rows if r.get("claim_token") == token]
        if "AS has_work" in text:
            return [{"has_work": bool(matched)}]
        if "AS pending" in text:
            count = len(matched)
            return [
                {
                    "pending": count,
                    "pending_discovered": count,
                    "pending_scanned": 0,
                    "pending_expand": 0,
                    "pending_enrich": 0,
                    "pending_score": 0,
                }
            ]
        return []

    def _match(self, kwargs):
        facility = kwargs.get("facility")
        rows = [
            r for r in self.rows if facility is None or r.get("facility_id") == facility
        ]
        excluded = kwargs.get("excluded_prefixes")
        if excluded:
            rows = [
                r
                for r in rows
                if not any(str(r["path"]).startswith(p) for p in excluded)
            ]
        scope = kwargs.get("scope_prefixes")
        if scope is None:
            scope = kwargs.get("prefixes")
        if scope:
            rows = [r for r in rows if any(str(r["path"]).startswith(p) for p in scope)]
        return rows


class RecordingStub(StubGraphClient):
    def __init__(self, rows):
        super().__init__(rows)
        self.calls = []

    def query(self, cypher, **kwargs):
        self.calls.append(kwargs)
        super().query(cypher, **kwargs)
        return []


class CypherRecorder(StubGraphClient):
    """Capture the rendered query text as well as the bound parameters."""

    def __init__(self, rows):
        super().__init__(rows)
        self.queries = []

    def query(self, cypher, **kwargs):
        self.queries.append((" ".join(cypher.split()), kwargs))
        return super().query(cypher, **kwargs)


def _path(path, **extra):
    row = {"id": path, "facility_id": FACILITY, "path": path, "depth": 0}
    row.update(extra)
    return row


def _code_file(path, **extra):
    row = {"id": path, "facility_id": FACILITY, "path": path}
    row.update(extra)
    return row


def _exclusion_config(path_prefixes):
    cfg = ExclusionConfig()
    cfg.path_prefixes = list(path_prefixes)
    return cfg


def _rows_for(module):
    return _path if module == PATHS else _code_file


def _call(fn, rows, target, **kwargs):
    with (
        patch(DOC_CFG, return_value=_exclusion_config([EXCLUDED_PREFIX])),
        patch(target, return_value=StubGraphClient(rows)),
    ):
        return fn(FACILITY, **kwargs)


@pytest.mark.parametrize("module,name,target,extra", CLAIM_CASES)
def test_claim_takes_only_the_outside_row(module, name, target, extra):
    fn = getattr(importlib.import_module(module), name)
    factory = _rows_for(module)
    rows = [factory(EXCLUDED_PREFIX + "/a"), factory(OUTSIDE_PREFIX + "/b")]
    claimed = _call(fn, rows, target, **extra)
    assert [c["path"] for c in claimed] == [OUTSIDE_PREFIX + "/b"]


@pytest.mark.parametrize("module,name,target,extra", PREDICATE_CASES)
def test_predicate_ignores_excluded_row(module, name, target, extra):
    fn = getattr(importlib.import_module(module), name)
    factory = _rows_for(module)
    excluded = _call(fn, [factory(EXCLUDED_PREFIX + "/a")], target, **extra)
    outside = _call(fn, [factory(OUTSIDE_PREFIX + "/b")], target, **extra)
    assert excluded is False
    assert outside is True


def test_scoped_claim_binds_scope_and_exclusion_separately():
    from imas_codex.discovery.paths.parallel import claim_paths_for_scanning

    recorder = RecordingStub([_path(OUTSIDE_PREFIX + "/b")])
    with (
        patch(DOC_CFG, return_value=_exclusion_config([EXCLUDED_PREFIX])),
        patch(GRAPH_GRAPH, return_value=recorder),
    ):
        claim_paths_for_scanning(FACILITY, limit=10, root_filter=[SCOPE_PREFIX])
    scoped = [c for c in recorder.calls if c.get("scope_prefixes")]
    assert scoped, "the scoped claim bound no scope prefix list"
    assert scoped[0]["scope_prefixes"] == [SCOPE_PREFIX]
    assert scoped[0]["excluded_prefixes"] == [EXCLUDED_PREFIX]


def test_claim_excludes_exactly_the_path_prefix_family():
    from imas_codex.discovery.code.graph_ops import claim_files_for_triage

    cfg = _exclusion_config([EXCLUDED_PREFIX])
    samples = [
        EXCLUDED_PREFIX + "/dropped",
        EXCLUDED_PREFIX + "/nested/deeper",
        OUTSIDE_PREFIX + "/kept",
        "/analysis/src/other/kept",
    ]
    expected = []
    for path in samples:
        excluded, reason = cfg.should_exclude(path)
        if excluded:
            assert reason == f"path_prefix:{EXCLUDED_PREFIX}"
        else:
            expected.append(path)

    rows = [_code_file(path) for path in samples]
    claimed = _call(claim_files_for_triage, rows, GRAPH_OPS, limit=10)
    assert [c["path"] for c in claimed] == expected


EXCLUSION_MARKER = "none(excluded IN $excluded_prefixes"


def _rendered_exclusion_query(module, name, target, extra):
    fn = getattr(importlib.import_module(module), name)
    recorder = CypherRecorder([_rows_for(module)(OUTSIDE_PREFIX + "/b")])
    with (
        patch(DOC_CFG, return_value=_exclusion_config([EXCLUDED_PREFIX])),
        patch(target, return_value=recorder),
    ):
        fn(FACILITY, **extra)
    marked = [text for text, _ in recorder.queries if EXCLUSION_MARKER in text]
    assert marked, f"{name} rendered no exclusion clause at all"
    return marked[0]


def _predicate_governing_exclusion(text):
    idx = text.index(EXCLUSION_MARKER)
    where = text.rindex("WHERE ", 0, idx)
    return text[where + len("WHERE ") : idx]


def _has_top_level_or(expr):
    depth = 0
    upper = expr.upper()
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and upper.startswith(" OR ", i):
            return True
    return False


@pytest.mark.parametrize("module,name,target,extra", CLAIM_CASES + PREDICATE_CASES)
def test_exclusion_clause_is_anded_onto_the_whole_predicate(
    module, name, target, extra
):
    """The exclusion must bind the whole predicate, never one disjunct.

    Cypher binds AND tighter than OR, so an exclusion clause appended after an
    unparenthesised disjunction applies only to the last disjunct. A row filter
    cannot see this: it reads the parameters, not where the clause sits. This
    reads the rendered text and requires no top-level OR before the clause.
    """
    text = _rendered_exclusion_query(module, name, target, extra)
    expr = _predicate_governing_exclusion(text)
    assert not _has_top_level_or(expr), (
        f"{name} leaves a top-level OR in front of the exclusion clause, so the "
        f"clause binds only the final disjunct: {expr!r}"
    )
