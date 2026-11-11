"""Re-measure chain-cap identities with no sourced ancestor, and the share a
whole-catalog cut would publish.

Chain-cap means ``chain_length >= 3``. An identity is *unsourced* when no
``StandardNameSource`` reaches it through ``PRODUCED_NAME``. An ancestor is
*sourced* when a bounded ``REFINED_FROM`` walk reaches a name a producer
reaches. Those definitions are carried verbatim from the 2026-09-01 record and
its 2026-09-17 re-measure, so the numbers stay comparable.

The publish half does not restate the export rules. It calls the export gate's
own population read, eligibility classifier and score gate -- what a
whole-catalog cut runs -- so a rule change in the gate shows up here instead of
being shadowed by a copy.

Modes
  capture  read the live graph and record the census and the published share
  control  plant fixtures in a disposable graph and require the same counters
           to separate them, so a live zero is a reading rather than a
           predicate that never fires
  verify   re-assert the accounting identities over a recorded capture

Graph reads only; nothing is written to the live graph. Spend USD 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from imas_codex.graph.client import GraphClient  # noqa: E402

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).resolve().stem.removesuffix("-run")
CAPTURE_JSON = HERE / f"{STEM}.json"

CHAIN_CAP = "sn.chain_length >= 3"
UNSOURCED = "NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }"
SOURCED = "EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }"
ANCESTOR_BOUND = 10

WITH_ANCESTOR = (
    "EXISTS { MATCH (sn)-[:REFINED_FROM*1..10]->(a:StandardName) "
    "WHERE EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(a) } }"
)
WITHOUT_ANCESTOR = (
    "NOT EXISTS { MATCH (sn)-[:REFINED_FROM*1..10]->(a:StandardName) "
    "WHERE EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(a) } }"
)

#: key -> (label, cypher). Every statement is a MATCH/count over the
#: StandardName label, filtered to the chain-cap cohort.
CENSUS_QUERIES: dict[str, tuple[str, str]] = {
    "chain_cap_total": (
        "chain-cap total",
        f"MATCH (sn:StandardName) WHERE {CHAIN_CAP} RETURN count(sn) AS n",
    ),
    "unsourced_total": (
        "unsourced chain-cap",
        f"MATCH (sn:StandardName) WHERE {CHAIN_CAP} AND {UNSOURCED} "
        "RETURN count(sn) AS n",
    ),
    "with_sourced_ancestor": (
        "unsourced chain-cap with a sourced ancestor",
        f"MATCH (sn:StandardName) WHERE {CHAIN_CAP} AND {UNSOURCED} "
        f"AND {WITH_ANCESTOR} RETURN count(sn) AS n",
    ),
    "without_sourced_ancestor": (
        "unsourced chain-cap with no sourced ancestor",
        f"MATCH (sn:StandardName) WHERE {CHAIN_CAP} AND {UNSOURCED} "
        f"AND {WITHOUT_ANCESTOR} RETURN count(sn) AS n",
    ),
}

#: The identity lists the same predicates select. Counts are cross-checked
#: against these, so a predicate cannot report a number its row set contradicts.
CENSUS_IDS: dict[str, str] = {
    "unsourced_total": (
        f"MATCH (sn:StandardName) WHERE {CHAIN_CAP} AND {UNSOURCED} "
        "RETURN sn.id AS id ORDER BY id"
    ),
    "with_sourced_ancestor": (
        f"MATCH (sn:StandardName) WHERE {CHAIN_CAP} AND {UNSOURCED} "
        f"AND {WITH_ANCESTOR} RETURN sn.id AS id ORDER BY id"
    ),
    "without_sourced_ancestor": (
        f"MATCH (sn:StandardName) WHERE {CHAIN_CAP} AND {UNSOURCED} "
        f"AND {WITHOUT_ANCESTOR} RETURN sn.id AS id ORDER BY id"
    ),
}

#: Positive controls. Each must be non-zero wherever an absence is reported, so
#: "no row" cannot stand in for "cannot match".
CONTROL_QUERIES: dict[str, str] = {
    "produced_names": f"MATCH (sn:StandardName) WHERE {SOURCED} RETURN count(sn) AS n",
    "produced_edges": (
        "MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(:StandardName) "
        "RETURN count(*) AS n"
    ),
    "chain_length_populated": (
        "MATCH (sn:StandardName) WHERE sn.chain_length IS NOT NULL "
        "RETURN count(sn) AS n"
    ),
    "refined_from_edges": (
        "MATCH (:StandardName)-[:REFINED_FROM]->(:StandardName) RETURN count(*) AS n"
    ),
}


def _rows(gc, cypher: str, **params) -> list[dict]:
    return list(gc.query(cypher, **params) or [])


def _scalar(gc, cypher: str) -> int:
    rows = _rows(gc, cypher)
    return int(rows[0]["n"]) if rows else 0


def run_census(gc) -> dict:
    timings: dict[str, float] = {}

    def timed(key: str, fn):
        started = time.perf_counter()
        value = fn()
        timings[key] = round(time.perf_counter() - started, 4)
        return value

    counts = {
        key: timed(key, lambda cypher=cypher: _scalar(gc, cypher))
        for key, (_, cypher) in CENSUS_QUERIES.items()
    }
    ids = {
        key: timed(key, lambda cypher=cypher: [row["id"] for row in _rows(gc, cypher)])
        for key, cypher in CENSUS_IDS.items()
    }
    controls = {
        key: timed(key, lambda cypher=cypher: _scalar(gc, cypher))
        for key, cypher in CONTROL_QUERIES.items()
    }
    return {
        "counts": counts,
        "ids": ids,
        "controls": controls,
        "timings": timings,
        "queries": {
            "census": {key: cypher for key, (_, cypher) in CENSUS_QUERIES.items()},
            "identities": CENSUS_IDS,
            "controls": CONTROL_QUERIES,
        },
    }


def check_census(reading: dict) -> list[str]:
    """The accounting identities, checked rather than inferred."""
    problems: list[str] = []
    counts, ids = reading["counts"], reading["ids"]

    for key, id_key in (
        ("unsourced_total", "unsourced_total"),
        ("with_sourced_ancestor", "with_sourced_ancestor"),
        ("without_sourced_ancestor", "without_sourced_ancestor"),
    ):
        if counts[key] != len(ids[id_key]):
            problems.append(
                f"count/row-set disagreement on {key}: "
                f"{counts[key]} counted, {len(ids[id_key])} identified"
            )
    if (
        counts["with_sourced_ancestor"] + counts["without_sourced_ancestor"]
        != counts["unsourced_total"]
    ):
        problems.append(
            "with_sourced_ancestor + without_sourced_ancestor "
            "does not close against unsourced_total"
        )
    both = set(ids["with_sourced_ancestor"]) & set(ids["without_sourced_ancestor"])
    if both:
        problems.append(f"the two ancestry parts are not disjoint: {sorted(both)}")
    if len(ids["unsourced_total"]) != len(ids["with_sourced_ancestor"]) + len(
        ids["without_sourced_ancestor"]
    ):
        problems.append("the ancestry parts do not partition the unsourced cohort")
    if counts["unsourced_total"] > counts["chain_cap_total"]:
        problems.append("unsourced total exceeds the chain-cap total")
    return problems


def run_published(gc) -> dict:
    """What a whole-catalog cut publishes, through the export gate's own rules."""
    from imas_codex.standard_names.export import (
        DEFAULT_BOUND_ADJACENT_HALF_WIDTH,
        _classify_export_population,
        _fetch_export_population,
        _run_gate_c,
    )

    def client():
        return gc

    with patch("imas_codex.graph.client.GraphClient", client):
        population = _fetch_export_population(batch=None, require_docs_review=True)
        candidates, eligibility = _classify_export_population(
            population, domain=None, names_only=False
        )
        gate_c, scored, _, _ = _run_gate_c(
            candidates,
            min_score=0.65,
            include_unreviewed=False,
            min_description_score=None,
            bound_adjacent_half_width=DEFAULT_BOUND_ADJACENT_HALF_WIDTH,
        )
    return {
        "population": [candidate["id"] for candidate in population],
        "eligible": [candidate["id"] for candidate in candidates],
        "published": [candidate["id"] for candidate in scored],
        "eligibility_exclusions": {
            record.standard_name_id: record.reason for record in eligibility
        },
        "score_exclusions": {issue["name"]: issue["type"] for issue in gate_c.issues},
        "rules": {
            "min_score": 0.65,
            "bound_adjacent_half_width": DEFAULT_BOUND_ADJACENT_HALF_WIDTH,
            "include_unreviewed": False,
            "domain": None,
            "names_only": False,
        },
    }


def measure(gc) -> dict:
    reading = run_census(gc)
    reading["published"] = run_published(gc)
    no_ancestor = set(reading["ids"]["without_sourced_ancestor"])
    reading["chain_cap_at_publish_boundary"] = sorted(
        no_ancestor & set(reading["published"]["published"])
    )
    reading["counts"]["no_ancestor_in_population"] = len(
        no_ancestor & set(reading["published"]["population"])
    )
    reading["counts"]["no_ancestor_publish_eligible"] = len(
        no_ancestor & set(reading["published"]["eligible"])
    )
    reading["counts"]["no_ancestor_would_publish"] = len(
        reading["chain_cap_at_publish_boundary"]
    )
    reading["counts"]["published_total"] = len(reading["published"]["published"])
    reading["problems"] = check_census(reading)
    return reading


# ---------------------------------------------------------------------------
# plantable control
# ---------------------------------------------------------------------------

FIXTURE_IDS = (
    "ctrl_chaincap_childless",
    "ctrl_chaincap_sourced_ancestry",
    "ctrl_chaincap_unsourced_parent",
)


def plant(gc) -> None:
    """Create three chain-cap fixtures that differ only where the tests read."""
    gc.query("MATCH (n) DETACH DELETE n")
    gc.query(
        """
        UNWIND $ids AS id
        CREATE (sn:StandardName {
            id: id, chain_length: 3, name_stage: 'accepted', status: 'draft',
            origin: 'pipeline', validation_status: 'valid',
            validated_at: datetime(), docs_stage: 'accepted',
            physics_domain: ['core_profiles'], reviewer_score_name: 0.9,
            description: 'Fixture description written by the planted control.'
        })
        CREATE (review:StandardNameReview {
            id: 'review-' + id, review_axis: 'docs',
            review_group_id: 'group-' + id,
            resolution_method: 'single_review', reviewed_at: datetime()
        })
        CREATE (sn)-[:HAS_REVIEW]->(review)
        """,
        ids=list(FIXTURE_IDS),
    )
    # A live child is the one route an unsourced identity has into a cut, so
    # two of the three carry one; the third is the childless case.
    gc.query(
        """
        MATCH (parent:StandardName)
        WHERE parent.id IN $parents
        CREATE (child:StandardName {
            id: parent.id + '_child', chain_length: 1, name_stage: 'accepted',
            status: 'draft', origin: 'pipeline', validation_status: 'valid',
            validated_at: datetime(), description: 'Fixture child.'
        })
        CREATE (child)-[:HAS_PARENT]->(parent)
        """,
        parents=["ctrl_chaincap_unsourced_parent", "ctrl_chaincap_sourced_ancestry"],
    )
    # Two hops to a producer-bearing ancestor, so the ancestry walk is
    # exercised past depth one.
    gc.query(
        """
        CREATE (ancestor:StandardName {
            id: 'ctrl_sourced_ancestor', chain_length: 1, name_stage: 'accepted',
            status: 'draft', origin: 'pipeline', description: 'Fixture ancestor.'
        })
        CREATE (mid:StandardName {
            id: 'ctrl_mid_lineage', chain_length: 2, name_stage: 'accepted',
            status: 'draft', origin: 'pipeline', description: 'Fixture mid.'
        })
        CREATE (producer:StandardNameSource {
            id: 'ctrl:dd:fixture', source_type: 'dd', status: 'composed'
        })
        CREATE (producer)-[:PRODUCED_NAME]->(ancestor)
        CREATE (mid)-[:REFINED_FROM]->(ancestor)
        """
    )
    # REFINED_FROM runs successor to predecessor, so the tip points at its
    # intermediate and the intermediate at the producer-bearing ancestor.
    gc.query(
        """
        MATCH (tip:StandardName {id: 'ctrl_chaincap_sourced_ancestry'}),
              (mid:StandardName {id: 'ctrl_mid_lineage'})
        CREATE (tip)-[:REFINED_FROM]->(mid)
        """
    )


def run_control() -> int:
    from imas_codex.graph import temp_neo4j

    def port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    scratch = os.environ.get("TMPDIR") or tempfile.gettempdir()
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="chaincap-control-", dir=scratch) as directory:
        root = Path(directory)
        bolt, http = port(), port()
        process, _log = temp_neo4j.start_temp_neo4j(root, bolt, http, load_dump=False)
        uri = f"bolt://127.0.0.1:{bolt}"

        def connect():
            return GraphClient(
                uri=uri, username="neo4j", password="", graph_name="chaincap-control"
            )

        try:
            with connect() as gc:
                empty = gc.query("MATCH (n) RETURN count(n) AS n")[0]["n"]
                print(
                    f"CONTROL endpoint={uri} empty_nodes={empty} "
                    f"live_graph_excluded_by_explicit_uri=true",
                    flush=True,
                )
                if empty != 0:
                    failures.append("the disposable instance was not empty")
                plant(gc)
                reading = measure(gc)
                with connect() as check:
                    ids = sorted(
                        row["id"]
                        for row in check.query(
                            "MATCH (n:StandardName) RETURN n.id AS id"
                        )
                    )
                    print(f"CONTROL planted_names={ids}", flush=True)
        finally:
            process.terminate()
            process.wait(timeout=60)

    counts = reading["counts"]
    print(
        "CONTROL census "
        f"chain_cap_total={counts['chain_cap_total']} "
        f"unsourced_total={counts['unsourced_total']} "
        f"with_sourced_ancestor={counts['with_sourced_ancestor']} "
        f"without_sourced_ancestor={counts['without_sourced_ancestor']}",
        flush=True,
    )
    print(
        "CONTROL publish "
        f"population={len(reading['published']['population'])} "
        f"eligible={len(reading['published']['eligible'])} "
        f"published={len(reading['published']['published'])} "
        f"reasons={json.dumps(reading['published']['eligibility_exclusions'], sort_keys=True)}",
        flush=True,
    )
    print(
        "CONTROL intersection "
        f"chain_cap_at_publish_boundary={json.dumps(reading['chain_cap_at_publish_boundary'])}",
        flush=True,
    )

    no_ancestor = set(reading["ids"]["without_sourced_ancestor"])
    with_ancestor = set(reading["ids"]["with_sourced_ancestor"])
    published = set(reading["published"]["published"])

    expected_no_ancestor = {"ctrl_chaincap_childless", "ctrl_chaincap_unsourced_parent"}
    expected_with_ancestor = {"ctrl_chaincap_sourced_ancestry"}
    expected_published = {
        "ctrl_chaincap_unsourced_parent",
        "ctrl_chaincap_sourced_ancestry",
    }
    if no_ancestor != expected_no_ancestor:
        failures.append(
            f"no-ancestor cohort read as {sorted(no_ancestor)}, "
            f"expected {sorted(expected_no_ancestor)}"
        )
    if with_ancestor != expected_with_ancestor:
        failures.append(
            f"sourced-ancestor cohort read as {sorted(with_ancestor)}, "
            f"expected {sorted(expected_with_ancestor)}"
        )
    if published != expected_published:
        failures.append(
            f"published set read as {sorted(published)}, "
            f"expected {sorted(expected_published)}"
        )
    reason = reading["published"]["eligibility_exclusions"].get("ctrl_chaincap_childless")
    if reason != "no_producing_source":
        failures.append(
            f"the childless identity was not refused on no_producing_source (read {reason!r})"
        )
    if not reading["chain_cap_at_publish_boundary"]:
        failures.append("the planted control produced no publishable no-ancestor identity")
    if not reading["controls"]["produced_names"]:
        failures.append("the producer positive control fired zero on the planted graph")
    for problem in reading["problems"]:
        failures.append(f"accounting: {problem}")

    for failure in failures:
        print(f"CONTROL-FAIL {failure}", flush=True)
    print(
        "CONTROL-RESULT " + ("PASS" if not failures else f"FAIL count={len(failures)}"),
        flush=True,
    )
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# capture and verify
# ---------------------------------------------------------------------------


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def summarise(reading: dict) -> dict:
    """The recorded measures, without the full population and exclusion ledgers.

    The complete capture names every published identity and every exclusion
    reason; that is several hundred kilobytes and belongs beside the run rather
    than in the repository. The summary keeps every number this record cites and
    the identities the census itself names.
    """
    published = reading["published"]
    no_ancestor = set(reading["ids"]["without_sourced_ancestor"])
    reason_counts = Counter(published["eligibility_exclusions"].values())
    return {
        "revision": reading["revision"],
        "captured_at": reading["captured_at"],
        "spend_usd": reading["spend_usd"],
        "counts": reading["counts"],
        "ids": reading["ids"],
        "controls": reading["controls"],
        "timings": reading["timings"],
        "problems": reading["problems"],
        "chain_cap_at_publish_boundary": reading["chain_cap_at_publish_boundary"],
        "publish": {
            "rules": published["rules"],
            "population_size": len(published["population"]),
            "eligible_size": len(published["eligible"]),
            "published_size": len(published["published"]),
            "no_ancestor_in_population": sorted(
                no_ancestor & set(published["population"])
            ),
            "no_ancestor_exclusion_reasons": {
                name: published["eligibility_exclusions"][name]
                for name in sorted(
                    no_ancestor & set(published["eligibility_exclusions"])
                )
            },
            "no_ancestor_score_exclusions": {
                name: published["score_exclusions"][name]
                for name in sorted(no_ancestor & set(published["score_exclusions"]))
            },
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        },
        "queries": reading["queries"],
    }


def run_capture(out: Path | None) -> int:
    started = datetime.now(UTC)
    with GraphClient() as gc:
        print(f"CONNECTED uri={getattr(gc, '_uri', 'unknown')}", flush=True)
        reading = measure(gc)
    reading["revision"] = _revision()
    reading["captured_at"] = started.isoformat()
    reading["module"] = str(Path(__file__).resolve())
    reading["module_sha256"] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    reading["spend_usd"] = 0

    for key in (
        "chain_cap_total",
        "unsourced_total",
        "with_sourced_ancestor",
        "without_sourced_ancestor",
        "no_ancestor_in_population",
        "no_ancestor_publish_eligible",
        "no_ancestor_would_publish",
        "published_total",
    ):
        print(f"CAPTURE {key}={reading['counts'][key]}", flush=True)
    print(f"CAPTURE controls={json.dumps(reading['controls'], sort_keys=True)}", flush=True)
    print(
        "CAPTURE max_statement_seconds="
        f"{max(reading['timings'].values(), default=0.0)}",
        flush=True,
    )
    print(f"CAPTURE timings={json.dumps(reading['timings'], sort_keys=True)}", flush=True)
    print(
        "CAPTURE identities_without_sourced_ancestor="
        + json.dumps(sorted(reading["ids"]["without_sourced_ancestor"])),
        flush=True,
    )
    print(
        "CAPTURE published_members_without_sourced_ancestor="
        + json.dumps(reading["chain_cap_at_publish_boundary"]),
        flush=True,
    )
    print(f"CAPTURE problems={json.dumps(reading['problems'])}", flush=True)
    print("CAPTURE-RESULT " + ("PASS" if not reading["problems"] else "FAIL"), flush=True)

    summary = summarise(reading)
    CAPTURE_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {CAPTURE_JSON} bytes={CAPTURE_JSON.stat().st_size}", flush=True)
    if out is not None:
        out.write_text(json.dumps(reading, indent=2, sort_keys=True) + "\n")
        print(f"WROTE {out} bytes={out.stat().st_size} (full capture)", flush=True)
    return 0 if not reading["problems"] else 1


def run_verify(path: Path) -> int:
    reading = json.loads(path.read_text())
    problems = check_census(reading)
    no_ancestor = set(reading["ids"]["without_sourced_ancestor"])
    boundary = reading["chain_cap_at_publish_boundary"]
    if not no_ancestor.issuperset(boundary):
        problems.append(
            "the recorded publish boundary names identities outside the "
            "no-sourced-ancestor cohort")
    if len(boundary) != reading["counts"]["no_ancestor_would_publish"]:
        problems.append(
            f"recorded publish intersection {reading['counts']['no_ancestor_would_publish']} "
            f"differs from its own identity list of {len(boundary)}"
        )
    published_size = reading.get("publish", {}).get("published_size")
    if published_size is not None and published_size != reading["counts"]["published_total"]:
        problems.append("the recorded published size disagrees with the counts block")
    full_published = reading.get("published", {}).get("published")
    if isinstance(full_published, list):
        recomputed = sorted(no_ancestor & set(full_published))
        if recomputed != boundary:
            problems.append("the publish boundary does not recompute from the full capture")
    for problem in problems:
        print(f"VERIFY-FAIL {problem}", flush=True)
    print("VERIFY-RESULT " + ("PASS" if not problems else "FAIL"), flush=True)
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", choices=("capture", "control", "verify"), default="capture"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path for the full capture; the summary always goes beside "
        "the instrument",
    )
    parser.add_argument("--capture", type=Path, default=CAPTURE_JSON)
    args = parser.parse_args(argv)

    print(
        "# revision: "
        + _revision()
        + f" tree: {ROOT} command: "
        + shlex.join([sys.executable, *sys.argv]),
        flush=True,
    )
    if args.mode == "control":
        return run_control()
    if args.mode == "verify":
        return run_verify(args.capture)
    return run_capture(args.out)


if __name__ == "__main__":
    raise SystemExit(main())