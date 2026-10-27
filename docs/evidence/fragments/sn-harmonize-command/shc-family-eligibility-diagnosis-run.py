"""Trace which eligibility predicate stops a locus family's docs drain.

Read-only against the live graph. For each member of the
``electrostatic_potential`` pattern family this evaluates the predicates the
harmonize apply passes on its way to the docs pools, in the order the apply
reaches them:

1. pattern-family enumeration (``families.py``)
2. the reset/scope-binding step the campaign runs before draining
   (``reset_standard_name_docs``; ``graph_docs_persistence.py``)
3. the generate_docs claim (``claim_generate_docs_batch``;
   ``graph_docs_persistence.py``) plus the scope clause the claim layer ANDs
4. the review_docs claim (``claim_review_docs_batch``;
   ``graph_review_persistence.py``) plus the same scope UUID clause
5. the scoped name set ``_scope_name_ids`` reads (``loop.py``)

Each predicate's Cypher is transcribed here and the transcription is checked
against the function source before it runs, so a predicate cannot silently
drift from the code it claims to measure.

Usage::

    python shc-family-eligibility-diagnosis-run.py measure   # write the JSON
    python shc-family-eligibility-diagnosis-run.py check     # gate the finding
"""

from __future__ import annotations

import inspect
import json
import sys
import uuid
from pathlib import Path

BASE = "electrostatic_potential"
ANCHOR_SEED = "electrostatic_potential_at_langmuir_probe_position"
NODE_ID = "shc-family-eligibility-diagnosis"
JSON_OUT = Path(__file__).parent / "shc-family-eligibility-diagnosis.json"

# ── predicates, transcribed ────────────────────────────────────────────────
# (symbol, module, fragment) — the fragment is asserted present in the
# symbol's source, so each query below is provably the code's own predicate.

PREDICATE_FRAGMENTS = {
    "reset_eligibility": (
        "reset_standard_name_docs",
        "imas_codex.standard_names.graph_store.graph_docs_persistence",
        "IN ['accepted', 'exhausted', 'reviewed', 'drafted']",
    ),
    "reset_scope_stamp": (
        "reset_standard_name_docs",
        "imas_codex.standard_names.graph_store.graph_docs_persistence",
        "coalesce(sn.docs_stage, 'pending') = 'pending'",
    ),
    "docs_claim_core": (
        "claim_generate_docs_batch",
        "imas_codex.standard_names.graph_store.graph_docs_persistence",
        "sn.name_stage = 'accepted' AND sn.docs_stage = 'pending'",
    ),
    "claim_scope_clause": (
        "_claim_sn_atomic",
        "imas_codex.standard_names.graph_store.graph_cost_and_claims",
        "AND sn.run_id = $scope_run_id",
    ),
    "review_claim_core": (
        "claim_review_docs_batch",
        "imas_codex.standard_names.graph_store.graph_review_persistence",
        "(sn.docs_stage = 'drafted' OR",
    ),
    "family_exclusion": (
        "_pattern_family_rows_query",
        "imas_codex.standard_names.families",
        "coalesce(m.name_stage, '') <> $superseded",
    ),
    "scope_name_ids": (
        "_scope_name_ids",
        "imas_codex.standard_names.loop",
        "sn.run_id = $rid",
    ),
}


def verify_transcriptions() -> dict[str, bool]:
    """Confirm every transcribed fragment exists in the function it cites.

    Without this, a query here could be measuring a predicate the code no
    longer carries.
    """
    import importlib

    out: dict[str, bool] = {}
    for key, (symbol, module_name, fragment) in PREDICATE_FRAGMENTS.items():
        module = importlib.import_module(module_name)
        out[key] = fragment in inspect.getsource(getattr(module, symbol))
    return out


# ── queries ────────────────────────────────────────────────────────────────

_STATE_QUERY = """
MATCH (sn:StandardName)
WHERE sn.id IN $ids
RETURN sn.id AS id,
       sn.name_stage AS name_stage,
       sn.docs_stage AS docs_stage,
       sn.reviewer_score_name AS reviewer_score_name,
       sn.reviewer_score_docs AS reviewer_score_docs,
       sn.validation_status AS validation_status,
       sn.claimed_at IS NOT NULL AS claimed,
       sn.run_id AS run_id,
       sn.harmonized_anchor AS pinned_anchor,
       COUNT { (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
               WHERE src.status = 'composed' } AS composed_sources
ORDER BY sn.id
"""

#: The reset step's own eligibility: which members it would snapshot and
#: reset, and therefore whether it reaches its scope-stamping query at all.
_RESET_ELIGIBILITY_QUERY = """
MATCH (sn:StandardName)
WHERE sn.id IN $ids
  AND sn.name_stage = 'accepted'
  AND coalesce(sn.docs_stage, '')
      IN ['accepted', 'exhausted', 'reviewed', 'drafted']
RETURN sn.id AS id
ORDER BY sn.id
"""

#: The second query in the reset step: binds the run's scope onto names that
#: are already pending. The function runs it only after its eligibility count.
_RESET_SCOPE_STAMP_QUERY = """
MATCH (sn:StandardName)
WHERE sn.id IN $ids
  AND sn.name_stage = 'accepted'
  AND coalesce(sn.docs_stage, 'pending') = 'pending'
RETURN sn.id AS id
ORDER BY sn.id
"""

#: The generate_docs claim's own WHERE, with the name-score gate omitted
#: exactly as the function omits it whenever a scope_run_id is bound.
_DOCS_CLAIM_QUERY = """
MATCH (sn:StandardName)
WHERE sn.name_stage = 'accepted' AND sn.docs_stage = 'pending'
  AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])
  AND NOT (coalesce(sn.name_stage, '') IN $frozen_name_stages)
  AND (sn.claimed_at IS NULL
       OR sn.claimed_at < datetime() - duration($cutoff))
  AND sn.id IN $ids
RETURN sn.id AS id
ORDER BY sn.id
"""

#: The same claim with the scope clause the claim layer ANDs under a scoped
#: run. This is the predicate a scoped drain actually applies.
_DOCS_CLAIM_SCOPED_QUERY = _DOCS_CLAIM_QUERY.replace(
    "RETURN sn.id AS id",
    "AND sn.run_id = $scope_run_id\nRETURN sn.id AS id",
)

_REVIEW_CLAIM_QUERY = """
MATCH (sn:StandardName)
WHERE (sn.docs_stage = 'drafted'
       OR ($drain_scope_id IS NOT NULL AND sn.docs_stage = 'reviewed'
           AND sn.reviewer_score_docs >= $min_score))
  AND NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])
  AND sn.id IN $ids
RETURN sn.id AS id
ORDER BY sn.id
"""

_REVIEW_CLAIM_SCOPED_QUERY = _REVIEW_CLAIM_QUERY.replace(
    "RETURN sn.id AS id",
    "AND sn.run_id = $scope_run_id\nRETURN sn.id AS id",
)

_SCOPE_IDS_QUERY = """
MATCH (sn:StandardName)
WHERE sn.run_id = $rid
   OR exists {
        (:StandardNameSource {run_id: $rid})-[:PRODUCED_NAME]->(sn)
      }
RETURN DISTINCT sn.id AS id
ORDER BY sn.id
"""

_FROZEN = ["approved", "contested"]
_CUTOFF = "PT900"
_MIN_SCORE = 0.85


def _ids(rows) -> list[str]:
    return [r["id"] for r in rows]


def measure() -> dict:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.families import fetch_pattern_families

    transcribed = verify_transcriptions()
    fresh_scope = str(uuid.uuid4())
    with GraphClient() as gc:
        families = fetch_pattern_families(gc, seeds=[ANCHOR_SEED])
        family = next(
            (f for f in families if f["physical_base"] == BASE),
            {"members": [], "invariant": (), "physical_base": BASE, "n": 0},
        )
        member_ids = sorted(m["id"] for m in family["members"])

        state = {r["id"]: dict(r) for r in gc.query(_STATE_QUERY, ids=member_ids)}

        reset_eligible = _ids(gc.query(_RESET_ELIGIBILITY_QUERY, ids=member_ids))
        scope_stampable = _ids(gc.query(_RESET_SCOPE_STAMP_QUERY, ids=member_ids))
        claim_eligible = _ids(
            gc.query(
                _DOCS_CLAIM_QUERY,
                ids=member_ids,
                frozen_name_stages=_FROZEN,
                cutoff=_CUTOFF,
            )
        )
        claim_scoped_fresh = _ids(
            gc.query(
                _DOCS_CLAIM_SCOPED_QUERY,
                ids=member_ids,
                frozen_name_stages=_FROZEN,
                cutoff=_CUTOFF,
                scope_run_id=fresh_scope,
            )
        )
        claim_scoped_by_own = {
            mid: _ids(
                gc.query(
                    _DOCS_CLAIM_SCOPED_QUERY,
                    ids=[mid],
                    frozen_name_stages=_FROZEN,
                    cutoff=_CUTOFF,
                    scope_run_id=(state.get(mid) or {}).get("run_id") or "",
                )
            )
            for mid in member_ids
        }
        review_claim = _ids(
            gc.query(
                _REVIEW_CLAIM_QUERY,
                ids=member_ids,
                drain_scope_id=None,
                min_score=_MIN_SCORE,
            )
        )
        review_claim_scoped_fresh = _ids(
            gc.query(
                _REVIEW_CLAIM_SCOPED_QUERY,
                ids=member_ids,
                drain_scope_id=None,
                min_score=_MIN_SCORE,
                scope_run_id=fresh_scope,
            )
        )
        scope_ids_by_own = {
            mid: _ids(
                gc.query(
                    _SCOPE_IDS_QUERY,
                    rid=(state.get(mid) or {}).get("run_id") or "",
                )
            )
            for mid in member_ids
        }
        scope_ids_fresh = _ids(gc.query(_SCOPE_IDS_QUERY, rid=fresh_scope))

    def _exclusion(mid: str) -> dict:
        """The first predicate in apply order that excludes *mid*."""
        st = state[mid]
        checks = [
            (
                "docs_claim:name_stage",
                "imas_codex/standard_names/graph_store/graph_docs_persistence.py:54",
                st["name_stage"] == "accepted",
                f"name_stage={st['name_stage']!r} (claim requires 'accepted')",
            ),
            (
                "docs_claim:docs_stage",
                "imas_codex/standard_names/graph_store/graph_docs_persistence.py:54",
                st["docs_stage"] == "pending",
                f"docs_stage={st['docs_stage']!r} (claim requires 'pending')",
            ),
            (
                "docs_claim:scope_run_id",
                "imas_codex/standard_names/graph_store/graph_cost_and_claims.py:329",
                mid in claim_scoped_fresh,
                (
                    "a scoped drain binds $scope_run_id to the uuid this apply "
                    f"just minted; the node carries {st['run_id']!r} — "
                    + (
                        "and the reset step never stamped the new scope because "
                        "its eligibility count was 0 for this batch"
                        if mid not in reset_eligible
                        else "the new scope was never stamped on this node"
                    )
                ),
            ),
        ]
        for name, where, passed, detail in checks:
            if not passed:
                return {"predicate": name, "where": where, "detail": detail}
        return {"predicate": None, "detail": "eligible for a scoped docs drain"}

    return {
        "node": NODE_ID,
        "family": {
            "physical_base": family["physical_base"],
            "invariant": list(family["invariant"]),
            "n": family["n"],
            "members": member_ids,
        },
        "scope": {
            "scope_run_id_used": fresh_scope,
            "fresh_scope_ids": scope_ids_fresh,
            "by_member_run_id": scope_ids_by_own,
        },
        "transcription_verified": transcribed,
        "predicates": {
            "reset_eligibility": {
                "file": "imas_codex/standard_names/graph_store/graph_docs_persistence.py",
                "line": 406,
                "excluded": sorted(set(member_ids) - set(reset_eligible)),
                "matched": reset_eligible,
            },
            "reset_scope_stamp": {
                "file": "imas_codex/standard_names/graph_store/graph_docs_persistence.py",
                "line": 420,
                "matched": scope_stampable,
                "note": "this query is skipped when reset eligibility is 0",
            },
            "docs_claim_core": {
                "file": "imas_codex/standard_names/graph_store/graph_docs_persistence.py",
                "line": 54,
                "matched": claim_eligible,
            },
            "docs_claim_scoped_fresh_scope": {
                "file": "imas_codex/standard_names/graph_store/graph_cost_and_claims.py",
                "line": 329,
                "scope_run_id": fresh_scope,
                "matched": claim_scoped_fresh,
            },
            "docs_claim_scoped_own_run_id": {
                "file": "imas_codex/standard_names/graph_store/graph_cost_and_claims.py",
                "line": 329,
                "note": "positive control: the same claim, scoped to the run id "
                "the node already carries",
                "by_member": claim_scoped_by_own,
            },
            "review_claim_core": {
                "file": "imas_codex/standard_names/graph_store/graph_review_persistence.py",
                "line": 689,
                "matched": review_claim,
            },
            "review_claim_scoped_fresh_scope": {
                "file": "imas_codex/standard_names/graph_store/graph_cost_and_claims.py",
                "line": 329,
                "scope_run_id": fresh_scope,
                "matched": review_claim_scoped_fresh,
            },
        },
        "members": {
            mid: dict(
                state[mid],
                reset_eligible=mid in reset_eligible,
                scope_stampable=mid in scope_stampable,
                claimable_unscoped=mid in claim_eligible,
                claimable_under_fresh_scope=mid in claim_scoped_fresh,
                first_excluding_predicate=_exclusion(mid),
            )
            for mid in member_ids
        },
        "guards": {"graph_writes": 0, "llm_calls": 0, "spend_usd": 0.0},
    }


def check(result: dict | None = None) -> int:
    """Gate the finding. Returns 0 only when it reproduces as recorded."""
    result = result or measure()
    failures: list[str] = []

    def require(label: str, ok: bool, detail: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'} {label}: {detail}")
        if not ok:
            failures.append(label)

    require(
        "family has five members",
        result["family"]["n"] == 5,
        f"n={result['family']['n']} members={result['family']['members']}",
    )
    for name, verified in result["transcription_verified"].items():
        require(f"predicate {name} matches source", verified, str(verified))

    anchor = result["members"][ANCHOR_SEED]
    require(
        "anchor is docs-eligible in principle",
        anchor["name_stage"] == "accepted" and anchor["docs_stage"] == "pending",
        f"name_stage={anchor['name_stage']!r} docs_stage={anchor['docs_stage']!r} "
        f"score_name={anchor['reviewer_score_name']!r} "
        f"composed_sources={anchor['composed_sources']}",
    )
    require(
        "anchor excluded by the reset step's eligibility",
        ANCHOR_SEED not in result["predicates"]["reset_eligibility"]["matched"],
        f"reset_eligible={result['predicates']['reset_eligibility']['matched']}",
    )
    require(
        "anchor would be scope-stamped had the reset not returned early",
        ANCHOR_SEED in result["predicates"]["reset_scope_stamp"]["matched"],
        f"scope_stampable={result['predicates']['reset_scope_stamp']['matched']}",
    )
    require(
        "anchor passes the docs claim without the scope clause",
        ANCHOR_SEED in result["predicates"]["docs_claim_core"]["matched"],
        f"claim_eligible={result['predicates']['docs_claim_core']['matched']}",
    )
    require(
        "anchor is not claimed under a fresh scope",
        ANCHOR_SEED
        not in result["predicates"]["docs_claim_scoped_fresh_scope"]["matched"],
        f"matched={result['predicates']['docs_claim_scoped_fresh_scope']['matched']}",
    )
    require(
        "anchor's first excluding predicate is the scope binding",
        anchor["first_excluding_predicate"]["predicate"] == "docs_claim:scope_run_id",
        str(anchor["first_excluding_predicate"]),
    )
    require(
        "positive control: the same scoped claim matches the anchor under the "
        "run id the node already carries",
        ANCHOR_SEED
        in result["predicates"]["docs_claim_scoped_own_run_id"]["by_member"][
            ANCHOR_SEED
        ],
        "by_member="
        f"{result['predicates']['docs_claim_scoped_own_run_id']['by_member'][ANCHOR_SEED]}",
    )
    require(
        "the scoped name set is empty for a fresh scope",
        result["scope"]["fresh_scope_ids"] == [],
        f"ids={result['scope']['fresh_scope_ids']}",
    )
    require(
        "no member is claimable by review_docs before generation",
        result["predicates"]["review_claim_core"]["matched"] == [],
        f"matched={result['predicates']['review_claim_core']['matched']}",
    )

    out = JSON_OUT
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(f"EXIT={1 if failures else 0}")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    command = (
        argv[1] if len(argv) > 1 and argv[1] in {"measure", "check"} else "measure"
    )
    if command == "check":
        return check()
    result = measure()
    out = JSON_OUT
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
