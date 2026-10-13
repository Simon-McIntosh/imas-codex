"""Measure signal-search rankings and the ported docs/code and embed-text arms.

Run as a module from the worktree, reusing the main checkout's environment:

    UV_PROJECT_ENVIRONMENT=<main>/.venv PYTHONPATH=$PWD uv run --no-sync \
        python -m tests.llm.measure_search_rankings [passes] [--section NAME]

The rerank arms spend on the discovery-relevance seat, which is why they are
scored here rather than inside pytest: the suite refuses a paid request, so its
arms carry a switch and never call one. Each section can be run alone with
``--section``:

``signal``     the fused order alone (``rrf``) and both rerank arms over
               ``passes`` passes. Each rerank arm builds its candidate state
               with the product's ``_signal_rerank_state``; the generic arm is
               judged on the generic relevance question and the role-aware arm
               on the role-aware one. The rerank order varies between runs, so
               each arm is scored once per pass and the mean over the passes is
               reported beside the per-pass figures.
``docs_code``  the ported section 4 measurement: the docs and code questions in
               ``fixtures/docs_code_search_questions.json`` are retrieved live
               and reranked live, and labelled-only nDCG@10 at pool 60 is
               reported for the embedding order and the reranked order, the
               figure section 4 landed.
``embed_text`` the ported section 5 measurement: every eligible signal is
               embedded in memory twice, on its description alone and on the
               composed name, description and unit, and each text is scored on a
               vector-only arm and an RRF arm, without writing a stored vector.

Only ``signal`` and ``docs_code`` spend Jev; ``embed_text`` ranks over freshly
computed embeddings and reads no paid endpoint. Each Jev section reports the
spend read from the response usage the judged call layer returns, not a count
derived from the text it sent.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

HARNESS_DIR = Path(__file__).parent
DOCS_CODE_FIXTURE = HARNESS_DIR / "fixtures" / "docs_code_search_questions.json"

# The pool the ported section 4 measurement reranks and scores at.
DOCS_CODE_POOL = 60

# The questions whose true answer the census found ingested and embedded but
# ranked below the top 30. A retrieved candidate carrying one of these names is
# relevant for that question, at whatever rank it sits.
CENSUS = {
    "q05": ("eqgrd", "eqequb"),
    "q12": ("shtrd",),
    "q14": ("use_compiler",),
    "q18": ("use_login",),
}


def _harness():
    from tests.llm import test_signal_search_goldens as harness

    return harness


# ---------------------------------------------------------------------------
# response-usage spend recorder
# ---------------------------------------------------------------------------


class SpendRecorder:
    """Sum the USD cost each decisions response reports, and count the calls.

    The judged call layer books a call's reported cost to its caller, but the
    rerank seam discards it once the ordering is built, so the spend is read at
    the one point every judged call passes through. ``install`` must precede the
    spend and ``restore`` must follow it: the patch is on a module attribute
    every rerank in the process reads.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.usd = 0.0
        self._real = None

    def install(self) -> None:
        from imas_codex.discovery.base import judgment

        self._real = judgment.acall_decisions
        judgment.acall_decisions = self._call

    def restore(self) -> None:
        if self._real is None:
            return
        from imas_codex.discovery.base import judgment

        judgment.acall_decisions = self._real
        self._real = None

    async def _call(self, model, state, questions, *, service=None, **kwargs):
        answers, spent = await self._real(
            model, state, questions, service=service, **kwargs
        )
        self.calls += 1
        self.usd += spent or 0.0
        return answers, spent

    def reset(self) -> None:
        self.calls = 0
        self.usd = 0.0


# ---------------------------------------------------------------------------
# signal arms (Jev)
# ---------------------------------------------------------------------------


def _score_arm(arm: str, ctx, questions) -> tuple[list[dict], list[str]]:
    """Score one arm, returning its rows and the questions it refused.

    A refusal is a question whose arm could not be scored because it lifted a
    candidate the fixture does not label; it is recorded rather than raised so
    the remaining questions are still measured.
    """
    harness = _harness()
    rows: list[dict] = []
    refused: list[str] = []
    for question in questions:
        ctx.embedding = ctx.embedding_for(question)
        try:
            ranked = harness.arm_ranking(question, ctx)
            harness.require_labelled(arm, question, ranked)
        except harness.UnlabelledCandidates:
            refused.append(question.id)
            continue
        row = {"arm": arm, "question": question.id, "held_out": question.held_out}
        row.update(harness.score_ranking(question, ranked))
        rows.append(row)
    return rows, refused


def measure_signal_arms(passes: int) -> dict:
    """Score ``rrf`` and both rerank arms over ``passes`` passes."""
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.graph.client import GraphClient

    harness = _harness()
    gc = GraphClient()
    encoder = Encoder(EncoderConfig())
    eligible = harness.eligible_signal_ids(gc)

    spend = SpendRecorder()
    table: dict[str, dict] = {}
    for arm in ["rrf", *harness.JEV_ARMS]:
        spend.reset()
        spend.install()
        per_pass_ndcg: list[float] = []
        per_pass_prec: list[float] = []
        refused: list[str] = []
        try:
            for _ in range(passes):
                ctx = harness.ArmContext(
                    gc=gc,
                    encoder=encoder,
                    arm=arm,
                    eligible=eligible,
                    spend_jev=arm in harness.JEV_ARMS,
                )
                rows, refused = _score_arm(arm, ctx, harness.QUESTIONS)
                per_pass_ndcg.append(harness.mean(rows, "ndcg@10"))
                per_pass_prec.append(harness.mean(rows, "grade3_precision@5"))
        finally:
            spend.restore()
        table[arm] = {
            "passes": passes,
            "mean_ndcg@10": round(sum(per_pass_ndcg) / len(per_pass_ndcg), 4)
            if per_pass_ndcg
            else None,
            "mean_grade3_precision@5": round(sum(per_pass_prec) / len(per_pass_prec), 4)
            if per_pass_prec
            else None,
            "worst_pass_ndcg@10": round(min(per_pass_ndcg), 4)
            if per_pass_ndcg
            else None,
            "per_pass_ndcg@10": [round(v, 4) for v in per_pass_ndcg],
            "refused_questions": refused,
            "jev_calls": spend.calls,
            "jev_usd": round(spend.usd, 6),
        }
    return table


# ---------------------------------------------------------------------------
# the ported section 4 docs and code measurement (Jev)
# ---------------------------------------------------------------------------


def _load_docs_code_fixture() -> dict:
    return json.loads(DOCS_CODE_FIXTURE.read_text())


def _retrieve_docs_code(query, kind, facility, gc, encoder) -> list[dict]:
    """Return the query's candidates in the search tool's own fused order."""
    from imas_codex.llm import search_tools as st

    captured: dict = {}
    orig_rerank = st.rerank_candidates
    orig_code = st.format_code_report
    orig_docs = st.format_docs_report

    def bypass(q, cands, **kwargs):
        return list(cands), None

    if kind == "code":

        def _code(rows, scores):
            captured["rows"] = rows
            return orig_code(rows, scores)

        st.rerank_candidates = bypass
        st.format_code_report = _code
        try:
            st._search_code(
                query, facility=facility, k=DOCS_CODE_POOL, gc=gc, encoder=encoder
            )
        finally:
            st.rerank_candidates = orig_rerank
            st.format_code_report = orig_code
        rows = captured["rows"]
    else:

        def _docs(chunks, documents, scores, signals=None):
            captured["rows"] = chunks + documents
            return orig_docs(chunks, documents, scores, signals=signals)

        st.rerank_candidates = bypass
        st.format_docs_report = _docs
        try:
            st._search_docs(
                query, facility=facility, k=DOCS_CODE_POOL, gc=gc, encoder=encoder
            )
        finally:
            st.rerank_candidates = orig_rerank
            st.format_docs_report = orig_docs
        rows = captured["rows"]

    cands = []
    for row in rows:
        path = (
            row.get("source_file")
            or row.get("page_id")
            or row.get("path")
            or row.get("id")
            or "unknown"
        )
        cands.append(
            {
                "id": row.get("id", "?"),
                "path": path,
                "source_file": path,
                "function_name": row.get("function_name") or "module-level",
                "page_title": row.get("page_title") or row.get("title"),
                "text": (row.get("text") or "").strip()
                or ((row.get("title") or "") + ". " + (row.get("description") or "")),
            }
        )
    return cands


def _dcg(rels: list[int], k: int = 10) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def _ndcg_at_10(rels: list[int]) -> float:
    ideal = _dcg(sorted(rels, reverse=True))
    return round(_dcg(rels) / ideal, 4) if ideal else 0.0


def _mrr(rels: list[int]) -> float:
    for i, r in enumerate(rels):
        if r:
            return round(1.0 / (i + 1), 4)
    return 0.0


def _ranked_metrics(ranked, relevant, hand) -> dict:
    rels = [relevant(c) for c in ranked]
    labeled_only = [relevant(c) for c in ranked if relevant(c) or c["id"] in hand]
    return {
        "ndcg@10": _ndcg_at_10(rels),
        "ndcg@10_labeled_only": _ndcg_at_10(labeled_only),
        "mrr": _mrr(rels),
        "relevant_in_pool": sum(1 for r in rels if r),
    }


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def measure_docs_code() -> dict:
    """Reproduce section 4's docs-and-code figure from the vendored fixture."""
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm import search_tools as st

    spec = _load_docs_code_fixture()
    facility = spec["facility"]
    gc = GraphClient()
    encoder = Encoder(EncoderConfig())

    spend = SpendRecorder()
    results: dict[str, dict] = {}
    for q in spec["questions"]:
        qid = q["id"]
        cands = _retrieve_docs_code(q["query"], q["kind"], facility, gc, encoder)
        ids = [c["id"] for c in cands]
        hand = {row["id"]: row["grade"] for row in q["labels"]}
        names = CENSUS.get(qid, ())

        def relevant(c, names=names, hand=hand):
            if any(name in c["path"] for name in names):
                return 1
            return hand.get(c["id"], 0)

        embedding = _ranked_metrics(cands[:DOCS_CODE_POOL], relevant, hand)

        spend.reset()
        spend.install()
        try:
            ordered, note = st.rerank_candidates(
                q["query"], cands, budget_seconds=120.0, concurrency=32
            )
        finally:
            spend.restore()

        idset = set(ids[:DOCS_CODE_POOL])
        ranked = [c for c in ordered if c["id"] in idset]
        rerank = _ranked_metrics(ranked, relevant, hand)
        rerank["note"] = note
        rerank["jev_calls"] = spend.calls
        rerank["jev_usd"] = round(spend.usd, 6)

        results[qid] = {
            "query": q["query"],
            "kind": q["kind"],
            "n_retrieved": len(cands),
            "embedding": embedding,
            "rerank": rerank,
        }
        print(
            f"{qid} {q['kind']} n={len(cands)} "
            f"emb={embedding['ndcg@10']:.4f} rerank={rerank['ndcg@10']:.4f} "
            f"labelled={rerank['ndcg@10_labeled_only']:.4f} jev_calls={spend.calls}",
            flush=True,
        )

    return {
        "pool": DOCS_CODE_POOL,
        "embedding_ndcg@10": _mean(
            [r["embedding"]["ndcg@10"] for r in results.values()]
        ),
        "embedding_ndcg@10_labeled_only": _mean(
            [r["embedding"]["ndcg@10_labeled_only"] for r in results.values()]
        ),
        "rerank_ndcg@10": _mean([r["rerank"]["ndcg@10"] for r in results.values()]),
        "rerank_ndcg@10_labeled_only": _mean(
            [r["rerank"]["ndcg@10_labeled_only"] for r in results.values()]
        ),
        "jev_calls": sum(r["rerank"]["jev_calls"] for r in results.values()),
        "jev_usd": round(sum(r["rerank"]["jev_usd"] for r in results.values()), 6),
        "questions": results,
    }


# ---------------------------------------------------------------------------
# the ported section 5 embed-text measurement (no Jev)
# ---------------------------------------------------------------------------


def compose_name_desc_unit(name: str, description: str, unit: str) -> str:
    """A signal's name, description and unit as one text: 'name - desc (unit)'."""
    name = (name or "").strip()
    description = (description or "").strip()
    unit = (unit or "").strip()
    core = f"{name} - {description}" if name and description else (name or description)
    return f"{core} ({unit})" if unit else core


def measure_embed_text() -> dict:
    """Score description-only against name+description+unit on the fixture."""
    import numpy as np

    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.graph.client import GraphClient
    from imas_codex.llm.search_tools import _text_search_signals

    harness = _harness()
    gc = GraphClient()
    encoder = Encoder(EncoderConfig())
    facility = harness.json_fixture["eligibility"]["facility"]

    eligible = harness.eligible_signal_ids(gc)
    rows = gc.query(
        "UNWIND $ids AS sid MATCH (s:FacilitySignal {id: sid}) "
        "RETURN s.id AS id, coalesce(s.name,'') AS name, "
        "coalesce(s.description,'') AS description, coalesce(s.unit,'') AS unit",
        ids=sorted(eligible),
    )
    signals = {r["id"]: r for r in rows}
    ids = sorted(signals)

    def normalize(mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    texts = {
        "description_only": [signals[sid]["description"] for sid in ids],
        "name_description_unit": [
            compose_name_desc_unit(
                signals[sid]["name"], signals[sid]["description"], signals[sid]["unit"]
            )
            for sid in ids
        ],
    }
    vecs_by_text: dict[str, object] = {}
    for mode, corpus in texts.items():
        t0 = time.time()
        emb = np.asarray(encoder.embed_texts(corpus), dtype=np.float32)
        vecs_by_text[mode] = normalize(emb)
        print(
            f"embedded {mode}: {emb.shape} in {time.time() - t0:.1f}s",
            flush=True,
        )

    qvecs = {
        q.id: normalize(np.asarray(encoder.embed_texts([q.query]), dtype=np.float32))[0]
        for q in harness.QUESTIONS
    }

    table: dict[str, dict] = {}
    for mode in ("description_only", "name_description_unit"):
        vecs = vecs_by_text[mode]
        for arm in ("vector_only", "rrf"):
            sel: list[dict] = []
            refused = 0
            for q in harness.QUESTIONS:
                sims = vecs @ qvecs[q.id]
                vec_ids = [
                    ids[i]
                    for i in np.argsort(-sims, kind="stable")[: harness.VECTOR_POOL]
                ]
                if arm == "vector_only":
                    ranked = harness.filter_eligible(vec_ids, eligible)[: harness.K]
                else:
                    text_ids = [
                        r["id"]
                        for r in _text_search_signals(
                            gc, q.query, facility, harness.TEXT_POOL
                        )
                    ]
                    fused = harness._rrf_fuse(
                        [harness._rf_ranks(vec_ids), harness._rf_ranks(text_ids)]
                    )
                    ranked = harness.filter_eligible(
                        [sid for sid, _ in fused], eligible
                    )[: harness.K]
                missing = harness.unlabelled_in_top_k(q, ranked, harness.TOP_K)
                if missing:
                    refused += 1
                    continue
                row = {"arm": arm, "question": q.id, "held_out": q.held_out}
                row.update(harness.score_ranking(q, ranked))
                sel.append(row)
            table[f"{mode}|{arm}"] = {
                "scored_questions": len(sel),
                "refused_questions": refused,
                "mean_ndcg@10": harness.mean(sel, "ndcg@10"),
                "mean_grade3_precision@5": harness.mean(sel, "grade3_precision@5"),
                "mean_grade3_recall@30": harness.mean(sel, "grade3_recall@30"),
            }
    return table


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    passes = 1
    sections: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--section":
            i += 1
            sections.append(argv[i])
        elif arg.isdigit():
            passes = int(arg)
        else:
            raise SystemExit(f"unknown argument: {arg}")
        i += 1
    if not sections:
        sections = ["signal", "docs_code", "embed_text"]

    out: dict = {"passes": passes, "sections": sections}
    if "signal" in sections:
        out["signal"] = measure_signal_arms(passes)
    if "docs_code" in sections:
        out["docs_code"] = measure_docs_code()
    if "embed_text" in sections:
        out["embed_text"] = measure_embed_text()

    print("=== measure_search_rankings ===", flush=True)
    print(json.dumps(out, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
