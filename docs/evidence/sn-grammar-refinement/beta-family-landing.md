# n-sgr-beta-is-the-total-beta — the sanctioned route landed a blocker, not acceptance

Executed the established rescore route against the live graph. Step 1 (the real
`sn rescore beta`) ran and advanced beta's name axis out of `exhausted`, but the
inline review could not reach `accepted`: the review pipeline's semantic gate
rejects the bare anchor name `beta` for **any** description within reach, and
both downstream steps (supersede, docs) are gated on `accepted`. The endpoint is
unreachable through sanctioned CLI routes until a pipeline guard is changed.

## Route executed (real mutations, in order)

1. Verified embedding-server reachability first (the one unverified
   precondition; it is up, so no blocker on that axis):

   ```
   $ imas-codex embed status
   Server (http://98dci4-gpu-0002:18765):  ✓ Healthy
   Model: Qwen/Qwen3-Embedding-0.6B, Dimension: 256

   $ python -c "... ensure_embedding_ready() ..."
   READY
   Server ready: Qwen/Qwen3-Embedding-0.6B on 98dci4-gpu-0002.iter.org
   ```

2. Rescore eligibility dry-run at this node's base revision:

   ```
   $ imas-codex sn rescore beta --dry-run
   would rescore beta (exhausted → drafted, run_id=sn-rescore-20260905T134702Z)
   ```

3. The real rescore (default inline review):

   ```
   $ imas-codex sn rescore beta
   rescored beta (exhausted → drafted, run_id=sn-rescore-20260905T140457Z)

   Inline review
   ┌──────────────┬─────────────────┬──────────┬────────┐
   │ StandardName │ Outcome         │ Stage    │ Score  │
   ├──────────────┼─────────────────┼──────────┼────────┤
   │ beta         │ below threshold │ reviewed │ 0.30   │
   └──────────────┴─────────────────┴──────────┴────────┘
   Inline review cost: $0.0000
   ```

   Beta is now `name_stage='reviewed'`, `validation_status='valid'` (the fresh
   LLM-free admission gate passed and cleared the stale quarantine). The
   terminal outcome the mechanism's own help promised — accept, or safe restore
   to `exhausted` — happened on neither side: it parked at `reviewed` with a
   **quorum shortfall**, cause recorded on the node:

   ```
   review_resolution_method  = semantic_similarity_gate
   review_quorum_shortfall   = fewer reviewer seats scored than the chain
                               defines (method=semantic_similarity_gate)
   reviewer_comments_name    = semantic_similarity_gate: sim=0.488 below
                               critical 0.55. Name is semantically ambiguous
   semantic_sim              = 0.4877
   ```

## Root cause: the semantic gate is structurally unreachable for a bare anchor

`review_name` (standard_names/workers.py) runs `semantic_similarity_check`
(name-as-text → description embedding cosine, critical threshold **0.55**) on
every non-derived name **before** the LLM review. Below 0.55 it skips the LLM
chain entirely and persists a synthetic 0.30 with a quorum shortfall. The intent
is sound: do not spend review seats on a name whose spelling does not stand
alone.

Measured against the live embedding server (pure reads, zero graph writes, zero
LLM spend), the bare name `beta` never clears 0.55 regardless of description:

| Description | sim(name, desc) | verdict at 0.55 |
|---|---|---|
| current node text — toroidal beta (β_tor), ratio of volume-averaged total perpendicular plasma pressure to magnetic pressure B0²/2μ0 | 0.488 | below |
| total-beta v1 — "Beta (β), the total plasma beta: ratio of total plasma pressure to magnetic pressure B0²/(2μ0), such that 1/β = 1/β_poloidal + 1/β_toroidal" | 0.503 | below |
| total-beta v2 — "Total beta (β), ratio of total plasma pressure to magnetic pressure, 1/β = 1/β_poloidal + 1/β_toroidal; characterizes the plasma as a whole" | 0.475 | below |
| total-beta v3 — "Plasma beta (β): dimensionless ratio of total plasma pressure to magnetic pressure; total beta, 1/β = 1/β_p + 1/β_t" | 0.508 | below |

Control — accepted multi-word members of the same family, same gate, same
server, all far above threshold:

| Name | sim(name, desc) |
|---|---|
| `plasma_beta` (accepted) | 0.779 |
| `normalized_toroidal_plasma_beta` (accepted) | 0.849 |
| `poloidal_beta` (accepted) | 0.770 |
| `toroidal_beta` (accepted) | 0.808 |

The endpoint (the lead's ruling) makes `beta` a deliberately terse one-word
family anchor whose children carry the qualifiers; that is precisely the shape
the embedding gate cannot admit. **No description fix clears the gate**, so a
fresh `sn rescore` or any pool review re-fires it deterministically. The gate is
not transiently down or misconfigured — it is a structural incompatibility
between the endpoint and the review safeguard.

## Downstream refusals (both gated on beta accepted, both confirmed by dry run)

Supersede (step 2) — target is `reviewed`, not `accepted`:

```
$ imas-codex sn supersede plasma_beta --into beta --dry-run
Error: target 'beta' is name_stage='reviewed', not 'accepted'
```

Docs redaction (step 3, correction pass) — docs edits require an accepted name:

```
$ imas-codex sn edit beta --docs "<total-beta text>" --reason "..." --dry-run
BLOCKED — target name_stage='reviewed' — docs edits require an accepted name
(name_stage='accepted')
```

The docs pool itself (`claim_generate_docs_batch`, graph_ops.py) requires
`name_stage='accepted'`, and `persist_generated_docs` additionally refuses
non-accepted names — so the ordinary docs axis cannot settle a non-accepted
name by design, closing the description-repair loop from that side too.

## Family state after this node (poloidal_beta and toroidal_beta untouched)

| id | name_stage | docs_stage | validation_status | notes |
|---|---|---|---|---|
| `beta` | **reviewed** (was exhausted) | pending | **valid** (was quarantined) | stale validation scalar cleared; parked at reviewed by the gate |
| `plasma_beta` | accepted | accepted | valid | untouched, still carries its two PRODUCED_NAME sources |
| `poloidal_beta` | accepted | accepted | valid | untouched, correct, still a child of beta |
| `toroidal_beta` | accepted | pending | valid | untouched, correct, still a child of beta |
| `normalized_toroidal_plasma_beta` | accepted | accepted | valid | untouched |

The REFINED_FROM history edges involving `beta`,
`normalized_toroidal_plasma_beta` and `normalized_toroidal_beta` are untouched;
no edge was created or deleted.

## Blocker

The ordered route cannot complete. Step 1's expected outcome (accepted) is
unreachable because the semantic gate rejects the bare anchor `beta` for every
description measured; steps 2 and 3 are then refused by their own accepted-name
guards. The necessary change — a sanctioned way for the review pipeline to admit
a bare family-anchor name (an exemption in `review_name`'s semantic gate, shaped
like the existing deterministic-parent exemption at workers.py:8197-8206, or an
alternative accept authority) — is a pipeline change plus a design decision the
plan does not settle. It exceeds this node's write scope (docs/evidence only),
so it is reported, not implemented.

## Options

1. Add a family-anchor exemption to the semantic gate (a code change in
   `imas_codex/standard_names/workers.py`, analogous to the existing
   deterministic-parent skip), then re-run `sn rescore beta`. The exemption
   should key on the name being a bare base with `HAS_PARENT` children carrying
   the disambiguating qualifiers — the same structural test the deterministic
   parents use, generalised.
2. Run `sn rescore beta --stage-only` repeated-with-acceptance is not possible;
   there is no CLI accept authority for a below-gate name. The `sn approve`
   route is for folded catalog PRs, not pipeline names.
3. Reconsider the endpoint under the gate's evidence: if the family anchor must
   stay bare, the gate needs the exemption; if the gate is authoritative, the
   anchor spelling needs the qualifier after all (contradicting the lead ruling).

Leaning: option 1 — it matches the precedent and the lead's ruling, and the
measured family (poloidal/toroidal members at 0.77–0.85) shows the qualifier is
exactly what the gate rewards. Cost if wrong: a wrongly-scoped exemption would
let genuinely ambiguous bare names through; it must be pinned to the
parent-with-children structural test and gated by the standard-names suite.

## Evidence inputs for the report

- `sn rescore beta` real output: `rescored beta (exhausted → drafted,
  run_id=sn-rescore-20260905T140457Z)`; inline review "below threshold",
  stage `reviewed`, score 0.30.
- Gate measurements: table above, computed via
  `imas_codex.standard_names.audits.semantic_similarity_check` against the live
  embedding server; current-node sim 0.4877 reproduces the value the pipeline
  stored (`semantic_sim=0.4877`).
- Dry-run refusals: supersede → "target 'beta' is name_stage='reviewed', not
  'accepted'"; docs edit → "docs edits require an accepted name".
- LLM spend: the rescore inline review cost $0.0000 (no LLM seats answered; the
  gate short-circuited before the chain). No further spend was made.
