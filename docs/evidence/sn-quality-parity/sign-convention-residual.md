# Conditional sign-convention residual closure

Snapshot: 2026-08-25, live `codex` graph. Status: **qualified pass**. The
owned deterministic cohort is clean and `magnetic_field` earned fresh
documentation acceptance through the ordinary quorum. A graph-wide rescore is
deliberately deferred until the concurrent documentation drain reaches
`no_eligible_work`.

## Outcome

The three identities that had become accepted outside the frozen 280-row
repair were re-diagnosed against fresh live DD authority and repaired under
their own signed deterministic manifest:

| Identity | Fresh authority | Exact change | Model spend |
|---|---|---|---:|
| `length_of_poloidal_magnetic_field_probe` | `magnetics/b_field_pol_probe/length`, whose DD `cocos_transformation_type` is null | Removed only the unsupported final sign-convention paragraph; retained `one_like`, scalar COCOS 17, and the single `HAS_COCOS` edge | USD 0.00 |
| `radial_coordinate_at_inboard_midplane` | `equilibrium/time_slice/profiles_1d/r_inboard`, whose DD `cocos_transformation_type` is null | Same bounded paragraph removal and metadata preservation | USD 0.00 |
| `ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope` | `spectrometer_visible/channel/isotope_ratios/isotope/density_ratio`, whose DD `cocos_transformation_type` is null | Same bounded paragraph removal and metadata preservation | USD 0.00 |

The manifest SHA-256 is
`dff5809f306f2f3662d21376cb9e645d6ed4e38e0ff4a5b8b4686370b6eed0ea`.
It admitted and changed 3/3 rows with zero refusals, removed 463 characters in
total with a maximum per-document delta of 198 characters, and wrote durable
change id
`sn-change:sign-convention-document-repair:dff5809f306f2f3662d21376cb9e645d6ed4e38e0ff4a5b8b4686370b6eed0ea`.
All three documents remain accepted at their pre-repair documentation scores:
0.875, 0.91875, and 0.88125 respectively. The node-scoped `LLMCost` census is
0 calls / USD 0.00.

Representative bounded change: the probe-length document previously ended
with “Sign convention: Positive when the probe extent is measured along the
assigned local normal sensing-axis direction.” The repair removed that final
paragraph and no preceding character. The accepted identity and its
DD-authoritative source binding are unchanged.

## `magnetic_field` ordinary quorum

Fresh structural evaluation reproduces the diagnosis exactly: `magnetic_field`
has 16 children, 15 survive the production eligibility rule, and their sole
non-null transformation class is `b0_like`, supplied by
`vacuum_magnetic_field`. The node is `origin=catalog_edit`, so the derived-parent
materializer is not its authority and correctly returned no eligible row. No
hand-written Cypher or direct acceptance was used.

The sanctioned `sn edit` path was dry-run first, then applied with self-only
scope and run id `sn-edit-20260825T024100Z`. Its reason records that this is the
single identity excluded from the deterministic repair because it is genuinely
COCOS-sensitive rather than invariant. Its hint required the documentation to
state the sign convention implied by `b0_like` while preserving the existing
definition, equation, symbol definitions, semantic scope, relationships, and
valid links.

The exact scope ran through the ordinary generate-docs and review-docs pools.
It completed with `stop_reason=no_eligible_work`, `docs_stage=accepted`, and
`edit_status=applied`; no claim remains. The two fresh reviewer scores were
0.8625 and 0.9625, aggregating to **0.9125**, so no refine-docs call was needed.
The accepted document now ends with:

> Sign convention: Positive when the vacuum toroidal magnetic-field component
> is directed along increasing toroidal angle $\phi$.

Scored against the freshly established structural `b0_like` authority, the
conditional gate returns **pass** with reason “COCOS-sensitive quantity has
canonical sign-convention prose.” The run cost **USD 0.092721** against a USD
1.00 ceiling: USD 0.005402 for generation and USD 0.087319 for the two-reviewer
quorum. This half made 3 model calls and no refinement call.

## Graph-wide gate

The documented post-frozen-repair baseline was **325 pass / 4 fail / 2,411
not-evaluable** across 2,740 accepted documents. After the three exact
deterministic repairs, an interim observation read **331 pass / 9 fail / 2,528
not-evaluable** across 2,868 accepted documents. That change is not a regression
of this cohort: its exact three rows score 3/3 pass after repair.

The graph-wide numbers moved because a concurrent documentation drain was
promoting backlog drafted under the pre-fix prompt. The producer fix prevents
newly generated invariant documentation from carrying the forbidden paragraph,
but it cannot rewrite already-drafted text before review promotes it. Accepted
coverage rose from 2,800 at this node's first live census to 2,868 at the
interim census, so a graph-wide zero was not a stable done-when during this
campaign.

No further graph-wide rescore was attempted after `magnetic_field` landed. The
next meaningful graph-wide measurement is a separate post-drain node, run only
after the concurrent campaign records `stop_reason=no_eligible_work`. The
interim 331/9/2,528 result is retained as an observation of campaign sequencing,
not as this node's verdict.

## Verdict

**Qualified pass.** The owned invariant cohort was repaired deterministically
at **USD 0.00**, under a fresh signed manifest, and is **3/3 pass**. The one
genuinely sensitive identity was regenerated through sanctioned `sn edit`,
earned ordinary-quorum acceptance at **0.9125**, and carries the canonical
`b0_like` sign paragraph at **USD 0.092721**. Total node model spend is therefore
**USD 0.092721**: zero for the deterministic half and USD 0.092721 for the
reviewed half. Graph-wide closure is intentionally sequenced after the active
documentation drain and is not claimed here.
