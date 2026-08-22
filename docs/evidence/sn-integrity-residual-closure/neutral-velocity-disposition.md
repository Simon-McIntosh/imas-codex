# Neutral internal-state momentum velocity — earned disposition

## Outcome

**Accepted through the ordinary review pool.** The sanctioned rename
invocation replaced
`poloidal_neutral_internal_state_momentum_convected_velocity` with
`poloidal_neutral_internal_state_momentum_convection_velocity`. The replacement
passed the name quorum at **0.9625**, above the **0.85** acceptance threshold by
**0.1125**, and re-read from the production graph as `name_stage=accepted` and
`validation_status=valid`.

No direct acceptance or graph-text edit was used. The predecessor remains as
`name_stage=superseded` history. The successor still has zero producing sources,
so this operation does not claim that attachment has already occurred; it earns
the accepted lifecycle required by the separately governed attachment boundary.

## Live pre-state

The preflight joined the exact identity to incoming
`StandardNameSource-[:PRODUCED_NAME]` relationships and read the following
state before the rename:

| Measure | Live value |
|---|---:|
| Identity | `poloidal_neutral_internal_state_momentum_convected_velocity` |
| Description | Poloidal convective velocity for neutral state momentum transport |
| `name_stage` | `reviewed` |
| `validation_status` | `valid` |
| `reviewer_score_name` | **0.3625** |
| Producing-source count | **0** |
| `refine_attempts` | **0** (null property, canonical zero) |
| Review resolution | `max_cycles_reached` |
| Quorum shortfall | blind seats disagreed and the escalator did not resolve them |

The exact DD realization is
`plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol`, unit
`m.s^-1`. Its DD description identifies the effective poloidal convection
velocity for neutral-species momentum transport in one-dimensional radial
profiles. There was no `StandardNameSource` row for
`dd:plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol`.

The correction was grounded in existing review evidence rather than selected
by preference. The prior primary reviewer and authoritative escalator both
suggested the exact absent replacement
`poloidal_neutral_internal_state_momentum_convection_velocity`; the toroidal
counterpart also uses the `momentum_convection_velocity` construction. The
replacement did not already exist in the graph.

The live plan read at **2026-08-22T21:45:00Z** recorded **6** live unsourced
identities immediately before this node began. The target was one of those six.

## Sanctioned invocation

The nonmutating preflight passed with exit code 0 and planned one self-scoped
rename, no descendant cascade, and direct entry to `review_name`:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
PYTHONPATH="$PWD" uv run --no-sync imas-codex sn edit \
  poloidal_neutral_internal_state_momentum_convected_velocity \
  --rename poloidal_neutral_internal_state_momentum_convection_velocity \
  --reason "The DD path plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol denotes the effective poloidal convection velocity in the neutral internal-state momentum equation; replace the non-canonical convected qualifier while preserving direction, neutral internal-state resolution, momentum transport, and velocity." \
  --scope self --cost-limit 1.00 --dry-run
```

The same command without `--dry-run` applied and completed with exit code 0.
The edit scope id was `sn-edit-20260822T215008Z`; its durable `SNRun` id was
`9dd3cfb2-3454-4f41-9790-25465b258c5e`.

## Earned review result

The live successor and both name-axis review rows were re-read after the CLI
finished:

| Measure | Result |
|---|---:|
| Successor | `poloidal_neutral_internal_state_momentum_convection_velocity` |
| Successor `name_stage` | **`accepted`** |
| Successor `validation_status` | **`valid`** |
| Primary blind-seat score | **0.9375** |
| Secondary blind-seat score | **0.9875** |
| Resolved `reviewer_score_name` | **0.9625** |
| Acceptance threshold | **0.85** |
| Margin above threshold | **+0.1125** |
| Resolution | **`quorum_consensus`** |
| Quorum decision | **accept** |
| Successor producing-source count | **0** |
| Successor `refine_attempts` | **1** |

The predecessor re-read as `name_stage=superseded`, still carrying its
historical score **0.3625** and zero producing sources. The successor's one
refinement attempt is the persisted rename transition; it is not a hidden
hand-accept. The scoped command subsequently completed successor documentation
through the ordinary docs pools as `docs_stage=accepted` with score **0.875**.

## Cost and unsourced-name accounting

The completed `SNRun` re-read as `status=completed`,
`stop_reason=no_eligible_work`, and `cost_is_exact=true`.

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| Live unsourced Standard Names | **6** | **6** | **0** |
| Target-lineage live unsourced identities | **1** old identity | **1** accepted successor | **0** |
| Producing sources on the live target lineage | **0** | **0** | **0** |

The post-read used the canonical ledger predicate: materialized Standard Names
whose `name_stage` is neither `superseded` nor `exhausted`, excluding
deterministic error siblings, with no incoming `PRODUCED_NAME`. It returned six
rows and included the accepted successor. The unchanged count is the exact
effect of this rename: the predecessor leaves the live cohort as the successor
enters it, and neither side owns a producer. This preserves the provenance
truth while lifting the lifecycle refusal.

| LLM accounting | Result |
|---|---:|
| Authorized ceiling | **$1.000000** |
| Exact run cost | **$0.252514** |
| Unspent ceiling | **$0.747486** |
| Ledger events | **5** |

The five events comprise two name-review calls, documentation generation, and
two docs-review calls. The name decision itself was reached by the two blind
review seats; no escalator call was needed.

## Disposition and remaining boundary

The former circular gate is broken without weakening either side:

1. hint steering remains correctly unavailable to a zero-source identity;
2. `sn edit --rename` admitted the reviewer-suggested full replacement without
   requiring a producer, then subjected it to grammar validation and the
   ordinary quorum;
3. the quorum earned `accepted` and `valid` lifecycle for the replacement;
4. the exact DD path remains unattached, so the source ledger still tells the
   truth.

The attachment lifecycle condition is now satisfied. A follow-on may build a
fresh signed attachment proposal for
`plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol` to the
accepted successor and apply it through the governed ordinary-source program.
That attachment is deliberately not folded into this disposition.
