# Mid-pipeline flush receipt

Date: 2026-09-15

This record measures one unscoped downstream drain. The command was deliberately
run with `--flush`, so it could review, refine, generate and review documentation,
and enrich parents, but could neither auto-seed sources nor run the
`generate_name` pool.

## Command and terminal receipt

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv
PYTHONPATH=<assigned-worktree>
uv run --no-sync imas-codex sn run --flush -c 40 -t 35
```

The command exited 0 after 870.811 seconds. The authoritative `SNRun` receipt is
`5027c365-ba86-44e2-9f92-7e13753ab71a`:

| Field | Receipt |
| --- | ---: |
| status | `completed` |
| stop reason | `no_eligible_work` |
| actual spend | **$1.197691** |
| authorised cap | **$40.00** |
| names composed | **0** |
| names enriched | **5** |
| names reviewed | **7** |
| names regenerated | **0** |

The complete terminal transcript is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T123716488637-n-swcr-the-mid-pipeline-backlog-is-drained/flush-output.log`
(318,147 bytes). Its last line records `COMMAND_EXIT_CODE="0"`.

## Before and after census

The baseline positive control saw 5,125 `StandardName` nodes, with `name_stage`
present on all 5,125 and `docs_stage` present on 5,123. This proves the census
was aimed at populated properties before interpreting any count. The supplied
starting figures reproduce exactly when the documentation remainder is split by
stage: 23 accepted names were at `docs_stage=reviewed`, with one further accepted
name pending and one exhausted.

| Population | Before | After | Delta |
| --- | ---: | ---: | ---: |
| all `StandardName` nodes | 5,125 | 5,130 | +5 |
| `name_stage=drafted` | **21** | **22** | +1 |
| `name_stage=reviewed` | **113** | **113** | 0 |
| `name_stage=pending` | **11** | **11** | 0 |
| accepted with `docs_stage!=accepted` | **25** | **25** | 0 |
| accepted with `docs_stage=reviewed` | **23** | **23** | 0 |
| accepted with `docs_stage=pending` | **1** | **1** | 0 |
| accepted with `docs_stage=exhausted` | **1** | **1** | 0 |

The five-node population increase is observed concurrent work, not hidden name
composition by this flush. During the run window, two named identities entered
from the separately running focused compose cohort:
`neutral_beam_charge_number` and
`poloidal_effective_ion_state_momentum_diffusivity`. The flush also materialised
three derived parents while enriching the downstream queue:
`effective_ion_state_momentum_diffusivity`,
`hard_xray_half_width_at_emissivity_peak`, and
`half_width_at_emissivity_peak`. The first concurrent name reached accepted on
both axes through this drain; the second exhausted after a refinement returned
the identical spelling. The raw global stage totals therefore include arrivals
as well as departures and must not be read as the run's own throughput counter.

The run still reached the relevant operational terminal condition: its live
pending-pool watchdog reported zero for `review_name`, `refine_name`,
`generate_docs`, `review_docs`, `refine_docs`, and `enrich_parents` for the
required quiet window, then stopped with `no_eligible_work`. The raw remainder is
therefore not claimable by these pools under the current lifecycle and validation
guards. This is a partial population drain, not a claim that the 171 rows in the
stage census were rewritten.

## The generation pool was untouched

The cost instrument first proved its fields were populated: all 39,687
`LLMCost` rows carried both `pool` and `for_run`. Filtering that same instrument
to the receipt above accounted for all 28 calls and the full $1.197691 spend:

| Pool recorded on this run | Calls | Spend |
| --- | ---: | ---: |
| `enrich_parents` | 3 | $0.000000 |
| `generate_docs` | 5 | $0.027371 |
| `refine_name` | 1 | $0.202595 |
| `review` | 13 | $0.618064 |
| `review_name` | 6 | $0.349661 |
| **`generate_name`** | **0** | **$0.000000** |

This agrees with the separate receipt counter `names_composed=0`. The command's
transcript also names every started pool and contains no
`pool[generate_name#…] starting` line. These are three independent observations
of the same fence: selected-pool structure, per-run cost ledger, and terminal
receipt.

## Preserved negative evidence

Two `x-ai/grok-4.5` reviewer calls returned provider-capacity errors. The quorum
guard fired and deferred the affected documentation review rather than accepting
one successful opinion. The run continued and later accepted the retryable
documentation where complete quorum became available.

`poloidal_effective_ion_state_momentum_diffusivity` scored 0.688 on review and
then exhausted when refinement returned the identical name. This reproduces the
already-recorded no-op-refinement defect: a semantically unchanged proposal still
spent a rotation. Repairing that implementation is outside this node's exclusive
documentation scope, so the failure is preserved as a follow-on rather than
hidden or repaired here.

Startup and post-drain maintenance also reported existing population findings:
21 live sourceless names, 7 names fed by absent DD paths, 9 inconsistent accepted
attachments left protected, and 125 missing derived-parent targets refused for
lack of complete authority. Those global findings predate or sit outside this
node's drain measure. None changes the receipt that the eligible downstream pools
reached zero.
