# Generation-evidence origin recovery

## Verdict

**REFUSAL — the generation-evidence predicate is observable but is not an
origin authority.** The required held-answer error count is **0**. At the live
measurement time, **2026-08-25T15:03:54.031637Z**, the proposed predicate
misclassified **1,372 of 1,798** accepted controls. It projected every control
to `pipeline`: all 426 declared `pipeline` controls were correct, but all 1,372
declared `catalog_edit` controls were wrong.

The result explains all five earlier import-marker counterexamples, but that
five-case success is not sufficient evidence for recovery. The same predicate
also classifies the complete accepted `catalog_edit` cohort as pipeline output.
No `origin` or `status` value was written.

The complete machine-readable measurement is retained at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T145736650094-n-originrecovery/logs/origin-rule-final-measurement.json`.

## Rule tested

The read-only rule was evaluated exactly as follows:

```text
generation_evidence = (
    model is present
    or generated_at is present
    or chain_length is present
    or an LLMCost.standard_name_ids entry equals the StandardName.id
)

if generation_evidence:
    project origin = pipeline
else:
    project origin = catalog_edit
```

This is deliberately broader than the three direct scalar fields: it recovers
surviving provider-call evidence through the schema-owned multivalued
`LLMCost.standard_name_ids` property. It does not query receipts by operation
name and it does not use `imported_at`, `catalog_commit_sha`, or `source_types`
as origin evidence.

## Schema and zero-result sanity

The live graph held **4,658** `StandardName` candidates. All **4,658** carried
`id` and `name_stage`; 3,567 carried `origin`, 2,319 carried `status`, 2,669
carried `model`, 2,669 carried `generated_at`, and 3,087 carried
`chain_length`. The graph also held **34,914** `LLMCost` candidates; all
34,914 carried both `id` and the aimed property `standard_name_ids`, covering
3,891 distinct Standard Name identities.

These counts prove the instrument was aimed at live schema properties. In
particular, `count(LLMCost.sn_ids)` was **0**, while
`count(LLMCost.standard_name_ids)` was **34,914**. The former is a real zero
for a non-schema legacy guess and was not used; the latter is the positive
property control used by the rule.

Every material zero in this report is paired with the population and the
schema-positive instrument that produced it:

| Reported zero | Population searched | Sanity anchor |
|---|---:|---|
| Accepted controls without any of the four evidence signals: **0** | 1,798 accepted controls | 4,658/4,658 `StandardName.id`; 34,914/34,914 `LLMCost.standard_name_ids` |
| Accepted `catalog_edit` identities without any evidence signal: **0** | 1,372 accepted `catalog_edit` identities | 2,669 `model`; 2,669 `generated_at`; 3,087 `chain_length`; 3,891 cost-linked identities |
| Five old counterexamples left unexplained: **0** | five named accepted `pipeline` controls | all five were found by `StandardName.id`; every one had a cost link through `standard_name_ids` |
| Null-origin identities without any evidence signal: **0** | 1,091 null-origin identities | the same 4,658-name and 34,914-cost schema census |
| Graph-state changes during measurement: **0** | 4,658 identities before and after | identical ordered `id/origin/status/name_stage` SHA-256 before and after |

The ordered graph-state digest was
`9edf3c90fb45c1da138645648f34a086ba5d24efffed20fbaefae0834351ea07`
both before and after the measurement. That equality, together with the use of
read-only `MATCH ... RETURN` queries, verifies that this node performed no
writes. This confirms **no graph mutation**.

## Held-answer measurement

The accepted control predicate was `name_stage = 'accepted'` with a declared
origin of either `pipeline` or `catalog_edit`. It reproduced the complete held
set: 426 pipeline controls plus 1,372 catalog-edit controls, for 1,798 total.

| Declared origin | Controls | Projected `pipeline` | Projected `catalog_edit` | Misclassified | Required |
|---|---:|---:|---:|---:|---:|
| `pipeline` | 426 | 426 | 0 | **0** | 0 |
| `catalog_edit` | 1,372 | 1,372 | 0 | **1,372** | 0 |
| **Total** | **1,798** | **1,798** | **0** | **1,372** | **0** |

Thus the measured error rate is **76.31%** (1,372/1,798), not the required
zero. The predicate detects historical LLM interaction; it does not invert the
declared most-recent editorial origin. This distinction is visible in the
LLMCost phases: a cost link can record name generation, name review, document
generation, document review, or document refinement. The existence of any one
of those calls does not prove that the identity's latest editorial origin is
pipeline generation.

## Size of the supposed human-curated cohort

The graph contains exactly **2,096** identities with
`origin = 'catalog_edit'` across all lifecycle stages. Of those, **1,372** are
accepted. Under the requested four-signal test, **1,372/1,372 accepted
catalog-edit identities (100%)** carry surviving generation evidence; **0** do
not.

The evidence is not limited to one field:

| Surviving signal among accepted `catalog_edit` identities | Count |
|---|---:|
| `model` present | 248 |
| `generated_at` present | 248 |
| `chain_length` present | 410 |
| linked from any `LLMCost.standard_name_ids` | 1,305 |
| union of the four signals | **1,372** |

If the predicate were accepted literally, it would call the entire accepted
catalog-edit cohort pipeline output relabelled by import, not merely a small
contaminated subset of the quoted 2,096. The failed held-answer gate shows why
that number must not be promoted to a recovered-origin fact: the evidence
measures historical pipeline participation, while declared `origin` records
editorial provenance. The measured accepted overlap is real; its interpretation
as mislabelling is not validated.

## The five shared-import counterexamples

All five controls previously missed by the import-marker rule are explained by
the generation-evidence predicate: **5/5 project to `pipeline`; 0/5 remain
unexplained**. Neo4j renders their common instant as
`2026-07-04T21:21:17.079000000+00:00`, which is the same timestamp as
`2026-07-04T21:21:17.079Z`. They also share catalog commit
`a2f8831cf9d14af2f7120969c728f990bdd923cf`.

| Held `pipeline` identity | Direct evidence | LLMCost evidence | Rule result |
|---|---|---|---|
| `electron_density` | none | 15 rows; docs generation/review/refinement and name review | `pipeline` |
| `normalized_toroidal_flux_coordinate` | model, generated-at, chain-length | 117 rows, including name generation | `pipeline` |
| `safety_factor` | none | 17 rows, including name generation | `pipeline` |
| `toroidal_magnetic_field` | model, generated-at, chain-length | 28 rows, including name generation | `pipeline` |
| `vertical_coordinate_of_camera` | model, generated-at, chain-length | 20 rows, including name generation | `pipeline` |

`safety_factor` is the positive control for the LLMCost join: its three direct
generation fields are absent, both import markers are present, yet its 17 cost
rows include name generation and link back through `standard_name_ids`. The
instrument therefore fires on the richer evidence and is aimed at the requested
identity property. `normalized_toroidal_flux_coordinate` separately controls
the direct-field branch because all three scalar signals are present. These
controls prove observability, not the rule's discriminating power; the full
1,798-control measurement supplies that separate test and fails it.

## Null-origin projection, not recovery

All **1,091** null-origin identities carry at least one requested evidence
signal, including all **278** accepted null-origin identities. The untrusted
rule therefore projects:

| Cohort | Projected origin | Projected status | Count |
|---|---|---|---:|
| accepted null-origin | `pipeline` | `draft` | 278 |
| drafted, reviewed, or exhausted null-origin | `pipeline` | `draft` | 108 |
| superseded null-origin | `pipeline` | `superseded` | 705 |
| **All null-origin identities** | **`pipeline`** | **386 `draft`; 705 `superseded`** | **1,091** |

The status projection follows lifecycle semantics rather than import markers:
non-superseded pipeline identities remain `draft`, while identities whose
`name_stage` is `superseded` project to `status = 'superseded'`. The live stage
split is 278 accepted, 13 drafted, 41 exhausted, 54 reviewed, and 705
superseded; 978 currently have null status and 113 already have superseded
status. The 705 superseded identities remain outside the proposed live mutation
scope.

Because the origin rule fails the zero-error authority gate, these are only
counterfactual projections. Writing the 278 accepted identities as
`pipeline`/`draft`, or writing any part of the 1,091-row total, would treat a
historical-interaction predicate as editorial-origin authority. This node
therefore leaves every identity untouched.

## Required follow-on

Recovery needs an event that records the most recent editorial action, or a
generation event that can be ordered against a genuine catalog-edit event.
Presence-only scalar fields and undifferentiated LLMCost membership cannot
provide that ordering. Any replacement rule must again reproduce all 1,798
accepted held answers with **0** errors before it can authorize a graph write.
