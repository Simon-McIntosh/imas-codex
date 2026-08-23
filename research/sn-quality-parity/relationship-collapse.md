# Why the essential-relationships score collapses outside the holdout

## Verdict

The **gate definition, not injected related-name context, produces the observed
80/85 versus 14/53 split**. Both evaluation populations were rendered with
exactly **zero candidate-specific related Standard Names per row**, for both
passing and failing rows. The gate then classifies a document as passing when it
contains *any* Markdown link or one of a short list of relationship phrases. On
the outside-holdout receipt, all 14 passes contain a Markdown link and all 39
failures contain neither a link nor a recognized phrase.

This does not establish that 39 outside-holdout documents lack an essential
semantic relationship. It establishes that the generator emitted the lexical
witness accepted by this gate for 14 of them. A representative false negative
is `vertical_coordinate_of_antenna_strap` at
`ic_antennas/antenna/module/strap/geometry/thick_line/second_point/z`: its prose
says that the quantity works “together with the other cylindrical coordinates”
to locate the geometry, but the gate is false because that relationship is
neither linked nor expressed with a phrase in the gate's allow-list. By
contrast, `carbon_density_at_divertor_target` at
`summary/local/divertor_target/n_i/carbon/value` passes because it contains the
inline link `[carbon density](name:carbon_density)`
([bulk-sample.json:162-186](/home/ITER/mcintos/.local/share/imas-codex/receipts/standard-names/sn-quality-parity/bulk-sample.json:162),
[bulk-sample.json:243-267](/home/ITER/mcintos/.local/share/imas-codex/receipts/standard-names/sn-quality-parity/bulk-sample.json:243)).

## Quantitative result

The receipts report **80/85 = 0.941176** on the holdout and
**14/53 = 0.264151** outside it, a pass-rate difference of **-0.677026**
([comparison receipt:85-90](/home/ITER/mcintos/.local/share/imas-codex/receipts/standard-names/sn-quality-parity/comparison-result-independent-gates.json),
[bulk receipt:109-116](/home/ITER/mcintos/.local/share/imas-codex/receipts/standard-names/sn-quality-parity/bulk-sample.json)).

Re-rendering the exact receipt populations through the evaluator's item
constructor gives the following candidate-specific related-name input census:

| Population | Rows | Gate pass/fail | Injected candidate related Standard Names per row | Population total |
|---|---:|---|---:|---:|
| Holdout | 80 | pass | 0 (range 0–0) | 0 |
| Holdout | 5 | fail | 0 (range 0–0) | 0 |
| Outside holdout | 14 | pass | 0 (range 0–0) | 0 |
| Outside holdout | 39 | fail | 0 (range 0–0) | 0 |

The reconstruction checked all **138** rendered prompts and found zero
candidate-specific parent, component, base, derivative, peer, related-quantity,
nearby-name, or sibling-family blocks in every group
([rendered-context check:1-6](/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T092120584012-inv-relcollapse/rendered-context-check.log)).
On the outside-holdout receipt, the lexical witness census is exact: **14/14
passes have a Markdown link; only 2/14 also have a recognized relationship
phrase; 0/39 failures have either witness**
([relationship analysis:4-8](/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T092120584012-inv-relcollapse/relationship-analysis.log)).

## Causal chain

| Link | Mechanism and evidence |
|---|---|
| 1. Holdout rows become minimal candidates | `_candidate_for()` supplies only the catalog name, one DD path, catalog description, unit/kind/domain placeholders and source path; `_generation_item()` retains only those scalar fields and adds empty review/history fields. It does not attach parent, component, base, derivative, peer, neighbour, nearby-name, locus, or sibling-family candidates ([docs_holdout_eval.py:74-97](../../imas_codex/standard_names/docs_holdout_eval.py)). |
| 2. The holdout uses the benchmark generator | The evaluator constructs those candidates and calls `generate_docs_for_candidates()` directly ([docs_holdout_eval.py:192-204](../../imas_codex/standard_names/docs_holdout_eval.py), [docs_holdout_eval.py:228-238](../../imas_codex/standard_names/docs_holdout_eval.py)). |
| 3. The generator discards enrichment-shaped extras | For every call, `generate_docs_for_candidates()` rebuilds the prompt item from only name, unit, kind, domain, description, source paths, review fields and empty history, then renders the user prompt. Even if a caller had supplied additional graph context on the candidate, this constructor would not forward it ([benchmark.py:820-861](../../imas_codex/standard_names/benchmark.py)). The bulk run used this same function with the same minimal candidate shape; its receipt records the input name, description, domain, unit and path beside the output, but no related-name context ([bulk-sample.json:160-186](/home/ITER/mcintos/.local/share/imas-codex/receipts/standard-names/sn-quality-parity/bulk-sample.json)). |
| 4. Related-name sections are conditional and therefore absent | The prompt emits parent/component/base/derivative blocks only when those item fields exist ([generate_docs_user.md:154-206](../../imas_codex/llm/prompts/sn/generate_docs_user.md)); it emits nearest peers and relationship neighbours only when populated ([generate_docs_user.md:245-260](../../imas_codex/llm/prompts/sn/generate_docs_user.md)); and the authoritative nearby-name list only when batch context supplies it ([generate_docs_user.md:288-301](../../imas_codex/llm/prompts/sn/generate_docs_user.md)). None is supplied by links 1–3. |
| 5. The model is nevertheless ordered to produce a relationship | The system prompt independently requires a relationship to the nearest parent/component/total/normalized/averaged/derivative/integral quantity and asks for an inline link when an available name exists ([generate_docs_system.md:40-67](../../imas_codex/llm/prompts/sn/generate_docs_system.md)). With no candidate list, the model can only infer a relation or related identifier from the Standard Name and draft description. |
| 6. The scorer checks a lexical proxy, not contextual correctness | `_has_essential_relationship()` returns true for any Markdown link **or** any occurrence of `relates to`, `relationship`, `depends on`, `proportional to`, `integral of`, `gradient of`, `derivative of`, `normalized by`, `defined from`, or `obtained from` ([docs_gates.py:53-59](../../imas_codex/standard_names/docs_gates.py), [docs_gates.py:110-111](../../imas_codex/standard_names/docs_gates.py)). It does not require that the link target was supplied, exists, is the nearest relevant quantity, or that the prose states the correct relationship. Link hygiene separately checks only target syntax and bare brackets ([docs_gates.py:129-135](../../imas_codex/standard_names/docs_gates.py)). |
| 7. Population composition controls how often the model invents that witness | The holdout's 85 path rows repeat only **13 catalog identities**, while the outside sample has **53 unique identities**. The repeated holdout is dominated by readily relational families such as `poloidal_magnetic_flux` (20 rows), magnetic-field components (28 rows), `electron_temperature` (13 rows), and `electron_density` (9 rows); only five rows fail, across `elongation_of_plasma_boundary`, `poloidal_magnetic_flux`, `radial_magnetic_field`, and `vertical_magnetic_field` ([comparison receipt:875-933](/home/ITER/mcintos/.local/share/imas-codex/receipts/standard-names/sn-quality-parity/comparison-result-independent-gates.json), [relationship analysis:9-11](/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T092120584012-inv-relcollapse/relationship-analysis.log)). The broad sample supplies one identity per row, so incidental derivation of a link from name morphology is much less common. |

## Boundary with the production pipeline

The receipts use the production documentation model and prompt, but **not the
production context-enrichment path**. The live pool enriches each accepted name
with DD source material, related graph neighbours, nearest Standard Name peers,
parent/component/derivative structure and sibling-family context
([workers.py:8623-8646](../../imas_codex/standard_names/workers.py)); it then
loads batch-wide nearby accepted names and places them in the prompt context
before rendering ([workers.py:9068-9097](../../imas_codex/standard_names/workers.py),
[workers.py:9143-9167](../../imas_codex/standard_names/workers.py)). The benchmark
generator bypasses that enrichment.

Therefore the receipts do **not** show that production-injected relationship
context generalizes poorly: they inject none. They show that the current
`essential_relationships` gate is a brittle presence test whose score is highly
sensitive to whether a model, given only a name and draft description, happens
to emit a syntactically valid link or one of nine phrases. The correct
attribution for the 80/85 versus 14/53 discrepancy is **the gate definition
acting on population-dependent lexical output**, not a difference in injected
candidate context.

The actionable consequence is to keep the generalization result qualified. A
future comparison intended to authorize the WEST refresh must either (a) run
both populations through the production enrichment path and record the injected
candidate list per row, or (b) rename this gate as a link/relationship-phrase
presence proxy. Strengthening it into a semantic relationship gate would also
require validating the target and the stated mathematical/semantic relationship;
the current `link_hygiene` check does neither.
