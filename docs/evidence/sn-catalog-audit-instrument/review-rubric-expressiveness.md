# Can the reviewer rubric express the defects the audit found?

provisional: false — all three questions are answered, every quoted line is verified against the file it names, and the result table is closed.

The WEST name audit found 74 defective bindings over 61 identities that the review pipeline had
already scored and accepted. Before asking why the reviewer missed them, this asks a prior
question: **if a correct reviewer had noticed each defect, was there a dimension to score it on and
a field to carry the correction?** A rubric that cannot express a defect makes a reviewer's
correctness irrelevant.

Read: `imas_codex/llm/prompts/sn/review_names.md` (438 lines),
`review_names_system.md` (198), `review_names_user.md` (227), `review_docs.md` (220),
`review_docs_system.md` (185), `review_docs_user.md` (119), `review_docs_parent_system.md` (120),
`review_docs_parent_user.md` (65), `review_description_system.md` (73),
`review_description_user.md` (33) and `review.md` (334), together with the structured output models
in `imas_codex/standard_names/models.py` the reviewer must fill.

## 1. The scoring dimensions, and what a low score means

The name-only path — the one the WEST cut ran — scores **four dimensions, 0–20 each, normalised
over 80** (`StandardNameQualityScoreNameOnly`, `models.py:1371`). The full path adds
`documentation` and `compliance` for six over 120 (`StandardNameQualityScore`, `models.py:1268`).
The numeric total is the decision, stated in the prompt at `review_names_system.md:173`:

> The numeric score is the decision — downstream code accepts when `score >= min_score`. **Do not**
> add a separate accept/reject vote.

| dimension | what the prompt says it measures | what a low score actually means |
| --- | --- | --- |
| `grammar` (0–20) | "**Round-trip + controlled base.** The base slot is a controlled vocabulary" (`review_names_system.md:121`) | the name does not parse, or uses a token no registry declares. A *mechanical* verdict — the grammar library can answer it. |
| `semantic` (0–20) | "**Self-descriptiveness + cross-name consistency + provenance + physical correctness.** This is the most important dimension" (`:130`) | four different failures share one number: opaque name, disagreement with siblings, wrong DD feature, wrong physics. A reader of the score cannot tell which. |
| `convention` (0–20) | "**Readability + clash hunt + style.**" (`:146`) | snake_case, segment order, abbreviations, sibling clash, readability. |
| `completeness` (0–20) | missing segments — "No missing `subject` when required" (`:157`) | a physically relevant segment is absent. |

Tier mapping is `outstanding` 68–80, `good` 48–67, `inadequate` 32–47, `poor` 0–31
(`review_names_system.md:165-169`).

**The first structural finding is in that table.** `semantic` carries four independent questions and
returns one integer. The prompt itself calls self-descriptiveness "worth up to 10 of 20 points"
(`:132`) and source-fidelity a hard cap "≤ 6/20" (`:142`), so two of the four sub-criteria can each
sink the dimension alone — which means a name that is perfectly self-descriptive and bound to the
wrong object scores the same 6 as a name that is opaque and correctly bound. The dimension is
expressive enough to *dock*, and not expressive enough to *say what is wrong* without the free-text
comment beside it.

## 2. The carrier fields, which is where most of the answer lives

Before the per-class verdicts, the fields a reviewer can actually write into, on the name-only path
(`StandardNameQualityReviewNameOnly`, `models.py:1406`):

| field | type | what it can carry |
| --- | --- | --- |
| `scores` | 4 × int | the numbers above |
| `comments` | 4 × `str \| None` | one per dimension, "a one-sentence reason" (`review_names_system.md:181`) |
| `reasoning` | `str` | "Specific justification per dimension" |
| `revised_name` | `str \| None` | a corrected spelling |
| `suggested_name` | `str \| None` | a second corrected spelling |
| `suggestion_justification` | `str \| None` | "1–3 sentence justification for `suggested_name`" |
| `issues` | `list[str]` | free text, unstructured |
| `dd_gaps` | `list[DDGapEvidence]` | "Flag-only DD declaration evidence; **independent of review scores and name-stage decisions**" |

Two properties of this set govern every verdict below.

**A correction is a single string.** `revised_name` and `suggested_name` each hold one name. There
is no field of any shape that can carry *two* names, so a defect whose correction is a **split**
has no representation at all — not a weak one, none. The reviewer's only option is prose in
`issues` or `reasoning`, which no downstream consumer parses.

**A suggestion that does not parse is silently deleted.** `_clear_unparseable_suggestion`
(`models.py:1445`) runs `compose(parse(suggested_name).ir)` and on any exception sets both
`suggested_name` and `suggestion_justification` to `None`, logging a warning. So a reviewer who
correctly identifies that the right name needs a token the grammar does not have loses the
suggestion entirely — the mechanism the prompt provides for that case is a `vocab_gap`, but
`vocab_gap` is a *generator* output, not a field on any review model.

## 3. The six defect classes, against dimension and field

| # | defect class | dimension that expresses it | field that carries the correction | verdict |
| --- | --- | --- | --- | --- |
| 1 | redundant spelling of a base the catalog already standardises | **yes** — `convention` | **yes** — `revised_name` | **expressible** |
| 2 | name asserts more than the DD text supports | **yes** — `semantic` (source-fidelity) | **yes** — `revised_name` | **expressible** |
| 3 | name bound to the wrong object within its container | **yes** — `semantic` (source-fidelity) | **yes** — `revised_name` | **expressible, but see the sibling blindness below** |
| 4 | wrong modifier that is a physics error, not a style issue | **partly** — `semantic`, undifferentiated | **yes** — `revised_name` | **expressible, unreportable as physics** |
| 5 | identity whose bindings span loci of different kinds | **no dimension names it** | **no field of any shape** | **NOT EXPRESSIBLE** |
| 6 | documentation contradicts its own bindings | `physics_accuracy` on the docs path only | `revised_documentation` | **NOT REACHABLE — the prompt forbids the check** |

### 1. A redundant spelling — expressible, and the prompt asks for it by name

`convention` covers it explicitly (`review_names_system.md:153`):

> **Clash with siblings**: does the name closely mirror a `same_base_neighbour` while differing only
> in arbitrary or noisy ways (extra/missing trailing token, alternate spelling)? Dock and cite the
> sibling.

`revised_name` carries the fix and `comments.convention` carries the citation. **The rubric is not
the reason this class survived into the cut.** The audit's example is instructive about what is:
`hard_xray_brightness` sits at `m^-2.s^-1.sr^-1` beside four accepted `photon_radiance` names, but
they carry a *different* `physical_base`, so they are not `same_base_neighbours` and the clause
above never fires. The dimension exists; the comparator that would trigger it does not. That is
§3's unit-anchored comparator, and it is a *input* gap rather than a rubric gap.

### 2. Asserting more than the DD supports — expressible, with the strongest language in the rubric

`semantic` carries it as the source-fidelity criterion (`review_names_system.md:142`), and the
prompt calls it the top failure mode in its own words:

> **Source-fidelity (CRITICAL — hard cap the whole dimension at ≤ 6/20):** every locus / subject /
> feature token in the name MUST denote the SAME physical feature named in the DD `source_paths`.
> … This is the **#1 silent failure**: when the generator cannot express the exact DD feature with
> a registered token, it substitutes the nearest registered one and the name looks fine. Your job
> is to catch it.

The rubric is fully adequate here. A reviewer noticing `surface_temperature` on an *apparent*
temperature has a dimension, a hard cap, a comment slot and `revised_name`.

### 3. Bound to the wrong object within its container — expressible, and the prompt's own example is a WEST row

The same source-fidelity clause covers it, and the example the prompt chose is from this very
cohort (`review_names_system.md:142`):

> a source path `.../strike_point_inner_r` names the **inner strike point**, so
> `radial_coordinate_of_inner_divertor_target` substitutes a different feature … and MUST be capped
> ≤ 6/20 with the mismatch cited.

**The rubric names the exact family the audit later found defective and still shipped it.** So for
this class the finding is sharp: expressiveness is not the constraint, and the failure is upstream
of the rubric — in whether the reviewer is shown the *sibling* spellings inside one container.
`review_names_user.md:121` does pass the bindings —

> `{% if item.source_paths %}- **Source paths** (authoritative bound-source cohort): {{ item.source_paths | join(', ') }}`

— but nothing passes *what the other coordinates of the same DD container are called*. The
aperture-centre defect is that `.../centre/phi` says `_of_aperture` while `.../centre/r` says
`_of_measurement_position`; a reviewer holding only its own row cannot see the disagreement, because
the disagreeing name is a different review item in a different batch.

### 4. A wrong modifier that is physics rather than style — expressible, unreportable as physics

A reviewer can dock it: `semantic` is defined to include "physical correctness"
(`review_names_system.md:130`). What it cannot do is **say that it is physics**. There is no
`physics` dimension on the name path — `physics_accuracy` exists only on the *docs* rubric
(`StandardNameQualityScoreDocs`, `models.py:1475`), and the name path's four dimensions are
`grammar`, `semantic`, `convention`, `completeness`.

So `total_power_due_to_ion_cyclotron_heating` on coupled-rather-than-launched power and an opaque
compound both land as "low semantic", and the only thing separating them is one free-text sentence
in `comments.semantic`. **A triage reading scores cannot rank a physics error above a readability
complaint**, because after scoring they are the same number on the same axis. That is not a missing
capability — it is a missing *distinction*, and it matters exactly when a human is deciding which
of 230 identities to look at first.

### 5. Bindings spanning loci of different kinds — NOT EXPRESSIBLE

This is the one class with **no dimension and no field**.

- **No dimension names it.** `grammar`, `semantic`, `convention` and `completeness` are all
  properties of one name against one context. None asks whether the identity's *set* of bindings is
  homogeneous.
- **No field can carry the correction**, and this is the harder half. The repair for
  `radial_coordinate_of_measurement_position` bound to an ECE position, a camera aperture centre and
  a magnetic probe position is **three names**. The model offers `revised_name: str | None` and
  `suggested_name: str | None` (`models.py:1406`) — two slots, each one string, and neither is
  addressed to a particular binding. There is no per-binding structure anywhere in
  `StandardNameQualityReviewNameOnly`.

A reviewer who saw the whole cohort, reasoned correctly, and wanted to say *"split this into three,
here is which binding gets which"* would have to write it into `issues: list[str]` as prose. Nothing
downstream reads `issues` as a correction.

**This is the class the WEST audit found 10 times, covering 24 bindings, and it is the class the
audit called the one a reader cannot repair.** The rubric agrees with that assessment by
construction: it cannot represent the repair either.

### 6. Documentation contradicting its own bindings — the prompt forbids the check

The measured case: `vertical_coordinate_of_strike_point` is bound to both
`summary/boundary/strike_point_inner_z/value` and `.../strike_point_outer_z/value`, and its
documentation says it applies specifically to the **inner** leg — affirmatively false about half its
bindings. Its radial twin documents the opposite and correct convention.

On the face of it the docs rubric handles this. `physics_accuracy` is defined as
(`review_docs_system.md:111-122`):

> **This dimension is a claim-level verification against the DD Ground Truth block, not a fluency
> judgment.** … **Contradiction**: any definitional claim that contradicts the DD ground truth or an
> accepted sibling's definition → **physics_accuracy ≤ 5** and an entry in `issues` naming the claim
> and the contradicting source.

And `revised_documentation` exists to carry the fix (`models.py:1545`). **But the docs prompt hands
the reviewer the bindings under an instruction to penalise their use** (`review_docs_user.md:19`,
and identically `review_docs.md:148`):

> `- **Source paths** (provenance context — dock if cited in output): {{ item.source_paths | join(', ') }}`

Compare the name path's framing of the same field — "**authoritative bound-source cohort**"
(`review_names_user.md:121`). The two prompts disagree about what the bindings are *for*: on the
name path they are the ground truth a name is checked against; on the docs path they are provenance
the documentation must not mention.

The consequence is precise. A docs reviewer that checked "does this documentation hold for every
path in `source_paths`?" would find the strike-point contradiction immediately — both paths are in
the list, the parent text of each names its own leg. The rubric would then let it score
`physics_accuracy ≤ 5` and write `revised_documentation`. **The dimension and the field are both
there; the instruction that would make the reviewer look is not, and the instruction that is there
points the other way.** So this class is not a rubric gap but a prompt gap, and it is a one-line
change with a dimension already waiting for it.

## 4. Does the rubric encode that ordinal dimensions are not carried in names?

**Yes, on the name path, and it is the most forcefully stated rule in the whole prompt set.**
`review_names_system.md:20-38` carries it under its own heading, flagged HARD:

> ## Positional samples never enter Standard Name identity — HARD
>
> Never emit, propose, approve, or refine a Standard Name that encodes an ordered sample or
> endpoint. Positional words such as **first, second, third, start, end** and equivalent
> sample-position labels remain in the DD path and source description as provenance, never in the
> identity. Apply this rule only when the source structure proves that the word indexes a point or
> sample; do not strip a registered semantic token such as `first_wall`, or `start`/`end` when it
> names a state or process rather than sample position.

The block is well-built in three ways a reviewer needs. It **names the verbs** — "approve, or
refine" — so it binds the reviewer and not only the generator. It **carves out the false
positives** (`first_wall`, a `start` that names a state), which is the failure a blunter rule would
cause. And it **says what to do instead**, closing the loop the rubric otherwise leaves open
(`:31-38`):

> Dropping the positional label must preserve the exact quantity, owner, carrier, geometry
> representation, axis, mechanism, and locus. If the same non-ordinal identity needs an unavailable
> carrier or locus token, emit a `vocab_gap` for that exact token. Never borrow `line_of_sight` or
> another object's identity. Thus `radial_coordinate_of_arc_of_circle_start_point` is forbidden …

It is also **backed in code rather than only asserted in prose**, which is unusual for a prompt
rule: `_is_ordinal_point_sample` (`imas_codex/standard_names/workers.py:2685`) recognises the shape
structurally — "the DD's `point`/`points` sampling noun and/or an ordinal position word, rather
than by an exhaustive path list" — and distinguishes it from a genuine second field of a device
(`camera/direction` vs `camera/up`).

**Two gaps, both in reach rather than in content.**

*The block sits on the name path only.* Of the ten reviewer prompt files, three carry it —
`review_names_system.md`, `review_names.md:50` and `review.md:48` — and **seven do not**:
`review_names_user.md`, `review_docs.md`, `review_docs_system.md`, `review_docs_user.md`,
`review_docs_parent_system.md`, `review_description_system.md`, `review_description_user.md`. A
docs reviewer writing `revised_documentation` for a line-of-sight identity has no instruction that
"the first reference point" is provenance rather than identity — and the WEST audit's most-repeated
description defect is exactly that: a shared line-of-sight description naming "the first reference
point" while the identity is bound to second and third points across 14 bindings.

*Nothing scores it.* The rule is stated HARD but lands on no dimension. `grammar` is a parse
question and an ordinal-qualified name parses; `convention` is style; `semantic` would take it under
"cross-name consistency" only by a reviewer's own choice. So a reviewer following the rule has no
told place to put the dock, and a reviewer proposing an ordinal-qualified `suggested_name` is not
refused by the model validator either — `_clear_unparseable_suggestion` (`models.py:1445`) only
rejects what fails a grammar parse, and `radial_coordinate_of_arc_of_circle_start_point` parses
cleanly. **The one rule with a code-level recogniser has no code-level enforcement at the review
boundary**, so an ordinal-qualified suggestion reaches downstream intact and must be rejected by
hand later.

## Result

| question | answer |
| --- | --- |
| scoring dimensions on the path the WEST cut ran | **4** (`grammar`, `semantic`, `convention`, `completeness`), 0–20 each over 80 |
| dimensions on the full path | 6 over 120; `physics_accuracy` exists on the **docs** rubric only |
| defect classes expressible with both a dimension and a correction field | **4 of 6** |
| classes with **no** dimension and **no** field | **1** — bindings spanning loci of different kinds |
| classes with dimension and field but an instruction pointing away | **1** — documentation contradicting its own bindings |
| classes expressible but not separable from style after scoring | **1** — a wrong modifier that is a physics error |
| reviewer prompt files carrying the ordinal rule | **3 of 10** |
| dimensions the ordinal rule can be scored on | **0** — stated HARD, docked nowhere |
| correction fields able to carry more than one name | **0** |

**The headline is the last row.** Four of the six classes are expressible and the rubric's language
for two of them is sharper than the audit's own — it calls source-fidelity "the **#1 silent
failure**" and uses a WEST strike-point path as its worked example. So for most of what the audit
found, *a correct reviewer had somewhere to put it*, and the reason the defects shipped is upstream
of the rubric: in the comparators the reviewer is shown, and in the fact that a per-identity review
never holds two members of a container at once.

The genuine expressiveness gap is narrow and specific: **a review model whose only correction slots
are two single strings cannot propose a split**, and a split is the repair for the one class the
audit called unrepairable by a reader. Widening that is not a prompt edit — it needs a field, and
the natural shape is a per-binding list rather than a second string.

The second finding is cheaper and sharper: the docs prompt tells the reviewer to **dock the output
for citing the source paths** while the name prompt calls the same field the **authoritative
bound-source cohort**. One of those two sentences is wrong about what bindings are for, and the
measured strike-point documentation defect is what the disagreement costs.
