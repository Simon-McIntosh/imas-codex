# Can the reviewer rubric express the defects the audit found?

provisional: true — findings are appended as they land; the closing pass clears this line.

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
The numeric total is the decision, stated in the prompt at `review_names_system.md:172`:

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
(`models.py:1444`) runs `compose(parse(suggested_name).ir)` and on any exception sets both
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
