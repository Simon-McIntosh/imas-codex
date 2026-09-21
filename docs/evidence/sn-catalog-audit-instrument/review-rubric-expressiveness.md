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
