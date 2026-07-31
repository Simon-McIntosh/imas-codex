# Standard-Names Agent Notes

Scoped to `imas_codex/standard_names/**` and `tests/standard_names/**`. Pipeline
architecture, lifecycle axes, pool semantics and CLI flags live in the repo root
`AGENTS.md` and `docs/architecture/standard-names.md` — this file holds only what
you need when *editing these files*.

## Naming-hygiene keep-list (calibration)

`~/.agents/AGENTS.md` mandates a pre-stage check for plan/stage/bug labels and
changelog prose in filenames, symbols and comments. Its letter class is
deliberately generic (`\b[A-Z][0-9]+[A-Za-z]?\b`), so it is noisy here. These
matches are **legitimate and must be kept** — verified, with the reason:

| Match | Why it stays |
|---|---|
| `S0`–`S11` (`sources/dd_qualifier.py`) | A local numbered rule catalogue, each item with its own description. `tests/standard_names/test_dd_qualifier.py` cross-references them **by id** in ~24 docstrings. |
| `Rule 1`–`Rule 6` (`error_siblings.py`) | Same shape; `test_error_siblings{,_gate}.py` assert on them. |
| `R1`–`R5` (`vocab_token_filter.py`) | Emitted **into** `TokenVerdict.reason` at runtime (`"R4: token contains digits"`); tests assert the substring. Removing them breaks assertions. |
| `D{n}` (compose prompt gallery) | Rendered prompt CONTENT — `test_prompt_completeness` counts `f"D{n} —"` in the rendered system prompt. |
| `rotation_cap` (~40 sites) | A real claim-query kwarg and function parameter. |
| ISN `rc14`/`rc21`/`rc22`/`rc34`/`rc39`/`rc41` | Dependency version contracts recording when a token became valid or a segment opened. |
| `--only <phase>`, `phase_caps`, `LLMCost.phase`, `_PHASE_TO_POOL` | `phase` is a real CLI flag and a real budget/cost dimension here. |
| `W74+`, `W$^{1+}$` | Tungsten charge states in LaTeX descriptions. |
| `T^2`, `m^-2`, `Wb`, `eV`, `A.m^-2` | Physical units. |
| `COCOS 17`, `DDv3`/`DDv4` | Real conventions and DD major versions. |
| `E2E` | "end-to-end". Note the contrast: `E1.`/`E2.` section ids in the *same* file are labels and come out. |
| `noqa` ids (`E402`, `F841`, `S608`, `D401`) | Lint rule codes. |
| `T0 = datetime(...)`, `s1`/`s2`, `m0`/`m1`, `h1`/`h2` | Ordinary locals and constants. |
| `Wave 2D …` in `definitions/clusters/labels.json` | **Physics** — a plasma wave, in two dimensions. Generated cluster labels; a naive `Wave [0-9]` rule corrupts the vocabulary. |

The discriminator that resolved most of the hard cases: **numbering that another
component references by number is load-bearing**; a plan-wide taxonomy id with no
adjacent prose is not. Contrast `S0`–`S11` (local, described, asserted on) with
`_L7_REVISION_MODEL` (a lever id meaning nothing without the taxonomy) — the
first stays, the second gets renamed to what it does.

## Attachment consistency (source → name)

`workers._is_attachment_consistent` is the single guard deciding whether a DD
path may realize a standard name. `attachment_audit.py` re-asks it of every
stored edge and detaches what it rejects, which makes the guard retroactive
across every writer at once.

**Three of its rules were wrong because a vocabulary was hardcoded in codex and
had drifted from ISN.** Expect that failure mode; derive from
`get_grammar_context()` at runtime and add a drift test asserting every ISN token
in the relevant segment is covered:

- rate-ness is expressed by SUFFIX as well as prefix (`..._source_rate`,
  `rotation_frequency`), so a name can be rate-natured without a leading
  `change_in_`/`rate_of_`;
- the DD uses `_dt` for **deuterium–tritium** as well as d/dt — the genuine
  derivative form carries a leading `d` on the differentiated quantity
  (`ddensity_dt_total`, `dphase_dt`);
- state resolution is ISN's `state` segment (`charge_state`, `internal_state`)
  plus the `_state`-suffixed subjects — not a fixed tuple.

**Ordinality never enters a name.** `…/line_of_sight/{first,second,third}_point/r`
must share ONE name, so `_vector_fields_conflict` must not treat ordinal samples
of one object as distinct vector fields. Genuinely distinct fields (a camera's
`direction` vs `up`) and distinct geometry primitives (`rectangle` centre vs
`oblique` reference corner) still conflict.

**A pairwise rule must be applied with compose semantics.** Compose accumulates
only ACCEPTED siblings, so one representative of a conflicting group survives;
passing every sibling to every row rejects the whole group. Applied
order-independently once, that would have stripped 127 names of every source.

**A whole-name wipeout is a NAME defect, not an attachment defect.** When every
source of a name fails the same rule the sources are consistently grouped and the
NAME is wrong — repair with `sn edit`, never by detaching, which would orphan the
name and rewind its sources to paid recomposition.

### Write paths

The guard is consulted at compose time. Paths that migrate a source set onto a
**different** name — the refine-successor migration, and the exclusive rebind /
retarget used by the edit cascade — need it at write time too: the set is
historical but the *pairing* is new, and a new pairing is what the guard exists
to judge. Paths that re-establish an edge against the **same** name (the
provenance reattach, the orphan-parent repair) must NOT be gated: gating a repair
with a rule that never governed the creation turns silent loss into permanent
loss, and the retroactive pass will detach a genuinely bad pair *with* an audit
record, which is the correct order.

A geometry-representation rule is still missing: a path under
`…/geometry/{thick_line,outline,rectangle,oblique,arcs_of_circle}/…` describes a
conductor cross-section, not an optical path, and must not realize a
`line_of_sight`/`beam` name. Both sides are metres, so the dimensionality rule is
silent and the guard currently accepts these.

## Units are DD-authoritative

The model never supplies `unit`, `cocos` or `physics_domain` — all three are
injected post-LLM. Consequences when a unit looks wrong:

- **Fix the DD side, not the name.** `units/dd_unit_exceptions.yaml` is the single
  registry. Most entries only *suppress* a mismatch (the DD is wrong, the name is
  right, and the axis keeps reporting it). Flag `correct_in_graph: true` only when
  the DD contradicts **itself** on one quantity, so there is no single declaration
  to mirror.
- **A registry entry alone is not enough.** The correction is applied by the DD
  build, so an entry added afterwards never reaches already-stored paths.
  `reconcile_dd_unit_corrections` (wired into the `sn run` startup sweep) is what
  makes the registry self-applying.
- **A name carries exactly one unit.** `HAS_UNIT` is cardinality-one in the
  schema, and the guard compares dimensionality — a name holding both `1` and
  `m^-2` admits sources of either. Both writers self-heal, but a terminal-stage
  name keeps the residue forever because nothing recomposes it.
- **Do NOT re-derive a name's unit from its sources in bulk.** ~40 accepted names
  correctly carry a dimensionless unit against a DD path the registry records as
  wrong (charge numbers tagged `e`, unit vectors tagged `m`); a bulk re-derivation
  would clobber every one of them.

## Operators live outside `SEGMENT_TOKEN_MAP` — read the grammar through one accessor

`SEGMENT_TOKEN_MAP` is **not** the whole grammar vocabulary. Operators
(`square`, `inverse`, `flux_surface_averaged`, `derivative_with_respect_to`, …)
are a separate mechanism composing through `operator_token`, so they occupy no
segment slot. A consumer reading only that map cannot see 51 legal tokens and
treats every one of them as an unregistered proposal.

**This has now generated the same bug five separate times**, each fixed at one
site: single-operator gap classification, the state/rate attachment rules, the
compose prompt's operator block, the decomposition classifier's multi-operator
compounds, and the plural-dedup check. Read the grammar through
`segments.grammar_tokens_by_segment()` (per-class tokens, operators included) or
its reverse `grammar_token_index()`. `tests/standard_names/test_vocab_consumers_see_operators.py`
fails on any new module importing `SEGMENT_TOKEN_MAP` that is not listed there
with a reason.

Two sets that look mergeable and are not: `reportable_segments()` is what a gap
may be **reported against** (wider — includes the model-layer slot names
`transformation`/`decomposition`); `known_segments()` is what the **parser** slots
tokens into. Merging them makes the response model reject valid composer output.

A compound spelled with `over` is a **division** — the binary `ratio` operator
over two operands, not a compound base. Never "fix" it by letting the cover walk
step over the word: that stops the token being `absent` while emitting guidance
that folds the operators into one base and silently drops the division. An honest
`absent` costs less than confident wrong guidance.

### Open ISN-side findings (not codex's to fix, no PR — ISN's MCP surface is in flux)

- **Locus tokens reachable by the parser but by no vocabulary enumeration.**
  `active_wall_point`, `beam_path` and `primary` are in the locus registry yet in
  no `SEGMENT_TOKEN_MAP` segment, so a consumer listing the vocabulary cannot emit
  them and they never reach a prompt. Same shape as the `normalizing_qualifiers`
  token `gyrocenter`, which is in no segment either. Pinned per-token in
  `tests/standard_names/test_grammar_vocabulary_drift.py::UNREACHED`.
- **Qualifier ordering structure is not exposed.** `load_qualifier_categories()`
  exists but `get_grammar_context()` never returns it. Note precisely what is and
  is not wrong here: the qualifier *tokens* all reach every prompt seat (they are
  qualifier names), so "qualifier_categories is missing from the prompt" is false.
  What is missing is the **category structure**, while the prompt asks for
  *ordered* qualifier stacking — a plausible driver of ordering errors, and it
  matters more while the qualifier class is being decomposed into ordered
  binding-depth segments.

## The catalog is not the source of truth

The graph plus the review pipeline is. Catalog-origin names
(`origin='catalog_edit'`, `source_types=['catalog']`) came from one bulk import of
this pipeline's own earlier output and sit at `status='draft'`; they are an old
method to be resolved through review, not authoritative content to protect.

This makes the `origin='catalog_edit'` exemption in
`supersede_prior_source_names` a defect rather than a safeguard: it stops the
one-live-name-per-source dedup from folding an imported name, which is why one
coordinate axis resolved to a single name while another stayed split across two.

## Acceptance

Never hand-accept, and never edit graph text with Cypher. Acceptance is earned
only through the RD-quorum review pool; corrections go through `sn edit`
(`--hint` to steer regeneration, `--rename`/`--docs` to replace and go straight
to review). `--stage-only` stages many for one batch review — the budget-efficient
path for a bulk repair, since compose is free and review is the only paid stage.

The single sanctioned structural accept is `ENRICH_PARENTS` for placeholder
derived parents, which the quorum systematically penalises for being abstractions.

## Exact-source compose steering

Use `sn source-hint <exact-dd-path> --hint ... --reason ...` only before an
eligible DD source without a live name binding composes. It is not a source
mutation: DD snapshot, unit, COCOS, domain, identity, and grammar validation
remain authoritative. Preview with `--dry-run`; replacing an open hint requires
`--replace`.

The write is claim-fenced compare-and-set. A missing or non-DD source, active
claim, non-`extracted` state, attempt cap, live binding, or unacknowledged open
hint is a refusal, not a reason to bypass the CLI with Cypher. An open hint rides
pooled compose and is consumed atomically only when that exact source binds a
name. Retry, failure, interruption, claim expiry, validation rejection, and
collision preserve it. Use `sn run --focus <same-exact-path>` to fence the
subsequent work; model choice comes from the configured compose seat unless the
run explicitly overrides it.

Do not confuse the three operator surfaces: `sn retry --reason` audits why a
blocked source may attempt again, `sn source-hint` steers the next successful
binding of one exact source, and `sn edit <name> --hint` steers an existing name
through review. `rejected` is reserved in the hint-status schema; no current CLI
transition sets it.
