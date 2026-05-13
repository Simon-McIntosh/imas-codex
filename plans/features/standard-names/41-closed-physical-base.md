# Plan 41: Closed Physical Base Vocabulary

## Problem

The ISN `physical_bases.yml` contains 296 entries, many of which are compound phrases
(`collisional_power_density`, `bootstrap_current_density`, `center_of_mass_velocity`).
The parser's greedy longest-match on these compounds prevents qualifier extraction,
causing the ISNC catalog to display 296 unique group headings instead of ~50 dimensional ones.

The root cause: the parser's `_match_base_with_qualifiers()` mechanism already supports
recursive qualifier stripping, but only 19 hardcoded modifier tokens exist. Expanding the
qualifier vocabulary and reducing the base vocabulary fixes grouping at the grammar level.

## Approach

Reduce `physical_bases.yml` to ~50-80 dimensional base quantities and promote the qualifier
mechanism from a hardcoded 19-token set to a YAML-loaded closed vocabulary (~80 tokens).
This is a vocabulary reorganization, not an architectural change.

## Design Decisions (from RD review)

### D1: Qualifier ordering — insertion-order, NOT alphabetical
The existing `render_qualifiers()` sorts lexicographically. Under this plan, qualifiers
MUST preserve insertion order (left-to-right as parsed). The IR `qualifiers` field is an
**ordered list**, and compose emits them in list order. Round-trip test validates ordering.

### D2: Subject/qualifier disambiguation — subjects win, compounds stay atomic
Compound subject tokens (`trapped_fast_particle`, `co_passing_fast_particle`) remain atomic
subjects in `subjects.yml`. They are NOT decomposed into qualifier chains. The rule:
- Parser strips subject (longest match) FIRST at Stage 3
- Qualifier extraction operates only on the residue AFTER subject removal
- A token cannot be both a subject AND a qualifier in the same name

### D3: Process vs qualifier — processes only via template
Tokens in the process vocabulary (`bootstrap`, `sawtooth`, etc.) may ONLY appear as processes
via the `_due_to_{token}` suffix template. If a process token appears as a prefix
(e.g., `bootstrap_current_density`), it is treated as a **qualifier**, not a process.
Parser priority: qualifier prefix > process suffix for the same token string.

### D4: `_of_` operator trap — keep multi-word qualifiers atomic
`center_of_mass` stays as a multi-word qualifier token (not decomposed by `_of_` operator
matching). The qualifier YAML supports multi-word tokens. Operator matching (Stage 4)
runs before qualifier extraction (Stage 5), so any `_of_` in the qualifier must not match
as an operator. Solution: register `center_of_mass` as an atomic qualifier token.

### D5: Base count target — ~50-80, not 45
The original 45 target is too aggressive. Many entries are genuinely atomic
(`beta_poloidal`, `ballooning_stability_parameter`). Target: reduce to 50-80 bases,
verified against the 479-name corpus before implementation.

## Phases

### Phase A: ISN Grammar Redesign (ISN repo only)

**Gate: 100% offline round-trip on 479-name corpus before proceeding to Phase B.**

#### A1: Create `qualifiers.yml` vocabulary file
Extract qualifier tokens from displaced compound bases. Categories:
- Orbit-class: `trapped`, `co_passing`, `counter_passing`, `passing`
- Species: `fast_particle`, `thermal`, `fast_electron`, `fast_ion`
- Collision: `collisional`, `neoclassical`, `classical`
- Source: `bootstrap`, `ohmic`, `beam`, `rf`, `nbi`
- Geometry: `center_of_mass`, `cross_field`, `parallel`, `perpendicular`
- State: `equilibrium`, `fluctuating`, `mean`
- ~80 total, loaded from YAML

#### A2: Reduce `physical_bases.yml` to dimensional bases
Target: 50-80 entries that are irreducible dimensional quantities.
Kept: `velocity`, `power_density`, `current_density`, `pressure`, `temperature`,
`density`, `energy`, `flux`, `frequency`, `resistivity`, `conductivity`, ...
Removed: `collisional_power_density`, `bootstrap_current_density`, etc.
(these become qualifier+base combinations)

#### A3: Add missing process tokens
- `coulomb_collisions_with_ion` → Process enum
- `coulomb_collisions_with_electron` → verify exists
- Any others found during corpus validation

#### A4: Update parser — load qualifiers from YAML
- Replace hardcoded `modifier_quals` frozenset with YAML-loaded vocabulary
- `_match_base_with_qualifiers()` uses expanded qualifier set
- Qualifier field on `StandardName` model: `qualifier: list[str] | None`
- Ordered list (insertion-order preserved from parse)

#### A5: Update compose — emit qualifiers in order
- `compose_standard_name()` emits qualifiers as underscore-joined prefix before base
- Order: `{subject}_{qualifier1}_{qualifier2}_{physical_base}_{process_suffix}`
- Remove alphabetical sort in render_qualifiers

#### A6: Round-trip regression suite
- Parse → compose → parse for all 479 corpus names
- Must achieve 100% (zero regression)
- Test multi-qualifier ordering explicitly
- Test subject/qualifier boundary cases

#### A7: Cut ISN RC release
- Version bump, push tag, GitHub Release
- Bump ISN dep in codex

### Phase B: Codex Pipeline Update

#### B1: Bump ISN dep in codex pyproject.toml
- Both occurrences (main deps + dependency-groups)
- `uv sync` + verify version

#### B2: Update grammar context and prompts
- `imas_codex/standard_names/context.py` — expose qualifier vocab to LLM
- `imas_codex/llm/prompts/shared/sn/_grammar_reference.md` — document qualifier segment
- Compose prompt: teach LLM to produce qualifiers separately from base

#### B3: Clear graph and run initial rotation
- `uv run imas-codex sn clear`
- Run first $5 rotation: `uv run imas-codex sn run -c 5`
- Evaluate parse success rate, vocab gaps

### Phase C: Iterative Bootstrap (5× $5 rotations)

Each rotation:
1. Run `sn run -c 5` (or `--focus` for debugging specific failures)
2. Inspect failures: vocab gaps, parse errors, process token misses
3. Fix ISN vocabulary (add qualifier/process tokens as needed)
4. Cut new ISN RC, bump dep in codex
5. Clear affected names, re-run

Between rotations: fix systematic errors, adjust prompts, expand vocab.

### Phase D: Catalog & Export

#### D1: Export validated names
- `sn export --staging ./staging`
- Catalog renderer uses ISN parser → clean dimensional grouping automatically

#### D2: Preview and publish
- `sn preview --staging ./staging`
- Verify hierarchical grouping on catalog site
- `sn publish --staging ./staging --isnc ../isnc --push`

## Risks

| Risk | Mitigation |
|------|-----------|
| Round-trip regression on corpus | Phase A gate — offline test before graph work |
| Subject/qualifier ambiguity | D2 rule — subjects win, compounds stay atomic |
| LLM produces wrong qualifier order | Validation layer rejects; compose+parse round-trip |
| Too many vocab gaps in Phase C | Iterative rotation design — fix between each $5 |
| Base count too low/high | D5 — target 50-80, validated against corpus first |

## Documentation Updates

| Target | Update needed |
|--------|--------------|
| AGENTS.md | Update grammar segment description, qualifier vocabulary rules |
| ISN AGENTS.md | Qualifier vocabulary maintenance rules |
| `_grammar_reference.md` | Document qualifier segment for LLM |
| Schema reference | Auto-rebuilds from `build-models` |
