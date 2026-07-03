---
name: sn-edit
description: >-
  Propose a correction or steering direction for a standard name or its
  documentation. Use this any time a StandardName's name or docs need to
  change — a wrong DD-derived name, an ambiguous description, a physics
  error caught in review. NEVER hand-edit graph text via Cypher and never
  bypass the generate→review→score pipeline; `sn edit` is the only sanctioned
  path. Covers redesign vs hint mode, the mandatory reason, axis/scope rules,
  the dry-run-first workflow, and following an edit through review.
allowed-tools: Bash(*)
---

# sn-edit — steer a standard name through the pipeline, never hand-edit it

## Golden rule

A standard name or its docs are wrong → run `imas-codex sn edit`, not a Cypher
`SET`. Hand-editing graph text bypasses style rules, grammar validation,
RD-quorum review, and scoring — the exact machinery that makes the catalog
trustworthy. `sn edit` rides the proposal through that machinery as a steered
candidate instead of a fait accompli.

```
imas-codex sn edit <standard_name> (--hint TEXT | --rename NAME | --docs TEXT) --reason TEXT
                   [--axis name|docs|both] [--scope self|family|subtree] [--dry-run]
```

Both `--hint`/`--rename`/`--docs` and `--reason` are mutually required: no
edit without a reason.

## Pick a mode

- **You know the exact end state** — the correct name, or the correct
  documentation text, in full — use **redesign**: `--rename NAME` or
  `--docs TEXT`. This skips `GENERATE_NAME`/`GENERATE_DOCS` entirely and
  enters directly at review (`REVIEW_NAME` / `REVIEW_DOCS`).
- **You know the direction but want the LLM to compose the specifics** under
  the grammar rules — use **hint**: `--hint TEXT`. The text is injected into
  the generate/refine prompts as steering context; the candidate still goes
  through the full generate→review cycle.

Internally these are three `EditMode` values (`hint`, `rename`, `docs`), not
two — `--rename` and `--docs` each imply their own axis. `--axis` only
matters in hint mode, where a single direction can steer the name axis, the
docs axis, or both.

## Write a reason that survives review

`--reason` is not a changelog entry — it is intent context shown to the
reviewer: *"a domain expert has deliberately steered this candidate for the
following reason: …; judge on merits given this intent; do not penalize
merely for differing from a prior variant."* Review still scores
independently and **can reject the edit** — a weak reason doesn't buy
immunity, it just fails to counter the reviewer's pull back toward the
previously-accepted variant.

A good reason grounds in physics and the DD path, not preference:

- ❌ "current name is ambiguous"
- ✅ "DD 'area' is the poloidal cross-sectional area enclosed by the
  boundary, NOT the swept toroidal surface (that is
  `surface_area_of_flux_surface`). The bare 'area' is ambiguous; name it
  explicitly as the poloidal cross-section."

## Scope and the sibling-desync guard

- `self` (`only_self` internally — LinkML can't name a value `self`) — this
  StandardName only.
- `family` — this SN plus siblings sharing the edited segment.
- `subtree` — this SN as parent plus every `HAS_PARENT` descendant.

Defaults: editing a parent → `subtree`; editing a leaf → `self`. **A leaf
edit that changes a base segment shared with siblings is blocked** unless you
pass `--scope family` explicitly — this is the sibling-desync guard,
preventing one sibling's name from silently drifting out of step with the
rest of its family.

**Cascade semantics:** the root rename is the only reviewed decision.
Descendant renames are deterministic and ISN-round-trip-validated, and are
applied atomically — only once the root's review **accepts**. There is no
half-edited family: either the whole cascade lands or none of it does.

## Always dry-run first

```bash
imas-codex sn edit <name> --rename <new_name> --reason "..." --dry-run
```

Confirm the proposed candidate (and, for `family`/`subtree` scope, the full
cascade set) before attaching it for real. Drop `--dry-run` only once the
preview looks right.

## Follow the edit through review

Attaching an edit does not score it — it just enters the queue.

1. `imas-codex sn status <name>` — shows `edit_status`:
   `open → applied | exhausted | rejected`.
2. Drive the pipeline forward with a review rotation, e.g.
   `imas-codex sn run --focus <name>` or `imas-codex sn review --ids <name>`.
3. `applied` = the steered candidate was accepted — the edit achieved its
   intended change. `exhausted` = the refine rotation cap was hit without
   acceptance. `rejected` = review terminally scored it below threshold.
   All three are terminal.

Eligible rename targets include names in terminal stages: a `superseded`
name with **no successor** (an orphaned DD path) can be resurrected by a
redesign edit.

## Worked example

`equilibrium/time_slice/global_quantities/area` is the poloidal
cross-sectional area enclosed by the plasma boundary — not the swept
toroidal surface (that's `surface_area_of_flux_surface`). The pipeline's bare
`area_of_plasma_boundary` was ambiguous between the two and died in review
(superseded, no successor). Fix it:

```bash
imas-codex sn edit area_of_plasma_boundary \
    --rename poloidal_cross_section_of_plasma_boundary \
    --reason "DD 'area' is the poloidal cross-sectional area enclosed by the boundary, NOT the swept toroidal surface (that is surface_area_of_flux_surface). The bare 'area' is ambiguous; name it explicitly as the poloidal cross-section." \
    --scope self
```

## Agent access via MCP

Agents that call tools rather than shells use the mirrored MCP write tool
`edit_standard_name` — same parameters, same underlying implementation
(`imas_codex.standard_names.edit.apply_edit`) as the CLI. Prefer it over the
CLI when already operating through MCP; behavior, review gating, and the
mandatory `--reason` requirement are identical.
