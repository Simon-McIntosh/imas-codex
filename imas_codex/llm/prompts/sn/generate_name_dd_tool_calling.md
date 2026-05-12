---
name: sn/generate_name_dd_tool_calling
description: Variant C — lean compose prompt that fetches DD context on-demand via tool calls
used_by: scripts/prompt_ab.py (plan 32 Phase 2 research harness)
task: composition
dynamic: true
schema_needs: []
---

Generate Standard Names for the following IMAS Data Dictionary paths.

## Core rules

- **Unit is authoritative** and comes from the DD `HAS_UNIT` relationship.
  Do not include unit in your output — it is injected at persistence time.
  Use it only to disambiguate the physical quantity.
- **Name = `physical_base` [+ modifiers]**, lowercase snake_case, never
  include abbreviations, symbols, measurement methods, or processing
  adjectives (`filtered_`, `reconstructed_`, `averaged_`). Position tokens
  go after the physical base (`electron_temperature_core`, not
  `core_electron_temperature`).
- **No unit strings, no IDS names, no method names** in the Standard Name.
- **Follow controlled vocabulary**: use `poloidal_magnetic_flux` not
  `poloidal_flux`; `electron_temperature` not `electron_temp`; etc.

## Tool-calling policy (variant C)

You have access to three tools that fetch additional context **only when
you need it**. The goal is to keep the prompt lean and let you pull in
siblings, reference exemplars, or version history on demand rather than
front-loading all of it:

- `fetch_cluster_siblings(cluster_id)` — returns names already assigned
  to paths in the same semantic cluster. Use when you are unsure whether
  a similar name has been established.
- `fetch_reference_exemplar(concept)` — returns a published exemplar
  Standard Name that matches a concept (e.g. `"electron temperature"`).
  Use to confirm controlled-vocabulary choices.
- `fetch_version_history(path)` — returns DD version change history for
  one path. Use when the path description references a renamed or
  repurposed field.

**Budget:** at most 2 tool calls per batch. Prefer to emit the name
directly if the context is obvious.

## Output

**You do NOT output a `standard_name` string.** You fill individual IR segment
fields. Code assembles the canonical name via ISN's `compose()` function.

Return a JSON array of objects. Each object includes:
- `path`: the DD path
- `base_token`: irreducible base quantity from the registry
- `base_kind`: `"quantity"` or `"geometry"`
- `projection_axis`, `projection_shape`: for vector/coordinate projections (null if none)
- `qualifiers`: list of qualifier tokens (empty list if none)
- `locus_token`, `locus_relation`, `locus_type`: for postfix locus (null if none)
- `process_token`: for `_due_to_` (null if none)
- `operator_token`, `operator_kind`: for operators (null if none)
- `rationale`: one sentence citing the physical quantity and any tool call evidence

{{ paths_block }}
