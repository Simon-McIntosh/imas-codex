# Spectral qualifier parse and structural-rewire verdict

## Verdict

The reverse qualifier edge is **not a parser defect and is not a legitimate
derivation**. The current pinned ISN parser decomposes
`spectral_signal_to_noise_ratio_of_spectrometer_channel` correctly: `spectral`
is its sole qualifier, `signal_to_noise_ratio` is its physical base,
`spectrometer_channel` is its locus, and it has no operator. Removing the one
qualifier therefore yields
`signal_to_noise_ratio_of_spectrometer_channel`, not a logarithm identity.

The graph nevertheless contains both directed relationships:

| Child | Stored edge | Parent | Verdict |
|---|---|---|---|
| `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` | `operator='logarithm'`, `operator_kind='unary_prefix'` | `spectral_signal_to_noise_ratio_of_spectrometer_channel` | legitimate unary-operator peel |
| `spectral_signal_to_noise_ratio_of_spectrometer_channel` | `operator='spectral'`, `operator_kind='qualifier'` | `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` | defective successor migration; not emitted by the parser |

The earlier recurring-producer diagnosis got the first step wrong: structural
derivation does not directly desire the logarithm target. It desires the
superseded plain-ratio target, after which a successor-rewire pass replaces the
target while preserving the qualifier relationship properties.

## Parse output that settles the question

The exact current parse and derived edge output was:

```text
spectral_signal_to_noise_ratio_of_spectrometer_channel
  operators: []
  qualifiers: [{category: diagnostic, token: spectral}]
  base: {kind: quantity, token: signal_to_noise_ratio}
  locus: {relation: of, token: spectrometer_channel}
  HAS_PARENT:
    spectral_signal_to_noise_ratio_of_spectrometer_channel
      -- {operator: spectral, operator_kind: qualifier} -->
    signal_to_noise_ratio_of_spectrometer_channel

logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel
  operators: [{kind: unary_prefix, op: logarithm}]
  qualifiers: [{category: diagnostic, token: spectral}]
  base: {kind: quantity, token: signal_to_noise_ratio}
  locus: {relation: of, token: spectrometer_channel}
  HAS_PARENT:
    logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel
      -- {operator: logarithm, operator_kind: unary_prefix} -->
    spectral_signal_to_noise_ratio_of_spectrometer_channel
```

Both parses compose losslessly back to their input spellings. The legitimate
edge removes the outer `logarithm` operator. The defective edge would have to
remove `spectral` while simultaneously adding `logarithm`; no such operation
exists in the parse tree.

## The actual producer

The complete ordinary-maintenance path is:

1. `derivation._derive_structural()` takes the qualifier branch and returns the
   correct plain-ratio parent.
2. `graph_ops._write_standard_name_edges()` admits that edge, deletes stale
   structural edges not in the current derived set, and `MERGE`s the correct
   child-to-plain-ratio relationship.
3. `graph_ops.rederive_structural_edges()` then calls
   `graph_ops._rewire_has_parent_off_superseded()`.
4. The plain-ratio parent is superseded. The rewire follows its longest
   `REFINED_FROM` path to an accepted tip:

```text
logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel
  -> logarithm_of_signal_to_noise_ratio_at_spectral_line
  -> ratio_of_spectral_power_to_reference_spectral_power
  -> spectral_signal_to_noise_ratio
  -> signal_to_noise_ratio
  -> reference_signal_to_noise_ratio
  -> signal_to_noise_ratio_of_spectrometer_channel
```

5. `_rewire_has_parent_off_superseded()` considers that tip a compatible rename
   because it compares only `physical_base`, `geometric_base`, `subject`, and
   `component`. The old parent and tip both have
   `physical_base='signal_to_noise_ratio'` and null values for the other three
   compared fields, even though the tip adds
   `transformation='logarithm'` and retains `spectral`. The function deletes the
   correct edge to the superseded parent and `MERGE`s an edge to the tip with
   the old `{operator: spectral, operator_kind: qualifier}` property map.

The durable prevention boundary is therefore
`imas_codex/standard_names/graph_ops.py::_rewire_has_parent_off_superseded`, at
the compatibility predicate and subsequent `MERGE`, not the qualifier parser
and not the `MERGE` in `_write_standard_name_edges`. A safe rule is: migrate an
incoming structural edge to a refinement tip only when the current
`derive_edges(child)` result itself authorizes that tip as the parent with
compatible edge semantics. Otherwise the refinement lineage is not authority
to rewrite a grammar edge. Special-casing these two names in the writer would
hide the wider mechanism.

## Exact live impact of that prevention rule

The impact simulation used the same live-name predicate and the same
`_filter_admissible_parents(..., full_rebuild=True)` gate as
`rederive_structural_edges()`, then applied the current successor-rewire
predicate without writing anything.

| Measure | Exact count |
|---|---:|
| Live names processed by structural re-derivation | 2,475 |
| Raw parser-derived `HAS_PARENT` rows | 2,123 |
| Rows retained by the production parent-admission gate | 1,512 |
| Superseded-parent/live-tip pairs passing the current four-field rewire guard | 320 |
| Admitted rows reaching the rewire before its self-loop check | 88 |
| Rows already suppressed as self-loops | 10 |
| Distinct non-self pairs the current code would `MERGE` | **78** |
| Subject pair in this report | 1 |
| **Other live child-to-tip pairs affected by the prevention rule** | **77** |
| Of the 78 migrated tips independently authorized by current derivation | **0** |

Thus the change is not a one-pair special case. It prevents this pair plus
exactly 77 other admitted, non-self structural relationships from being moved
to refinement tips that the current parser does not name as their parent.
Representative other affected pairs include:

| Parser-derived child → parent | Current rewire tip |
|---|---|
| `turn_count_of_toroidal_magnetic_field_probe` → `turn_count` | `effective_turn_count_of_passive_loop` |
| `flux_surface_averaged_effective_charge_at_plasma_boundary` → `effective_charge_at_plasma_boundary` | `effective_charge_at_separatrix` |
| `perturbed_particle_pressure` → `particle_pressure` | `total_plasma_pressure` |
| `radial_neutral_momentum` → `neutral_momentum` | `neutral_momentum_source` |
| `ion_charge_state_energy_flux` → `energy_flux` | `energy_flux_at_control_surface` |

The existing graph already shows the same class as residue: 711 live
qualifier-kind relationships were compared with current derivation; 52 do not
match, comprising this edge and 51 other live qualifier pairs. That is a
current-state observation. The 77 figure above is the forward impact of fixing
the rewire producer after applying the production admission gate.

## Schema and zero sanity

No zero in this report comes from a guessed property or reversed relationship:

| Probe | Candidates | Required-key coverage |
|---|---:|---:|
| `StandardName.id` | 4,656 | 4,656 |
| `StandardName.name_stage` | 4,656 | 4,656 |
| Authored `StandardName -[:HAS_PARENT]-> StandardName` | 1,485 | source id 1,485; target id 1,485; `operator_kind` 1,485 |
| Authored `StandardName -[:REFINED_FROM]-> StandardName` | 1,561 | both endpoints constrained by declared `StandardName.id` in the impact query |

All three named endpoints resolve exactly once by `StandardName.id`: the
spectral name is accepted, the plain ratio is superseded, and the logarithm
name is accepted. The six-hop `REFINED_FROM` path resolves exactly once. The
live graph contains exactly one bidirectional `HAS_PARENT` pair—the subject
pair—and **zero other bidirectional pairs**; that zero was evaluated only after
the 4,656/4,656 identity and stage coverage and 1,485/1,485 relationship-endpoint
coverage checks above. The independent-authorization zero is likewise a
comparison of 78 fully enumerated, admitted pairs against lossless current
`derive_edges()` output, not an aggregate over a nullable graph property.

## Evidence inputs and limits

- Exact parser and `derive_edges()` output: `/tmp/n-qualifierparse-live-graph.log`
- Focused live endpoint, edge, and lineage evidence:
  `/tmp/n-qualifierparse-lineage.log` and
  `/tmp/n-qualifierparse-lineage-path.log`
- Current qualifier-edge and bidirectional-pair census:
  `/tmp/n-qualifierparse-impact-summary.log`
- Raw rewire simulation: `/tmp/n-qualifierparse-rewire-simulation.log`
- Production-admission-aware impact simulation:
  `/tmp/n-qualifierparse-admitted-rewire-impact.log`

This investigation was read-only. It changed no graph state, called no model,
and supplies no authority to delete, migrate, accept, supersede, or rewrite any
identity or relationship.
