# Paired documentation-arm measurement on the current gates

## Result

The production-enriched arm and its catalog counterpart were scored over the same
85 immutable holdout rows. Generation admitted all 85 rows and made 85 calls for
**USD 0.311539**, below both the **USD 0.990188** zero-call projection and the
hard **USD 5.00** ceiling. The zero-call preflight made **0 calls** and spent
**USD 0.00**.

The result is qualified rather than a blanket quality pass. Four gates are fully
numeric and one COCOS-conditional gate is numeric over its 24 evaluable rows. The
generated arm's `defining_equation` gate is **not evaluable on all 85 rows** under
the current honest-binding behavior, so its module-defined pass rate and signed
rate delta are undefined. Reporting those rows as failures would recreate the
absence-versus-contradiction error that the tri-state result was introduced to
prevent.

## Cost and durability

| Measure | Zero-call admission | Actual run |
|---|---:|---:|
| Holdout rows | 85 | 85 |
| Calls | **0** | **85** |
| Projected exposure | **USD 0.990188** | **USD 0.990188** |
| Actual spend | **USD 0.00** | **USD 0.311539** |
| Hard ceiling | **USD 5.00** | **USD 5.00** |
| Scored rows | 0 | **85** |
| Rows excluded for missing relationship context | 0 | **0** |
| Rows carrying non-empty generated documentation in the receipt | 0 | **85/85** |

The durable receipt is
`/home/ITER/mcintos/.local/share/imas-codex/receipts/standard-names/sn-quality-parity/paired-arm-current-gates.json`
(SHA-256 `eea5ceef492143de9ddd447a2ac63627edfc0405bd54220573e5f521600f7cf7`).
Every one of its 85 row records contains the full non-empty
`generated_documentation` string, the paired `catalog_documentation`, pinned
physics context, and both current gate vectors. Future gate changes can therefore
rescore this exact arm without another generation call. The separate zero-call
receipt is `paired-arm-current-gates-dry-run.json` (SHA-256
`3303af2ab99daeca99e5f87e6f9a100073661d891b5633bd51455c2a330c1ece`).

The recorded validation log is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T205837489403-n-armreplay/paired-arm-validation.log`.
It records the admission and actual figures together, the six gate rows, both
required outcome splits, receipt text counts, gate-source revision
`12806f0f8ae67918704ce6bece837ed4d35a35f2`, and successful invariants that all
splits sum to 85.

## Per-gate paired results

The module's pass rate divides by evaluable rows only. The final column supplies
an always-defined signed comparison over the common 85-row population: generated
pass fraction minus catalog pass fraction. It is included so every gate has a
signed paired delta without silently counting `not_evaluable` as `fail`.

| Current gate | Generated pass rate | Catalog pass rate | Signed evaluable-rate delta | Signed all-row pass-fraction delta |
|---|---:|---:|---:|---:|
| `defining_equation` | undefined (0/0 evaluable) | 0.886792 (47/53) | undefined | **-0.552941** (0/85 - 47/85) |
| `symbol_definitions` | 0.988235 (84/85) | 0.800000 (68/85) | **+0.188235** | **+0.188235** |
| `relationship_link` | 0.976471 (83/85) | 1.000000 (85/85) | **-0.023529** | **-0.023529** |
| `sign_convention` | 0.916667 (22/24) | 0.916667 (22/24) | **+0.000000** | **+0.000000** (22/85 - 22/85) |
| `link_hygiene` | 1.000000 (85/85) | 1.000000 (85/85) | **+0.000000** | **+0.000000** |
| `minimum_word_count` | 1.000000 (85/85) | 1.000000 (85/85) | **+0.000000** | **+0.000000** |

## Required tri-state outcome splits

Every row appears exactly once in each split.

| Gate and arm | pass | fail | not_evaluable | total |
|---|---:|---:|---:|---:|
| `defining_equation`, generated | **0** | **0** | **85** | **85** |
| `defining_equation`, catalog | **47** | **6** | **32** | **85** |
| `sign_convention`, generated | **22** | **2** | **61** | **85** |
| `sign_convention`, catalog | **22** | **2** | **61** | **85** |

The defining-equation result is diagnostically sharp: persisted generated prose
is present on 85/85 rows, but the conservative relation parser cannot bind every
generated relation to stated symbol units, so it makes no contradiction claim.
The catalog arm retains 53 evaluable rows and six contradictions. For sign
conventions, the 24 rows pinned to a sensitive or invariant transformation class
are evaluable; the remaining 61 lack per-row transformation authority and remain
explicit abstentions.

## Representative source-path bindings

| DD source path | Standard Name | Pinned authority | Generated outcome | Catalog outcome | Representative generated opening |
|---|---|---|---|---|---|
| `magnetics/flux_loop/area` | `area_of_flux_loop` | unit `m^2`; no transform class | equation `not_evaluable`; sign `not_evaluable` | equation `fail`; sign `not_evaluable` | “The effective area is the area-equivalent factor of a flux-loop sensor.” |
| `equilibrium/time_slice/profiles_1d/q` | `safety_factor` | unit `1`; transform `q_like` | equation `not_evaluable`; sign `fail` | equation `not_evaluable`; sign `pass` | “The safety factor is the signed field-line winding ratio assigned to a magnetic flux surface...” |
| `magnetics/b_field_phi_probe/field` | `toroidal_magnetic_field` | unit `T`; transform `one_like` | equation `not_evaluable`; sign `fail` | equation `pass`; sign `fail` | “The toroidal magnetic field is the signed component of the local total magnetic-induction vector...” |

These examples are bindings, not hand-selected score overrides. Their complete
prose and gate reasons remain in the receipt alongside all other rows.
