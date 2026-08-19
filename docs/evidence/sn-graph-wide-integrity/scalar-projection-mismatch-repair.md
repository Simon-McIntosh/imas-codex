# Scalar and projection mirror repair

## Outcome

The live `codex` graph was repaired through one hash-bound transactional
instrument at source commit `d292d8a898be8525cd2f931b2931e57a6f2ce824`.
The production semantic-source invariant supplied the exact cohort after the
dual-target disposition apply had landed: **144 scalar mismatches and 2
projection mismatches**. The signed preview admitted all **146 of 146** rows
with zero refusals. Apply changed 144 `produced_sn_id` scalars, created the two
missing `HAS_STANDARD_NAME` projections, and wrote one `StandardNameChange`
receipt. Remeasurement reduced both mismatch classes to **zero**.

| Integrity measure | Before | After apply and replay |
|---|---:|---:|
| Multiple-live-target sources | 87 | 87 |
| Scalar mismatches | 144 | **0** |
| Projection mismatches | 2 | **0** |
| Names with no live producing source | 85 | **85** |
| `StandardNameChange` rows | 7,452 | 7,453 |
| `LLMCost` rows | 27,467 | **27,467** |

The 87 multiple-target rows were not part of this instrument. Their unchanged
count is expected: the sole-live-target precondition makes the scalar or
projection target mechanically authoritative only after the dual-target class
has been resolved.

## Signed exact authority

`repair_scalar_projection_mismatches` derives each action from one and only one
non-terminal `PRODUCED_NAME` target. It signs the complete selected source,
binding, target, origin-backing, and projection closure; locks those exact nodes
and relationships; reconstructs the manifest under lock; and requires the
second SHA-256 to match before writing. Scalar changes use a null-safe prior
value and claim compare-and-set. Projection additions require the signed source,
backing, and target element identities and refuse duplicate target projections.
Missing sources, claims, non-live lifecycle states, multiple live targets,
unsupported source types, backing ambiguity, source/backing identity mismatch,
and closure drift are fail-closed outcomes.

The executable manifest SHA-256 is
`94eb4533048d7f947f5b61be71b8c18eb16c3f78183ca9d0401de498de65f8de`.
It created exactly one receipt:

`sn-change:semantic-mirror-repair:94eb4533048d7f947f5b61be71b8c18eb16c3f78183ca9d0401de498de65f8de`

The quiet-window preflight measured zero active `StandardName` claims, zero
active `StandardNameSource` claims, and zero active `SNRun` rows. The immediate
replay returned `already_applied` with `changed=0`; it rechecked all 146 recorded
source-to-target postconditions and rolled its transaction back without a
second write.

## Collateral invariants

Every `StandardNameSource` outside the 146-row allowlist was projected into a
normalized closure containing its properties, `PRODUCED_NAME` bindings,
upstream origin, backing identity, and all backing `HAS_STANDARD_NAME`
projections. The complete **9,352-row** closure hashed identically before and
after:

`4a8b15ed9a98eae61e0aae99a1338cd14392b1306bb04f38c22180a73ef7266c`

The live count of names with no producing source remained exactly 85. The
`LLMCost` count remained 27,467, proving this deterministic repair made no
provider call. The single-row `StandardNameChange` increase is exactly the
manifest-stamped receipt above.

## Representative repaired bindings

| DD source | Before | Accepted live identity after repair | Why the result is different |
|---|---|---|---|
| `camera_ir/channel/target_surface_center/z` | scalar `vertical_coordinate_of_divertor_target` | `z_coordinate_of_divertor_target` (name score 0.90) | The live edge and DD projection already selected the Cartesian Z-coordinate identity; the stale scalar named a different vertical-coordinate spelling. The accepted description is “Vertical Cartesian coordinate of the geometric center of a divertor target surface, identifying its location along the facility's vertical axis.” |
| `core_profiles/global_quantities/li_3` | scalar `internal_inductance` | `normalized_plasma_internal_inductance` (0.925) | The DD quantity is the normalized `li_3` equilibrium parameter, not unqualified inductance. The binding and projection already carried that normalization; only the stale scalar disagreed. |
| `core_sources/source/profiles_1d/electrons/particles_decomposed/explicit_part` | missing DD projection | `electron_source_rate` (0.68125) | The scalar and sole live binding already agreed on the volume-integrated rate of net electron production or removal; the instrument restored only the missing backing projection. |
| `edge_sources/source/ggd/electrons/particles/values` | missing DD projection | `electron_source_rate` (0.68125) | As above, identity was unchanged and the absent `HAS_STANDARD_NAME` mirror was created against the exact signed backing. |

## Regression evidence

The changed test file passed completely against a fresh empty Neo4j 2026.01.4
instance on `sun_debug`: **12 passed, 0 failed, 0 skipped**. The replay case was
first rerun alone after correcting a Neo4j-invalid nested aggregate and passed
1/1. The graph regressions cover combined scalar-plus-projection apply,
out-of-allowlist byte stability, unchanged unsourced-name count, exactly one
receipt, write-free replay, ambiguity refusal, and post-preview projection
drift refusal. Ruff check, Ruff format check, and `git diff --check` passed.

Durable logs:

- `live-exact-apply.log`, SHA-256
  `643c7345ae98578a47d181c94b3b78caa7e6e1715a91e6b3cb20d8840ce0111c`;
- `replay-targeted.log`, SHA-256
  `30b3d65ed18f8cfdb355a1672f16f3f3b1356ea59ccfcff9972be2e976590c33`;
- `disposable-tests-final.log`, SHA-256
  `9ae252f68531ea19a1f949bda3a4ad09db8f2c0c98744ccb409dfa84eb54af58`.

All three are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T193934640772-scalar-projection-mismatch-repair/`.
