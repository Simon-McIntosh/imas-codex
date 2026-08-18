# DD 4.1.1 legacy-row upstream provenance manifest

Retrieval date: **2026-08-10** (Europe/Paris)  
Mode: **read-only detached-worktree investigation; no repository, graph, service, pipeline, catalog, model, or provider mutation**  
Repository base: `208c5d64d47d95a7e417c4908a3f6db145705ed5`  
Official upstream: <https://github.com/iterorganization/IMAS-Data-Dictionary>  
Observed upstream `develop`: [`d4c6345f3689a9bc905be527c061a0340a974c61`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/d4c6345f3689a9bc905be527c061a0340a974c61)  
Latest published tag: [`4.1.1`](https://github.com/iterorganization/IMAS-Data-Dictionary/releases/tag/4.1.1), tag object `5a9bfab9aca1d09527f565394ef2f886c51bdb72`, peeled commit [`023e94be394b94db9ed1bc314628fee296594a42`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/023e94be394b94db9ed1bc314628fee296594a42)  
Previous tag: `4.1.0`, object `1ef4fbc145bc00a546d4ed4800bb29f594a0b95f`, peeled commit [`90f446e408b0ea2fc2333d15421499b4bbbdd39a`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/90f446e408b0ea2fc2333d15421499b4bbbdd39a)

## Authority result

**No audited row becomes eligible solely from this research.** Exact official solution commits or merge requests were found for 21 row mappings, but `U19` is only a six-path subset of a fourteen-path row, two solution changes remain open, and the two merged solution families landed after the latest published 4.1.1 tag. All 62 rows still lack the imas-codex governed approval fields required by the machine evidence: `approval_receipt`, `approved_by`, `approved_at`, `governed_decision_reason`, and `resolution_revision`. Upstream code-owner review/merge is upstream provenance, not an imas-codex approval receipt.

This work can reduce only `upstream_ref` for exact, row-bounded subsets. It creates no reviewer, actor, reason, receipt, token, active resolution, or graph authority. Release/tag diffs are evidence, never governed approval.

Machine facts retained unchanged:

- 62 rows: `U01`–`U34`, `O01`–`O28`.
- Installed `imas-data-dictionaries 4.1.1`, `imas-python 2.2.0`, 82 IDSs, 44,150 paths, XML CRC `2172880527`.
- DD XML SHA-256 `22f6cef9e5937c13a4888d94462fd0f597cf66dca637e85e2a0fcb129335499c`.
- 45 rows have raw-tuple matches. No matches: `U20,U31,U34,O01,O02,O03,O04,O05,O06,O07,O08,O10,O18,O19,O26,O27,O28`.
- Conflict-held rows: `U01,U02,U08,U09,U10,U19,U20,U23,U30,U31,U33,U34`.
- 57 graph/release conflicts: 35 absent graph source paths and 22 raw-unit mismatches.
- No DDResolution nodes, version bindings, governed receipts, or lifecycle receipts were present.
- Transition state remains 49 `CANDIDATE`, `O06`–`O08` normalization-only, `O19`–`O24` `HOLD`, `O25`–`O28` `SKIP`, 0 `ACTIVE`.

Classification: **E** exact solution commit/MR; **I** issue only; **V** version-diff inference; **N** no provenance found; **C** graph evidence conflict prevents binding. Combinations are deliberate.

## Deduplicated official references

| Ref | Immutable reference and status | Exact binding and published range | Authority fields satisfied |
|---|---|---|---|
| A | [PR 242](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242); commits [`fd0c145cb897770738c20de4a426c27b2d8d1a2d`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/fd0c145cb897770738c20de4a426c27b2d8d1a2d), [`721638233cd87f5ca3f9e71b36d66c46e146af2e`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/721638233cd87f5ca3f9e71b36d66c46e146af2e); merge [`cb0d86de388dbbdf62acca36de7b7f8c62bb9889`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/cb0d86de388dbbdf62acca36de7b7f8c62bb9889). Merged 2026-06-15. | `U11`–`U16`, `O20`–`O24`: DD 4.1.1 vector components `m → 1`. Fixed on post-tag `develop`; no fixed published tag. Broad O rows remain local HOLD. | Candidate exact `upstream_ref` only; five governed fields remain missing. |
| B | [Issue 272](https://github.com/iterorganization/IMAS-Data-Dictionary/issues/272), [open PR 273](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/273), commit [`f34c85d33497f2bd777db7eaf0f6fb93fddc66f2`](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/273/commits/f34c85d33497f2bd777db7eaf0f6fb93fddc66f2). | `U21,U22,U32`: neutral kinetic/recombination incident values `m^-2.s^-1 → W.m^-2`. Defect introduced 3.38.0; published 3.38.0–4.1.1 affected; no fixed version. | Candidate `upstream_ref`; open, unmerged, unreleased. U21 still has dual `type_wiring`/`unit_defect` local facts. |
| C | [Issue 277](https://github.com/iterorganization/IMAS-Data-Dictionary/issues/277), [open PR 280](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/280), commit [`30a5ddd4b7037b9f93a8f00f7837809403349d99`](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/280/commits/30a5ddd4b7037b9f93a8f00f7837809403349d99). | Exactly six DD 4.1.1 paths: edge/plasma profiles `.../ionisation_potential`, `/values`, `/coefficients`, `e → eV`. Covers both O17 base paths but only 6/14 U19 paths. Does not bind absent U20/O18 spelling. Proposed `change_nbc_version=4.2.0`; no fixed release. | Candidate `upstream_ref` for six paths only. PR expressly says adjacent `z_min,z_max,z_average,z_square_average` legitimately remain `e`. |
| D | [Issue 278](https://github.com/iterorganization/IMAS-Data-Dictionary/issues/278), [PR 281](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/281), head [`35c146031bf98028911b8266d286dcdf6ee85e2e`](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/281/commits/35c146031bf98028911b8266d286dcdf6ee85e2e), merge [`d07172e814e91900cb4ed5d0b5f41547be3eef90`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/d07172e814e91900cb4ed5d0b5f41547be3eef90). Merged 2026-08-06. | `U25`–`U29`: `reconstructed 1 → as_parent`, resolving to `Pa,Pa,m^-3,A.m^-2,A.m^-2`. DD 4.1.1 affected; fixed post-tag on develop; no fixed published tag. | Candidate exact `upstream_ref` only. |
| E | [Open issue 279](https://github.com/iterorganization/IMAS-Data-Dictionary/issues/279). | `U10/O16`: reports 36 `vibrational_level` declarations as `e`, argues a dimensionless quantum number, explicitly requests maintainer ruling. No PR/commit/fixed version; export expands to 66 raw paths and U10 has 3 absent graph paths. | Discussion only; satisfies no exact solution or approval field. |
| F | [Issue 102](https://github.com/iterorganization/IMAS-Data-Dictionary/issues/102), [PR 103](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/103), head [`6081186cc1b3f381e7c3f39db40fae3972d93f24`](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/103/commits/6081186cc1b3f381e7c3f39db40fae3972d93f24), merge [`81ec3ea69d1c365ee7a59304d3fa7e88c224dc49`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/81ec3ea69d1c365ee7a59304d3fa7e88c224dc49). | Historical fix only for `reflectometer_profile/position/psi` and `/channel/position/psi`, `W → Wb`. Current U33 raw paths are ECE and `reflectometer_fluctuation`, so this is not a U33 solution. | Version/history evidence only for U33. |
| G | [Published 4.0.0 notes](https://github.com/iterorganization/IMAS-Data-Dictionary/releases/tag/4.0.0), commit [`6ed980c`](https://github.com/iterorganization/IMAS-Data-Dictionary/commit/6ed980c); [4.1.0 notes](https://github.com/iterorganization/IMAS-Data-Dictionary/releases/tag/4.1.0). | Notes cover American `-ize`, obsolete transport structures, and `_tor → _phi`, with 4.1.0 naming DISTRIBUTIONS missed renames. Context for U20/U34 absent paths, not proof of unit changes. | Version-diff evidence only. |

## Row-complete table

`Raw → effective` repeats the audited legacy assertion and does not approve it. Count is exact DD 4.1.1 tuple count; Conf is conflict count.

### U rows

| Row | Pattern | Raw → effective | Count/Conf | Class/ref | Binding result and version range |
|---|---|---|---:|---|---|
| U01 | `*/z_ion` | `e → 1` | 53/3 | N+C | No solution; 3 obsolete transport-solver paths absent. |
| U02 | `*/z_n` | `e → 1` | 102/4 | N+C | No unit solution; 4 paths absent. G records type, not unit, change. |
| U03 | `*/z_n/value` | `e → 1` | 1/0 | N | 4.1.1 raw fact only; no official solution. |
| U04 | `*/z_average` | `e → 1` | 8/0 | N | No solution; C explicitly calls `z_average` legitimately `e`. |
| U05 | `*/z_average/values` | `e → 1` | 2/0 | N | No solution; cannot inherit authority by analogy. |
| U06 | `*/z_square_average` | `e → 1` | 8/0 | N | No solution; C explicitly calls it legitimately `e`. |
| U07 | `*/z_square_average/values` | `e → 1` | 2/0 | N | No solution; unresolved child semantics. |
| U08 | `*/state/z_max` | `e → 1` | 41/3 | N+C | No solution; 3 paths absent; C says sibling legitimately `e`. |
| U09 | `*/state/z_min` | `e → 1` | 41/3 | N+C | No solution; 3 paths absent; C says sibling legitimately `e`. |
| U10 | `*/vibrational_level` | `e → 1` | 66/3 | I+C (E) | Issue-only; 36 declarations discussed, no solution; 3 paths absent. 4.1.1 affected, no fixed version. |
| U11 | `*/direction/[xyz]` | `m → 1` | 12/0 | E (A) | Exact merged fix; 4.1.1 affected; develop fixed, no fixed tag. |
| U12 | `*/direction_second/[xyz]` | `m → 1` | 3/0 | E (A) | Exact merged fix; same range. |
| U13 | `*/up/[xyz]` | `m → 1` | 3/0 | E (A) | Exact merged fix; same range. |
| U14 | `*/injection_direction/[xyz]` | `m → 1` | 3/0 | E (A) | Exact merged fix; same range. |
| U15 | `*/unit_vector_major/[xyz]` | `m → 1` | 3/0 | E (A) | Exact SPI fix; same range. |
| U16 | `*/unit_vector_minor/[xyz]` | `m → 1` | 3/0 | E (A) | Exact SPI fix; same range. |
| U17 | `ec_launchers/beam/direction/kphi` | `m^-1 → 1` | 1/0 | N | No solution. [4.1.1 schema](https://github.com/iterorganization/IMAS-Data-Dictionary/blob/4.1.1/schemas/ec_launchers/dd_ec_launchers.xsd) defines `(kR,kphi,kz)` as wavevector components, all `m^-1`; A is unrelated. |
| U18 | `wall/global_quantities/power_density_*_target_max` | `W → W.m^-2` | 2/0 | N | No exact official issue/MR/commit/tag/changelog item found. |
| U19 | `*/ggd/ion/state/ionisation_potential*` | `e → eV` | 14/8 | E(partial)+C (C) | Exact only for base, values, coefficients in edge/plasma (6). Four error-index paths absent; four index fields have empty unit. Split required. |
| U20 | `*/ggd/ion/state/ionization_potential*` | `e → eV` | 0/9 | V+C (G) | All American-spelling graph paths absent. Broad spelling note and C cannot bind absent paths. |
| U21 | `*/energy_fluxes/kinetic/neutral/state/incident/values` | `m^-2.s^-1 → W.m^-2` | 1/0 | E (B) | Exact open solution; affected 3.38.0–4.1.1; no fixed version. Dual local gap kinds remain. |
| U22 | `*/energy_fluxes/kinetic/neutral/incident/values` | `m^-2.s^-1 → W.m^-2` | 1/0 | E (B) | Exact open solution; 3.38.0–4.1.1; no fixed version. |
| U23 | `waves/coherent_wave/*k_perpendicular*` | `V.m^-1 → m^-1` | 10/5 | N+C | No solution; 3 error-index paths absent, 2 index fields empty-unit. |
| U24 | `spi/injector/*_gas/flow_rate` | `s^-1 → Pa.m^3.s^-1` | 2/0 | N | No exact solution for fragmentation/propellant flow rates. |
| U25 | `equilibrium/.../pressure/reconstructed` | `1 → Pa` | 1/0 | E (D) | Exact merged `as_parent`; 4.1.1 affected; no fixed tag. |
| U26 | `.../pressure_rotational/reconstructed` | `1 → Pa` | 1/0 | E (D) | Exact merged `as_parent`; same range. |
| U27 | `.../n_e/reconstructed` | `1 → m^-3` | 1/0 | E (D) | Exact merged `as_parent`; same range. |
| U28 | `.../j_phi/reconstructed` | `1 → A.m^-2` | 1/0 | E (D) | Exact merged `as_parent`; same range. |
| U29 | `.../j_parallel/reconstructed` | `1 → A.m^-2` | 1/0 | E (D) | Exact merged `as_parent`; same range. |
| U30 | `gyrokinetics_local/*/angle_pol` | `1 → rad` | 2/1 | N+C | No solution; `species_all/angle_pol` graph path absent. |
| U31 | `distribution_sources/source/ggd/particles/values` | claimed `m^-6.s^2 → m^-3.s^-1` | 0/1 | N+C | Claimed raw identity conflicts: 4.1.1 publishes `(m.s^-1)^-3.m^-3.s^-1`. No solution. |
| U32 | `*/energy_fluxes/recombination/neutral/incident/values` | `m^-2.s^-1 → W.m^-2` | 1/0 | E (B) | Exact open solution; 3.38.0–4.1.1; no fixed version. |
| U33 | `*/position/psi` | `W → Wb` | 3/16 | V+C (F) | F fixes different historical paths. Current raw: ECE plus 2 reflectometer_fluctuation. 15 graph claims already raw Wb; j_tor absent. No current solution. |
| U34 | `distributions/distribution/global_quantities/current_tor` | claimed `N.m → A` | 0/1 | V+C (G) | Path absent. G proves rename history, not claimed unit correction. |

### O rows

| Row | Pattern | Raw → effective | Count/Conf | Class/ref | Binding result and state |
|---|---|---|---:|---|---|
| O01 | `**/element/multiplicity` | `Elementary Charge Unit → 1` | 0/0 | N | No tuple or solution; CANDIDATE. |
| O02 | `**/ionisation_potential` | `Elementary Charge Unit → eV` | 0/0 | N | C concerns raw `e`, not this raw spelling; CANDIDATE. |
| O03 | `**/ionization_potential` | `Elementary Charge Unit → eV` | 0/0 | N | No tuple; spelling history not solution; CANDIDATE. |
| O04 | `**/binding_energy` | `Elementary Charge Unit → eV` | 0/0 | N | No tuple/solution; CANDIDATE. |
| O05 | `**/z_n` | `Elementary Charge Unit → 1` | 0/0 | N | No tuple/solution; CANDIDATE. |
| O06 | `**/element/a` | `Atomic Mass Unit → u` | 0/0 | V/N | Normalization-only representation, not DD fix; NORM. |
| O07 | `**/atomic_mass` | `Atomic Mass Unit → u` | 0/0 | V/N | Normalization-only; NORM. |
| O08 | `**/a` | `Atomic Mass Unit → u` | 0/0 | V/N | Normalization-only; NORM. |
| O09 | `**/z_n` | `e → 1` | 102/0 | N | G changes type, not unit; CANDIDATE. |
| O10 | `**/charge_number` | `e → 1` | 0/0 | N | No tuple/solution; CANDIDATE. |
| O11 | `**/z_ion` | `e → 1` | 53/0 | N | No solution; CANDIDATE. |
| O12 | `**/z_average` | `e → 1` | 8/0 | N | C says legitimately `e`; CANDIDATE. |
| O13 | `**/z_square_average` | `e → 1` | 8/0 | N | C says legitimately `e`; CANDIDATE. |
| O14 | `**/z_min` | `e → 1` | 42/0 | N | C says legitimately `e`; CANDIDATE. |
| O15 | `**/z_max` | `e → 1` | 42/0 | N | C says legitimately `e`; CANDIDATE. |
| O16 | `**/vibrational_level` | `e → 1` | 66/0 | I (E) | Issue-only, no solution/fixed version; CANDIDATE. |
| O17 | `**/ionisation_potential` | `e → eV` | 2/0 | E (C) | Exact for two base paths; open, proposed 4.2.0, no fixed tag; CANDIDATE. |
| O18 | `**/ionization_potential` | `e → eV` | 0/0 | V (G) | No tuple; spelling context is not solution; CANDIDATE. |
| O19 | `**/z` | `e → 1` | 0/0 | N | No tuple/solution; broad semantic glob remains HOLD. |
| O20 | `**/*unit_vector*/*` | `m → 1` | 1188/0 | E (A) | Exact upstream template provenance; only 6 overlap U15/U16 evidence; HOLD. |
| O21 | `**/direction/*` | `m → 1` | 36/0 | E (A) | Exact upstream change; only 12 overlap U11 evidence; HOLD. |
| O22 | `**/direction_second/*` | `m → 1` | 9/0 | E (A) | Exact upstream change; only 3 overlap U12; HOLD. |
| O23 | `**/up/*` | `m → 1` | 9/0 | E (A) | Exact upstream change; only 3 overlap U13; HOLD. |
| O24 | `**/injection_direction/*` | `m → 1` | 9/0 | E (A) | Exact upstream change; only 3 overlap U14; HOLD. |
| O25 | `**` | `m^dimension → dd_unit_unresolvable` | 18/0 | N | Context qualification, not solution; SKIP. |
| O26 | `pulse_schedule/**/reference` | `1 → context-dependent` | 0/0 | N | No tuple; qualification, not solution; SKIP. |
| O27 | `pulse_schedule/**/reference/data` | `1 → context-dependent` | 0/0 | N | No tuple; qualification, not solution; SKIP. |
| O28 | `pulse_schedule/**/reference_waveform/data` | `1 → context-dependent` | 0/0 | N | No tuple; qualification, not solution; SKIP. |

## Representative identities and semantic boundaries

- `spi/injector/shatter_cone/unit_vector_major/x`: 4.1.1 raw `m`; A changes the unit-vector template to `1`.
- `ec_launchers/beam/direction/kphi`: 4.1.1 raw `m^-1`; upstream calls the containing structure wavevector `(kR,kphi,kz)`. It is not A's geometrical unit vector.
- `wall/description_ggd/ggd/energy_fluxes/kinetic/neutral/state/incident/values`: `m^-2.s^-1`; B retargets particle-neutral to energy-neutral, yielding `W.m^-2`.
- `equilibrium/time_slice/constraints/pressure/reconstructed`: `1`; D changes to `as_parent`, yielding `Pa`, and similarly yields the four U26–U29 units.
- `edge_profiles/ggd/ion/state/ionisation_potential/values`: `e`; C changes to `eV`. But `grid_index` has empty raw unit and `values_error_index` is absent, so U19 cannot be bound whole.
- F changes `reflectometer_profile/channel/position/psi`, while U33 current raw `W` paths are ECE and two `reflectometer_fluctuation` paths. Historical precedent is not current provenance.
- U31's immutable raw tuple is `(m.s^-1)^-3.m^-3.s^-1`, not claimed `m^-6.s^2`.
- C explicitly preserves charge-number siblings as `e`; the ionization correction cannot be generalized into charge-row authority.

## Conflict-held detail

- U01: 3 absent transport-solver `/ion/z_ion` paths.
- U02: absent gas-injection path plus 3 transport-solver `/ion/z_n` paths.
- U08/U09/U10: 3 absent transport-solver paths each.
- U19: 4 absent `*_error_index` paths plus 4 index fields whose immutable unit is empty.
- U20: all 9 American-spelling graph paths absent.
- U23: 3 absent error-index paths and 2 empty-unit index fields.
- U30: `gyrokinetics_local/species_all/angle_pol` absent.
- U31: graph-claimed raw unit conflicts with immutable published raw unit.
- U33: 15 claims publish immutable `Wb`, not claimed `W`; `j_tor` path absent.
- U34: `current_tor` absent.

These require a fresh version-matched raw-DD export and exact reviewed row/path binding. Never reverse current graph rewrites to invent a solution.

## Residual governed approval requirements

For every tuple, including A–D positives:

1. Pin registry row, resolver kind, exact enumerated paths, raw/effective value, and DD version 4.1.1.
2. Attach immutable exact upstream SHA/MR/tag and its state. For post-tag merges record no fixed published tag; for open PRs record unmerged/unreleased.
3. Resolve conflicts and split mixed rows such as U19 before binding.
4. Record a governed reason independent of legacy comments and this report.
5. Record authorized reviewer/actor and timestamp.
6. Record immutable approval receipt and resolution revision.
7. Bind the exact version range, including a fixed release/upper boundary when available.
8. Before mutation, regenerate the fresh action manifest and use normal dry-run, guarded transaction, and independent pre/postflight. This report grants no mutation authority.

The supplied machine manifest names the missing fields exactly: `upstream_ref`, `approval_receipt`, `approved_by`, `approved_at`, `governed_decision_reason`, `resolution_revision`. This research can fill only candidate `upstream_ref`, and only for listed subsets.

## Negative findings and limitations

- No exact solution found for U01–U09, U17–U18, U23–U24, U30–U31, current U33 paths, or U34's claimed unit change.
- No exact O-row solution except A mappings and C's O17; E is issue-only; O06–O08 are normalization, O25–O28 qualifications.
- F is confined to historical reflectometer_profile paths; no current U33 solution found.
- No official source supports applying A to EC `kphi`.
- No official source supports broad charge conversion from C; C provides contrary sibling semantics.
- No tag after 4.1.1 was visible. A/D are merged only on develop; B/C are open.
- GitHub unauthenticated API quota was exhausted during enumeration. Work continued via official public HTML, immutable commit/blob pages, releases/tags, raw official source, and `git ls-remote`. Negative findings are bounded to retrievable public official surfaces, not private discussions.

## Input hashes

| Input | SHA-256 |
|---|---|
| transition audit | `fd5bb1e70ab0945df501550c38b0b202740eea0046b53d6d17c7b1c88aabbaf1` |
| evidence export Markdown | `51340f7777aeb112f18f3959b6469d3db6372e7d1a896be5e0b39fddb5483945` |
| evidence export JSON | `93ad2ae36bbb9e322591bbf6f71539b3c170d09672059ca7078f74ed9129512e` |
| `docs/sn-dd-gaps.html` | `c24075b7c15035db2bda29db51f2222a5ff5b87ad0a910fef3642d0291168070` |
| root `AGENTS.md` | `4f1ad328104048b528c13e4e66b83dd46895a1e228266281ecb5ae19b775b3b4` |
| SN `AGENTS.md` | `3abf0cdacbf6c2f01caaa893abad723322776b5a0bc87e5156ff5e8250851e9a` |
| DD 4.1.1 XML | `22f6cef9e5937c13a4888d94462fd0f597cf66dca637e85e2a0fcb129335499c` |

The final whole-file SHA-256 is computed after the final write and reported in the supervisor handoff; embedding it would change the file.

## Exact commands and queries

```text
git status --short
git branch --show-current
git rev-parse HEAD
git stash list
git ls-remote --heads --tags https://github.com/iterorganization/IMAS-Data-Dictionary.git refs/heads/develop 'refs/tags/4.1.1*' 'refs/tags/4.1.0*'
jq -r '.rows[] | [.row_id,.pattern,(.raw_unit // "-"),(.legacy_effective_unit // .skip_reason // "-"),(.exact_raw_tuple_path_count|tostring),([.graph_claim_conflicts_against_immutable_release[]?]|length|tostring)] | @tsv' /tmp/reckon-s8-scope/dd-resolution-evidence-export.json
jq -r '.rows[] | select((.graph_claim_conflicts_against_immutable_release|length)>0) | .row_id as $id | .graph_claim_conflicts_against_immutable_release[] | [$id,.path,(.claimed_observed_value//"null"),(.immutable_raw_unit//"ABSENT")] | @tsv' /tmp/reckon-s8-scope/dd-resolution-evidence-export.json
sha256sum /tmp/reckon-s8-scope/dd-resolution-transition-audit.md /tmp/reckon-s8-scope/dd-resolution-evidence-export.md /tmp/reckon-s8-scope/dd-resolution-evidence-export.json docs/sn-dd-gaps.html AGENTS.md imas_codex/standard_names/AGENTS.md
git show --stat --oneline --decorate --no-renames 208c5d64d47d95a7e417c4908a3f6db145705ed5
```

Official GETs: issue/PR `102/103,242,272/273,277/280,278/281,279`; release/tag pages `4.0.0,4.1.0,4.1.1`; each immutable A–F commit; tag-4.1.1 EC-launcher XSD; and GitHub issues API pages 1–3 (`state=all&per_page=100`).

Repository-restricted searches:

```text
site:github.com/iterorganization/IMAS-Data-Dictionary "power_density_radiated_target_max"
site:github.com/iterorganization/IMAS-Data-Dictionary "flow_rate" "Pa.m^3.s^-1"
site:github.com/iterorganization/IMAS-Data-Dictionary "k_perpendicular" "V.m^-1"
site:github.com/iterorganization/IMAS-Data-Dictionary "angle_pol" units
site:github.com/iterorganization/IMAS-Data-Dictionary "distribution_sources/source/ggd/particles/values"
site:github.com/iterorganization/IMAS-Data-Dictionary "current_tor" "N.m"
site:github.com/iterorganization/IMAS-Data-Dictionary "position/psi" "Wb"
site:github.com/iterorganization/IMAS-Data-Dictionary "ec_launchers" "kphi"
```

## Pinned-base audit and final status

```text
208c5d64 (HEAD, origin/main, main) docs: record typed dispatch review hold
 docs/plans/llm-context-integrity.html | 5 ++++-
 1 file changed, 4 insertions(+), 1 deletion(-)
```

Final repository status: **clean** (`git status --short` empty). Only this external manifest was written.
