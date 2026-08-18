# DD resolution transition authority audit

Audit date: 2026-08-10 (Europe/Paris)  
Repository: `/home/ITER/mcintos/Code/imas-codex`  
Detached audit worktree: `/run/user/39486/reckon-worktrees/imas-codex-c994bf55fb01/20260810-authority-runtime/raw-dd-authority`  
Base SHA: `32aa552d6f9a69339bb78c1be58b4af6a73905c0`  
Scope manifest: `/tmp/reckon-s8-scope/dd-resolution-scope.md`  
Scope manifest SHA-256: `164bdd74c3a856cbd0510c3594eb84dfd463a7deead18ac9f0e875a467493af4`

## 1. Decision

**HOLD removal of every semantics-changing legacy unit rule.** The integrated
typed resolver is structurally suitable, but its packaged behavior manifest is
intentionally empty (`imas_codex/standard_names/config/dd_resolutions.yaml:1-2`).
No tracked row supplies the complete tuple required to activate a resolution:
exact IDS path, exact DD version, exact raw value/hash from an immutable release
fact, exact compatible DDGap identity, observation identities, evidence token,
human approval identity/time/receipt, and exact upstream solution URL/ref.

The 35 DDGap facts and 451 evidence relationships are evidence, not behavior
authority. The plan keeps flags evidence-only (`docs/sn-dd-gaps.html:158-181`),
and the landed resolver independently requires exact path/version/field
applicability and complete approval/evidence/upstream provenance
(`imas_codex/standard_names/dd_resolutions.py:152-174,309-406,481-522`). The
empty active cohort therefore means that removing the current 11 graph-rewrite
rules or the extraction replacements would expose raw incorrect units to
Standard Names, while restoring raw graph authority before consumer cutover
would split scalar/edge/effective semantics.

There is also one inventory correction to the prior scope document: the current
tracked override file contains **24 replacements and 4 skips**, not 25 and 3.
The total of 28 is correct. The fourth skip is the global `m^dimension`
qualification rule (`unit_overrides.yaml:193-202`). This audit uses current
tracked content, not the stale 25/3 subdivision.

## 2. Evidence basis and hashes

Read completely:

- live plan v13: `docs/sn-dd-gaps.html`, SHA-256
  `f0928a117f5f5522717d2bd24913d335ebe5881a1f99bbcf1cb03448026d245a`;
- landed foundation archive: `docs/archive/sn-dd-gaps-foundation-landed.html`,
  SHA-256 `fba013fde56a27f507026b3823a87fdd9e10b47f9af05a9d3acff5159625e1e4`;
- landed policy archive: `docs/archive/sn-dd-gaps-policy-landed.html`, SHA-256
  `474d8ad518fc422a3e4d5a5468d8250bb14c33b44312a0eae81618852e76d70f`;
- prior scope manifest (path/hash above);
- root and Standard Names `AGENTS.md` files;
- integrated N1 resolver/schema and all legacy registries/tests at the base SHA.

Key tracked hashes:

| artifact | SHA-256 |
|---|---|
| `imas_codex/units/dd_unit_exceptions.yaml` | `81dcd027ff5f14c87a8afc6334e41414e5cc67c14c062d897438dbaaf31bcdf6` |
| `imas_codex/standard_names/config/unit_overrides.yaml` | `13310c50b546cc4860779e641495195f79f24875ae0c058d430141cc8fdb151d` |
| `imas_codex/standard_names/config/dd_resolutions.yaml` | `64c20eb0405022f33265e4bc222919c25f51b1c98b00b6e473ff615c963b33cf` |
| `imas_codex/standard_names/dd_resolutions.py` | `dfb4ce3b365b9b2304e5d36f067e327986c8aaea24c6a87dd0ca98593c1b543f` |
| `imas_codex/schemas/standard_name.yaml` | `73e750d38b674e9fd61156550a5c07a24eb27ceb6fe18add1e32a1271345c2c2` |
| `imas_codex/schemas/imas_dd.yaml` | `94b4c8baea23d836795854c078c228c3fb2208b64ff24a000ef6b561eeb3609f` |

N1 source commit is `99cc1bb43e14242ee6c8aecf4c267527686b2b5d`,
integrated by merge `c5fa4e3c`; base documentation records the contract as
landed while explicitly retaining raw restoration and all consumers as open
(`docs/sn-dd-gaps.html:278-282`).

### 2.1 Current runtime behavior

- `units_agree()` suppresses an SN/raw-DD mismatch when path, raw unit, and
  effective unit match a curated DD-unit-bug row
  (`dd_unit_exceptions.py:83-99,131-147`).
- The 11 `correct_in_graph: true` rows additionally feed
  `graph_unit_correction()` (`dd_unit_exceptions.py:102-128`). The DD build
  writes the latest raw string to `IMASNode.unit` but builds `HAS_UNIT` through
  the path-aware correction (`build_dd.py:2058-2118,3577-3616`). The startup
  reconciler then rewrites both the scalar and edge to the correction
  (`dd_graph_ops.py:687-799`; `loop.py:1364-1380`).
- The override engine is ordered, first-match behavior. A replacement changes
  the extraction row and records lightweight provenance; a skip returns no
  effective unit and records qualification metadata
  (`unit_overrides.py:35-49,52-83,86-147`). Extraction drops skipped rows and
  persists their reason (`sources/dd.py:222-311`); worker enrichment reapplies
  the same engine (`workers.py:2235-2271`).
- The packaged typed resolver never reads its graph mirror and only applies
  active exact records from packaged YAML (`dd_resolutions.py:1-7,684-765`).
  The package currently has zero such records.

### 2.2 Provenance notation used below

- **P0** — the curated DD-unit-bug row is statically present. Registry sync is
  coded to create a `registered_exception` DDGap plus exact observations for
  every current path match (`dd_gaps.py:428-478`). The landed record says this
  produced 34 registered-exception facts and 451 total evidence edges, but the
  tracked repository does not contain the expanded exact path set,
  observation IDs, or evidence tokens. No row has an approval receipt or exact
  DD-version binding.
- **P1** — P0 plus a tracked issue URL. This is still not an exact upstream
  solution ref, raw release fact, or approval receipt.
- **O0** — override-only row: no independent DDGap/evidence/upstream/approval
  provenance is present in the row. An overlap with a P0 row is only a possible
  evidence source after exact path/raw/version reconciliation; it is not
  inherited automatically.
- **Version `unbound`** — the row has no exact version. The repository currently
  selects DD `4.1.1` (`pyproject.toml:258-259`), but configuration selection is
  not an immutable raw release fact and cannot be copied into every record.

Disposition codes:

- **NORM** — safe path-independent representation normalization; no resolution
  record needed, but prove the parser mapping before deleting the path rules.
- **ACTIVE** — all authority is present; eligible now for an active typed record.
- **CANDIDATE** — physics correction appears defensible, but one or more exact
  path/version/raw/evidence/upstream/approval artifacts are missing.
- **SKIP** — explicit qualification outcome, not an effective DD value.
- **RETIRE** — obsolete now and safe to remove without replacement.
- **HOLD** — the glob is too broad or semantically ambiguous as a whole; split
  and adjudicate exact paths before any replacement/removal.

## 3. Row-by-row DD-unit-bug inventory (34)

All 34 rows are semantics-changing (`raw != effective`); none is mere spelling
normalization. For rows with `correct_in_graph=false`, current behavior is
mismatch suppression only. For `true`, current behavior is suppression plus
build-time edge correction and startup scalar+edge mutation. Every row has P0
unless stated otherwise, but **none has complete active-resolution authority**.

| ID / source lines | pattern; raw -> effective | current behavior / `correct_in_graph` | overlap and concrete tracked examples | exact version and provenance | disposition |
|---|---|---|---|---|---|
| U01 `:30-33` | `*/z_ion`; `e` -> `1` | suppress; false | O11 exact-pair overlap. `core_profiles/profiles_1d/ion/z_ion` (`test_dd_unit_exceptions.py:41-44`) | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE: expand exact paths and approve |
| U02 `:34-37` | `*/z_n`; `e` -> `1` | suppress; false | O09 overlap. `nbi/unit/species/z_n` (`test_dd_unit_exceptions.py:41-45`) | unbound; P0; all activation provenance missing | CANDIDATE |
| U03 `:38-41` | `*/z_n/value`; `e` -> `1` | suppress; false | no override row reaches the `/value` leaf; no tracked concrete fixture beyond the pattern | unbound; P0; all activation provenance missing | CANDIDATE |
| U04 `:42-45` | `*/z_average`; `e` -> `1` | suppress; false | O12 overlap; parameterized concrete `core_profiles/profiles_1d/ion/z_average` (`test_unit_overrides.py:103-119`) | unbound; P0; all activation provenance missing | CANDIDATE |
| U05 `:46-51` | `*/z_average/values`; `e` -> `1` | suppress; false | no override leaf match; `edge_profiles/ggd/ion/state/z_average/values` (`test_dd_unit_exceptions.py:65-70`) | unbound; P0; all activation provenance missing | CANDIDATE |
| U06 `:52-55` | `*/z_square_average`; `e` -> `1` | suppress; false | O13 overlap; parameterized `core_profiles/profiles_1d/ion/z_square_average` | unbound; P0; all activation provenance missing | CANDIDATE |
| U07 `:56-61` | `*/z_square_average/values`; `e` -> `1` | suppress; false | no override leaf match; `plasma_profiles/ggd/ion/state/z_square_average/values` (`test_dd_unit_exceptions.py:65-70`) | unbound; P0; all activation provenance missing | CANDIDATE |
| U08 `:62-68` | `*/state/z_max`; `e` -> `1` | suppress; false | subset of broader O15; `core_profiles/profiles_1d/ion/state/z_max` (`test_dd_unit_exceptions.py:60-63`) | unbound; P0; all activation provenance missing | CANDIDATE |
| U09 `:69-75` | `*/state/z_min`; `e` -> `1` | suppress; false | subset of broader O14; `waves/coherent_wave/profiles_2d/ion/state/z_min` (`test_dd_unit_exceptions.py:60-64`) | unbound; P0; all activation provenance missing | CANDIDATE |
| U10 `:76-79` | `*/vibrational_level`; `e` -> `1` | suppress; false | O16 overlap; parameterized `core_profiles/profiles_1d/ion/vibrational_level` | unbound; P0; all activation provenance missing | CANDIDATE |
| U11 `:84-87` | `*/direction/[xyz]`; `m` -> `1` | suppress; false | narrow subset of O21. `camera_ir/channel/camera/direction/x` (`test_dd_unit_exceptions.py:46-48`) | unbound; P0; all activation provenance missing | CANDIDATE for exact x/y/z paths |
| U12 `:88-91` | `*/direction_second/[xyz]`; `m` -> `1` | suppress; false | narrow subset of O22. `operational_instrumentation/sensor/direction_second/y` (`test_unit_overrides.py:154-169`) | unbound; P0; all activation provenance missing | CANDIDATE for exact x/y/z paths |
| U13 `:92-95` | `*/up/[xyz]`; `m` -> `1` | suppress; false | narrow subset of O23. `camera_ir/channel/camera/up/z` (`test_unit_overrides.py:154-169`) | unbound; P0; all activation provenance missing | CANDIDATE; retain distinct-vector semantics separately |
| U14 `:96-99` | `*/injection_direction/[xyz]`; `m` -> `1` | suppress; false | narrow subset of O24. `spi/injector/shatter_cone/injection_direction/x` (`test_unit_overrides.py:154-169`) | unbound; P0; all activation provenance missing | CANDIDATE |
| U15 `:100-103` | `*/unit_vector_major/[xyz]`; `m` -> `1` | suppress; false | subset of broad O20. `spi/injector/shatter_cone/unit_vector_major/z` (`test_dd_unit_exceptions.py:46-49`) | unbound; P0; all activation provenance missing | CANDIDATE |
| U16 `:104-107` | `*/unit_vector_minor/[xyz]`; `m` -> `1` | suppress; false | subset of broad O20; no distinct tracked positive fixture located | unbound; P0; all activation provenance missing | CANDIDATE |
| U17 `:108-113` | exact `ec_launchers/beam/direction/kphi`; `m^-1` -> `1` | suppress; false | O21 path shape but not activation overlap because O21 requires raw `m`; concrete path is exact in row | unbound; P0; all activation provenance missing | CANDIDATE |
| U18 `:118-123` | `wall/global_quantities/power_density_*_target_max`; `W` -> `W.m^-2` | suppress; false | no override. Tracked concrete `wall/global_quantities/power_density_outer_target_max` (`definitions/physics/domain_gold_set.json:981`) | unbound; P0; exact matched cohort/upstream/approval/raw fact missing | CANDIDATE |
| U19 `:133-140` | `*/ggd/ion/state/ionisation_potential*`; `e` -> `eV` | suppress + graph rewrite; true | O17 overlaps the exact leaf; U19 also includes descendants because `fnmatch *` crosses `/`. Exact fixtures: `edge_profiles/.../ionisation_potential`, `plasma_profiles/.../ionisation_potential/values` (`test_dd_unit_resolution.py:192-199`) | row says defect introduced in 4.1.0; test asserts 4.1.1 behavior (`:180-199`); neither is a release fact. P0; no upstream/ref/approval | CANDIDATE, highest priority; emit separate exact records per version/path |
| U20 `:141-146` | `*/ggd/ion/state/ionization_potential*`; `e` -> `eV` | suppress + graph rewrite; true | O18 overlaps exact US-spelling leaf; descendants only U20 | unbound; P0; no exact tracked release fact/upstream/ref/approval | CANDIDATE, highest priority |
| U21 `:151-158` | `*/energy_fluxes/kinetic/neutral/state/incident/values`; `m^-2.s^-1` -> `W.m^-2` | suppress; false | no override; exact expansion absent from tracked fixtures | unbound; P1 issue URL `.../issues/272`, but registry code creates that URL on a separate `type_wiring` gap. Unit resolutions accept only `unit_defect`/`self_contradiction` (`dd_resolutions.py:200-203`); plan mentions PR 273 but no bound upstream ref/approval/raw fact | CANDIDATE only after compatible exact unit-gap evidence and PR/ref receipt are bound |
| U22 `:159-164` | `*/energy_fluxes/kinetic/neutral/incident/values`; `m^-2.s^-1` -> `W.m^-2` | suppress; false | no override; non-state sibling of U21 | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE; do not infer U21 provenance |
| U23 `:169-174` | `waves/coherent_wave/*k_perpendicular*`; `V.m^-1` -> `m^-1` | suppress; false | no override. Fixtures `waves/coherent_wave/profiles_1d/k_perpendicular` and `.../full_wave/k_perpendicular/values` (`test_dd_unit_exceptions.py:72-78`) | unbound; P0; broad middle/suffix expansion and all activation provenance missing | CANDIDATE after exact expansion |
| U24 `:181-187` | `spi/injector/*_gas/flow_rate`; `s^-1` -> `Pa.m^3.s^-1` | suppress + graph rewrite; true | no override. `fragmentation_gas` fixture (`test_dd_unit_exceptions.py:80-87`); reporting fixtures also name `propellant_gas` (`test_dd_gap_reporting.py:830-835`) | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE, highest priority; two tracked examples are not an exhaustive release cohort |
| U25 `:196-202` | exact `equilibrium/.../pressure/reconstructed`; `1` -> `Pa` | suppress + graph rewrite; true | no override; exact fixture (`test_dd_unit_exceptions.py:109-117`) | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE, highest priority |
| U26 `:203-209` | exact `equilibrium/.../pressure_rotational/reconstructed`; `1` -> `Pa` | suppress + graph rewrite; true | no override; exact path only | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE, highest priority |
| U27 `:210-216` | exact `equilibrium/.../n_e/reconstructed`; `1` -> `m^-3` | suppress + graph rewrite; true | no override; exact fixture (`test_dd_unit_exceptions.py:118-122`) | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE, highest priority |
| U28 `:217-223` | exact `equilibrium/.../j_phi/reconstructed`; `1` -> `A.m^-2` | suppress + graph rewrite; true | no override; exact fixture (`test_dd_unit_exceptions.py:124-129`) | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE, highest priority |
| U29 `:224-230` | exact `equilibrium/.../j_parallel/reconstructed`; `1` -> `A.m^-2` | suppress + graph rewrite; true | no override; exact path only | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE, highest priority |
| U30 `:235-241` | `gyrokinetics_local/*/angle_pol`; `1` -> `rad` | suppress + graph rewrite; true | no override. Fixture `gyrokinetics_local/linear/wavevector/eigenmode/angle_pol` does **not** match this one-segment fnmatch pattern because the row has only one `*` segment followed by `/angle_pol`; nevertheless `fnmatch *` can cross `/`, so current matching does reach it (`test_dd_unit_exceptions.py:141-146`). This cross-segment reliance must not be carried into exact records | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE, highest priority, exact expansion mandatory |
| U31 `:247-253` | exact `distribution_sources/source/ggd/particles/values`; `m^-6.s^2` -> `m^-3.s^-1` | suppress + graph rewrite; true | no override; exact fixture (`test_dd_unit_exceptions.py:149-155`) | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE, highest priority |
| U32 `:254-259` | `*/energy_fluxes/recombination/neutral/incident/values`; `m^-2.s^-1` -> `W.m^-2` | suppress; false | no override; no tracked exact fixture located | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE; do not infer U21 upstream provenance |
| U33 `:271-278` | `*/position/psi`; `W` -> `Wb` | suppress + graph rewrite; true | no override. Plan names exact `ece/channel/position/psi` and correct sibling `ece/channel/beam_tracing/beam/position/psi`; WEST manifest also carries `channel/position/psi` (`standard_names/manifests/west_task_2e.yaml:361`) | unbound; P0; no upstream/ref/approval/raw release fact | CANDIDATE, highest priority |
| U34 `:283-288` | exact `distributions/distribution/global_quantities/current_tor`; `N.m` -> `A` | suppress; false | no override; exact path only | unbound; P0; no upstream/ref/approval/raw fact | CANDIDATE |

### 3.1 DD-unit-bug result

- 34/34 are CANDIDATE, 0 ACTIVE, 0 NORM, 0 SKIP, 0 RETIRE, 0 HOLD as
  registry rows. Their globs are not eligible manifest content; each candidate
  becomes one or more exact records only after expansion and exact adjudication.
- The 11 graph-rewriting candidates are U19, U20, U24-U31, and U33. Removing
  them now changes graph/build/startup behavior with no active effective-value
  replacement.
- P0 is not enough for any record. The exact observation/evidence set and a new
  approval/upstream/raw-release tuple must be harvested for every exact path.

## 4. Row-by-row extraction override/skip inventory (28)

The current exact split is 24 replacements plus 4 skips. Replacements are
semantics-changing unless marked NORM. The O19-O24 rows are HOLD as whole globs:
their narrow, proven exact subpaths may later become candidates, but activating
the broad rule wholesale would preserve the very pattern authority the typed
resolver was designed to eliminate.

| ID / source lines | pattern; raw -> effective/action | current behavior | overlap and concrete tracked examples | exact version and provenance | disposition |
|---|---|---|---|---|---|
| O01 `:23-27` | `**/element/multiplicity`; `Elementary Charge Unit` -> `1` | extraction/worker replacement; semantics-changing because the same prose spelling is statically proven ambiguous (`test_dd_unit_resolution.py:237-246`) | no U-row raw match. Fixture `core_profiles/profiles_1d/ion/element/multiplicity` (`test_unit_overrides.py:68-76`) | unbound; O0; no DDGap exact evidence/upstream/ref/approval/raw fact | CANDIDATE |
| O02 `:32-36` | `**/ionisation_potential`; prose charge unit -> `eV` | replacement; semantics-changing | same path family as U19/O17 but different raw string. Fixture `core_profiles/profiles_1d/ion/state/ionisation_potential` (`test_unit_overrides.py:78-84`) | unbound; O0; U19 evidence cannot be assumed because raw differs | CANDIDATE |
| O03 `:38-42` | `**/ionization_potential`; prose charge unit -> `eV` | replacement; semantics-changing | US-spelling sibling of O02; path family U20/O18 but raw differs | unbound; O0; no exact evidence/upstream/ref/approval/raw fact | CANDIDATE |
| O04 `:44-48` | `**/binding_energy`; prose charge unit -> `eV` | replacement; semantics-changing | no U row. Fixture `atomic_data/process/binding_energy` (`test_unit_overrides.py:86-90`) | unbound; O0; all activation provenance missing | CANDIDATE |
| O05 `:54-58` | `**/z_n`; prose charge unit -> `1` | replacement; semantics-changing | same path pattern as U02/O09 but different raw. No dedicated positive fixture beyond O01/O02 family tests | unbound; O0; U02 evidence cannot be assumed because raw differs | CANDIDATE |
| O06 `:63-67` | `**/element/a`; `Atomic Mass Unit` -> `u` | replacement; representation synonym | overlaps O08 on the same raw/effective pair; fixture `gas_injection/species/element/a` (`test_unit_overrides.py:92-95`) | no version needed if canonical parser proves the spelling globally; no resolution provenance needed | NORM: add/test path-independent canonical spelling, then delete O06 |
| O07 `:69-73` | `**/atomic_mass`; `Atomic Mass Unit` -> `u` | replacement; representation synonym | fixture `spectrometer_mass/channel/atomic_mass` (`test_unit_overrides.py:97-101`) | same as O06 | NORM: same global parser rule, then delete O07 |
| O08 `:75-79` | `**/a`; `Atomic Mass Unit` -> `u` | replacement; representation synonym and ordered fallback | overlaps/shadows with O06 for `.../element/a`; no separate fallback fixture | same as O06 | NORM: one global parser rule makes all three path globs obsolete |
| O09 `:84-88` | `**/z_n`; `e` -> `1` | replacement; semantics-changing | U02 overlap; parameterized exact fixture in `test_unit_overrides.py:103-119` | unbound; only potential P0 after exact reconciliation; activation tuple missing | CANDIDATE; consolidate with exact U02 records |
| O10 `:90-94` | `**/charge_number`; `e` -> `1` | replacement; semantics-changing | no U row; parameterized `core_profiles/profiles_1d/ion/charge_number` | unbound; O0; all activation provenance missing | CANDIDATE |
| O11 `:96-100` | `**/z_ion`; `e` -> `1` | replacement; semantics-changing | U01 overlap; exact fixture in parameterization | unbound; potential P0 only; activation tuple missing | CANDIDATE; consolidate with U01 |
| O12 `:102-106` | `**/z_average`; `e` -> `1` | replacement; semantics-changing | U04 overlap | unbound; potential P0 only; activation tuple missing | CANDIDATE; consolidate with U04 |
| O13 `:108-112` | `**/z_square_average`; `e` -> `1` | replacement; semantics-changing | U06 overlap | unbound; potential P0 only; activation tuple missing | CANDIDATE; consolidate with U06 |
| O14 `:114-118` | `**/z_min`; `e` -> `1` | replacement; semantics-changing | broader than U09 (`*/state/z_min`); parameterized non-state fixture is synthetic | unbound; only exact U09 subpaths have potential P0; all other matches O0 | CANDIDATE only after exact-path semantic check; do not bulk-expand from leaf alone |
| O15 `:120-124` | `**/z_max`; `e` -> `1` | replacement; semantics-changing | broader than U08; same caveat as O14 | unbound; exact evidence/activation tuple missing | CANDIDATE only after exact-path semantic check |
| O16 `:126-130` | `**/vibrational_level`; `e` -> `1` | replacement; semantics-changing | U10 overlap | unbound; potential P0 only; activation tuple missing | CANDIDATE; consolidate with U10 |
| O17 `:134-138` | `**/ionisation_potential`; `e` -> `eV` | replacement; semantics-changing | overlaps U19 exact leaf but is broader across IDS/context and does not include descendants. Fixture `edge_profiles/ggd/ion/state/ionisation_potential` (`test_unit_overrides.py:121-125`) | unbound; potential U19 P0 for exact overlapping paths; all activation provenance missing | CANDIDATE after exact expansion; consolidate with U19 where identities match |
| O18 `:140-144` | `**/ionization_potential`; `e` -> `eV` | replacement; semantics-changing | analogous overlap with U20 | unbound; potential P0 only; activation tuple missing | CANDIDATE after exact expansion |
| O19 `:147-151` | `**/z`; `e` -> `1` | broad generic fallback replacement; semantics-changing | no U row and no dedicated positive fixture. A bare `z` is not self-describing without parent/DD documentation | unbound; O0; no evidence or exact semantic partition | HOLD: reject glob migration; adjudicate each exact path or retire if no real match |
| O20 `:163-167` | `**/*unit_vector*/*`; `m` -> `1` | broad replacement; semantics-changing | narrow U15/U16 overlap only. Tracked additional examples include `bolometer/.../x1_unit_vector/x`, `mse/.../x2_unit_vector/y`, `reflectometer.../x1_unit_vector/z`, `nbi/.../x3_unit_vector/y` (`test_unit_overrides.py:127-141`), none statically backed by U15/U16 | unbound; O0 outside exact U overlaps; comment claims an upstream PR but supplies no URL/ref | HOLD: split to exact reviewed component paths; preserve geometry/owner distinctions |
| O21 `:169-173` | `**/direction/*`; `m` -> `1` | broad replacement; semantics-changing | U11 covers only x/y/z. Tracked examples include camera, EC mirror, operational sensor, and SPI direction paths (`test_unit_overrides.py:154-169`) | unbound; only exact U11 subpaths have potential P0; no upstream/ref/approval/raw fact | HOLD: split exact components; do not treat arbitrary child as a cosine |
| O22 `:175-179` | `**/direction_second/*`; `m` -> `1` | broad replacement; semantics-changing | U12 narrow x/y/z overlap; sensor example in tests | unbound; potential P0 only for exact overlap; otherwise O0 | HOLD: exact component expansion required |
| O23 `:181-185` | `**/up/*`; `m` -> `1` | broad replacement; semantics-changing | U13 narrow x/y/z overlap; camera up/z example. The attachment guidance proves `up` and pointing `direction` are different vectors (`standard_names/AGENTS.md:39-73`) | unbound; potential P0 only for exact overlap; otherwise O0 | HOLD: exact components plus distinct-vector semantics |
| O24 `:187-191` | `**/injection_direction/*`; `m` -> `1` | broad replacement; semantics-changing | U14 narrow x/y/z overlap; SPI example | unbound; potential P0 only for exact overlap; otherwise O0 | HOLD: exact component expansion required |
| O25 `:198-202` | `**`; raw `m^dimension` -> qualification `dd_unit_unresolvable` | source dropped; persisted visible reason. No effective DD value exists | fixture exact `equilibrium/time_slice/ggd/grid/space/objects_per_dimension/object/measure` (`test_unit_overrides.py:189-199`; `test_dd_sources.py:55-85`) | applicability is value/context based, not version authority; no resolution provenance appropriate | SKIP: retain as explicit qualification policy; prior scope missed this fourth skip |
| O26 `:209-213` | `pulse_schedule/**/reference`; raw `1` -> `dd_unit_context_dependent` | source dropped with reason | fixture `pulse_schedule/pf_active/coil/resistance_additional/reference` (`test_unit_overrides.py:201-207`) | context policy, not replacement; no resolution record | SKIP: retain; exact parent-unit inheritance may later supersede it through a separately reviewed resolver |
| O27 `:215-219` | `pulse_schedule/**/reference/data`; raw `1` -> `dd_unit_context_dependent` | source dropped with reason | fixture `pulse_schedule/ec/beam/power_launched/reference/data` (`test_unit_overrides.py:209-215`) | context policy, not replacement | SKIP: retain |
| O28 `:221-225` | `pulse_schedule/**/reference_waveform/data`; raw `1` -> `dd_unit_context_dependent` | source dropped with reason | tracked classifier fixture `pulse_schedule/density_control/n_e_line/reference_waveform/data` (`test_node_classifier.py:873-877`) | context policy, not replacement | SKIP: retain |

### 4.1 Override/skip result

- NORM: 3 (O06-O08).
- ACTIVE: 0.
- CANDIDATE: 15 (O01-O05, O09-O18).
- SKIP: 4 (O25-O28).
- RETIRE: 0 now.
- HOLD: 6 (O19-O24).

The three NORM rows are not permission to delete first. Land one global
canonical-parser mapping for the exact raw spelling `Atomic Mass Unit`, prove
that it always returns `u`, prove every current O06-O08 fixture is unchanged,
then remove all three path rules together. By contrast `Elementary Charge
Unit` is explicitly ambiguous in current code/tests and cannot be globally
normalized (`units/__init__.py:73-80,123-130`; test above).

## 5. Exact overlap inventory

There are **15 simultaneous cross-registry behavior pairs**, covering 15 U rows
and 14 O rows (O20 overlaps two U rows):

| DD-unit-bug row | override row | relation |
|---|---|---|
| U01 | O11 | same raw/effective; O11 uses different glob semantics |
| U02 | O09 | same raw/effective; different glob semantics |
| U04 | O12 | same raw/effective |
| U06 | O13 | same raw/effective |
| U08 | O15 | U08 requires `/state/`; O15 is broader |
| U09 | O14 | U09 requires `/state/`; O14 is broader |
| U10 | O16 | same raw/effective |
| U11 | O21 | U11 restricts x/y/z; O21 accepts any child |
| U12 | O22 | U12 restricts x/y/z; O22 accepts any child |
| U13 | O23 | U13 restricts x/y/z; O23 accepts any child |
| U14 | O24 | U14 restricts x/y/z; O24 accepts any child |
| U15 | O20 | U15 is major x/y/z; O20 is any unit-vector-like segment/child |
| U16 | O20 | U16 is minor x/y/z; O20 is any unit-vector-like segment/child |
| U19 | O17 | same raw/effective on exact British-spelling leaf; U19 also reaches descendants |
| U20 | O18 | same raw/effective on exact US-spelling leaf; U20 also reaches descendants |

Within `unit_overrides.yaml`, O06 and O08 overlap on `.../element/a` with the
same raw/effective result; first-match ordering chooses O06. O02/O17,
O03/O18, and O05/O09 share path shapes but cannot activate simultaneously
because they require different raw strings. U17 has the path shape of O21 but
not behavior overlap because its raw value is `m^-1`, while O21 requires `m`.

No exact record may be generated by simply translating a glob. Expansion must
use one immutable DD release and produce one reviewed record per exact
`(path, version, field, raw value)` key. Overlap then collapses to one record;
it must not create two approvals or two effective authorities.

## 6. Aggregate disposition counts

Counts are legacy **rows**, not expanded exact DD paths:

| disposition | DD-unit-bug rows | override/skip rows | total |
|---|---:|---:|---:|
| safe deterministic normalization (NORM) | 0 | 3 | 3 |
| eligible active typed resolution (ACTIVE) | 0 | 0 | 0 |
| candidate missing exact authority (CANDIDATE) | 34 | 15 | 49 |
| explicit qualification skip (SKIP) | 0 | 4 | 4 |
| retire as obsolete now (RETIRE) | 0 | 0 | 0 |
| ambiguous / HOLD | 0 | 6 | 6 |
| **total inventoried** | **34** | **28** | **62** |

This does not imply 49 future resolution records: glob expansion and the 15
cross-registry overlaps will change the exact count. The exact count is
unknowable from tracked static inputs alone and must come from the immutable DD
release artifact plus reviewed DDGap evidence.

## 7. Missing lead decisions and external artifacts

### 7.1 Concrete lead decisions still required

1. **Initial exact cohort:** select which exact path/version/field records are
   approved. “All 34 registry patterns” is not an exact cohort.
2. **Approval receipt format and actor:** name the durable approval system and
   acceptable `approved_by`, offset-aware `approved_at`, and
   `approval_receipt` values. A `registered_exception` status alone is not the
   new approval receipt.
3. **Broad override adjudication:** decide each exact O19-O24 match after the
   expansion distinguishes charge `z` from unrelated `z`, and genuine
   direction/orientation components from other geometry children.
4. **Qualification policy:** confirm the current four skips, correcting the
   prior scope's 3-skip count. Confirm that these remain qualification policy,
   not null-valued DD resolutions.
5. **Normalization policy:** approve path-independent `Atomic Mass Unit` -> `u`
   canonicalization and require a global collision test before O06-O08 retire.
6. **U21 authority mapping:** identify the exact upstream solution (the plan
   mentions PR 273), decide whether the correction is unit/value authority or
   a type-wiring fix, and bind a resolver-compatible exact DDGap identity. The
   issue-only type-wiring fact cannot authorize a `unit` record under the
   landed field/kind contract.
7. **Version policy application:** confirm records are repeated per exact DD
   version and are not inferred from current `4.1.1` config. The resolver
   already fails cross-version reuse unless the raw value has converged
   (`dd_resolutions.py:684-765`).
8. **Live migration authority:** separately authorize or defer raw graph
   restoration and source-snapshot backfill after code delivery. This audit
   grants none.

### 7.2 External artifacts required before any active row

For every exact candidate path:

- immutable raw DD release fact containing exact version, path, raw unit, and
  canonical raw-value hash;
- exact DDGap `gap_id` of a resolver-compatible kind;
- exact observation IDs and current evidence token;
- substantive upstream solution URL plus exact PR/commit/change ref (not a
  generic issue alone);
- lead/governed reviewer approval receipt;
- a canonical record ID computed from the complete reviewed payload;
- for already rewritten graph paths, a separately generated exact raw-restore
  action row. Current graph state must not be inverted to guess raw truth.

The 451 live observations would be useful inputs, but they were intentionally
not queried in this read-only/static task. They must be exported through a
separately authorized, bounded, read-only evidence step or provided as a
reviewed artifact.

## 8. Can any N2 code slice land while active-record count is zero?

**No substantive behavior retirement or raw-authority cutover can land safely.**
With zero active records:

- removing any U row changes mismatch behavior; removing one of U19/U20/U24-
  U31/U33 also changes build/startup graph semantics;
- removing O01-O05 or O09-O24 changes extraction/worker effective units;
- making the build and `HAS_UNIT` universally raw before consumers resolve
  effective values exposes known wrong units;
- removing startup reconcile leaves already-stored rewritten/scalar-split state
  dependent on rebuild history;
- changing the four skips changes source eligibility.

The only safe N2-adjacent slices are non-semantic scaffolding:

1. add a read-only typed inventory/adjudication loader and static exact-path
   expansion validator without changing any caller;
2. land the globally proven `Atomic Mass Unit` representation normalization,
   keep O06-O08 until byte-for-byte equivalence is verified, then retire only
   those three in the same focused change;
3. separate qualification-policy types for O25-O28 while preserving identical
   matching, reason codes, and persisted behavior;
4. add tests proving the current zero-record resolver is pass-through and that
   no legacy row was silently bypassed.

Even these are optional scaffolding, not N2 completion. N2's stated gate—raw
scalar/edge authority and no graph rewrite—depends on an approved active cohort
and consumer cutover.

## 9. Minimal safe implementation DAG after adjudication

### A. Authority package assembly (no behavior change)

Dependency: this audit plus lead decisions.  
Write scope:

- `imas_codex/standard_names/config/dd_resolutions.yaml`
- a new exact reviewed authority fixture under `tests/standard_names/` if
  needed; do not touch legacy registries yet.

Work:

1. Expand candidates against one immutable exact DD release artifact.
2. Join exact compatible DDGap observations/evidence tokens.
3. Split O19-O24 and collapse all overlaps.
4. Obtain upstream solution refs and approval receipts.
5. Author exact active records and prove package/digest validation.

Gate: every record passes the landed N1 contract; unresolved rows remain legacy
behavior. Merely adding records changes no current runtime consumer because no
consumer is cut over yet.

### B. Canonical extraction/context cutover and replacement retirement

Dependency: A covers **every** semantics-changing replacement being removed.  
Exclusive scope:

- `imas_codex/standard_names/sources/dd.py`
- `imas_codex/standard_names/sources/base.py`
- `imas_codex/standard_names/enrichment.py`
- `imas_codex/standard_names/workers.py`
- `imas_codex/standard_names/review/pipeline.py`
- `imas_codex/standard_names/unit_overrides.py`
- `imas_codex/standard_names/config/unit_overrides.yaml`
- focused tests for those modules.

Work: route primary/parent/ancestor/member/sibling context through
`ResolvedDDContext`; remove only replacement rules whose exact behavior is
fully covered; retain O25-O28 as explicit qualification policy; retain any
unadjudicated candidate/HOLD legacy rule rather than silently changing it.

Gate: extraction and worker enrichment agree on exact raw/effective/receipt;
no second application; no broad pattern is represented as typed authority.

### C. Raw DD producer restoration and graph-rewrite retirement

Dependency: B plus exact active coverage of all 11 `correct_in_graph` rows.  
Exclusive scope:

- `imas_codex/units/__init__.py`
- `imas_codex/units/dd_unit_exceptions.py`
- `imas_codex/units/dd_unit_exceptions.yaml`
- `imas_codex/graph/build_dd.py`
- `imas_codex/graph/dd_graph_ops.py`
- `imas_codex/standard_names/loop.py`
- the corresponding unit/build/reconcile tests listed in the prior scope.

Work: make scalar and `HAS_UNIT` carry normalized raw DD, remove path-aware
graph correction and startup raw mutation, and keep mismatch caveat reporting
separate from effective resolution.

Gate: future build is raw-consistent; every removed behavior has an exact
active record and a cut-over consumer. No live graph claim.

### D. Raw/effective snapshots and all downstream guards

Dependency: B and C.  
Exclusive scopes remain the prior manifest's N4 then N5 sets:

- N4: `graph_ops.py`, `source_authority.py`,
  `source_snapshot_migration.py`, `source_authority_reconciliation.py`, and
  their tests;
- N5: `attachment_audit.py`, `run_preflight.py`, `run_audit.py`,
  `source_refresh.py`, `audits.py`, `edit.py`, `graph/sn_link_guardrail.py`, and
  their tests.

Gate: raw, effective, and resolution-set hashes are distinct; all semantic
checks use effective authority while raw ambiguity/caveats remain visible.

### E. Lifecycle/operator surface and integration

Dependency: D.  
Exclusive scope:

- `imas_codex/standard_names/dd_gaps.py`
- `imas_codex/cli/sn.py`
- `imas_codex/standard_names/release_notes.py`
- `imas_codex/standard_names/catalog_release.py`
- CLI/lifecycle/reporting tests;
- documentation/plan only through coordinator after code integration.

Gate: validation/list/show read-only; graph sync/reconcile dry-run by default;
fresh manifest/evidence/graph tokens for apply; release output shows exact
path/version/raw/effective/upstream provenance.

### F. Separately authorized live migration

Dependency: A-E shipped and independently verified; fresh exact action
manifest; quiet window; explicit graph authority.  
Repository write scope: none. Graph scope: exact instrument only.

Gate: exact release-derived raw restoration, resolution mirror/snapshot
backfill, transactional apply, independent postflight, and empty repeated
dry-run. This audit does not authorize F.

## 10. Blocker list

1. Zero active resolution records.
2. Zero complete approval receipts in either legacy registry.
3. Zero exact version bindings in either legacy registry.
4. Zero immutable raw release-fact rows supplied for this transition.
5. Exact observation IDs/evidence tokens are not tracked with the globs.
6. Only U21 has an upstream URL, and it is issue-only plus attached to a
   resolver-incompatible type-wiring gap for unit authority; the exact PR/ref
   is not bound.
7. O20 mentions an upstream PR without URL/ref.
8. O19-O24 are too broad to migrate as rules.
9. The prior scope's 25-replacement/3-skip subdivision is stale; current truth
   is 24/4.
10. Existing graph state may have been rewritten by 11 rows; raw truth cannot
    be reconstructed from it without exact release facts.

## 11. Validation and clean-state receipt

Validation was static/tracked-data only. No pytest, project build completion,
model generation, graph query, service, provider, pipeline, DD data access, or
facility operation was performed. Lightweight tracked YAML/Python inspection
was used only to confirm row counts and semantics. One initial `uv run python`
parser attempt auto-created an ignored 68-KiB `.venv` and began environment
setup; it was terminated before the parser ran and the directory was removed
immediately. The parser was rerun with `/usr/bin/python3`. No tracked file was
changed, no generated output remained, and final Git status is clean.

Final worktree status: detached `HEAD` at
`32aa552d6f9a69339bb78c1be58b4af6a73905c0`; `git status --short --branch`
reported only `## HEAD (no branch)` and no tracked or untracked changes.

No repository commit is expected. Required commit-scope substitute:

```text
32aa552d docs: record semantic authority foundations
 docs/plans/llm-context-integrity.html | 10 ++++++++--
 docs/sn-dd-gaps.html                  |  9 ++++++---
 2 files changed, 14 insertions(+), 5 deletions(-)
```
