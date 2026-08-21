# Reviewed candidate attachments for unsourced identities

## Outcome

This artifact consolidates the reviewed candidate attachment cohort from
`orphan-resourcing-search.md`, `orphan-uncertain-adjudication.md`, and
`orphan-negative-retention.md`. It contains **31 candidate attachments for 31
unique unsourced identities**: the exact 36-identity census minus the four
physics/DD holds and the one physics retirement recorded by the uncertain-row
adjudication.

Every row below names one exact DD path, the Standard Name's declared unit, the
DD unit, an explicit unit verdict, and an explicit semantic verdict. All 31
candidate rows have `AGREE` unit verdicts and `ATTACH` semantic verdicts. The
two real unit disagreements in the 36-identity census are the held sensor
direction-cosine rows and therefore do not enter this attachment cohort.

`ATTACH` is a reviewed candidate disposition, not graph-mutation authority. No
row is signed for application here. A later exact workflow must re-read the
live source/name closure, run the ordinary unit and semantic guards, bind the
exact manifest hash, and refuse on drift or collateral change.

Authority inputs:

- `orphan-resourcing-search.md` — 36 identities partitioned as 25 recoverable,
  8 uncertain, and 3 initially missed by rank-limited search; SHA-256
  `4dea85aacb5d1e786711bfca68d578644f44964ddf8eb947a328b626a1f99759`;
- `orphan-uncertain-adjudication.md` — 3 attach, 4 hold, and 1 retire;
  SHA-256
  `5231adee725aa00e9c8fa5d567369490f0b18010e1429838dd0642e90d15cc80`;
- `orphan-negative-retention.md` — all 3 initially negative identities
  recovered through their pointed DD clusters; SHA-256
  `e725c96573771881f2fa72402b198852d08ef132fc38ca520483d5c037f5aec6`.

## Candidate cohort

Algebraically equivalent unit spellings are agreements: `Hz` with `s^-1`,
`N.m^-2` with `m^-2.N`, `N.m` with `m.N`, and order-only products such as
`W.m^-3` with `m^-3.W`. Where DD field metadata omits a literal unit, the DD
unit is explicitly marked as derived rather than silently presented as stored.

| Row | Standard name | Exact DD path | Standard-name unit | DD unit | Unit verdict | Semantic verdict | Reviewed semantic basis |
|---|---|---|---|---|---|---|---|
| 1 | `capacitance_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/matching_element/capacitance` | `F` | `F` | **AGREE** | **ATTACH** | The leaf is the capacitance of the antenna module's matching element, preserving the RF owner and circuit quantity. |
| 2 | `cross_section_of_flux_surface` | `core_profiles/profiles_1d/grid/area` | `m^2` | `m^2` | **AGREE** | **ATTACH** | The grid-area leaf stores poloidal flux-surface cross-sectional area; the plan explicitly withdrew the earlier deletion disposition after this 0.92 recovery. |
| 3 | `forward_wave_phase_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/phase_forward` | `rad` | `rad` | **AGREE** | **ATTACH** | The leaf preserves forward-wave direction, phase, antenna ownership, and module locus. |
| 4 | `impurity_ion_photon_radiance_of_spectral_line_due_to_charge_exchange` | `charge_exchange/channel/spectrum/processed_line/radiance` | `m^-2.s^-1.sr^-1` | `m^-2.0.s^-1.0.sr^-1` | **AGREE** | **ATTACH** | The processed charge-exchange spectral-line leaf stores photon radiance with the required channel and line semantics. |
| 5 | `line_integrated_electron_density` | `interferometer/channel/n_e_line` | `m^-2` | `m^-2` | **AGREE** | **ATTACH** | The interferometer leaf is electron density integrated along the beam path, the canonical column-density observable. |
| 6 | `minimum_magnetic_field_magnitude` | `equilibrium/time_slice/profiles_1d/b_field_min` | `T` | `T` | **AGREE** | **ATTACH** | The equilibrium profile leaf directly stores the flux-surface minimum magnetic-field magnitude. |
| 7 | `minimum_of_safety_factor` | `equilibrium/time_slice/global_quantities/q_min/value` | `1` | `1` (derived; DD metadata absent) | **AGREE** | **ATTACH** | The global `q_min` value is the minimum safety factor; safety factor is dimensionless even though the leaf omits literal unit metadata. |
| 8 | `neutral_state_power_density` | `plasma_sources/source/profiles_1d/neutral/state/energy` | `W.m^-3` | `m^-3.W` | **AGREE** | **ATTACH** | The state-resolved neutral energy-source leaf is a volumetric power term and preserves the neutral-state owner. |
| 9 | `neutron_flux_due_to_fusion` | `neutron_diagnostic/neutron_flux_total` | `Hz` | `s^-1` | **AGREE** | **ATTACH** | The diagnostic total neutron-rate leaf represents the total fusion-neutron count rate; `Hz` and `s^-1` are identical dimensions. |
| 10 | `parallel_current_density_due_to_ohmic_current_drive` | `core_profiles/profiles_1d/j_ohmic` | `A.m^-2` | `A.m^-2` | **AGREE** | **ATTACH** | The `j_ohmic` profile is explicitly the ohmic parallel-current-density contribution. |
| 11 | `parallel_mach_number` | `langmuir_probes/reciprocating/plunge/mach_number_parallel` | `1` | `1` (derived; DD metadata absent) | **AGREE** | **ATTACH** | The reciprocating-probe leaf explicitly stores parallel Mach number; Mach number is dimensionless despite absent literal unit metadata. |
| 12 | `parallel_neutral_momentum_diffusion_coefficient` | `plasma_transport/model/profiles_1d/neutral/state/momentum/d_parallel` | `m^2.s^-1` | `m^2.0.s^-1` | **AGREE** | **ATTACH** | The state-resolved neutral momentum equation's parallel diffusion coefficient preserves owner, axis, and transport mechanism. |
| 13 | `poloidal_neutral_internal_state_momentum_convected_velocity` | `plasma_transport/model/profiles_1d/neutral/state/momentum/v_pol` | `m.s^-1` | `m.s^-1` | **AGREE** | **ATTACH** | The momentum-equation leaf is the explicit poloidal convected-velocity coefficient for a resolved neutral state. |
| 14 | `poloidal_neutral_state_particle_convection_velocity` | `plasma_transport/model/ggd/neutral/state/particles/v_pol` | `m.s^-1` | `m.s^-1` | **AGREE** | **ATTACH** | The particle-equation leaf explicitly distinguishes particle convection from the momentum and energy equation siblings. |
| 15 | `poloidal_straight_field_line_angle` | `distributions/distribution/profiles_2d/grid/theta_straight` | `rad` | `rad` | **AGREE** | **ATTACH** | The distribution-grid leaf directly stores the straight-field-line poloidal angle. |
| 16 | `radial_effective_electron_diffusivity` | `edge_transport/model/ggd/electrons/particles/d_radial` | `m^2.s^-1` | `m^2.0.s^-1` | **AGREE** | **ATTACH** | The electron particle-transport leaf is explicitly the effective radial diffusion coefficient. |
| 17 | `radial_effective_ion_diffusivity` | `plasma_transport/model/ggd/ion/particles/d_radial` | `m^2.s^-1` | `m^2.0.s^-1` | **AGREE** | **ATTACH** | The ion particle-transport leaf is explicitly the effective radial diffusion coefficient. |
| 18 | `radial_effective_neutral_diffusivity` | `edge_transport/model/ggd/neutral/particles/d_radial` | `m^2.s^-1` | `m^2.0.s^-1` | **AGREE** | **ATTACH** | The neutral particle-transport leaf is explicitly the effective radial diffusion coefficient. |
| 19 | `radial_thermal_ion_charge_state_energy_diffusion_coefficient` | `plasma_transport/model/ggd/ion/state/energy/d_radial` | `m^2.s^-1` | `m^2.0.s^-1` | **AGREE** | **ATTACH** | The leaf preserves radial axis, thermal-ion charge-state resolution, energy-equation ownership, and diffusive mechanism. |
| 20 | `toroidal_co_passing_thermal_electron_torque_density_due_to_collisions` | `distributions/distribution/profiles_2d/co_passing/collisions/electrons/torque_thermal_phi` | `N.m^-2` | `m^-2.N` | **AGREE** | **ATTACH** | The kinetic leaf preserves co-passing orbit class, electron collision partner, transfer to the thermal population, toroidal component, and density form. |
| 21 | `toroidal_line_integrated_impurity_ion_velocity` | `charge_exchange/channel/ion/velocity_phi` | `m.s^-1` | `m.s^-1` | **AGREE** | **ATTACH** | The charge-exchange channel's impurity-ion toroidal velocity is the line-of-sight-integrated rotation observable described by the identity. |
| 22 | `toroidal_thermal_ion_charge_state_torque_due_to_collisions` | `distributions/distribution/global_quantities/collisions/ion/state/torque_thermal_phi` | `N.m` | `m.N` | **AGREE** | **ATTACH** | The global leaf is integrated toroidal collisional torque transferred to the thermal ion population, resolved by charge state. |
| 23 | `toroidal_thermal_ion_torque_density_due_to_thermalization` | `distributions/distribution/profiles_2d/collisions/ion/torque_thermal_phi` | `N.m^-2` | `m^-2.N` | **AGREE** | **ATTACH** | DD documentation identifies momentum transfer from a non-Maxwellian distribution to background thermal ions, matching thermalization torque density. |
| 24 | `toroidal_trapped_fast_ion_charge_state_torque_density_due_to_collisions` | `distributions/distribution/profiles_2d/trapped/collisions/ion/state/torque_fast_tor` | `N.m^-2` | `m^-2.N` | **AGREE** | **ATTACH** | The leaf preserves trapped orbit class, fast-ion recipient, charge-state resolution, collisional process, toroidal component, and density form. |
| 25 | `variation_of_length_of_interferometer_beam` | `interferometer/channel/path_length_variation` | `m` | `m` | **AGREE** | **ATTACH** | The interferometer channel leaf directly stores beam optical-path-length variation. |
| 26 | `flux_surface_averaged_toroidal_flux_coordinate_gradient_magnitude` | `equilibrium/time_slice/profiles_1d/gm7` | `1` | `1` (derived; DD metadata absent) | **AGREE** | **ATTACH** | DD 4.1 defines `gm7` as the flux-surface average of `|grad rho_tor|`; metre-valued `rho_tor` over metre coordinates makes the gradient dimensionless, and `gm3` is the distinct squared metric. |
| 27 | `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | `edge_profiles/ggd/neutral/state/velocity_diamagnetic/parallel` | `m.s^-1` | `m.s^-1` | **AGREE** | **ATTACH** | The explicit child resolves the parallel projection beneath neutral state and diamagnetic velocity, preserving owner, state resolution, mechanism, and axis. |
| 28 | `poloidal_neutral_internal_state_convection_velocity` | `edge_transport/model/ggd/neutral/state/particles/v_pol` | `m.s^-1` | `m.s^-1` | **AGREE** | **ATTACH** | DD 4.1 calls this leaf particle effective convection and provides a separate momentum sibling, resolving the earlier equation-owner ambiguity. |
| 29 | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `summary/pedestal_fits/mtanh/b_field_pedestal_top_lfs/value` | `T` | `T` | **AGREE** | **ATTACH** | The pedestal-fit leaf explicitly preserves magnetic-field magnitude, pedestal-top locus, and low-field-side position. |
| 30 | `tendency_of_total_thermal_plasma_internal_energy` | `summary/global_quantities/denergy_thermal_dt/value` | `W` | `W` | **AGREE** | **ATTACH** | The summary global-quantity leaf is the time derivative of total thermal plasma energy rather than stored energy or an individual source power. |
| 31 | `toroidal_neutral_state_momentum_diffusivity` | `plasma_transport/model/ggd/neutral/state/momentum/d/phi` | `m^2.s^-1` | `m^2.s^-1` | **AGREE** | **ATTACH** | The GGD neutral-state momentum-diffusion tensor's `phi` leaf supplies the independently stored toroidal coefficient missed by the original top-eight search. |

## Excluded dispositions

The five excluded identities are not silently dropped. They are the exact
non-attachment set recorded by `orphan-uncertain-adjudication.md`:

| Standard name | Disposition | Why it is outside the candidate attachment cohort |
|---|---|---|
| `fast_ion_charge_state_power_at_inside_flux_surface` | **HOLD** | DD spelling and hierarchy suggest fast-ion power, but the prose says thermal-ion deposition and conflicts with the distinct thermal sibling. |
| `toroidal_ion_charge_state_torque_density` | **HOLD** | No DD path simultaneously supplies total, charge-state-resolved, and toroidal semantics. |
| `x_direction_unit_vector_of_sensor` | **HOLD** | The dimensionless direction-cosine name conflicts with the DD child's metre unit and coordinate wording. |
| `z_direction_unit_vector_of_sensor` | **HOLD** | The dimensionless direction-cosine name conflicts with the DD child's metre unit and coordinate wording. |
| `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | **RETIRE** | The candidate classifies a trapped source distribution transferring torque to a background thermal recipient; the identity incorrectly collapses both roles onto trapped thermal ions. |

## Quantitative validation

The following read-only command parses the three authority inputs and this
artifact. It checks the 36-row census, the exact four-hold/one-retirement
exclusion, the candidate table's row and identity uniqueness, required path and
unit fields, explicit verdicts, and exact set equality:

```sh
env -u VIRTUAL_ENV \
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
PYTHONPATH="$PWD" uv run --no-sync python - <<'PY'
from pathlib import Path
import re

root = Path("docs/evidence/sn-graph-wide-integrity")
search = (root / "orphan-resourcing-search.md").read_text()
uncertain = (root / "orphan-uncertain-adjudication.md").read_text()
cohort = (root / "orphan-attachment-candidates.md").read_text()

census = {
    match.group(1)
    for match in re.finditer(r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|", search, re.M)
}
uncertain_rows = re.findall(
    r"^\|\s*U\d+\s*\|\s*`([^`]+)`\s*\|\s*(ATTACH|HOLD|RETIRE)\s*\|",
    uncertain,
    re.M,
)
holds = {name for name, verdict in uncertain_rows if verdict == "HOLD"}
retirements = {name for name, verdict in uncertain_rows if verdict == "RETIRE"}

candidate_rows = re.findall(
    r"^\|\s*(?:[1-9]|[12]\d|3[01])\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*\*\*(AGREE|DISAGREE)\*\*\s*\|\s*\*\*(ATTACH)\*\*\s*\|",
    cohort,
    re.M,
)
candidates = [row[0] for row in candidate_rows]
candidate_set = set(candidates)
expected = census - holds - retirements

print(f"census_rows={len(census)}")
print(f"holds={len(holds)}")
print(f"retirements={len(retirements)}")
print(f"expected_candidates={len(expected)}")
print(f"candidate_rows={len(candidate_rows)}")
print(f"candidate_unique={len(candidate_set)}")
print(f"paths_present={sum(bool(row[1]) for row in candidate_rows)}")
print(f"dd_units_present={sum(bool(row[3].strip()) for row in candidate_rows)}")
print(f"unit_agree={sum(row[4] == 'AGREE' for row in candidate_rows)}")
print(f"unit_disagree={sum(row[4] == 'DISAGREE' for row in candidate_rows)}")
print(f"semantic_attach={sum(row[5] == 'ATTACH' for row in candidate_rows)}")
print(f"set_missing={sorted(expected - candidate_set)}")
print(f"set_extra={sorted(candidate_set - expected)}")

assert len(census) == 36
assert len(holds) == 4
assert len(retirements) == 1
assert len(expected) == 31
assert len(candidate_rows) == 31
assert len(candidate_set) == 31
assert all(row[1] and row[2] and row[3].strip() for row in candidate_rows)
assert all(row[4] in {"AGREE", "DISAGREE"} for row in candidate_rows)
assert all(row[5] == "ATTACH" for row in candidate_rows)
assert candidate_set == expected
print("PASS")
PY
```

Recorded output:

```text
census_rows=36
holds=4
retirements=1
expected_candidates=31
candidate_rows=31
candidate_unique=31
paths_present=31
dd_units_present=31
unit_agree=31
unit_disagree=0
semantic_attach=31
set_missing=[]
set_extra=[]
PASS
```

No graph mutation, attachment, deletion, retirement, provider call, pipeline
operation, or name acceptance was performed.
