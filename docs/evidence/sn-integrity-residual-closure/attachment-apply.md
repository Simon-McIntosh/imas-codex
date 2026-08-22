# Governed unbound-source attachment apply

## Outcome

**COMPLETE — three exact ordinary DD sources attached.** One production
invocation freshly regenerated the live genuine-orphan cohort and its reviewed
candidate-path join, derived the maximal existing-unbound subset, signed that
three-row authority, previewed it, applied it, reread every mirror, queried the
exact receipt, and replayed the same manifest hash.

The in-invocation preview returned `outcome=would_apply` with **3 authority
rows = 3 admitted + 0 refused** and `would_change=3`. The refusal list is the
empty array `[]`; consequently there is no omitted refusal reason. The signed
apply returned `outcome=applied`, `changed=3`, and `receipt_rows=3`. Its
manifest SHA-256 is
`08b5da2291527fb818a0448ba81b72a109c18390792e4ad4e39911741d128dba`.

The same-hash replay returned `outcome=already_applied`, `changed=0`, and
`persistent_writes=0`. The live genuine-orphan census fell from **12 before to
9 after**, a difference of **3**, exactly equal to the applied row count.

## Fresh cohort derivation

The applying process did not consume the prior rollback preview as a fixed
cohort. It reread live names satisfying all of these conditions:

- the name and catalog lifecycles are live;
- no `StandardNameSource -> PRODUCED_NAME` reaches the name; and
- no live child reaches the name through `HAS_PARENT`.

It then joined all 12 live identities to the independently reviewed
reverse-search evidence, augmented only by the exact reviewed steering retained
in the live change ledger for the newer X-ray spectroscopy identity. The
reverse-search input SHA-256 was
`b1b6714f22071edaf048baaf79c3b828c0f041ca777f54ff097ccb16d21946d7`.
For every candidate path, the invocation reread current source bindings and the
union of live `PRODUCED_NAME` and `HAS_STANDARD_NAME` owners. Only an existing
DD source with zero bindings and a path with zero live canonical owners entered
the signed authority.

| Fresh disposition | Count | Identities |
|---|---:|---|
| `signed-existing-unbound-source` | **3** | `capacitance_of_ion_cyclotron_heating_antenna`; `toroidal_ion_charge_state_torque_density`; `toroidal_line_averaged_plasma_velocity` |
| `adjudicate-collision` | **5** | `cross_section_of_flux_surface`; `fast_ion_charge_state_power_at_inside_flux_surface`; `tendency_of_total_thermal_plasma_internal_energy`; `x_direction_unit_vector_of_sensor`; `z_direction_unit_vector_of_sensor` |
| `source-already-bound` | **1** | `neutron_flux_due_to_fusion` |
| `no-existing-source` | **2** | `parallel_neutral_momentum_diffusion_coefficient`; `poloidal_neutral_internal_state_momentum_convected_velocity` |
| `no-candidate` | **1** | `toroidal_trapped_thermal_ion_charge_state_torque_density_due_to_collisions` |

This fresh partition explains why the earlier rollback diagnostic was not apply
authority. The parallel-neutral diffusion source existed only inside that
rolled-back preview and is absent live. The poloidal-neutral source is also
absent. The neutron source still has a `PRODUCED_NAME` binding to
`power_due_to_fusion_reactions`, even though that target is exhausted, so it is
not an unbound ordinary source. The five collision paths remain owned by other
live canonical identities. None of those nine rows was broadened into the
three-row closed program.

## Exact signed rows and four-mirror reread

| DD source | Accepted target identity | `PRODUCED_NAME` edges | Backing `HAS_STANDARD_NAME` edges | Source scalar/lifecycle | Target mirror occurrences |
|---|---|---:|---:|---|---:|
| `dd:ic_antennas/antenna/module/matching_element/capacitance` | `capacitance_of_ion_cyclotron_heating_antenna` | **1** | **1** | target id; `attached` | **1** |
| `dd:plasma_sources/source/ggd/ion/state/momentum/phi` | `toroidal_ion_charge_state_torque_density` | **1** | **1** | target id; `attached` | **1** |
| `dd:spectrometer_x_ray_crystal/channel/profiles_line_integrated/velocity_tor` | `toroidal_line_averaged_plasma_velocity` | **1** | **1** | target id; `attached` | **1** |

Before the apply, each of these sources had zero `PRODUCED_NAME` bindings and
each DD backing had zero live canonical owners. The post-apply reread proved,
for every row, exactly one source-to-target `PRODUCED_NAME`, exactly one
backing-to-same-target `HAS_STANDARD_NAME`, `source.status='attached'`,
`source.produced_sn_id` equal to the target, and exactly one matching `dd:` URI
in `target.source_paths`. Thus each target gained one producer beside its exact
backing projection and scalar/list mirrors, with no identity fold.

## Receipt, counters, and replay

The durable receipt was not inferred from an operation-name counter. The
invocation queried `StandardNameChange` using all three exact selectors:

- `run_id=r-20260822T211919039495-n-attachapply`;
- `manifest_sha256=08b5da2291527fb818a0448ba81b72a109c18390792e4ad4e39911741d128dba`;
  and
- `operation=attach_unbound_standard_name_source`.

That query returned exactly the three signed row ids, and every row pinned the
authority file and payload hashes shown below.

| Measure | Before | After apply | After same-hash replay | Apply delta |
|---|---:|---:|---:|---:|
| `StandardNameChange` | **7,875** | **7,878** | **7,878** | **+3** |
| `PRODUCED_NAME` | **5,774** | **5,777** | **5,777** | **+3** |
| Live genuine orphans | **12** | **9** | **9** | **−3** |

| Governed measure | Value |
|---|---|
| Authority file SHA-256 | `4c77ea17124abce5446505cba121e64489840a9cfc0e23acfad1656f04cfdc0a` |
| Signed payload SHA-256 | `71a4cb6e5477778dc65d57cb96406d1572a4290bec1fc853669d304eae27c054` |
| Apply manifest SHA-256 | `08b5da2291527fb818a0448ba81b72a109c18390792e4ad4e39911741d128dba` |
| Apply result | `outcome=applied`; `changed=3`; `receipt_rows=3`; admitted **3**; refused **0** |
| Replay result | `outcome=already_applied`; `changed=0`; `persistent_writes=0`; receipt rows still **3** |

## Durable artifacts

The complete machine-readable result is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T211919039495-n-attachapply/attachment-apply-result.json`
(SHA-256
`a7727b842cda073a11c5901238a5b32ede8230dfcea0053a323d0f1baa023bff`).
The exact signed authority is retained beside it as
`unbound-attachment-authority.json`; its file hash is the authority hash above.
The applying driver is `apply_unbound_attachments.py`, and the complete log is
`attachment-apply.log` (SHA-256
`9936b9b698dbdf0f05f3332c8b1e94ec60a9380203f2c73894ee62da14c0e5e2`).
The log terminates with `EXIT=0`.
