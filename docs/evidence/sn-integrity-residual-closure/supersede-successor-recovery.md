NEEDS-HELP: all four authorized successor recoveries applied and replayed exactly, but the required global null-successor decrease of at least 4 is mathematically impossible for a cohort whose pre-read contained only one already-superseded member; the measured decrease is 1.

tried: Re-read every predecessor and adjudicated successor from the live graph, then executed four sequential one-row signed invocations in one production runner. Each invocation constructed its authority only after its own live read, previewed one admitted supersede with no refusal, applied the preview hash, replayed that same hash, and verified the predecessor scalar plus its receipt. All four mutations and all four replays passed. The final assertion measured the global `name_stage=superseded AND superseded_by IS NULL` count at 1,626 before and 1,625 after, a decrease of 1 rather than the required minimum of 4.

options: (1) amend the count gate to the quantity the four authorized transitions can change: exact-cohort unresolved successor records, which fell 4 to 0; equivalently retain the global count and expect the mathematically exact decrease of 1. (2) Authorize three additional, specifically adjudicated recoveries from the pre-existing global null-successor backlog so the global decrease reaches 4; this requires new predecessor-to-successor authority beyond this cohort. (3) Require a new execution design and explicitly authorize reversal of the four valid receipts before retrying; reversal is destructive and still cannot make the three initially pending or accepted predecessors belong to the pre-apply null-superseded population without first recording invalid partial transitions.

leaning: Option 1. It measures the actual semantic obligation—each of the four named predecessors has its exact canonical successor—and preserves the valid immutable history. The global count is a catalog-wide backlog metric, not a cardinality measure for three transitions that became superseded and received their successor atomically.

cost-if-wrong: If the intended obligation was genuinely to reduce the unrelated global backlog by four, the orchestrator must adjudicate three more exact successor identities and issue new signed authorities. The four transitions recorded here remain valid, but this node stays blocked until those additional mutations apply or the gate is corrected.

# Supersede successor recovery — applied cohort and unsatisfiable aggregate gate

## Material outcome

The production graph now records the exact adjudicated successor for all four
predecessors, and each new immutable `StandardNameChange` row records that same
successor in `to_name`. The signed work itself is complete and write-free on
replay:

- signed applies: **4 of 4** returned `outcome=applied`, `changed=1`, and
  `receipt_rows=1`;
- same-hash replays: **4 of 4** returned `outcome=already_applied`, `changed=0`,
  and `persistent_writes=0`;
- post-apply predecessor states: **4 of 4** are
  `name_stage=superseded`, `status=superseded`, with `superseded_by` equal to
  the exact adjudicated successor;
- receipt targets: **4 of 4** have `from_name` equal to the predecessor and
  `to_name` equal to that successor;
- `StandardNameChange`: **7,878 → 7,882**, delta **+4**, exactly equal to the
  four receipt rows.

The node is nevertheless blocked on the aggregate count gate. The applying
invocation measured **1,626 → 1,625** live Standard Names at
`name_stage=superseded` with a null `superseded_by`, a decrease of **1** against
the required minimum decrease of **4**.

## Sequential signed invocations

The runner processed the rows strictly in the order below. It did not carry a
previously computed cohort. For each row it re-read the two identities, required
zero live structural children on the predecessor, required a null existing
successor scalar, required the successor to be accepted and valid, built and
signed a fresh one-row authority, previewed it, applied the returned manifest
hash, replayed that hash, and completed the persistent post-read before moving
to the next row.

| Sequence | Predecessor | Exact adjudicated successor | Authority file SHA-256 | Signed payload SHA-256 | Manifest SHA-256 | Apply | Same-hash replay |
|---:|---|---|---|---|---|---|---|
| 1 | `area_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | `059edce8222e123b0f5cd63c4795fd10d9397e7301d7a95506ef61fe49bacd70` | `6ef957c7600bd7529efc87fb5d82f74423b7af59fb4e14d8fdb23777d05cc793` | `6eb731cb3f0b5f8254bc7d346bb479c10106cf6de3de771e7404df68f4d1f038` | `applied`, changed 1, receipt 1 | `already_applied`, changed 0, writes 0 |
| 2 | `cross_section_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | `038f99475e9ce62d44e885d594482d65be0a33c9b0effc7fcec0b1a0bbd41d34` | `c56e857a60def490b4f5c87b915c16f290c40120da7c515ad90ecda1f16daf54` | `cd31a552db12bf6926aa9755a444ae7b8b5bc3f6b723ff7b1b6631dca3304976` | `applied`, changed 1, receipt 1 | `already_applied`, changed 0, writes 0 |
| 3 | `x_direction_unit_vector_of_sensor` | `x_first_measurement_direction_unit_vector_of_strain_gauge` | `b015273c317de9e9706d883978d44cc55e09dd93de12824e52ef650ff4ff2652` | `3ab3230ae351143b24aa8c403eeaaa9d77a4c5af8d3e21e707270c62ae4f2d58` | `89daa6bfa2c3a92ce1368488d6986867dbb44dbe5c986bb366e5f3db8910489f` | `applied`, changed 1, receipt 1 | `already_applied`, changed 0, writes 0 |
| 4 | `z_direction_unit_vector_of_sensor` | `z_first_measurement_direction_unit_vector_of_strain_gauge` | `7897e89d0ba99ede7e071618a46b4067403afc89825e5b44b49fe4641ba78f54` | `8b7fd610dba0d15d016bbe29c30c340d63550fece5ad167b4c5ed4b017ecf47f` | `c226ae0cf9881d1f40ea842e104b1b3aa9955462152d19b6178d890c065966d0` | `applied`, changed 1, receipt 1 | `already_applied`, changed 0, writes 0 |

Every preview returned `authority_rows=1`, `admitted=1`, `refused=0`, and
`would_change=1`. Each authority contains only one `supersede` mutation with an
explicitly signed `successor_id`; no invocation mixed mutation kinds.

## Persistent post-state and receipts

The independent read-only diagnostic re-read each predecessor, successor, and
receipt by the exact run id, operation, manifest digest, and row id.

| Predecessor | Post `name_stage` / vocabulary `status` | Post `superseded_by` | Receipt `to_name` | Receipt id |
|---|---|---|---|---|
| `area_of_flux_surface` | `superseded` / `superseded` | `poloidal_plane_cross_sectional_area_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | `sn-change:signed-manifest:6eb731cb3f0b5f8254bc7d346bb479c10106cf6de3de771e7404df68f4d1f038:faf45258d8d7587981353eea` |
| `cross_section_of_flux_surface` | `superseded` / `superseded` | `poloidal_plane_cross_sectional_area_of_flux_surface` | `poloidal_plane_cross_sectional_area_of_flux_surface` | `sn-change:signed-manifest:cd31a552db12bf6926aa9755a444ae7b8b5bc3f6b723ff7b1b6631dca3304976:730f1e60d64fc9c0a29d86b2` |
| `x_direction_unit_vector_of_sensor` | `superseded` / `superseded` | `x_first_measurement_direction_unit_vector_of_strain_gauge` | `x_first_measurement_direction_unit_vector_of_strain_gauge` | `sn-change:signed-manifest:89daa6bfa2c3a92ce1368488d6986867dbb44dbe5c986bb366e5f3db8910489f:956e84e1d3661003d74c68d6` |
| `z_direction_unit_vector_of_sensor` | `superseded` / `superseded` | `z_first_measurement_direction_unit_vector_of_strain_gauge` | `z_first_measurement_direction_unit_vector_of_strain_gauge` | `sn-change:signed-manifest:c226ae0cf9881d1f40ea842e104b1b3aa9955462152d19b6178d890c065966d0:e501c8a91f0e92b2ad3f12f6` |

All four successors re-read as accepted and valid. The applying runner also
hashed each successor's complete properties and producing-source closure before
and after its predecessor transition and required byte-equivalent hashes before
continuing.

## Why the aggregate gate cannot reach four

The applying invocation's pre-read recorded these predecessor lifecycle states:

| Predecessor | Stage before its signed invocation | Included in the global pre-count? |
|---|---|---|
| `area_of_flux_surface` | `superseded`, null `superseded_by` | yes |
| `cross_section_of_flux_surface` | `pending`, null `superseded_by` | no |
| `x_direction_unit_vector_of_sensor` | `accepted`, null `superseded_by` | no |
| `z_direction_unit_vector_of_sensor` | `accepted`, null `superseded_by` | no |

The signed operator writes `name_stage=superseded` and the non-null
`superseded_by` scalar atomically. The three live predecessors therefore never
enter the counted null-successor state. Within this exact authorized cohort,
only `area_of_flux_surface` can leave the global pre-count, so the maximum
possible decrease is **1**. The measured global decrease is exactly that:
**1,626 → 1,625**.

The semantically aligned exact-cohort measure did reach zero: predecessors that
were not yet superseded or lacked their exact successor fell **4 → 0**. This is
not substituted for the stated global gate; it is reported only to make the
necessary gate correction concrete.

## Durable artifacts

The production artifacts are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T213100350941-n-supersederecover/`:

- `supersede_successor_recovery_runner.py` — sequential live-read, authority
  derivation, signing, preview, apply, replay, and assertion runner;
- `01-area_of_flux_surface-authority.json` through
  `04-z_direction_unit_vector_of_sensor-authority.json` — the four signed
  one-row authorities;
- `supersede-successor-recovery-apply.log` — complete production transcript,
  ending `EXIT_MARKER=1` only because the aggregate decrease assertion observed
  1 rather than the impossible required 4;
- `supersede-successor-recovery-diagnostic.json` — compact persistent state,
  receipt, digest, and counter evidence;
- `supersede-successor-recovery-diagnostic.log` — independent read-only check,
  ending `EXIT_MARKER=0`.

No raw Cypher mutation, direct property edit, rollback, or mutation outside the
four signed predecessor rows was performed.
