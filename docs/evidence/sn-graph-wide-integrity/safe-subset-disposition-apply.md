# Guarded catalog-edit source-disposition subset apply

## Outcome

The exact safe subset derived from the unchanged 216-row signed catalog-edit
adjudication was previewed, applied to the live graph, and replayed without a
second mutation. The write-free subset preview returned `outcome=would_apply`
with **138 admitted and 0 refused rows**. Applying its exact manifest SHA-256
removed **138 `PRODUCED_NAME` bindings and 138 matching
`HAS_STANDARD_NAME` projections across 59 distinct removed targets**, changed
51 source scalars, and wrote one `StandardNameChange` receipt. The replay
returned `outcome=already_applied` with `changed=0`.

The 89 last-binding targets excluded by the complete adjudication preview were
not modified. Their target, incoming `PRODUCED_NAME`, and incoming projection
closure retained SHA-256
`3512947fbbf26721cc09b4d79cf5c4306defa76996e6b337df5200dc3b9dfded`
before and after the apply. The live unsourced-name census remained exactly
**85** before apply, after apply, and after replay: 69 accepted, 4 reviewed,
4 drafted, and 8 pending.

## Exact authority derivation

No adjudication row was edited, regenerated, or re-signed. The committed
authority remains:

- artifact file SHA-256:
  `5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e`;
- signed payload SHA-256:
  `c227e70ec5cd940577ca778ce5ec63e4df3a63bf68c3e845eba92d0a4b9a0efb`;
- 216 individually signed source rows.

The operator derives the subset rather than accepting a caller-supplied source
list. In one transaction it first constructs the complete 216-row guarded
authority, deterministically selects the actions admitted by that authority,
and then constructs a new refusal-free manifest over only those selected rows.
The executable manifest includes:

- the original adjudication payload and row-set digests;
- all 216 adjudication source IDs;
- the exact 138 selected and 78 excluded source IDs;
- every excluded refusal and its protected target IDs;
- the complete parent-preview manifest SHA-256; and
- the selected rows' source, backing, binding, projection, and global incoming
  target closures.

After locking the executable subset's source nodes, target nodes, and
relationships, the operator reconstructs both the complete parent authority
and the selected authority. Any selection, refusal, graph closure, scalar,
claim, or manifest-digest drift rolls the transaction back before deletion.
The resulting hashes and counts were:

| Measure | Result |
|---|---:|
| Complete parent rows | 216 |
| Parent admitted rows | 138 |
| Parent refused source rows | 78 |
| Executable subset rows | 138 |
| Executable subset refusals | 0 |
| Executable removed-target closures | 59 |
| Incoming bindings signed for those 59 targets | 569 |
| Complete parent manifest SHA-256 | `c4d617712e5ab80ba9b8c12ec7f84301781896efe1013615e3d031c33ebaefa0` |
| Executable subset manifest SHA-256 | `10ba388e0875b77e3d0bf0b3455fcf48a5aaf17795a9a1fa34abb703703846e7` |

The parent hash differs from the earlier 216-row write-free preview hash
because the operation reason is part of signed execution authority. Its row
partition, adjudication hashes, and protected-target set are unchanged.

## Applied receipt and write-free replay

The apply used only executable manifest
`10ba388e0875b77e3d0bf0b3455fcf48a5aaf17795a9a1fa34abb703703846e7`.
It created receipt:

`sn-change:signed-source-disposition:10ba388e0875b77e3d0bf0b3455fcf48a5aaf17795a9a1fa34abb703703846e7`

| Measure | Before | After apply | After replay |
|---|---:|---:|---:|
| Admitted sources at execution | — | 138 | 138 already represented by the receipt |
| Removed `PRODUCED_NAME` bindings | — | 138 | 0 additional |
| Removed `HAS_STANDARD_NAME` projections | — | 138 | 0 additional |
| Scalars changed | — | 51 | 0 additional |
| Distinct removed targets | — | 59 | 0 additional |
| `StandardNameChange` nodes | 7,451 | 7,452 | 7,452 |
| `LLMCost` nodes | 27,467 | 27,467 | 27,467 |
| Live unsourced names | 85 | 85 | 85 |

The replay returned `already_applied` and `changed=0`. The normalized selected
source state, protected-target closure, non-selected-source digest, counters,
and unsourced census hashed to
`ee4c02b56622ce8338948bc7f387d0989caa1081f5af85845387d72779dfb6e7`
both immediately before and immediately after replay.

## Protected and collateral closure

The complete parent authority excluded 78 source rows because their scheduled
removals would eliminate the final live producer of 89 targets. Those targets
remain governed by individual orphan-policy adjudication. The executable
subset's 59 removed targets have zero intersection with that protected set.

The protected-target closure hash was identical before and after:

`3512947fbbf26721cc09b4d79cf5c4306defa76996e6b337df5200dc3b9dfded`

The 9,360 `StandardNameSource` rows outside the executable 138-row allowlist
also retained an identical normalized source, binding, backing, and projection
closure:

`5f099ddd2da36e06e03c22d6b851f88290de090faa35d36d638ed84d26fc5165`

These two independent hashes prove that neither a protected last-binding target
nor a non-selected source was changed. The unchanged `LLMCost` count proves the
operation made no provider call.

## Representative source dispositions

The applied subset contained 87 `retain_scalar_target` rows and 51
`retarget_scalar_target` rows. All three `select_missing_scalar` rows remained
in the protected 78-row partition.

- `dd:amns_data/a` retained accepted derived `atomic_mass` and removed the
  catalog-edit competitor `neutral_species_atomic_mass` (review score 0.875).
  The DD path represents the species' mass generally; the removed name adds a
  neutral-only restriction not present in the source identity.
- `dd:bolometer/camera/channel/aperture/centre/phi` retained accepted
  `toroidal_coordinate_of_aperture` (review score 0.96875) and removed
  `toroidal_angle_of_measurement_position` (0.95625). The former preserves the
  aperture owner encoded by the exact path.
- `dd:charge_exchange/channel/spectrum/radiance_spectral` changed its scalar
  from `spectral_bremsstrahlung_radiance` (accepted, 0.9625) to accepted
  `spectral_radiance`. The source is a charge-exchange spectrum, so the removed
  identity's bremsstrahlung mechanism is an unsupported extra claim.
- `dd:coils_non_axisymmetric/coil/conductor/elements/end_points/phi` changed
  from `toroidal_angle_of_measurement_position` to
  `toroidal_angle_of_coil_conductor_element`, preserving the coil-conductor
  owner stated by the DD path.

One independent post-apply assertion initially counted a historical terminal
edge as live for
`dd:reflectometer_profile/channel/line_of_sight_emission/first_point/z`.
Inspection showed the extra target was
`vertical_coordinate_of_diagnostic_component_centre` with
`name_stage=superseded` and `validation_status=quarantined`. The governing
instrument, adjudication, semantic-source invariant, and replay check all
exclude terminal targets. The corrected read-only proof used that same
production predicate and found zero selected-source postcondition failures; no
second mutation attempt was made.

## Regression and durable evidence

The complete focused file passed against a disposable Neo4j instance:
**9 passed, 0 failed, 0 skipped**. The added regression constructs one safe
row and one last-binding row, then proves all four properties together:

1. the full signed authority partitions the rows deterministically;
2. the derived subset preview is `would_apply` with one admitted and zero
   refused rows;
3. apply changes only the safe row and leaves the protected row byte-identical;
4. replay is `already_applied`, `changed=0`, and byte-identical.

Durable machine-readable receipts and logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T073112500467-safe-subset-disposition-apply/live-subset-preview-receipt.json`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T073112500467-safe-subset-disposition-apply/live-subset-apply-receipt.json`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T073112500467-safe-subset-disposition-apply/live-subset-replay-receipt.json`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T073112500467-safe-subset-disposition-apply/live-subset-preview.log`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T073112500467-safe-subset-disposition-apply/live-subset-replay-proof.log`;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T073112500467-safe-subset-disposition-apply/disposable-tests.log`.

The 89 protected targets remain the only catalog-edit disposition follow-on;
they require replacement-source or lifecycle authority before their refused
rows can become executable.
