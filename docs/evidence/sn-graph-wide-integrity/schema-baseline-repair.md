# Standard Name graph schema baseline repair

Date: 2026-08-18

## Outcome

The credentialed live-graph schema suite now executes all nine tests with zero
skips and passes all nine. The repair reduced every measured baseline defect to
zero without provider calls.

| Defect | Before | After | Disposition |
|---|---:|---:|---|
| Retry nodes carrying undeclared terminal-recovery metadata | 9 | 0 | Declared six optional fields in LinkML; the nine receipts and their values remain intact. |
| `StandardNameSource.source_id IS NULL` | 27 | 0 | Filled the required bare DD identity through an exact compare-and-set manifest. |
| `Unit.symbol IS NULL` | 1 | 0 | Deleted the isolated non-unit node `as_parent_level_2`; it had no relationships and only an `id` property. |

The exact live mutation manifest SHA-256 is
`fa3167afa3007446c94e56197dfe2640d62a9cebd4f6829991d51688389064ed`.
It changed 27 required source fields and deleted one isolated invalid Unit. The
sorted changed-source-id SHA-256 is
`62260f34d2637efb6a91cf93e3d4eba3d1aaaaddd21cb83a2e941fd4646829a8`.

## Root-cause adjudication

The six properties on `StandardNameSourceRetry` are durable authority, not
accidental writer residue:

- `before_closure_hash` and `preserved_state_hash` prove the exact terminal
  source/DD/name closure and the state that must survive recovery;
- `manifest_hash` binds the event to the reviewed input;
- `run_id` groups the atomic invocation;
- `terminal_sn_id` and `terminal_stage` identify the terminal target and the
  lifecycle boundary at which its source was released.

`attachment_audit.py` reads these values during replay and preserved-state
verification. Stopping the writer would erase the evidence needed to prove an
event already current. The correct repair is therefore **declare in LinkML**.
The fields are optional because ordinary retry events do not represent a
terminal-attachment recovery.

All 27 null source identities were DD sources whose node identity already had
the required canonical form `dd:{bare-path}`. Eleven were live sources (six
`attached`, five `composed`) with matching `FROM_DD_PATH` authority; sixteen
were `stale` fixture residues without a backing edge. The exact repair derived
only the bare suffix already asserted by each stable node id. Representative
live bindings include:

- `dd:ece/channel/t_e_voltage` → source id `ece/channel/t_e_voltage` →
  `voltage_of_diagnostic_antenna`;
- `dd:equilibrium/time_slice/constraints/flux_loop/measured` → source id
  `equilibrium/time_slice/constraints/flux_loop/measured` →
  `poloidal_magnetic_flux_of_flux_loop`;
- `dd:spectrometer_x_ray_crystal/channel/energy_bound_lower` → source id
  `spectrometer_x_ray_crystal/channel/energy_bound_lower` →
  `lower_bound_energy_of_detector_pixel`.

The null-symbol Unit was not repaired by copying its id into `symbol`:
`as_parent_level_2` is a grammar-like token, not a physical unit, and the node
had zero incident relationships. Deleting that isolated invalid classification
preserves unit semantics instead of making the schema superficially green.

## Evidence

- Before suite:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T200931345566-sgwi-schema-baseline/schema-before-executed.log`
  — 9 executed, 0 skipped, 7 passed, 2 failed; SHA-256
  `304a17b397545e185fa5f28c3a15a50a12ae007ab29f66302e664ec7ef59c732`.
- Before census:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T200931345566-sgwi-schema-baseline/live-before.json`
  — exact 9/27/1 rows; SHA-256
  `629e24ad7812be038c82aa377628f809a1fa788547594173919dbbebea682ceb`.
- Apply receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T200931345566-sgwi-schema-baseline/live-apply-receipt.json`
  — exact manifest, before/after census, and 27-plus-1 cardinality proof;
  SHA-256
  `1cae615e8ccd7440d322b0e2280dbbfe7064f4af1bef530f13acec887c091c41`.
- After suite:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T200931345566-sgwi-schema-baseline/schema-after.log`
  — 9 executed, 0 skipped, 9 passed, 0 failed; SHA-256
  `7c7156eb19764b67943b7f74f78b7a81b506d36afb7ce4a41504249e259eaa5b`.

Provider calls: 0.
