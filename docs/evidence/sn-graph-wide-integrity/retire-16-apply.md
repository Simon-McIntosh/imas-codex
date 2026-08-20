NEEDS-HELP: the signed lifecycle operator refused all 16 production targets because 20 live bindings from 19 sources remain, so the fenced node cannot produce the authorized retirement delta.

# Signed non-structural retirement apply

## Outcome

**Blocked before mutation.** One production invocation loaded the committed
orphan-disposition authority, verified both its file and canonical payload
digests, derived the exact 16-target retirement cohort, and regenerated the
live retirement manifest. The operator returned `refused` with `changed=0`:
all 16 targets still have at least one live `PRODUCED_NAME` source.

No target lifecycle, source binding, source scalar, projection, or
`StandardNameChange` row was written by this attempt. The operator stopped at
its first live-authority gate, before the required `7496 -> 7512` ledger delta,
collateral baseline, apply, replay, or post-apply census could execute.

## Tried

- Read live plan version 191 and confirmed the sole prerequisite is shipped and
  this serialized node is the active `remaining-work` execution.
- Verified the authority file SHA-256 as
  `2c2d38f3241ec3057d24a5d05c27840f5e4ffe99520063059ab31c1e9d4bca36`.
- Verified the signed canonical authority SHA-256 as
  `4bac6110486390e95c1cab9620c4723df96fe6f2190b85e6496464c77fbba873`.
- Reproduced the declared cohort exactly: 16 targets, partitioned as 13
  accepted and 3 reviewed.
- Invoked `retire_signed_provenance_orphans` once in preview mode through the
  same production harness intended to continue directly into canonical-digest
  comparison, apply, postflight, and replay.

The regenerated manifest SHA-256 is
`5d367a6bb2ef9437c46061a239eaebe72a46e154fd660910d90e9f543fff4122`.
Its exact outcome is 16 requested, 0 admitted, 16 refused. Every refusal says
`name has a live producing source`.

## Observable blocker

The 16 lifecycle targets retain 20 live incoming bindings from 19 unique
sources. Fourteen targets retain one binding each; two retain three:

- `radial_total_thermal_electron_energy_flux` is still produced by the
  electron, ion, and neutral plasma-transport energy-flux sources;
- `surface_roughness_of_optical_element` is still produced by the infrared
  camera, visible camera, and visible spectrometer optical-element sources.

Representative one-source refusals include:

- `beam_area_of_neutral_beam_injector` from
  `dd:nbi/unit/source/surface`;
- `bremsstrahlung_count` and
  `time_derivative_of_bremsstrahlung_count_at_detector_pixel`, both from
  `dd:bremsstrahlung_visible/channel/intensity`;
- `rotation_frequency` from `dd:ntms/time_slice/mode/dphase_dt`;
- `poloidal_current_density` from
  `dd:plasma_profiles/ggd/j_total/poloidal`.

This is the operator's intended safety behavior. Its disposable-graph proof
requires the targets to be unsourced before lifecycle supersession, and its
production code refuses a target whenever any non-stale producer remains. The
preceding signed structural release removed bindings only for targets admitted
as structural identities; it deliberately excluded the non-structural retire
set. Therefore the live graph does not satisfy the proven operator's input
contract.

## Options

1. Add and independently prove a single signed transaction that releases the
   exact 20 bindings and supersedes the 16 targets atomically, with both the
   source-disposition authority and orphan-lifecycle authority hash-pinned.
2. Add and prove a signed binding-release operator for the retire disposition,
   apply that exact source cohort first, then immediately run the existing
   lifecycle operator. This creates an observable intermediate unsourced state
   and therefore needs its own rollback/recovery contract.
3. Change the signed source-disposition instrument so a retire disposition can
   authorize final-source release only when the same transaction also performs
   the ledgered lifecycle supersession. This is mechanically equivalent to the
   first option but may reuse more of the existing disposition machinery.

## Leaning

Use option 1: one atomic, dual-authority transaction. It preserves the current
fail-closed rule that a live name cannot lose its final source unless its
authorized lifecycle transition commits in the same transaction. It also
avoids the intermediate state and recovery burden of a two-operator sequence.

## Cost if wrong

Choosing a two-step release where atomic retirement is required would demand a
new recovery instrument for any failure between binding removal and lifecycle
supersession, a fresh signed manifest, and another production apply. Choosing
an overly broad combined transaction could remove source authority from
retain, re-source, or structural dispositions; all 89 signed targets and their
complete producer/child closures would then need re-audit before any retry.

## Durable artifacts

- production preflight log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T123938946491-sgwi-retire-16-apply/production-signed-retirement.log`;
- complete regenerated refusal receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T123938946491-sgwi-retire-16-apply/retirement-preview.json`;
- fail-closed harness:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T123938946491-sgwi-retire-16-apply/run_signed_retirement.py`.

The requested completion measures remain unmet by design: superseded targets
0/16; ledger rows 0/16; replay not reached; collateral closure comparison not
reached; post-apply structural-versus-genuine unsourced census not reached.
