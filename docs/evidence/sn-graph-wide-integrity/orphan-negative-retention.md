# Reclassification of the overlooked unsourced identities

## Outcome

The three identities previously classified as `NO-MEASURING-PATH` are all
reclassified as **RECOVERABLE**. Each has an exact DD 4 field whose quantity
and unit agree with the Standard Name. Therefore **zero of the original 36
unsourced identities is a delete candidate on no-measuring-path grounds**.

This record is read-only adjudication evidence. It does not attach a source,
delete or rename a name, accept an identity, sign a manifest, or mutate the
graph. Attachment belongs to a separately authorized exact workflow with its
ordinary semantic, unit, closure, and collateral guards.

The prior negative classification came from treating the first eight directly
ranked paths as the whole search result. Those paths missed the measuring
fields, while the same responses' semantic-cluster results already named the
correct families. Following the cluster evidence to concrete DD paths reverses
all three classifications. The failure was top-eight path under-coverage, not
absence from the Data Dictionary.

## Per-identity reclassification

| Row | Standard name | Previous class | Verdict | Recovering DD path | Unit evidence | Pointing cluster | Original query whose top-eight path rank missed the field |
|---:|---|---|---|---|---|---|---|
| 1 | `magnetic_field_at_pedestal_top_low_field_side_magnitude` | `NO-MEASURING-PATH` | **RECOVERABLE** | `summary/pedestal_fits/mtanh/b_field_pedestal_top_lfs/value` | name `T`; DD `T` — agreement | `Pedestal Top Magnetic Field Fits` | `magnetic field magnitude at the low-field-side top of the plasma pedestal` |
| 2 | `tendency_of_total_thermal_plasma_internal_energy` | `NO-MEASURING-PATH` | **RECOVERABLE** | `summary/global_quantities/denergy_thermal_dt/value` | name `W`; DD `W` — agreement | `Plasma Diamagnetic and Thermal Energy Content` | `time derivative of volume-integrated total thermal plasma energy` |
| 3 | `toroidal_neutral_state_momentum_diffusivity` | `NO-MEASURING-PATH` | **RECOVERABLE** | `plasma_transport/model/ggd/neutral/state/momentum/d/phi` | name `m^2.s^-1`; DD `m^2.s^-1` — agreement | `Toroidal Momentum Diffusivity GGD` | `toroidal momentum diffusivity for a resolved neutral state` |

The first identity also has the equivalent linear-fit field
`summary/pedestal_fits/linear/b_field_pedestal_top_lfs/value`, and the third
has the corresponding one-dimensional-profile field
`plasma_transport/model/profiles_1d/neutral/state/momentum/d/phi`. These
siblings reinforce the reclassification; the table retains one exact
recovering path per identity so the three-row accounting remains unambiguous.

## Method and quantitative closure

The original queries were independently re-run with
`search_dd_paths(dd_version=4, k=8)`. Their first eight path results did not
contain the exact fields, but their cluster results pointed to the three named
families above. Focused resolution within those families returned the concrete
paths and DD units recorded in the table. No result was promoted directly to a
graph attachment.

The following validation command counts only the three adjudication rows,
checks identity uniqueness, and requires every row to carry one named
recovering path:

```sh
awk -F'|' '
/^\| [1-3] \|/ {
  rows++
  identity = $3
  path = $6
  gsub(/^ +| +$/, "", identity)
  gsub(/^ +| +$/, "", path)
  identities[identity] = 1
  if (path ~ /^`[^`]+`$/) recovering_paths++
}
END {
  for (identity in identities) unique++
  printf "rows=%d\nunique=%d\nrecovering_paths=%d\n", rows, unique, recovering_paths
  if (rows == 3 && unique == 3 && recovering_paths == 3) print "PASS"
  else exit 1
}' docs/evidence/sn-graph-wide-integrity/orphan-negative-retention.md
```

Recorded output:

```text
rows=3
unique=3
recovering_paths=3
PASS
```

No graph mutation, attachment, deletion, provider call, or pipeline operation
was performed.
