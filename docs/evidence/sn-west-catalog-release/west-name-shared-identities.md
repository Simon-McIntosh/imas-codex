# WEST batch shared identities — one quantity, or several?

provisional: true — verdicts are appended as each group is judged; the closing
pass rewrites this line and adds the result table.

53 of the 230 accepted identities in the WEST batch cohort are bound to more
than one data-dictionary source path, covering 164 of the 341 bindings. Sharing
is the normal case: one geometric or global quantity measured by many
instruments is exactly what a standard name is for. The question each group is
judged on is narrower and has one answer:

> Do all of this identity's bindings denote **the same physical quantity**
> measured at different places or by different instruments — **ONE-QUANTITY**,
> one spelling is correct — or do two or more of them denote **physically
> different quantities** — **MUST-SPLIT**, because a reader given the name
> alone would attribute a value to the wrong object?

A shared geometric identity across many diagnostics is not a defect for being
shared. What makes a group MUST-SPLIT is that the name cannot be read back onto
the right object.

## Inputs, and why no query was issued

Every group and every binding is read from
[`west-name-shared-identities.json`](west-name-shared-identities.json), which
was drawn from the live graph by the node that produced it. Each binding
carries its source path, `sn_unit`, `dd_unit`, the name's description and the
data-dictionary text, with `dd_doc_parent` supplying the parent container's
documentation wherever the leaf's own text is empty or the literal `Value`. No
database was opened for this judgement and no graph query was issued.

Two further files are read, both already in the tree:
[`west-name-audit.md`](west-name-audit.md), the every-fourth-row physical
correctness audit whose closing section points at three of these groups, and
[`west-name-cohort-remainder.json`](west-name-cohort-remainder.json), which
supplies the **sibling spellings** — what the other coordinates of the same
data-dictionary container are called — that a container-level judgement needs.
`imas_codex/standard_names/manifests/west_production_dd_paths.yaml` supplies
the batch's own migration map, which turns out to settle one group outright.

### The unit comparator was controlled before its zero was reported

Comparing `sn_unit` against `dd_unit` over all 164 bindings returns **zero
disagreements**. A zero is a claim about the instrument until the instrument is
shown to see something known present, so the identical comparator was run over
the 255 bindings of the cohort remainder, where it returns **three**:

```
DISAGREE: turn_count_of_toroidal_magnetic_field_probe | magnetics/b_field_phi_probe/turns | sn= '1' dd= ''
DISAGREE: turn_count_of_poloidal_magnetic_field_probe | magnetics/b_field_pol_probe/turns | sn= '1' dd= ''
DISAGREE: atomic_count | spectrometer_visible/channel/isotope_ratios/isotope/element/atoms_n | sn= '1' dd= ''
```

The comparator fires. The zero over the shared-identity bindings is therefore a
real absence: **no group in this record carries a unit disagreement**, and the
unit half of the narrower-failure check is reported once here rather than
repeated as an empty line under all 53 verdicts.

### What is not relitigated

Three adjudications are recorded as settled and are taken as given wherever a
group touches them: the `*_of_flux_surface` family including
`volume_of_flux_surface` and `area_of_flux_surface`; the
`back_surface_distance_of_antenna_strap` spelling; and the etendue `_detector`
spelling. Where a group falls inside one of these, the verdict says so and
stops.

