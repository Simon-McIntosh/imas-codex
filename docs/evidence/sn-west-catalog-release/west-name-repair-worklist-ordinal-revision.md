# The WEST repair worklist under the no-ordinal convention

provisional: true — rows are being appended as they are classified; the result
table is not yet closed.

**The project lead's convention: ordinal and index dimensions are not carried in
standard names.** Inner and outer, main and secondary, primary, first and
second, upper and lower — *where they enumerate members of an array rather than
name a physical distinction* — are indices in the data structure, not name
segments.

This file re-reads the 61 repairs in
[the deduplicated repair worklist](west-name-repair-worklist) under that
convention and classifies each one **STANDS**, **WITHDRAWN** or
**RECLASSIFIED**. It judges no name afresh against the data dictionary and adds
no defect the records do not contain: it applies one convention to verdicts
that already exist and reports what the convention changes.

**The worklist and the six records it aggregates are not edited.** A peer
session has adopted the worklist as an independent test fixture for a review
instrument it is building, and amending it would destroy that independence.
This file is the revision; it links to the worklist rather than replacing it.

## What the convention decides, and in which direction

Three consequences, and the third is the one that is easy to lose.

1. **A split whose two halves differ only by such a qualifier is invalid.** The
   shared identity binding both members was the convention working rather than
   an ambiguity, so the repair is **WITHDRAWN**.
2. **A rename that ADDS such a qualifier is invalid for the same reason**, and
   is **WITHDRAWN**.
3. **Where the generic spelling is correct and a SIBLING binding carries the
   ordinal, the defect is real but inverted.** The repair is not to add the
   qualifier to the generic name; it is to *strip* it from the sibling and bind
   that sibling to the generic identity — which frequently already exists and is
   accepted. Such a row is **RECLASSIFIED**, and it names the surviving generic
   identity and states, from a graph lookup rather than from assumption, whether
   that identity exists.

A fourth thing survives a withdrawal and sits on a different axis. Where the
ruling makes a shared identity correct, its **description** may still name one
ordinal member as though the identity meant only that one. That is a
description repair, never a rename, and every WITHDRAWN row is checked for it
below.

## How the 61 were scanned, and why the scan is trustworthy

The rejected and proposed spellings of all 61 repairs were parsed out of the
worklist's two tables mechanically — 51 rename rows from part one and 10 split
sections from part two, each with its per-binding resolutions — and every
segment of every proposed spelling was tested against the token set `inner`,
`outer`, `main`, `secondary`, `primary`, `first`, `second`, `third`, `upper`,
`lower`, `innermost`, `outermost`.

The scan reports **six proposals carrying an ordinal token**, and the six are
listed here in full because the classification of the other 55 rests on the
claim that they carry none:

| proposal | token | verdict on the token |
| --- | --- | --- |
| `radial_coordinate_of_inner_strike_point` | inner | an array member — the convention applies |
| `radial_coordinate_of_outer_strike_point` | outer | an array member — the convention applies |
| `vertical_coordinate_of_inner_strike_point` | inner | an array member — the convention applies |
| `vertical_coordinate_of_outer_strike_point` | outer | an array member — the convention applies |
| `radial_coordinate_of_primary_x_point` | primary | an array member — the convention applies |
| `radiative_temperature_at_innermost_ece_channel` | innermost | a physical criterion, not an index — see its row |
| `radial_separation_of_inner_and_outer_separatrices_at_outboard_midplane` | inner, outer | names both, selects neither — see its row |
| `upper_bound_photon_energy` | upper | a band boundary, not an index — see its row |
| `total_launched_power_of_lower_hybrid_antennas` | lower | the wave mode, not an ordinal |
| `launched_power_of_lower_hybrid_antenna` | lower | the wave mode, not an ordinal |

The scan is not a blind instrument: it returns hits, it returns them on the
rows that were expected to carry them, and it separately finds the token set in
**one accepted cohort name** that no repair row proposes —
`vertical_coordinate_of_primary_x_point` — which is the sibling that makes the
RECLASSIFIED row below possible. An instrument that found nothing anywhere
would be reporting its own failure.

## Verdicts

### WITHDRAWN — the two strike-point splits

These are split rows 9 and 10 of the worklist's part two. Their two halves
differ by nothing but `inner` against `outer`, and `summary/boundary` holds
`strike_point_inner_r` and `strike_point_outer_r` as two members of one
conceptual pair. Under the convention the shared identity was correct.

#### 1. `radial_coordinate_of_strike_point` — **WITHDRAWN**

- worklist row: part two, shared group 42; rejected by audit row 72
  (`summary/boundary/strike_point_outer_r/value`) and by the
  spectrometer-visible→wall record row 212
  (`summary/boundary/strike_point_inner_r/value`)
- proposed split: `radial_coordinate_of_inner_strike_point` /
  `radial_coordinate_of_outer_strike_point`
- why withdrawn: the two proposed halves are identical but for `inner` and
  `outer`, which is the signature the convention names. Both records argue the
  split from the physics of the two legs — different heat flux, different
  geometry, different control significance — and that argument is not wrong
  about the divertor; it is the wrong conclusion about the *name*. The legs are
  two members of the strike-point array, and a catalog that carries the leg in
  the name carries the index in the name.
- **description check: clean.** The identity's description reads "Major-radius
  location of **an individual** magnetic strike point where **a** separatrix leg
  intersects **a** divertor target, expressed in the right-handed cylindrical
  (R, φ, Z) frame." It covers both legs without selecting either. This is the
  description the convention requires, and it is the model against which its own
  vertical twin fails.

#### 2. `vertical_coordinate_of_strike_point` — **WITHDRAWN**, and it carries the surviving defect

- worklist row: part two, shared group 51; rejected by the
  spectrometer-visible→wall record rows 213 and 214, both halves inside one
  range
- proposed split: `vertical_coordinate_of_inner_strike_point` /
  `vertical_coordinate_of_outer_strike_point`
- why withdrawn: the Z axis of the same pair, withdrawn for the same reason.
- **description check: FAILS, and this is the model for the class.** The
  identity's description reads:

  > "Signed vertical (Z) coordinate of the **inner** divertor strike point,
  > where the **inner** separatrix leg intersects the divertor target in the
  > right-handed cylindrical (R, φ, Z) frame."

  The identity is bound to `strike_point_inner_z` and `strike_point_outer_z`.
  So for the outer binding the description is not merely incomplete — it is
  **affirmatively false**: a reader who resolves the published entry for the
  outer strike point is told it is the inner leg. The shared-identities record
  calls this "the only group of the 53 where the description asserts the wrong
  member rather than an incomplete set."
- **the remedy is a description repair, not a rename.** The wall record's row
  214 concludes the opposite — "The defect is in the name, not the description:
  one identity cannot carry a description that is true of only one of the two
  paths it is bound to, so the split is the only remedy." Under the convention
  that inference runs backwards. The identity *may* cover both paths; what it
  may not do is describe one of them. Its own radial twin, one row above, shows
  the repair already written: replace the two occurrences of "inner" with the
  indefinite article, and the description covers both legs without selecting
  one, exactly as `radial_coordinate_of_strike_point` does today.
