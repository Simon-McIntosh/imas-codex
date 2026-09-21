# WEST accepted names, cohort rows 205–255 — physical-correctness audit

provisional: true — verdicts are appended as they are judged; the closing pass
rewrites this line and adds the result section.

This file judges the **51 accepted name bindings at indices 205 to 255** of the
unjudged remainder cohort recorded in
[`west-name-cohort-remainder.json`](west-name-cohort-remainder.json) — from
`spectrometer_visible/channel/line_of_sight/second_point/z` to
`wall/description_2d/mobile/unit/outline/r`. It is the fifth of the remainder
blocks and is written in the shape of
[`west-name-audit.md`](west-name-audit.md), which judged 86 rows of the same
341-binding cohort, so that the two can be **summed rather than compared**.

## What is judged, and from what

Every row was drawn from the live graph when the remainder cohort was cut, so
**no graph query was issued and no database was opened here**. Each row carries
`name`, `path`, `sn_unit`, `sn_description`, `dd_unit`, `dd_doc` and
`dd_doc_parent` — the parent container's text, used where the leaf's own
documentation is empty or the literal `Value` — and that is the whole evidence
base for the judgement.

Each name is judged on three questions:

1. Does the data-dictionary text of the path it is bound to describe the
   quantity the name claims?
2. Do `sn_unit` and `dd_unit` agree, and where they differ, which side is
   defensible?
3. Is the name self-descriptive to a reader who does not have the source path
   in hand?

A verdict is **correct** or **INCORRECT**. Every INCORRECT row shows the
rejected spelling beside a proposed one and states the semantic distinction in
plain language. A defect that lies in the **description** rather than in the
name is recorded as a **note** on a correct row and never as an INCORRECT
verdict; the result table counts notes separately.

### Adjudications carried over, not relitigated

The first half settled these and they are reused rather than reopened: the
`*_of_flux_surface` family including `volume_of_flux_surface` and
`area_of_flux_surface`; `back_surface_distance_of_antenna_strap`; and the
etendue `_detector` spelling. Where the first half accepted a spelling, this
half reuses it instead of minting a parallel one — the proposed
`total_coupled_power_due_to_lower_hybrid_heating` at row 238 is built from the
first half's proposed `total_coupled_power_due_to_ion_cyclotron_heating` at its
row 80, and the proposed `radial_outline_of_plasma_facing_component` at row 255
is built from the accepted `vertical_outline_of_plasma_facing_component` at its
row 86.

### Collisions are named here and adjudicated elsewhere

Twenty of these 51 rows carry an identity that is also bound to source paths
**outside** indices 205–255. Each such row says so and defers the collision
itself to the node that owns the whole-cohort collision sweep. Two collisions
lie **wholly inside** this range — `vertical_coordinate_of_strike_point` across
rows 213 and 214, and `launched_power_of_lower_hybrid_antenna` across rows 236
and 238 — and those are judged here, because both halves are in hand.

## Verdicts
