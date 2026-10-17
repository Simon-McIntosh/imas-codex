#!/usr/bin/env python3
"""Build source-anchored documentation claims from the acceptable judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = (
    ROOT
    / "docs/evidence/fragments/sn-judgment-evaluation/golden-documentation-acceptable.json"
)
OUTPUT = Path(__file__).with_name("golden-documentation-planted.json")
SEED = 17341

# Each row is a human-authored true claim, a single contradictory claim, and
# the exact DD wording that licenses the comparison. The false claim is appended
# as one sentence; no source document or model output is changed in place.
CLAIMS = """
12|locus|Upper inner squareness|The squareness describes the upper inner plasma boundary.|The squareness describes the lower inner plasma boundary.
19|relation|k_z / norm(k)|The normalized vertical component is k_z divided by the norm of k.|The normalized vertical component is the norm of k divided by k_z.
24|locus|horizontal, vertical|The detector extent includes horizontal and vertical camera dimensions.|The detector extent includes only the horizontal camera dimension.
26|relation|Total number of electrons|The quantity counts electrons in the plasma.|The quantity counts ions in the plasma instead of electrons.
30|relation|Nuclear charge|Atomic number denotes nuclear charge.|Atomic number denotes nuclear mass instead of nuclear charge.
34|direction|Component along Y axis|The image-up vector component is along the Y axis.|The image-up vector component is along the X axis.
35|direction|launched into the vacuum vessel|The beam power is launched into the vacuum vessel.|The beam power is launched out of the vacuum vessel.
38|locus|Resistance of the block element|This is the resistance of a vessel block element.|This is the resistance of the plasma column rather than a vessel block element.
41|direction|major radius axis|The force component is along the major-radius axis.|The force component is along the height axis rather than the major-radius axis.
42|direction|anti-clockwise when viewed from above|Positive vacuum-field sign is anti-clockwise when viewed from above.|Positive vacuum-field sign is clockwise when viewed from above.
49|relation|averaged on charge states and ion species|The ion temperature is averaged over charge states and ion species.|The ion temperature is the value of one charge state and one ion species without averaging.
51|locus|sum over all targets|Conducted power is summed over all divertor targets.|Conducted power includes only one divertor target and excludes the others.
58|relation|e(X)traordinary mode|The predicted radiation temperature belongs to the extraordinary mode.|The predicted radiation temperature belongs to the ordinary mode instead of the extraordinary mode.
60|relation|Duration of a sweep|The quantity is the duration of a launched-wave sweep.|The quantity is the waiting interval between two launched-wave sweeps.
62|direction|Parallel component|The momentum-velocity component is parallel.|The momentum-velocity component is perpendicular rather than parallel.
65|relation|real particles represented by each marker|Marker weight counts real particles represented by each marker.|Marker weight counts markers represented by each real particle.
93|relation|3/2 * int(p,dV)|Plasma energy is three halves of the pressure volume integral.|Plasma energy is two thirds of the pressure volume integral.
104|direction|Normal component|The vacuum error-field component is normal to the surface.|The vacuum error-field component is tangential rather than normal to the surface.
107|direction|positive/negative for divergent/convergent|Inverse phase-ellipse curvature is positive for divergent beams and negative for convergent beams.|Inverse phase-ellipse curvature is negative for divergent beams and positive for convergent beams.
115|direction|Component along Z axis|The divertor-target coordinate is along the Z axis.|The divertor-target coordinate is along the X axis.
125|relation|Fluid temperature|This quantity is the fluid temperature at the port.|This quantity is the fluid pressure at the port instead of its temperature.
131|direction|Height|The probe coordinate measures height.|The probe coordinate measures major radius rather than height.
133|relation|Source rate of thermal particle density|The quantity is a source rate for thermal particle density.|The quantity is a sink rate that removes thermal particle density.
135|relation|int(e_field.j.dV) / Ip^2|Plasma resistance divides the electric-field current-density integral by squared plasma current.|Plasma resistance multiplies the electric-field current-density integral by squared plasma current.
146|direction|Component along Y axis|The shatter-cone major-axis component is along Y.|The shatter-cone major-axis component is along X instead of Y.
147|relation|adding this dynamic quantity to the static resistance|Effective coil resistance adds the dynamic resistance to the static resistance.|Effective coil resistance subtracts the dynamic resistance from the static resistance.
148|relation|Pellet diameter|The quantity is the pellet diameter.|The quantity is the pellet radius rather than its diameter.
158|locus|Reference major radius where the vacuum toroidal magnetic field is given|The vacuum toroidal field is specified at a reference major radius.|The vacuum toroidal field is specified at a reference vertical height instead of a major radius.
164|direction|Height|The lower-hybrid antenna coordinate is a height.|The lower-hybrid antenna coordinate is a major radius rather than a height.
166|direction|Component along X axis|The mirror vector component is along X.|The mirror vector component is along Y instead of X.
169|direction|height axis|The vorticity component is along the height axis.|The vorticity component is along the major-radius axis instead of the height axis.
176|locus|halo region boundary|The toroidal flux is evaluated at the halo-region boundary.|The toroidal flux is evaluated at the magnetic axis instead of the halo-region boundary.
187|relation|Number of turns per unit length|This quantity counts coil turns per unit length.|This quantity counts the total coil turns without dividing by length.
189|direction|behind the antenna strap|The back-surface distance reaches a conductor behind the antenna strap.|The back-surface distance reaches a conductor in front of the antenna strap.
191|relation|incident angles of this incident species|The angle concerns the species incident on the wall.|The angle concerns particles emitted from the wall rather than the incident species.
194|relation|Minimum(modulus(B))|The quantity is the minimum magnetic-field magnitude on the flux surface.|The quantity is the maximum magnetic-field magnitude on the flux surface.
214|relation|mu_0 rho*rho/1.22/eta_neo|The resistive time is inversely proportional to neo-classical resistivity.|The resistive time is directly proportional to neo-classical resistivity.
221|direction|Toroidal angle|The channel offset is a toroidal angle.|The channel offset is a poloidal angle instead of a toroidal angle.
228|direction|Component along X axis|The minor-axis vector component is along X.|The minor-axis vector component is along Y instead of X.
242|direction|X2 direction|The second local curvature radius is in the X2 direction.|The second local curvature radius is in the X1 direction instead of X2.
250|relation|divided by the length of the line-of-sight|Line-averaged density divides the line integral by the line-of-sight length.|Line-averaged density multiplies the line integral by the line-of-sight length.
251|direction|Component along X axis|The camera unit-vector component is along X.|The camera unit-vector component is along Y instead of X.
264|direction|toroidal direction|The antenna-strap width is measured in the toroidal direction.|The antenna-strap width is measured in the poloidal direction instead of the toroidal direction.
269|direction|height axis|The vorticity component is along the height axis.|The vorticity component is along the major-radius axis instead of the height axis.
273|relation|Full-width of the Intermediate Frequency|Bandwidth is the full width of the intermediate-frequency bandpass.|Bandwidth is the center frequency of the intermediate-frequency bandpass rather than its full width.
276|direction|Radial profile|The emissivity is a radial profile.|The emissivity is a toroidal profile instead of a radial profile.
278|relation|Length measured by a displacement sensor|The displacement sensor measures a length.|The displacement sensor measures a velocity rather than a length.
282|direction|Y coordinates|The centroid coordinates are Y coordinates.|The centroid coordinates are X coordinates instead of Y coordinates.
287|direction|Component along Z axis|The camera unit-vector component is along Z.|The camera unit-vector component is along X instead of Z.
291|relation|o_mode_fraction = 1.0|All power in ordinary mode gives an ordinary-mode fraction of one.|All power in ordinary mode gives an ordinary-mode fraction of zero.
296|relation|cumulative volume integral of the absorbed power density|Power inside the flux surface is a cumulative volume integral of absorbed power density.|Power inside the flux surface is only the local absorbed power density at that surface.
311|relation|sum of pressure_parallel and pressure_perpendicular|Normalized pressure sums parallel and perpendicular pressure.|Normalized pressure subtracts perpendicular pressure from parallel pressure.
336|direction|counter-clockwise direction|The positive polarizer angle turns counter-clockwise.|The positive polarizer angle turns clockwise instead of counter-clockwise.
342|relation|fast particle population|This absorbed power belongs to the fast particle population.|This absorbed power belongs to the thermal electron population instead of the fast particle population.
354|relation|linear relation between magnetic field and probe coil current|Calibration assumes a linear field-current relation.|Calibration assumes a quadratic field-current relation instead of a linear one.
358|relation|ratio between the poloidal field at the midplane separatrix and the poloidal field at the strike-point|Flux expansion is midplane-separatrix poloidal field divided by strike-point poloidal field.|Flux expansion is strike-point poloidal field divided by midplane-separatrix poloidal field.
364|direction|counter-clockwise when viewing from above|The toroidal angle grows counter-clockwise when viewed from above.|The toroidal angle grows clockwise when viewed from above.
369|locus|entrance pupil|The aperture diameter is at the entrance pupil.|The aperture diameter is at the exit pupil instead of the entrance pupil.
372|relation|Total absorbed wave power|This is total absorbed wave power.|This is total launched wave power before absorption.
380|relation|Effective diffusivity|This quantity is an effective diffusivity.|This quantity is a convective velocity rather than an effective diffusivity.
386|relation|tritium atoms created for each fusion neutron|Breeding ratio counts tritium atoms created per fusion neutron.|Breeding ratio counts fusion neutrons per tritium atom created.
389|direction|coolant going through the loop|Coolant mass flow goes through the loop.|Coolant mass flow leaves the loop without passing through it.
399|relation|DT reactions|The emissivity is due to deuterium-tritium reactions.|The emissivity is due to deuterium-deuterium reactions instead of deuterium-tritium reactions.
404|locus|sum over all targets|Radiated power is summed over all divertor targets.|Radiated power includes only one divertor target and excludes the others.
405|direction|Previous sawtooth period|The quantity is the previous sawtooth period.|The quantity is the next sawtooth period rather than the previous one.
406|locus|mapped to the mid-plane|The divertor heat-decay length is mapped to the mid-plane.|The divertor heat-decay length is left at the divertor target and is not mapped to the mid-plane.
408|relation|4 int(p dV) / [R_0 * mu_0 * Ip^2]|Poloidal beta has four times the pressure integral in its numerator.|Poloidal beta has the reference radius times squared plasma current in its numerator and the pressure integral in its denominator.
417|direction|horiontal plane|The beamlet inclination is measured from the horizontal plane.|The beamlet inclination is measured from the vertical plane instead of the horizontal plane.
426|relation|incident-reflected|Net wall radiated power is incident power minus reflected power.|Net wall radiated power is incident power plus reflected power.
429|locus|detector level|Relative humidity is measured at the detector.|Relative humidity is measured at the radiation source instead of the detector.
433|relation|events occurring in the detector from one neutron energy bin|The response counts detector events from one emitted neutron-energy bin.|The response counts emitted neutron bins per detector event.
447|relation|Probability of detection of a photon|Spectral efficiency is the probability of detecting an incident photon.|Spectral efficiency is the probability that an incident photon is not detected.
450|direction|Pumped particle flux|This is a pumped electron-equivalent particle flux.|This is an injected electron-equivalent particle flux instead of a pumped one.
467|direction|X1 direction|The aperture width is in the X1 direction.|The aperture width is in the X2 direction instead of X1.
471|relation|cumulative volume integral of the absorbed power density|Thermal-electron power inside a flux surface is a cumulative volume integral.|Thermal-electron power inside a flux surface is only the local density at that surface.
476|relation|0 is perfectly specular, 1 is perfectly rough|Roughness zero is perfectly specular and one is perfectly rough.|Roughness zero is perfectly rough and one is perfectly specular.
481|direction|counter-clockwise when viewing from above|The detector toroidal coordinate grows counter-clockwise when viewed from above.|The detector toroidal coordinate grows clockwise when viewed from above.
484|relation|upstream to downstream average|The temperature is averaged from upstream to downstream.|The temperature is only the upstream value without downstream averaging.
486|relation|Normalised temperature|The quantity is normalized temperature.|The quantity is absolute temperature without normalization.
""".strip()

OFF_SOURCE = (
    "One metre equals one thousand millimetres.",
    "One kilometre equals one thousand metres.",
    "One hour contains sixty minutes.",
    "A circle has a diameter twice its radius.",
    "A right angle measures ninety degrees.",
    "One square metre equals ten thousand square centimetres.",
    "One kilogram equals one thousand grams.",
    "One megawatt equals one million watts.",
    "One millisecond equals one thousand microseconds.",
    "A Euclidean triangle has three sides.",
    "A rectangle has four right angles.",
    "The square root of sixteen is four.",
    "One gigahertz equals one billion hertz.",
    "One kilovolt equals one thousand volts.",
    "One newton equals one kilogram metre per second squared.",
    "One joule equals one newton metre.",
    "A complete revolution measures two pi radians.",
    "One millimetre equals one thousand micrometres.",
    "One microsecond equals one millionth of a second.",
    "The area of a rectangle equals its length times its width.",
)


def stratified_draw(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Cover each populated split/domain stratum, then allocate remaining seats."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["physics_domain"])].append(row)
    if len(groups) > count:
        raise ValueError("More populated strata than draw seats")
    selected = []
    remaining = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["draw_rank"])
        rng.shuffle(group)
        selected.append(group.pop())
        remaining.extend(group)
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    return sorted(
        selected,
        key=lambda row: (row["split"], row["physics_domain"], row["draw_rank"]),
    )


def build() -> dict:
    source_bytes = SOURCE.read_bytes()
    acceptable = {
        row["draw_rank"]: row for row in json.loads(source_bytes) if row["label"] == 1
    }
    authored = {}
    for line in CLAIMS.splitlines():
        rank, rule, anchor, original, planted = line.split("|", 4)
        rank = int(rank)
        if rank in authored or rank not in acceptable:
            raise ValueError(f"Invalid authored entry: {rank}")
        if rule not in {"direction", "locus", "relation"}:
            raise ValueError(f"Invalid rule: {rule}")
        source = next(
            (
                item
                for item in acceptable[rank]["dd_sources"]
                if anchor.casefold() in item["documentation"].casefold()
            ),
            None,
        )
        if source is None:
            raise ValueError(f"DD anchor absent for {rank}: {anchor}")
        authored[rank] = (rule, original, planted, source)
    rng = random.Random(SEED)
    contradictions = stratified_draw([acceptable[rank] for rank in authored], 60, rng)
    control_pool = [
        row
        for rank, row in acceptable.items()
        if rank not in {item["draw_rank"] for item in contradictions}
        and any(
            len(source["documentation"].strip()) > 25 for source in row["dd_sources"]
        )
    ]
    controls = sorted(
        rng.sample(control_pool, 20),
        key=lambda row: (row["split"], row["physics_domain"], row["draw_rank"]),
    )
    copies = []
    for kind, selected in (("contradicts", contradictions), ("says_nothing", controls)):
        for row in selected:
            rank = row["draw_rank"]
            if kind == "contradicts":
                rule, source_faithful, planted, source = authored[rank]
            else:
                source = next(
                    source
                    for source in row["dd_sources"]
                    if len(source["documentation"].strip()) > 25
                )
                rule, source_faithful, planted = (
                    "off_source",
                    None,
                    OFF_SOURCE[len(copies) - len(contradictions)],
                )
                if planted.casefold() in source["documentation"].casefold():
                    raise ValueError(f"Control supported by source: {rank}")
            original_documentation = row["documentation"]
            original_sentence = original_documentation.split(". ", 1)[0].strip()
            if not original_sentence.endswith("."):
                original_sentence += "."
            copies.append(
                {
                    "id": f"{kind}-{rank}",
                    "source_draw_rank": rank,
                    "source_unit": row["unit"],
                    "source_evidence": row["source_evidence"],
                    "split": row["split"],
                    "physics_domain": row["physics_domain"],
                    "label": 0 if kind == "contradicts" else 1,
                    "claim_label": kind,
                    "source_label": row["label"],
                    "rule": rule,
                    "original_sentence": original_sentence,
                    "source_faithful_sentence": source_faithful,
                    "planted_sentence": planted,
                    "dd_source": source,
                    "original_documentation": original_documentation,
                    "documentation": original_documentation.rstrip() + "\n\n" + planted,
                }
            )
    return {
        "schema": "golden-documentation-planted-claims",
        "seed": SEED,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "draw": {
            "method": "one random contradiction per populated split/domain stratum, then seeded shuffle of remaining seats; seeded sample of source-bearing silent controls",
            "eligible_contradictions": len(authored),
            "contradiction_ranks": [row["draw_rank"] for row in contradictions],
            "silent_control_ranks": [row["draw_rank"] for row in controls],
        },
        "counts": {
            "claim_label": dict(
                sorted(Counter(row["claim_label"] for row in copies).items())
            ),
            "entry_label": dict(
                sorted(Counter(row["label"] for row in copies).items())
            ),
            "rule": dict(sorted(Counter(row["rule"] for row in copies).items())),
            "split": dict(sorted(Counter(row["split"] for row in copies).items())),
            "domain": dict(
                sorted(Counter(row["physics_domain"] for row in copies).items())
            ),
        },
        "copies": copies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="refuse when the committed fixture differs"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="fixture path to write or check"
    )
    args = parser.parse_args()
    rendered = (
        json.dumps(build(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    if args.check:
        if args.output.read_text() != rendered:
            raise SystemExit("Generated fixture differs")
        print(
            f"MATCH bytes={len(rendered.encode())} sha256={hashlib.sha256(rendered.encode()).hexdigest()}"
        )
    else:
        args.output.write_text(rendered)
        print(f"WROTE {args.output} bytes={len(rendered.encode())}")


if __name__ == "__main__":
    main()
