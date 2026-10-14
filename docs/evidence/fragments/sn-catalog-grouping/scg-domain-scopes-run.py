#!/usr/bin/env python3
"""Adjudicate the confused domain pairs and relabel the measured DD paths by rule.

Reads the path measure's saved answers (dated 2026-10-14T19:11 logic, recorded
in ``scg-domain-path-measure-results.json``) and, for every pair the measure
flagged as confused, records one of two verdicts:

- ``defect`` — the two scopes claim the same quantity. One scope is rewritten in
  ``imas_codex/definitions/physics/domains.yaml`` so the two exclude each other,
  via an ``excludes`` entry naming the neighbour.
- ``overlap`` — the quantity genuinely belongs to both. Precedence settles it,
  and the guidance carries an ``excludes`` entry naming precedence as the reason.

The path-level label is then taken from the revised scopes plus precedence —
never from a classifier's answer, which would be circular. Each label that
changes is listed with the rule that changed it.

The re-score compares the tree and chat classifiers' *saved* answers against the
new labels. A saved answer predating the revised guidance is scored as recorded;
the next classifier run under the new guidance is what measures the repair.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GUIDANCE = REPO / "imas_codex/definitions/physics/domains.yaml"

# ---------------------------------------------------------------------------
# Per-pair adjudication.
#
# Key: (label, predicted) exactly as the measure names the pair. ``label`` is the
# domain the path's owner name inherited; ``predicted`` is one classifier's
# answer. Value: (verdict, winner, rule).
#
# ``winner`` is a domain string applied to the pair's paths, or a list of
# ``(glob, domain)`` splits when one pair holds paths on both sides (an antenna
# quantity and a wave quantity are the same pair). The winner is decided from the
# scopes and excludes text, not from the classifier's answer.
# ---------------------------------------------------------------------------

PAIRS: dict[tuple[str, str], tuple[str, object, str]] = {
    ("plasma_measurement_diagnostics", "radiation_measurement_diagnostics"): (
        "defect",
        "radiation_measurement_diagnostics",
        "A bolometer's channel geometry and sightline is radiation_measurement_diagnostics, "
        "the family that owns bolometry; plasma_measurement_diagnostics holds only an "
        "instrument no specific family owns.",
    ),
    ("auxiliary_heating", "waves"): (
        "defect",
        [("waves/*", "waves"), ("*", "auxiliary_heating")],
        "An antenna or launcher and the refractive index it launches are auxiliary_heating; "
        "a wave field inside the plasma and its absorbed power density are waves.",
    ),
    ("equilibrium", "core_plasma_physics"): (
        "defect",
        [
            ("core_profiles/global_quantities/ip", "core_plasma_physics"),
            ("*", "equilibrium"),
        ],
        "The plasma current is a global state of the plasma, core_plasma_physics; a "
        "profile or constraint that states the field's shape is equilibrium.",
    ),
    ("radiation_measurement_diagnostics", "spectroscopy"): (
        "defect",
        [
            ("spectrometer_*", "spectroscopy"),
            ("charge_exchange/*", "spectroscopy"),
            ("mse/*", "spectroscopy"),
            ("*", "radiation_measurement_diagnostics"),
        ],
        "A spectrometer's line-of-sight emission resolved in wavelength is spectroscopy; "
        "a bolometric power, a bremsstrahlung filter band or a camera image is "
        "radiation_measurement_diagnostics.",
    ),
    ("core_plasma_physics", "edge_plasma_physics"): (
        "defect",
        "edge_plasma_physics",
        "The same quantity resolved in the scrape-off layer, the pedestal or an edge "
        "profile is edge_plasma_physics; only the core resolution is core_plasma_physics.",
    ),
    ("general", "computational_workflow"): (
        "defect",
        "computational_workflow",
        "Atomic and molecular reference data is computational_workflow; general is for a "
        "physical quantity with no single subject, not for a reference-data service.",
    ),
    ("general", "core_plasma_physics"): (
        "defect",
        "core_plasma_physics",
        "A quantity whose subject is knowable takes the subject's own domain; general is a "
        "last resort, not a default for a core-plasma quantity.",
    ),
    ("core_plasma_physics", "spectroscopy"): (
        "defect",
        "spectroscopy",
        "A charge-exchange channel's ion concentration and charge state are spectroscopy; "
        "the charge_exchange IDS is a spectroscopic diagnostic.",
    ),
    ("core_plasma_physics", "transport"): (
        "defect",
        [
            ("bolometer/*", "radiation_measurement_diagnostics"),
            ("summary/global_quantities/power_radiated/*", "core_plasma_physics"),
            ("*", "core_plasma_physics"),
        ],
        "Radiated power measured by bolometry is radiation_measurement_diagnostics; a "
        "global energy-balance scalar is core_plasma_physics; a flux or diffusivity is "
        "transport.",
    ),
    ("edge_plasma_physics", "spectroscopy"): (
        "defect",
        "spectroscopy",
        "A visible spectrometer's isotope density ratio and hot-neutral velocity are "
        "spectroscopy; the emissive channel is a spectroscopic diagnostic, whatever plasma "
        "region it observes.",
    ),
    ("neutronics", "plasma_wall_interactions"): (
        "defect",
        "neutronics",
        "Nuclear power deposited in a wall unit is neutronics; the wall-surface erosion and "
        "retention that plasma_wall_interactions owns is a different quantity.",
    ),
    ("neutronics", "structural_components"): (
        "defect",
        "neutronics",
        "The neutron power and escaping flux in a breeding blanket are neutronics; the "
        "blanket's structure is structural_components.",
    ),
    ("structural_components", "equilibrium"): (
        "defect",
        "magnetic_field_systems",
        "A ferritic field-shaping object is magnetic_field_systems; equilibrium owns the "
        "field and its surfaces, and structural_components the load-bearing structure.",
    ),
    ("auxiliary_heating", "electromagnetic_wave_diagnostics"): (
        "defect",
        "auxiliary_heating",
        "An antenna module's mechanical position is auxiliary_heating; "
        "electromagnetic_wave_diagnostics owns a probe that transmits a wave and measures "
        "its phase, not the launcher's own hardware.",
    ),
    ("current_drive", "core_plasma_physics"): (
        "defect",
        "current_drive",
        "The part of the current driven by fusion reactions and its efficiency is "
        "current_drive, even when stored beside the plasma current in a summary product.",
    ),
    ("edge_plasma_physics", "core_plasma_physics"): (
        "defect",
        "edge_plasma_physics",
        "A quantity in the SOL or pedestal region is edge_plasma_physics; only a core "
        "resolution is core_plasma_physics.",
    ),
    ("electromagnetic_wave_diagnostics", "data_management"): (
        "defect",
        "equilibrium",
        "A measured or reconstructed line-density constraint on the equilibrium is "
        "equilibrium; it is an equilibrium constraint, not a wave diagnostic or an "
        "assembled data product.",
    ),
    ("electromagnetic_wave_diagnostics", "radiation_measurement_diagnostics"): (
        "defect",
        "radiation_measurement_diagnostics",
        "Electron cyclotron emission is radiated power the plasma emits, so its position "
        "correction is radiation_measurement_diagnostics; a probe that transmits a wave is "
        "electromagnetic_wave_diagnostics.",
    ),
    ("equilibrium", "edge_plasma_physics"): (
        "defect",
        "edge_plasma_physics",
        "A fitted pedestal position is a property of the pedestal region, so it is "
        "edge_plasma_physics; the flux-surface geometry it is expressed in stays "
        "equilibrium.",
    ),
    ("equilibrium", "electromagnetic_wave_diagnostics"): (
        "defect",
        "equilibrium",
        "A normalised flux coordinate along a ray path is a flux-surface mapping, so it is "
        "equilibrium; a probe that transmits a wave is electromagnetic_wave_diagnostics.",
    ),
    ("equilibrium", "magnetic_field_systems"): (
        "overlap",
        "equilibrium",
        "A reference major radius defining B0 is both a coordinate of the equilibrium "
        "coordinate system and a property of the toroidal field; precedence puts "
        "equilibrium first and settles the overlap.",
    ),
    ("equilibrium", "plasma_control"): (
        "defect",
        "plasma_control",
        "A gap-direction angle used as a position-control reference is plasma_control; the "
        "achieved equilibrium it targets is equilibrium.",
    ),
    ("equilibrium", "waves"): (
        "overlap",
        "equilibrium",
        "A normalised flux coordinate along a wave-code ray is both a flux-surface mapping "
        "and a quantity the wave code computes; precedence puts equilibrium first and "
        "settles the overlap.",
    ),
    ("fast_particles", "core_plasma_physics"): (
        "defect",
        "fast_particles",
        "Orbit frequency of a suprathermal distribution is fast_particles; the thermal bulk "
        "is core_plasma_physics.",
    ),
    ("fast_particles", "transport"): (
        "defect",
        "fast_particles",
        "A source term with a distribution source in the distribution_sources IDS is "
        "fast_particles; a flux or diffusivity is transport.",
    ),
    ("fueling", "core_plasma_physics"): (
        "defect",
        "fueling",
        "A pellet's own atom inventory is fueling; the plasma density it produces is "
        "core_plasma_physics.",
    ),
    ("magnetic_field_diagnostics", "equilibrium"): (
        "defect",
        "equilibrium",
        "The outline geometry of a flux-loop reconstruction is equilibrium; the coils' "
        "signals are magnetic_field_diagnostics.",
    ),
    ("magnetic_field_systems", "computational_workflow"): (
        "defect",
        "magnetic_field_systems",
        "A passive loop element's geometry is magnetic_field_systems; it is a machine "
        "component, not a computational geometry definition.",
    ),
    ("magnetohydrodynamics", "gyrokinetics"): (
        "defect",
        "magnetohydrodynamics",
        "A shear Alfven continuum in a linear MHD stability analysis is magnetohydrodynamics; "
        "a gyrokinetic eigenmode is gyrokinetics.",
    ),
    ("mechanical_measurement_diagnostics", "machine_operations"): (
        "defect",
        "mechanical_measurement_diagnostics",
        "A sensor attachment-point coordinate is the mechanical state of a component, so it "
        "is mechanical_measurement_diagnostics; machine_operations owns scheduling, not "
        "hardware geometry.",
    ),
    ("neutronics", "core_plasma_physics"): (
        "defect",
        "neutronics",
        "The neutron emission rate and its measurement are neutronics; the plasma's energy "
        "state is core_plasma_physics.",
    ),
    ("neutronics", "magnetic_field_diagnostics"): (
        "defect",
        "spectroscopy",
        "An MSE channel's detector geometry is spectroscopy; the motional Stark effect on "
        "the emitted light is a spectroscopic measurement, not neutronics or a magnetic "
        "sensor.",
    ),
    ("neutronics", "particle_measurement_diagnostics"): (
        "defect",
        "neutronics",
        "A neutron detector's aperture geometry is neutronics; particle_measurement_diagnostics "
        "owns a scattering or particle-counting instrument.",
    ),
    ("neutronics", "spectroscopy"): (
        "defect",
        "spectroscopy",
        "An MSE channel's detector geometry is spectroscopy (mse/*); neutronics owns a "
        "neutron detection or activation quantity.",
    ),
    ("particle_measurement_diagnostics", "mechanical_measurement_diagnostics"): (
        "defect",
        "mechanical_measurement_diagnostics",
        "A barometry gauge's gas pressure is mechanical_measurement_diagnostics; "
        "particle_measurement_diagnostics owns a scattering or particle-counting instrument.",
    ),
    ("particle_measurement_diagnostics", "plasma_measurement_diagnostics"): (
        "defect",
        "particle_measurement_diagnostics",
        "A Langmuir probe's ion saturation current is particle_measurement_diagnostics; "
        "plasma_measurement_diagnostics holds only an instrument no family owns.",
    ),
    ("plasma_control", "equilibrium"): (
        "defect",
        "equilibrium",
        "The current-centre vertical velocity is a global equilibrium quantity; the command "
        "that aims the control loop is plasma_control.",
    ),
    ("plasma_control", "magnetohydrodynamics"): (
        "defect",
        "equilibrium",
        "A global current-centre velocity is an equilibrium state; plasma_control owns the "
        "control response, and magnetohydrodynamics a coherent mode.",
    ),
    ("plasma_initiation", "magnetic_field_systems"): (
        "defect",
        "plasma_initiation",
        "The stray field at the plasma position during breakdown is a start-up physics "
        "quantity, so it is plasma_initiation; the coils that produce it are "
        "magnetic_field_systems.",
    ),
    ("plasma_measurement_diagnostics", "computational_workflow"): (
        "defect",
        "mechanical_measurement_diagnostics",
        "A barometry gauge's position is mechanical_measurement_diagnostics; "
        "computational_workflow owns geometry definitions and tooling, not instrument "
        "hardware.",
    ),
    ("plasma_measurement_diagnostics", "electromagnetic_wave_diagnostics"): (
        "defect",
        "radiation_measurement_diagnostics",
        "An ECE channel position is radiation_measurement_diagnostics; an ECE diagnostic "
        "records emitted cyclotron radiation rather than transmitting a probe wave.",
    ),
    ("plasma_measurement_diagnostics", "equilibrium"): (
        "defect",
        "radiation_measurement_diagnostics",
        "An ECE channel position is radiation_measurement_diagnostics; the flux surface it "
        "is expressed against is equilibrium.",
    ),
    ("plasma_measurement_diagnostics", "mechanical_measurement_diagnostics"): (
        "defect",
        "mechanical_measurement_diagnostics",
        "A barometry gauge's position is mechanical_measurement_diagnostics; "
        "plasma_measurement_diagnostics owns a plasma-parameter instrument.",
    ),
    ("plasma_measurement_diagnostics", "neutronics"): (
        "defect",
        "neutronics",
        "A neutron detector's aperture width is neutronics; "
        "plasma_measurement_diagnostics holds only an instrument no family owns.",
    ),
    ("plasma_measurement_diagnostics", "particle_measurement_diagnostics"): (
        "defect",
        "neutronics",
        "A neutron detector's aperture geometry is neutronics and, being a particle counter, "
        "also particle_measurement_diagnostics; neutronics owns the neutron-detection "
        "family, so the label follows the family.",
    ),
    ("plasma_measurement_diagnostics", "spectroscopy"): (
        "defect",
        "spectroscopy",
        "A charge-exchange channel's optical cone diameter is spectroscopy (charge_exchange/*); "
        "plasma_measurement_diagnostics holds only an instrument no family owns.",
    ),
    ("plasma_wall_interactions", "divertor_physics"): (
        "defect",
        "divertor_physics",
        "A divertor target's own extent is divertor_physics; the erosion of its surface is "
        "plasma_wall_interactions.",
    ),
    ("plasma_wall_interactions", "gyrokinetics"): (
        "defect",
        "transport",
        "An edge-transport model's flux-limiter coordinate is transport; the model's own "
        "quantity in machine coordinates is not gyrokinetics, and a wall surface "
        "property is plasma_wall_interactions.",
    ),
    ("plasma_wall_interactions", "magnetohydrodynamics"): (
        "defect",
        "magnetohydrodynamics",
        "A disruption halo-current wall point is part of the disruption evolution, so it is "
        "magnetohydrodynamics; plasma_wall_interactions owns the surface's erosion and "
        "retention.",
    ),
    ("plasma_wall_interactions", "radiation_measurement_diagnostics"): (
        "defect",
        "radiation_measurement_diagnostics",
        "An infrared camera's target-image coordinate is radiation_measurement_diagnostics; "
        "the wall surface's own response is plasma_wall_interactions.",
    ),
    ("plasma_wall_interactions", "transport"): (
        "defect",
        "transport",
        "An edge-transport model's flux-limiter coordinate is transport; a surface property "
        "is plasma_wall_interactions.",
    ),
    ("structural_components", "magnetic_field_systems"): (
        "defect",
        "magnetic_field_systems",
        "A passive conductor loop's poloidal geometry is magnetic_field_systems; "
        "structural_components owns the load-bearing structure that supports a magnet.",
    ),
    ("transport", "gyrokinetics"): (
        "defect",
        "transport",
        "A plasma_transport model's neutral parallel velocity is transport; gyrokinetics "
        "owns the gyrokinetic model's own coordinates.",
    ),
    ("waves", "auxiliary_heating"): (
        "defect",
        "auxiliary_heating",
        "An EC launcher beam's phase-front curvature is a launcher hardware quantity, so it "
        "is auxiliary_heating; a wave field inside the plasma is waves.",
    ),
    ("particle_measurement_diagnostics", "electromagnetic_wave_diagnostics"): (
        "defect",
        "particle_measurement_diagnostics",
        "A Thomson-scattering laser end point is particle_measurement_diagnostics "
        "(thomson_scattering/*); electromagnetic_wave_diagnostics owns a probe that "
        "transmits a wave to measure its propagation.",
    ),
    ("waves", "electromagnetic_wave_diagnostics"): (
        "defect",
        "waves",
        "An ECE beam's parallel refractive index is a wave-field quantity propagating in the "
        "plasma, so it is waves; a probe instrument is "
        "electromagnetic_wave_diagnostics.",
    ),
    ("general", "data_management"): (
        "defect",
        "computational_workflow",
        "Atomic and molecular reference data is computational_workflow; general is for a "
        "physical quantity with no single subject, and data_management for an assembled "
        "data product.",
    ),
    ("core_plasma_physics", "equilibrium"): (
        "defect",
        "equilibrium",
        "A density constraint stored under equilibrium/time_slice/constraints is a "
        "constraint on the equilibrium, so it is equilibrium even though the quantity it "
        "constrains is a core-plasma density.",
    ),
    ("electromagnetic_wave_diagnostics", "equilibrium"): (
        "defect",
        "equilibrium",
        "A line-density constraint stored under equilibrium/time_slice/constraints is a "
        "constraint on the equilibrium; the interferometer that measured it is "
        "electromagnetic_wave_diagnostics.",
    ),
    ("auxiliary_heating", "current_drive"): (
        "defect",
        "auxiliary_heating",
        "A launcher's own launched refractive index and reference-point geometry are "
        "auxiliary_heating; current_drive owns the current the launch drives, not the "
        "launcher.",
    ),
    ("core_plasma_physics", "electromagnetic_wave_diagnostics"): (
        "defect",
        "electromagnetic_wave_diagnostics",
        "An interferometer's electron line density is electromagnetic_wave_diagnostics; the "
        "interferometer IDS is the wave-probe family.",
    ),
    ("core_plasma_physics", "radiation_measurement_diagnostics"): (
        "defect",
        "radiation_measurement_diagnostics",
        "Power radiated as an integral over the plasma is radiation_measurement_diagnostics; "
        "a bolometer's own quantity is the family's, not core_plasma_physics.",
    ),
    ("edge_plasma_physics", "transport"): (
        "defect",
        "core_plasma_physics",
        "A global loss-power scalar is a core energy-balance quantity, so it is "
        "core_plasma_physics; a flux or diffusivity is transport, and an edge resolution is "
        "edge_plasma_physics.",
    ),
    ("equilibrium", "magnetohydrodynamics"): (
        "defect",
        "magnetohydrodynamics",
        "A disruption's radial profile is part of the MHD instability evolution, so it is "
        "magnetohydrodynamics; equilibrium owns the pre-disruption equilibrium.",
    ),
}


def match(globs: object, path: str) -> str | None:
    """The first winner whose glob matches *path*, else None."""
    if isinstance(globs, str):
        return globs
    for pattern, winner in globs:  # type: ignore[union-attr]
        if pattern == "*":
            return winner
        if pattern.endswith("*") and path.startswith(pattern[:-1]):
            return winner
        if path == pattern:
            return winner
    return None


def load_measure() -> dict:
    return json.loads(
        (HERE / "scg-domain-path-measure-results.json").read_text(encoding="utf-8")
    )


def pair_entries(measure: dict) -> list[dict]:
    """Every (classifier, label, predicted, paths) row the measure flagged."""
    rows = []
    for cls in ("confusion_tree", "confusion_chat"):
        for entry in measure[cls]:
            rows.append(
                {
                    "classifier": cls.removeprefix("confusion_"),
                    "label": entry["label"],
                    "predicted": entry["predicted"],
                    "paths": list(entry["paths"]),
                }
            )
    return rows


def relabel(measure: dict) -> tuple[list[dict], dict[str, dict]]:
    """Apply the pair verdicts to every path, by rule and precedence.

    A path can sit in several pairs; each pair proposes a winner. The winner
    whose domain ranks earliest in the declared precedence is taken, which is the
    same tie-break the guidance states. A path in no pair keeps its label.
    """
    guidance = yaml.safe_load(GUIDANCE.read_text(encoding="utf-8"))
    order: list[str] = []
    for band in guidance["precedence"]:
        order.extend(band["domains"])
    rank = {name: i for i, name in enumerate(order)}

    proposals: dict[str, list[tuple[tuple[str, str], str]]] = {}
    for row in pair_entries(measure):
        key = (row["label"], row["predicted"])
        if key not in PAIRS:
            continue
        _, winner_spec, _ = PAIRS[key]
        for path in row["paths"]:
            win = match(winner_spec, path)
            if win:
                proposals.setdefault(path, []).append((key, win))

    answers = measure["path_answers"]
    labels: dict[str, dict] = {}
    changed: list[dict] = []
    for path, answer in answers.items():
        props = proposals.get(path)
        if not props:
            continue
        win = sorted({w for _, w in props}, key=lambda d: rank.get(d, len(rank)))[0]
        labels[path] = {"old": answer["label"], "new": win}
        if win != answer["label"]:
            pair_keys = sorted({k for k, w in props if w == win})
            changed.append(
                {
                    "path": path,
                    "old": answer["label"],
                    "new": win,
                    "verdict": PAIRS[pair_keys[0]][0],
                    "rule": PAIRS[pair_keys[0]][2],
                }
            )
    return changed, labels


def rescore(measure: dict, labels: dict[str, dict]) -> dict:
    """Path-level accuracy of the saved classifier answers against the new labels.

    A saved answer is scored as recorded; only paths that carry a rule-based new
    label are scored, since a path in no confused pair already agrees with its
    owner's label.
    """
    answers = measure["path_answers"]

    def new_label(path: str) -> str:
        return labels.get(path, {}).get("new", answers[path]["label"])

    def answer_of(cls: str, path: str) -> str:
        return answers[path]["label"] if cls == "label" else answers[path][cls]

    by_split: dict[str, dict] = {}
    for split in sorted({a["split"] for a in answers.values()}):
        row: dict[str, dict] = {}
        for cls in ("tree", "chat", "label"):
            paths = [p for p, a in answers.items() if a["split"] == split]
            correct = sum(int(answer_of(cls, p) == new_label(p)) for p in paths)
            row[cls] = {
                "correct": correct,
                "total": len(paths),
                "accuracy": round(correct / len(paths), 4),
            }
        by_split[split] = row

    overall: dict[str, dict] = {}
    for cls in ("tree", "chat", "label"):
        correct = sum(int(answer_of(cls, p) == new_label(p)) for p in answers)
        overall[cls] = {
            "correct": correct,
            "total": len(answers),
            "accuracy": round(correct / len(answers), 4),
        }

    contested: dict[str, dict] = {}
    for cls in ("tree", "chat", "label"):
        correct = sum(int(answer_of(cls, p) == new_label(p)) for p in labels)
        contested[cls] = {
            "correct": correct,
            "total": len(labels),
            "accuracy": round(correct / len(labels), 4),
        }

    return {"by_split": by_split, "overall": overall, "contested": contested}


def main() -> int:
    measure = load_measure()
    changed, labels = relabel(measure)
    scores = rescore(measure, labels)

    verdicts = []
    for (label, predicted), (verdict, winner_spec, rule) in PAIRS.items():
        verdicts.append(
            {
                "pair": f"{label} -> {predicted}",
                "verdict": verdict,
                "winner": winner_spec
                if isinstance(winner_spec, str)
                else [list(x) for x in winner_spec],
                "rule": rule,
            }
        )

    out = {
        "generated_from": "scg-domain-path-measure-results.json",
        "guidance": "imas_codex/definitions/physics/domains.yaml",
        "pair_count": len(pair_entries(measure)),
        "distinct_pair_count": len(PAIRS),
        "verdicts": verdicts,
        "changed_labels": changed,
        "changed_count": len(changed),
        "scores": scores,
    }
    (HERE / "scg-domain-scopes-results.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"pairs adjudicated: {len(PAIRS)} distinct, {len(pair_entries(measure))} entries"
    )
    print(f"changed labels: {len(changed)}")
    for split, row in scores["by_split"].items():
        print(
            f"  {split:12s} tree={row['tree']['accuracy']:.4f} "
            f"chat={row['chat']['accuracy']:.4f} label(old)={row['label']['accuracy']:.4f}"
        )
    print(
        f"overall      tree={scores['overall']['tree']['accuracy']:.4f} "
        f"chat={scores['overall']['chat']['accuracy']:.4f} "
        f"label(old)={scores['overall']['label']['accuracy']:.4f}"
    )
    print(
        f"contested    tree={scores['contested']['tree']['accuracy']:.4f}".rstrip(
            "0"
        ).rstrip("."),
        f"chat={scores['contested']['chat']['accuracy']:.4f} "
        f"label(old)={scores['contested']['label']['accuracy']:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
