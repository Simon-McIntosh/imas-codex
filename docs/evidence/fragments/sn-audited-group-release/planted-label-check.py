"""Freeze source-only judgments before comparing planted labels and readers.

Collection projects identifiers only from the audit; no labels, citations or
reader verdicts enter the source-only packet. Graph access is read-only and
restricted to these identifiers, their producers and enclosing DD structures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "planted-label-check.json"
PAGE = ROOT / "planted-label-check.html"
BLIND_COMMIT = "0b058c4394856e777436e3761a6cb66b7b814d81"
RELATIVE_DATA = (
    "docs/evidence/fragments/sn-audited-group-release/planted-label-check.json"
)


def audit_module():
    spec = importlib.util.spec_from_file_location(
        "blind_audit", ROOT / "blind-audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def revision():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def write_data(data):
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def collect():
    if DATA.exists():
        raise ValueError("Refusing to overwrite the source-only record")
    audit = audit_module()
    # Only this projection of the label-bearing file is exposed to the judge.
    ids = sorted(
        row["id"]
        for row in json.loads((ROOT / "blind-audit.json").read_text())[
            "planted_detail"
        ]["rows"]
    )
    assert len(ids) == len(set(ids)) == 40
    from iter_codex.graph.client import GraphClient
    from iter_codex.standard_names.source_paths import split_source_paths
    from iter_codex.standard_names.source_reads import source_paths_projection
    from iter_codex.standard_names.worker_tasks.worker_common import (
        _ANCESTOR_CONTEXT_QUERY,
        ancestor_statements,
    )

    gc = GraphClient()
    try:
        rows = list(
            gc.query(
                f"""
                UNWIND $ids AS wanted
                MATCH (sn:StandardName {{id: wanted}})
                RETURN sn.id AS id, coalesce(sn.description,'') AS description,
                       coalesce(sn.documentation,'') AS documentation,
                       coalesce(sn.unit,'') AS unit,
                       coalesce(sn.physics_domain,'') AS physics_domain,
                       {source_paths_projection()} AS source_paths
                ORDER BY id
                """,
                ids=ids,
            )
        )
        assert {r["id"] for r in rows} == set(ids), "Missing planted identifiers"
        paths = sorted(
            {
                path
                for row in rows
                for path in split_source_paths(row["source_paths"]).get("dd", [])
            }
        )
        docs = {
            r["path"]: r
            for r in gc.query(
                """
                UNWIND $paths AS pid MATCH (n:IMASNode {id: pid})
                RETURN n.id AS path, coalesce(n.description,'') AS description,
                       coalesce(n.documentation,'') AS documentation,
                       coalesce(n.unit,'') AS unit
                """,
                paths=paths,
            )
        }
        assert set(docs) == set(paths), "Missing DD source text"
        duplicates = {
            r["path"]: sorted(r["names"])
            for r in gc.query(
                f"""
                UNWIND $paths AS pid
                MATCH (src:StandardNameSource {{id: 'dd:' + pid}})
                      -[:PRODUCED_NAME]->(sn:StandardName)
                WHERE coalesce(src.status,'') <> 'stale' AND {audit.LIVE}
                RETURN pid AS path, collect(DISTINCT sn.id) AS names
                """,
                paths=paths,
            )
        }
        for row in rows:
            row["dd_paths"] = split_source_paths(row["source_paths"]).get("dd", [])
            row["dd_docs"] = [docs[p] for p in row["dd_paths"]]
            row["ancestors"] = (
                ancestor_statements(
                    list(gc.query(_ANCESTOR_CONTEXT_QUERY, path=row["dd_paths"][0]))
                )
                if row["dd_paths"]
                else []
            )
            audit.attach_duplicate_candidates([row], duplicates)
            row["reader_visible_text"] = audit.render_user_prompt(row)
            audit.assert_blind(row["reader_visible_text"], row)
        receipt = {
            "requested_names": len(ids),
            "returned_names": len(rows),
            "requested_dd_paths": len(paths),
            "returned_dd_paths": len(docs),
            "names_with_dd_text": sum(bool(r["dd_docs"]) for r in rows),
            "names_with_ancestor_text": sum(bool(r["ancestors"]) for r in rows),
            "names_with_duplicate_candidates": sum(
                bool(r["other_live_names"]) for r in rows
            ),
            "producer_source_groups": len(duplicates),
            "self_producer_memberships": sum(
                row["id"] in duplicates.get(path, [])
                for row in rows
                for path in row["dd_paths"]
            ),
            "graph_write_statements": 0,
            "model_calls": 0,
        }
    finally:
        gc.close()
    rows = [
        {key: row[key] for key in ("id", "source_paths", "reader_visible_text")}
        for row in rows
    ]
    write_data(
        {
            "phase": "source-only",
            "collected_at": datetime.now(UTC).isoformat(),
            "collection_revision": revision(),
            "collection_receipt": receipt,
            "class_list": list(audit.CLASS_KEYS),
            "blindness": {
                "exposed_before_judgment": "Sorted identifiers and source-only packets",
                "withheld": ["known", "citation", "A", "B", "C"],
                "historical_packet_caveat": (
                    "The audit stores verdicts but no full per-name prompt snapshots; "
                    "these are current graph fields rendered by the original prompt "
                    "function, not proof the text has remained unchanged since reading."
                ),
                "events": ["Read live plan", "Collected source-only packets"],
            },
            "packets": rows,
            "packet_digest": digest(rows),
            "judgments": [],
        }
    )
    print(json.dumps(receipt, sort_keys=True))


def verify_sources():
    """Require the duplicate-candidate join to recover each known producer."""
    from iter_codex.graph.client import GraphClient

    data = json.loads(DATA.read_text())
    packets = data["packets"]
    ids = sorted({p for r in packets for p in r["source_paths"] if p.startswith("dd:")})
    gc = GraphClient()
    try:
        rows = list(
            gc.query(
                """
                UNWIND $ids AS sid
                MATCH (src:StandardNameSource {id: sid})-[:PRODUCED_NAME]->(n:StandardName)
                WHERE coalesce(src.status,'') <> 'stale'
                  AND coalesce(n.name_stage,'') <> 'superseded'
                  AND coalesce(n.status,'') <> 'superseded'
                RETURN sid AS source, collect(DISTINCT n.id) AS names
                """,
                ids=ids,
            )
        )
    finally:
        gc.close()
    by_source = {r["source"]: r["names"] for r in rows}
    self_matches = sum(
        p["id"] in by_source.get(src, []) for p in packets for src in p["source_paths"]
    )
    expected = sum(len(p["source_paths"]) for p in packets)
    assert len(by_source) == len(ids), "Producer join missed known source nodes"
    assert self_matches == expected, "Producer join missed known names"
    candidates = {
        p["id"]: sorted(
            {
                n
                for src in p["source_paths"]
                for n in by_source.get(src, [])
                if n != p["id"]
            }
        )[:10]
        for p in packets
    }
    assert not any(candidates.values()), (
        "Duplicate candidates differ from frozen packet"
    )
    receipt = {
        "producer_source_groups": len(by_source),
        "self_producer_memberships": self_matches,
        "expected_self_memberships": expected,
        "other_live_name_candidates": sum(len(v) for v in candidates.values()),
        "positive_control": "Every bound DD producer recovered its own planted name",
    }
    data["collection_receipt"]["duplicate_join_control"] = receipt
    write_data(data)
    print(json.dumps(receipt, sort_keys=True))


def check_blind(data):
    packets = {p["id"]: p for p in data["packets"]}
    judgments = data["judgments"]
    assert len(packets) == len(judgments) == 40, "Require forty unique judgments"
    assert {j["id"] for j in judgments} == set(packets), "Judgment coverage differs"
    assert digest(data["packets"]) == data["packet_digest"], (
        "Source packet digest changed"
    )
    assert digest(judgments) == data["blindness"]["judgment_digest"], (
        "Judgments changed"
    )
    for judgment in judgments:
        text = packets[judgment["id"]]["reader_visible_text"]
        assert judgment["evidence_quote"] in text, (
            "Quote absent from packet: " + judgment["id"]
        )
        assert judgment["evidence_path"] in text, (
            "Source binding absent: " + judgment["id"]
        )
        assert judgment["verdict"] in ("defective", "sound")
        assert bool(judgment["defect_classes"]) == (judgment["verdict"] == "defective")
        assert set(judgment["defect_classes"]) <= set(data["class_list"])
        assert isinstance(judgment["decidable_from_text"], bool)
        assert not ({"known", "citation", "A", "B", "C"} & judgment.keys())
    control = data["collection_receipt"]["duplicate_join_control"]
    assert (
        control["self_producer_memberships"] == control["expected_self_memberships"] > 0
    )
    print(
        "PASS: 40 source-only judgments; quotes, bindings, digests and producer control"
    )


def frozen_record():
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BLIND_COMMIT}:{RELATIVE_DATA}"]
    )
    record = json.loads(raw)
    assert record["phase"] == "source-only"
    assert not {"comparison", "scores", "reader_models"} & record.keys()
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BLIND_COMMIT, "HEAD"],
        check=True,
    )
    paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            BLIND_COMMIT,
        ],
        text=True,
    ).splitlines()
    assert set(paths) == {
        RELATIVE_DATA,
        "docs/evidence/fragments/sn-audited-group-release/planted-label-check.py",
    }, "Blind judgment commit must touch no reader data"
    check_blind(record)
    return record


def input_records():
    inputs = {}
    hashes = {}
    for filename in ("blind-audit.json", "frontier-third-reader.json"):
        path = ROOT / filename
        raw = path.read_bytes()
        relative = "docs/evidence/fragments/sn-audited-group-release/" + filename
        committed = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{BLIND_COMMIT}:{relative}"]
        )
        assert raw == committed, "Reader input changed since blind commit: " + filename
        inputs[filename] = json.loads(raw)
        hashes[filename] = hashlib.sha256(raw).hexdigest()
    return inputs, hashes


def comparison_rows(frozen, inputs):
    original = {
        r["id"]: r for r in inputs["blind-audit.json"]["planted_detail"]["rows"]
    }
    third = {
        r["name"]: r for r in inputs["frontier-third-reader.json"]["planted"]["rows"]
    }
    ids = {j["id"] for j in frozen["judgments"]}
    assert ids == set(original) == set(third)
    assert Counter(r["known"] for r in original.values()) == {
        "defective": 20,
        "sound": 20,
    }
    rows = []
    models = {r["seat"]: r["model"] for r in inputs["blind-audit.json"]["readers"]}
    models["C"] = inputs["frontier-third-reader.json"]["reader"]["model"]
    for judgment in frozen["judgments"]:
        name = judgment["id"]
        plant, later = original[name], third[name]
        assert later["read"] is True
        assert later["known"] == plant["known"]
        readers = {}
        for seat in ("A", "B", "C"):
            source = plant[seat] if seat != "C" else later[seat]
            classes = source["classes"]
            assert set(classes) <= set(frozen["class_list"])
            if seat != "C":
                assert set(later[seat]["classes"]) == set(classes)
                assert source["defective"] == bool(classes)
            readers[seat] = {
                "verdict": "defective" if classes else "sound",
                "classes": classes,
                "rationale": source["rationale"],
                "model": models[seat],
                "shares_judged_class": bool(
                    set(classes) & set(judgment["defect_classes"])
                ),
            }
        disposition = (
            "undecidable"
            if not judgment["decidable_from_text"]
            else "stands"
            if judgment["verdict"] == plant["known"]
            else "overturned"
        )
        rows.append(
            {
                "id": name,
                "planted_label": plant["known"],
                "original_citation": plant["citation"],
                "original_audited_source_path": plant["audited_source_path"],
                "independent_verdict": judgment["verdict"],
                "label_verdict": disposition,
                "source_sentence": judgment["evidence_quote"],
                "source_sentence_path": judgment["evidence_path"],
                "label_reason": judgment.get("uncertainty", judgment["reason"]),
                "readers": readers,
                "at_least_two_flag": sum(
                    r["verdict"] == "defective" for r in readers.values()
                )
                >= 2,
            }
        )
    return rows, models


def rate_interval(count, total):
    lower, upper = audit_module().wilson(count, total)
    point = count / total if total else None
    return {
        "count": count,
        "total": total,
        "rate": point,
        "ci_95": [lower, upper] if total else [None, None],
        "within_ten_points": total > 0 and max(point - lower, upper - point) <= 0.10,
    }


def score_rows(rows, label_field):
    results = {}
    defective = [r for r in rows if r[label_field] == "defective"]
    sound = [r for r in rows if r[label_field] == "sound"]
    for seat in ("A", "B", "C", "two_of_three"):

        def flag(row, reader_seat=seat):
            if reader_seat == "two_of_three":
                return row["at_least_two_flag"]
            return row["readers"][reader_seat]["verdict"] == "defective"

        recall = rate_interval(sum(flag(r) for r in defective), len(defective))
        alarm = rate_interval(sum(flag(r) for r in sound), len(sound))
        point_pass = bool(
            recall["rate"] is not None
            and alarm["rate"] is not None
            and recall["rate"] >= 0.90
            and alarm["rate"] <= 0.10
        )
        precision = recall["within_ten_points"] and alarm["within_ten_points"]
        results[seat] = {
            "recall": recall,
            "false_alarm": alarm,
            "meets_point_thresholds": point_pass,
            "meets_precision": precision,
            "fit_to_decide": point_pass and precision,
        }
    return results


def derived_report(frozen):
    inputs, hashes = input_records()
    rows, models = comparison_rows(frozen, inputs)
    summary = dict(Counter(r["label_verdict"] for r in rows))
    summary["original_defective_now_sound"] = sum(
        r["planted_label"] == "defective" and r["label_verdict"] == "overturned"
        for r in rows
    )
    summary["original_sound_now_defective"] = sum(
        r["planted_label"] == "sound" and r["label_verdict"] == "overturned"
        for r in rows
    )
    return {
        "comparison": rows,
        "reader_models": models,
        "input_sha256": hashes,
        "label_summary": summary,
        "scores": {
            "labels_that_stand": score_rows(
                [r for r in rows if r["label_verdict"] == "stands"], "planted_label"
            ),
            "all_decidable_checked_labels": score_rows(
                [r for r in rows if r["label_verdict"] != "undecidable"],
                "independent_verdict",
            ),
            "original_labels_reference_only": score_rows(rows, "planted_label"),
        },
        "method": {
            "interval": "Two-sided 95% Wilson score interval, z=1.96",
            "unit": "One name; any listed defect class counts as one flag",
            "majority": "At least two of three observed flags; no independence assumption",
            "primary_cohort": "Original labels that agree with the frozen source judgment",
            "supplement": "All decidable judgments, including overturned labels relabelled",
            "precision": "Each interval endpoint within 0.10 of the observed rate",
            "fitness": "Recall >= 0.90, false alarm <= 0.10, and both precision checks",
            "uncertainty": (
                "Binomial intervals describe sampling uncertainty only; they do not "
                "include label disagreement, selective planting, text drift, or "
                "uncertainty about a single manual adjudication."
            ),
            "provenance": (
                "Both planted classes were selected from earlier West audit rows "
                "labelled correct or INCORRECT, not from pipeline review scores. "
                "Selection checked live name and audited-source membership, not "
                "unchanged stored text or agreement across every current source."
            ),
            "historical_replay": (
                "Current graph fields use the original prompt renderer. The prior "
                "audit retained no complete per-name prompts or text hashes, so "
                "this is conditional calibration against checked current text, "
                "not an exact historical replay or a new physicist sign-off."
            ),
        },
        "conclusion": (
            "No individual reader or two-of-three rule meets both point thresholds "
            "on the labels that stand, and none meets interval precision. The "
            "checked set also remains too small to establish fitness. Do not use "
            "these results alone to authorize deciding group audits or a release "
            "bar. Enlarge an independently adjudicated set with versioned full "
            "source packets and measure again; any interim deciding rule requires "
            "the lead's recorded choice. Re-estimating catalog defect rates from "
            "this selected calibration set requires assumptions not tested here."
        ),
    }


def rate_text(value):
    n, total, point = value["count"], value["total"], value["rate"]
    if point is None:
        return f"{n}/{total}; not estimable"
    low, high = value["ci_95"]
    return f"{n}/{total} = {100 * point:.1f}% (95% {100 * low:.1f}–{100 * high:.1f}%)"


def score_table(scores):
    lines = [
        "<table><thead><tr><th>Reader / rule</th><th>Defects found / defective</th>"
        "<th>Sound flagged / sound</th><th>90% / 10% points</th>"
        "<th>Both intervals within ±10 points</th><th>Fit to decide</th>"
        "</tr></thead><tbody>"
    ]
    for seat, score in scores.items():
        label = "at least two of three" if seat == "two_of_three" else f"reader {seat}"
        lines.append(
            "<tr><td>"
            + label
            + "</td><td>"
            + rate_text(score["recall"])
            + "</td><td>"
            + rate_text(score["false_alarm"])
            + "</td><td>"
            + str(score["meets_point_thresholds"])
            + "</td><td>"
            + str(score["meets_precision"])
            + "</td><td>"
            + str(score["fit_to_decide"])
            + "</td></tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def render_page(data):
    summary = data["label_summary"]
    judgments = {j["id"]: j for j in data["judgments"]}
    lines = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<meta name="docs-project" content="iter-codex">',
        '<meta name="reckon-type" content="evidence-fragment">',
        '<meta name="plan-slug" content="sn-audited-group-release">',
        "<title>Planted labels checked against their sources</title>",
        "<style>body{font:16px/1.5 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;"
        "max-width:1500px;color:#222}table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "td,th{text-align:left;vertical-align:top;padding:.6rem;border-bottom:1px solid #ccc}"
        "code,pre{font-size:.85em}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;"
        "overflow-wrap:anywhere}blockquote{margin:.5rem 0;padding-left:.7rem;"
        "border-left:2px solid #929292}summary{cursor:pointer}td:first-child{min-width:10rem}"
        ".comparison{overflow-x:auto}.comparison table{min-width:1100px}"
        ".comparison td{max-width:24rem}.comparison th{min-width:8rem}"
        "</style></head><body><main>",
        '<h1 id="planted-label-check-title">Planted labels checked against their sources</h1>',
        '<p id="planted-label-check-summary"><strong>'
        + f"{summary['stands']} labels stand, {summary['overturned']} are overturned, "
        + f"{summary['undecidable']} are undecidable.</strong> "
        + "None of the three readers or the two-of-three rule meets both the 90% "
        + "recall and 10% false-alarm point thresholds on the standing labels. "
        + "None meets the interval precision required for fitness.</p>",
        "<p>The 40 judgments were written from the displayed sources before any "
        "per-name planted label, citation or reader verdict was opened, and "
        "committed in <code>" + BLIND_COMMIT + "</code>. That commit changes only "
        "the new script and source-only data file. Comparison was added afterward. "
        "The live plan and its aggregate reader results were read first. No model "
        "calls were made for these judgments, no stored text was changed, and "
        "additional metered spend was USD 0.</p>",
        "<h2>What the label check changes</h2>",
        "<p>"
        + str(summary["original_defective_now_sound"])
        + " originally defective labels are read sound; "
        + str(summary["original_sound_now_defective"])
        + " originally sound labels are read defective. The two provisional "
        "readings concern whether a power-conversion fluid is coolant and whether "
        "a limiter unit is one tile. They remain undecidable and are excluded "
        "from calibration. A sound judgment means no fixed defect class is "
        "established by the shown text; it is not an assertion about unshown context.</p>",
        "<p>For example, <code>vertical_coordinate_of_ece_channel</code> is supported "
        "by the source’s “Vertical (Z) coordinate of position” under “Position of "
        "the measurements”; its defective label is overturned. Conversely, "
        "<code>total_power_due_to_ohmic_dissipation</code> excludes halo-only power "
        "in its definition although one producer states “Ohmic power in the halo "
        "region”; its sound label is overturned. These are readings of the "
        "current full source portfolio, not a declaration that an earlier "
        "source-specific ruling was wrong when made.</p>",
        "<h2>Calibration on labels that stand</h2>",
        "<p>Primary, as requested: 15 standing defective labels and 11 standing "
        "sound labels. Overturned labels and the two undecidable labels are left "
        "out. Intervals are two-sided 95% Wilson score intervals (z = 1.96). A "
        "flag means any defect class; it need not be the same class identified "
        "by the independent judgment. The two-of-three rule is scored directly "
        "from the three flags, without assuming reader independence.</p>",
        score_table(data["scores"]["labels_that_stand"]),
        "<p>Reader C’s apparent false alarms shrink from 7/20 under the original "
        "labels to 1/11 on standing sound labels: six of its original seven flags "
        "are supported defects under this check. It still misses the stated "
        "cross-section scope in <code>height_of_poloidal_field_coil</code> and "
        "the conversion mechanism in <code>spectral_calibration_factor_at_line_of_sight</code>. "
        "The two-of-three rule misses four of the fifteen standing defects.</p>",
        "<h2>All decidable checked labels</h2>",
        "<p>Supplement: retain the twelve overturned labels with their new "
        "source-based verdicts. This gives 22 defective and 16 sound labels; "
        "the two undecidable cases are still excluded. This avoids silently "
        "discarding the defects found among the original sound plants. No reader "
        "or rule becomes fit under this alternative.</p>",
        score_table(data["scores"]["all_decidable_checked_labels"]),
        "<h2>What this permits for group audits</h2>",
        "<p>" + escape(data["conclusion"]) + "</p>",
        "<p>Even a rule with zero false alarms among 11 sound cases has a Wilson "
        "upper bound of 25.9%. Forty plants, reduced to 26 standing labels, cannot "
        "establish a ten-percentage-point precision margin here. The intervals "
        "cover binomial sampling uncertainty only. These selected plants do not "
        "measure catalog prevalence, and neither label uncertainty nor shared "
        "reader failure modes is included in those intervals.</p>",
        "<p>Reader C’s recall interval still includes 90%. The point estimate "
        "fails the required bar, but this small sample also cannot establish "
        "at 95% confidence that C’s underlying recall is below 90%.</p>",
        "<h2>Provenance and limits</h2>",
        "<p>" + escape(data["method"]["provenance"]) + "</p>",
        "<p>"
        + escape(data["method"]["historical_replay"])
        + " "
        + "The retained source packets were read at "
        + escape(data["collected_at"])
        + ". All 40 requested names and 177 DD paths were recovered; 39 names had "
        "DD text and 37 had enclosing-structure text. The zero other-live-name "
        "result has a positive control: the same source join recovered all "
        "177 expected producer/name memberships.</p>",
        '<p>Inputs: <a href="/imas-codex/evidence/fragments/sn-audited-group-release/sagr-blind-audit.html">'
        'the first audit</a>, <a href="/imas-codex/evidence/fragments/sn-audited-group-release/sagr-frontier-third-reader.html">'
        "the third reading</a>, and the source packets retained in "
        '<a href="/imas-codex/evidence/fragments/sn-audited-group-release/planted-label-check.json">'
        "the checked data</a>. The companion script verifies the frozen commit, "
        "input digests, every disposition, all counts and intervals, and this "
        "rendered page; changing a label disposition must make its check refuse.</p>",
        "<h2>Every plant beside its source judgment and the readers</h2>",
        '<div class="comparison"><table><thead><tr><th>Name and planted label</th>'
        "<th>Independent judgment, source sentence and reason</th><th>Label check</th>"
        "<th>Reader A</th><th>Reader B</th><th>Reader C</th>"
        "<th>At least two flag</th></tr></thead><tbody>",
    ]
    for row in data["comparison"]:
        judgment = judgments[row["id"]]
        lines.append(
            '<tr id="plant-'
            + escape(row["id"])
            + '"><td><code>'
            + escape(row["id"])
            + "</code><p>Planted: <strong>"
            + row["planted_label"]
            + "</strong></p><details><summary>Earlier ruling</summary>"
            + "<code>"
            + escape(row["original_citation"])
            + "</code><br><code>"
            + escape(row["original_audited_source_path"])
            + "</code></details></td>"
            + "<td><strong>"
            + judgment["verdict"]
            + "</strong><br><code>"
            + escape(", ".join(judgment["defect_classes"]) or "no fixed class")
            + "</code><blockquote>"
            + escape(judgment["evidence_quote"])
            + "</blockquote><code>"
            + escape(judgment["evidence_path"])
            + "</code><p>"
            + escape(judgment["reason"])
            + "</p></td>"
            + "<td><strong>"
            + row["label_verdict"]
            + "</strong><p>"
            + escape(row["label_reason"])
            + "</p></td>"
        )
        for seat in ("A", "B", "C"):
            reader = row["readers"][seat]
            lines.append(
                "<td><strong>"
                + reader["verdict"]
                + "</strong><br><code>"
                + escape(", ".join(reader["classes"]) or "no fixed class")
                + "</code><details><summary>Reader reason</summary><p>"
                + escape(reader["rationale"])
                + "</p></details></td>"
            )
        lines.append("<td>" + str(row["at_least_two_flag"]) + "</td></tr>")
    lines.extend(["</tbody></table></div>", "<h2>Frozen source-only packets</h2>"])
    for packet in data["packets"]:
        lines.append(
            "<details><summary><code>"
            + escape(packet["id"])
            + "</code></summary><pre>"
            + escape(packet["reader_visible_text"])
            + "</pre></details>"
        )
    lines.extend(
        [
            "<h2>Reader identities</h2><table><thead><tr><th>Seat</th><th>Recorded model</th>"
            "</tr></thead><tbody>",
            *[
                "<tr><td>"
                + seat
                + "</td><td><code>"
                + escape(model)
                + "</code></td></tr>"
                for seat, model in data["reader_models"].items()
            ],
            "</tbody></table></main></body></html>",
        ]
    )
    return "\n".join(lines) + "\n"


def compare():
    frozen = frozen_record()
    current = json.loads(DATA.read_text())
    assert current["judgments"] == frozen["judgments"], "Blind judgments changed"
    assert current["packets"] == frozen["packets"], "Blind packets changed"
    data = frozen | derived_report(frozen)
    data["phase"] = "compared"
    data["workflow"] = {
        "blind_judgment_commit": BLIND_COMMIT,
        "compared_after_revision": revision(),
        "compared_at": datetime.now(UTC).isoformat(),
        "order": [
            "Read live plan and aggregate context",
            "Read source-only packets and write forty judgments",
            "Commit all forty judgments without reader data",
            "Open planted labels and three readers and compare frozen judgments",
        ],
    }
    write_data(data)
    PAGE.write_text(render_page(data))
    print(json.dumps(data["label_summary"], sort_keys=True))
    for seat, score in data["scores"]["labels_that_stand"].items():
        print(
            seat,
            "recall",
            rate_text(score["recall"]),
            "false alarm",
            rate_text(score["false_alarm"]),
        )


def check(path):
    frozen = frozen_record()
    data = json.loads(path.read_text())
    assert data["phase"] == "compared", "Comparison not completed"
    for key, value in frozen.items():
        if key != "phase":
            assert data[key] == value, "Frozen source-only field changed: " + key
    assert data["workflow"]["blind_judgment_commit"] == BLIND_COMMIT
    for key, value in derived_report(frozen).items():
        assert data[key] == value, "Recomputed report differs: " + key
    assert PAGE.read_text() == render_page(data), "Rendered evidence page differs"
    assert path.stat().st_size <= 300_000, "Data file exceeds repository size ceiling"
    assert not any(
        s["fit_to_decide"] for s in data["scores"]["labels_that_stand"].values()
    )
    reference = data["scores"]["original_labels_reference_only"]
    for seat, recalled, alarmed in (("A", 9, 5), ("B", 11, 5), ("C", 13, 7)):
        assert reference[seat]["recall"]["count"] == recalled
        assert reference[seat]["false_alarm"]["count"] == alarmed
        assert reference[seat]["recall"]["total"] == 20
        assert reference[seat]["false_alarm"]["total"] == 20
        print(
            f"REPRODUCED original {seat}: recall {recalled}/20; false alarm {alarmed}/20"
        )
    for seat, score in data["scores"]["labels_that_stand"].items():
        print(
            seat,
            "recall",
            rate_text(score["recall"]),
            "false alarm",
            rate_text(score["false_alarm"]),
        )
    print(
        "PASS: 40 judgments frozen before comparison; 26 stand, 12 overturned, 2 undecidable; all four calibrations and rendered page verified"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--check-blind", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.verify_sources:
        verify_sources()
    elif args.check_blind:
        check_blind(json.loads(DATA.read_text()))
    elif args.compare:
        compare()
    elif args.check:
        check(args.data)
    else:
        parser.error("Choose --collect, --verify-sources, --compare or --check")


if __name__ == "__main__":
    main()
