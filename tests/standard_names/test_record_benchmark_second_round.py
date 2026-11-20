"""The second round's check refuses an altered record, a leaking packet, a broken key."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HARNESS = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "evidence"
    / "fragments"
    / "sn-case-file-generation"
    / "benchmark.py"
)


def load_harness():
    spec = importlib.util.spec_from_file_location("record_benchmark_harness", HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bm = load_harness()

MEMBERS = [
    {
        "id": "m1",
        "group": "g",
        "unit": "W",
        "right_names": ["n1"],
        "definition_points": ["p"],
        "verdict": "sound",
    },
    {
        "id": "m2",
        "group": "g",
        "unit": "W",
        "right_names": ["n2"],
        "definition_points": ["p"],
        "verdict": "definition_defect",
    },
]
CASES = {
    member["id"]: {"content": {"sources": []}, "digest": "digest-" + member["id"]}
    for member in MEMBERS
}


def row(member: str, arm: str, name: str, decision: str = "repair") -> dict:
    return {
        "phase": "held_out",
        "arm": arm,
        "member": member,
        "decision": decision,
        "record": {
            "name": name,
            "description": "definition",
            "documentation": "documentation",
            "unit": "W",
            "group": "g",
        },
        "blocker": "",
        "reason": "",
        "structure": "keep-or-repair",
        "model": "synthetic",
        "prompt_version": "v3",
        "prompt_hash": "hash",
        "cost_usd": 0.5,
        "input_tokens": 100,
        "output_tokens": 10,
        "seconds": 2.0,
        "changed_incumbent": True,
        "incumbent_right": True,
        "case_digest": "digest-" + member,
        "target_group": "g",
    }


RECORDS = [
    row("m1", "one-pass/frontier", "n1"),
    row("m2", "one-pass/frontier", "n2"),
    row("m1", "incumbent", "n1"),
    row("m2", "incumbent", "n3"),
]


def write_round(tmp_path: Path, records: list[dict]):
    packet, key = bm.build_packet_and_key(records, MEMBERS, CASES)
    digest = bm.records_digest(records)
    document = {
        "records_digest": digest,
        "records": {"lines": len(records)},
        "packet": {"lines": len(packet)},
        "spend": {
            "ceiling_usd": bm.SECOND_SPEND_CEILING_USD,
            "within_ceiling": True,
        },
    }
    records_path = tmp_path / "records.jsonl"
    packet_path = tmp_path / "packet.jsonl"
    key_path = tmp_path / "key.json"
    data_path = tmp_path / "round.json"
    records_path.write_text(
        "".join(json.dumps(line) + "\n" for line in records), encoding="utf-8"
    )
    packet_path.write_text(
        "".join(json.dumps(line) + "\n" for line in packet), encoding="utf-8"
    )
    key_path.write_text(json.dumps(key), encoding="utf-8")
    data_path.write_text(json.dumps(document), encoding="utf-8")
    return records_path, packet_path, key_path, data_path


def check(paths) -> list[str]:
    return bm.check_second_round(*paths)


def test_score_subcommand_passes_its_synthetic_self_test():
    assert bm.score_self_test() == []


def test_clean_round_passes(tmp_path):
    assert check(write_round(tmp_path, RECORDS)) == []


def test_an_altered_record_line_fails(tmp_path):
    paths = write_round(tmp_path, RECORDS)
    records_path = paths[0]
    lines = [json.loads(line) for line in records_path.read_text().splitlines()]
    lines[1]["record"]["description"] = "a different definition"
    records_path.write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    problems = check(paths)
    assert any("records digest" in problem for problem in problems)


def test_a_packet_line_that_reveals_an_arm_fails(tmp_path):
    paths = write_round(tmp_path, RECORDS)
    packet_path = paths[1]
    lines = [json.loads(line) for line in packet_path.read_text().splitlines()]
    lines[0]["arm"] = "keep-or-repair/frontier"
    packet_path.write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    problems = check(paths)
    assert any("reveals" in problem for problem in problems)


def test_a_key_id_absent_from_the_packet_fails(tmp_path):
    paths = write_round(tmp_path, RECORDS)
    key_path = paths[2]
    key = json.loads(key_path.read_text())
    key["packets"]["pkt-absent"] = {"member": "m1", "arms": [["m1", "A"]]}
    key_path.write_text(json.dumps(key), encoding="utf-8")
    problems = check(paths)
    assert any("disagree on packet ids" in problem for problem in problems)


@pytest.mark.parametrize("field", ["name", "definition", "unit"])
def test_a_packet_record_that_differs_from_its_record_fails(tmp_path, field):
    paths = write_round(tmp_path, RECORDS)
    packet_path = paths[1]
    lines = [json.loads(line) for line in packet_path.read_text().splitlines()]
    lines[0][field] = "something else"
    packet_path.write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    problems = check(paths)
    assert any("differs from the record line" in problem for problem in problems)
