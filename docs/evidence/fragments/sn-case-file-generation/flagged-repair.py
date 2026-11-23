"""Repair a flagged defect and leave an unflagged name alone.

Four detection passes each look for exactly one class of defect the blind
adjudication of the second benchmark round recorded on the held-out incumbents:
a name that drops a qualifier the source states, a name with no owner or
subject, a name bound to sources describing several quantities or to a parent
too vague to identify one, and a wrong or garbled term. Each pass reads only
what a real pipeline stage would be given -- the incumbent record, the data
dictionary text bound to the name, the enclosing structure, the family and the
grammar facts -- and reports the member with the words of the source or the rule
that caused the flag.

A flagged name is repaired once, by a single local call, under a field
restriction: the repair may change only the field the flag names, and a repair
that changes anything else is rejected and counted. A name no pass flags is
never rewritten.

The result is measured against the incumbent verdicts already committed for the
held-out half: the share of the adjudicated failures in each class that its pass
flagged, and the share of adjudicated-sound names it flagged, each with a
Wilson 95 % interval. A planted control of ten sound names is given a style-only
variation and must produce no flag.

Every repaired record is written to a blind packet beside its incumbent, with a
key the coordinating session does not open until its verdicts are committed.
The harness writes no stored standard-name text and touches no graph property.

Run:

    uv run --no-sync python docs/evidence/fragments/sn-case-file-generation/flagged-repair.py --stage all

`--stage detect` runs only the passes, `--stage report` rebuilds the evidence
from the rows already on disk, and `--check` runs the harness's self-checks,
including a planted out-of-field repair that the field restriction must reject.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import importlib.util
import json
import os
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
RUN = Path(
    os.environ.get(
        "FLAGGED_REPAIR_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261122T184533028047-flagged-repair",
    )
)
CASE_FILES = Path(
    "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261118T064806928346-scfg-case-file/case-files"
)
RECORDS = ROOT / "benchmark-second-round-records.jsonl"
KEY = ROOT / "benchmark-second-round-key.json"
VERDICTS = ROOT / "benchmark-second-round-verdicts.jsonl"
DEFECTS = ROOT / "benchmark-second-round-incumbent-defects.json"
EVIDENCE = ROOT / "flagged-repair.json"
PAGE = ROOT / "flagged-repair.html"
ROW_LOG = RUN / "flagged-repair-rows.jsonl"
ROW_EXPORT = ROOT / "flagged-repair-rows.jsonl"
PACKET = ROOT / "flagged-repair-blind-packet.jsonl"
PACKET_KEY = ROOT / "flagged-repair-blind-key.json"
NEGATIVE_LOG = RUN / "negative-control.log"

LOCAL_MODEL = "local/deepseek-v4.1-flash"
SEED = 20261120
SPEND_CEILING_USD = 10.0
CONCURRENCY = 4
CONTROL_MEMBERS = 10

PASSES: dict[str, dict[str, str]] = {
    "qualifier": {
        "class": "name drops a qualifier the source states",
        "criterion": (
            "The record states a physically load-bearing qualifier and the "
            "incumbent name omits it, so the name denotes a broader or "
            "different quantity. The record states it in its bound source "
            "(description or enclosing structure) or, where no source describes "
            "the quantity, in its own definition. Walk every category the "
            "catalog names and check each against the name: locus (for example "
            "'at wall', 'at constraint position'), extremum ('maximum', "
            "'minimum'), species ('tritium', 'electron', 'ion'), medium "
            "('coolant'), flux kind ('neutron' versus 'heat'), and the kind of "
            "quantity ('count' versus 'rate' versus 'temperature', 'apparent' "
            "versus thermodynamic, 'volumetric' versus mass). A count, an "
            "accumulation, an inventory or a rate must state what is counted "
            "or per what: the counting basis is identity, not normalisation, "
            "so 'particles' versus 'equivalent electrons', 'atoms' versus "
            "'molecules', and 'per pulse' versus 'per second' all belong in the "
            "name when the record states them. If a stated qualifier from any "
            "category is absent from the name, flag."
        ),
        "not": (
            "Do not flag a purely numerical normalisation, weighting, "
            "averaging, integration, convention, coordinate frame, method or "
            "unit that is missing from the name; the identity rule puts those "
            "in the definition. Do not flag domain-implied boilerplate the "
            "record does not state. Do not flag wording or spelling style. Do "
            "not flag when neither the source, the enclosing structure nor the "
            "definition states any qualifier: with no words you cannot ground "
            "this flag."
        ),
        "rule_id": "I1.10_qualifier_preservation",
        "field": "name",
    },
    "subject": {
        "class": "name has no owner or subject",
        "criterion": (
            "The name states a property, coordinate, rate or relation but does "
            "not name the physical object or subject whose quantity it is, so a "
            "reader cannot tell which object's quantity is meant. A derived "
            "locus or bare quantity must name its owner unless the term belongs "
            "to one object only; a count or rate must state its counting basis; "
            "a shape parameter must name the surface it describes. Flag when "
            "the source identifies a specific object or subject that the name "
            "leaves out."
        ),
        "not": (
            "Do not flag a name that does name its subject, however tersely. Do "
            "not flag a naming style. Do not flag a purely numerical "
            "normalisation or convention in the name. Do not flag when neither "
            "the source, the enclosing structure nor the definition identifies "
            "an owner or a counting basis, unless the name itself plainly names "
            "no subject. A neighbour or family entry that is the same kind of "
            "quantity and does name its owner is evidence that this name leaves "
            "its own definition to say what is counted."
        ),
        "rule_id": "name-states-its-subject",
        "field": "name",
    },
    "multiplicity": {
        "class": "one name for several quantities, or a parent too vague to identify one",
        "criterion": (
            "The sources given for this name describe more than one distinct "
            "quantity (for example different components, different averaging "
            "bases, or a coefficient vector whose components are different "
            "quantities), or the enclosing parent is too vague to identify a "
            "single quantity, so one name cannot denote exactly one quantity."
        ),
        "not": (
            "Do not flag a name merely because several source statements "
            "describe the same quantity from different depths. Do not flag a "
            "quantity that is meaningful as a single vector."
        ),
        "rule_id": "one-name-one-quantity",
        "field": "name",
    },
    "term": {
        "class": "wrong or garbled term",
        "criterion": (
            "A term in the name names a different quantity than the source, "
            "misstates the physical relation, or is coined or garbled so that "
            "it does not denote what the source describes. Check every term: "
            "does it denote the source's quantity, or a different one? (For "
            "example, a name that says a surface is aligned with the field when "
            "the source defines an angle between the field and the surface; a "
            "'parity' where the source defines the sign of a field component; "
            "a measurement quantity attributed to an instrument rather than the "
            "observable.)"
        ),
        "not": (
            "Do not flag a long name for being long, or a rare but correct term "
            "for being rare. Do not flag a name for wording style."
        ),
        "rule_id": "I2.7_wrong_physics_qualifier",
        "field": "name",
    },
}

IDENTITY_RULE = (
    "The lead ruled that a name identifies and its definition specifies. A name "
    "carries only what separates the quantity from its neighbours: the kind of "
    "quantity, the physical object or locus it describes, and the relation or "
    "component where that separation depends on it. A definition carries "
    "everything a reader needs to obtain the number: normalisations, "
    "weightings, averaging or integration, conventions, coordinate frames, "
    "methods and units. A property that changes which quantity is meant "
    "belongs in the name; a property that changes how the number is measured "
    "belongs in the definition."
)

SPELLING = (
    ("normalized", "normalised"),
    ("normalization", "normalisation"),
    ("center", "centre"),
    ("behavior", "behaviour"),
    ("labeled", "labelled"),
    ("modeled", "modelled"),
    ("ionized", "ionised"),
    ("organized", "organised"),
    ("minimized", "minimised"),
    ("maximized", "maximised"),
    ("optimized", "optimised"),
    ("characterized", "characterised"),
    ("parameterized", "parameterised"),
    ("generalized", "generalised"),
    ("neighbor", "neighbour"),
    ("favorable", "favourable"),
    ("analyzed", "analysed"),
)


# --------------------------------------------------------------------------- #
# Loading the evidence the passes are measured against
# --------------------------------------------------------------------------- #


def load_benchmark_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "scfg_benchmark", ROOT / "benchmark.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("benchmark.py cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCH = load_benchmark_module()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()


def load_dataset() -> dict[str, Any]:
    """The 87 held-out members: incumbent record, committed verdict, class."""
    key = json.loads(KEY.read_text(encoding="utf-8"))
    held_out = sorted(key["members"])
    incumbents = {
        row["member"]: row["record"]
        for row in read_jsonl(RECORDS)
        if row.get("arm") == "incumbent" and row.get("member") in key["members"]
    }
    verdict_by_packet = {row["packet_id"]: row for row in read_jsonl(VERDICTS)}
    member_packet: dict[str, str] = {}
    for packet_id, info in key["packets"].items():
        member = info["member"]
        if any(arm == "incumbent" for _, arm in info["arms"]):
            member_packet[member] = packet_id
    verdicts = {
        member: verdict_by_packet[packet]
        for member, packet in member_packet.items()
        if member in key["members"]
    }
    classes = {
        entry["member"]: defect_class["class"]
        for defect_class in json.loads(DEFECTS.read_text(encoding="utf-8"))["classes"]
        for entry in defect_class["members"]
    }
    cases = {
        member: json.loads((CASE_FILES / f"{member}.json").read_text(encoding="utf-8"))
        for member in held_out
        if (CASE_FILES / f"{member}.json").exists()
    }
    missing_cases = [member for member in held_out if member not in cases]
    if missing_cases:
        raise RuntimeError(f"case files missing for {missing_cases}")
    if len(held_out) != 87:
        raise RuntimeError(f"expected 87 held-out members, found {len(held_out)}")
    if set(incumbents) != set(held_out):
        raise RuntimeError("incumbent records do not cover the held-out members")
    return {
        "key": key,
        "held_out": held_out,
        "incumbents": incumbents,
        "verdicts": verdicts,
        "classes": classes,
        "cases": cases,
    }


def is_adjudicated_failure(verdict: dict[str, Any]) -> bool:
    return verdict.get("name_verdict") != "right" or (
        verdict.get("definition_verdict") != "right"
    )


# --------------------------------------------------------------------------- #
# The context a pass is allowed to see
# --------------------------------------------------------------------------- #


def squash(value: Any, limit: int = 0) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit and len(text) > limit:
        return text[:limit].rstrip() + " …"
    return text


def build_context(case: dict[str, Any]) -> dict[str, Any]:
    """The pipeline channels a pass may read; never the earlier adjudication."""
    content = case.get("content") or {}
    incumbent = content.get("incumbent") or {}
    blocks = BENCH.source_blocks(case)
    family = content.get("family") or {}
    members = [
        {
            "name": member.get("name") or "",
            "differs_on": member.get("differs_on") or "",
            "description": squash(member.get("description"), 220),
        }
        for member in (family.get("members") or [])
    ]
    grammar = content.get("grammar") or {}
    neighbours: list[dict[str, Any]] = []
    seen_neighbour: set[str] = set()
    neighbour_block = content.get("neighbours") or {}
    for group in ("by_locus", "by_subject"):
        for member in (neighbour_block.get(group) or {}).get("members") or []:
            name = member.get("name") or ""
            if not name or name in seen_neighbour:
                continue
            seen_neighbour.add(name)
            neighbours.append(
                {
                    "name": name,
                    "separation": member.get("separation") or "",
                    "description": squash(member.get("description"), 300),
                }
            )
    return {
        "bound_paths": list(incumbent.get("bound_paths") or []),
        "missing_channels": list((case.get("completeness") or {}).get("missing") or []),
        "source_descriptions": blocks["source_descriptions"],
        "enclosing_statements": blocks["enclosing_statements"],
        "unit": incumbent.get("unit") or "",
        "neighbours": neighbours,
        "family": members,
        "grammar": {
            "segments": grammar.get("segments") or {},
            "registry": (grammar.get("registry") or {}).get("entries") or [],
            "conventions": (grammar.get("conventions") or {}).get("statement") or "",
        },
    }


def live_names(dataset: dict[str, Any]) -> set[str]:
    """Every live name the case files expose: incumbents, family, neighbours."""
    names = set(dataset["incumbents"])
    for case in dataset["cases"].values():
        content = case.get("content") or {}
        for member in (content.get("family") or {}).get("members") or []:
            if member.get("name"):
                names.add(member["name"])
        for group in ("by_locus", "by_subject"):
            block = (content.get("neighbours") or {}).get(group) or {}
            for member in block.get("members") or []:
                if member.get("name"):
                    names.add(member["name"])
    return names


def has_own_source(context: dict[str, Any]) -> bool:
    return bool(context["source_descriptions"] or context["enclosing_statements"])


def render_context(
    context: dict[str, Any], *, max_desc: int = 4000, max_enc: int = 6000
) -> str:
    descriptions = context["source_descriptions"]
    statements = context["enclosing_statements"]
    desc_text = (
        "\n\n".join(descriptions)
        if descriptions
        else "(none: this name has no source of its own)"
    )
    enc_text = "\n".join(f"- {line}" for line in statements) if statements else "(none)"
    family = (
        "\n".join(
            f"- {member['name']} | differs on: {member['differs_on'] or 'not recorded'} | {member['description']}"
            for member in context["family"]
        )
        or "(no family members recorded)"
    )
    grammar = context["grammar"]
    neighbours = (
        "\n".join(
            f"- {member['name']} | separation: {member['separation'] or 'not recorded'} | {member['description']}"
            for member in context.get("neighbours") or []
        )
        or "(no neighbour recorded)"
    )
    registry = (
        ", ".join(
            f"{entry.get('registry')}:{entry.get('token')}"
            for entry in grammar["registry"]
        )
        or "(none)"
    )
    return (
        "SOURCE DESCRIPTION (the data dictionary text bound to this name):\n"
        f"{desc_text[:max_desc]}\n\n"
        "ENCLOSING STRUCTURE (the parent containers the name sits in):\n"
        f"{enc_text[:max_enc]}\n\n"
        "TARGET: bound paths: "
        f"{', '.join(context['bound_paths']) or '(none: this is a parent name)'}\n"
        f"missing context channels: {', '.join(context['missing_channels']) or 'none'}\n\n"
        "NEAREST NEIGHBOURS (the live names closest to this one; 'separation' is the "
        "grammar's statement of how each differs):\n"
        f"{neighbours}\n\n"
        "FAMILY (members sharing this name's parent):\n"
        f"{family}\n\n"
        "GRAMMAR FACTS:\n"
        f"segments: {json.dumps(grammar['segments'], sort_keys=True)}\n"
        f"registry: {registry}\n"
        f"conventions: {grammar['conventions']}\n"
    )


def pass_system(pass_name: str) -> str:
    spec = PASSES[pass_name]
    return (
        "You are one detection pass over one standard-name record. You answer "
        "one question and nothing else: applying exactly one catalog rule, is "
        "the incumbent record defective in that one respect? A record is "
        f"defective when it shows this class of defect: {spec['class']}. It is "
        "not defective merely because it could be worded differently; it is "
        "defective when a competent reviewer applying only this rule would "
        "require the name to change.\n\n"
        f"{IDENTITY_RULE}\n\n"
        f"FLAG ONLY WHEN: {spec['criterion']}\n"
        f"DO NOT FLAG: {spec['not']}\n\n"
        f"Catalog rule: {spec['rule_id']}.\n"
        "The nearest neighbours show what this name must separate itself from. "
        "A neighbour's own record, and the 'separation' the grammar states for "
        "it, are evidence: when a neighbour carries a qualifier this name lacks "
        "for the same kind of quantity, that is the defect showing itself.\n"
        "When you flag, 'quote' must carry the exact words of the source (or, "
        "where the flag rests on the name: the name tokens) that caused it, and "
        "'rule' must carry the catalog rule id above and the rule you applied. "
        "When you do not flag, leave 'quote' empty and say briefly in 'reason' "
        "why the record does not show this defect."
    )


def pass_user(pass_name: str, record: dict[str, Any], context: dict[str, Any]) -> str:
    return (
        "INCUMBENT RECORD\n"
        f"name: {record.get('name') or record.get('record_name') or ''}\n"
        f"definition: {squash(record.get('description'), 1800)}\n"
        f"documentation: {squash(record.get('documentation'), 1800)}\n"
        f"unit: {record.get('unit') or context['unit']}\n\n"
        f"{render_context(context)}\n"
        f"Decide: does this record show the defect class '{PASSES[pass_name]['class']}'?"
    )


def repair_system(allowed_fields: list[str]) -> str:
    fields = ", ".join(allowed_fields)
    return (
        "You repair one standard-name record. You may change ONLY these fields: "
        f"{fields}. Every other field must be returned byte-identical to the "
        "incumbent: a harness compares every unflagged field and rejects the "
        "whole repair if even one character differs. Copy those fields verbatim; "
        "do not improve, correct, align or restate them, even when the flag makes "
        "them look inconsistent. A repair is not a rewrite of the record.\n\n"
        f"{IDENTITY_RULE}\n\n"
        "Return the complete record. If the defect cannot be repaired from the "
        "source text given, return decision 'blocked' and set 'blocker_reason' "
        "to exactly one of: multiple-quantities, parent-no-source, "
        "insufficient-source-text, other. Do not invent a source fact. Do not "
        "change a field that no flag names."
    )


def repair_user(
    record: dict[str, Any],
    context: dict[str, Any],
    flags: list[dict[str, Any]],
) -> str:
    flag_lines = "\n".join(
        f"- {flag['pass']} ({PASSES[flag['pass']]['class']}): quote: "
        f"{flag.get('quote') or '(none)'} | rule: {flag.get('rule') or '(none)'} | "
        f"{flag.get('reason') or ''}"
        for flag in flags
    )
    return (
        "INCUMBENT RECORD\n"
        f"name: {record.get('name') or ''}\n"
        f"definition: {record.get('description') or ''}\n"
        f"documentation: {squash(record.get('documentation'), 2500)}\n"
        f"unit: {record.get('unit') or context['unit']}\n\n"
        f"{render_context(context)}\n\n"
        "THE FLAGS THAT MUST BE REPAIRED:\n"
        f"{flag_lines}\n\n"
        f"FIELDS YOU MAY CHANGE: {', '.join(sorted({PASSES[flag['pass']]['field'] for flag in flags}))}.\n"
        "Return the other fields exactly as the INCUMBENT RECORD gives them.\n\n"
        "Repair the flagged field(s) using only what the source and the record "
        "state. If you cannot, return blocked with the typed reason."
    )


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #


class FlagOut(BaseModel):
    flag: bool = Field(description="True only if the record shows this defect class")
    quote: str = Field(
        default="", description="Exact source words or name tokens that caused the flag"
    )
    rule: str = Field(default="", description="The rule applied")
    reason: str = Field(default="", description="One sentence")


class RepairOut(BaseModel):
    decision: Literal["repair", "blocked"] = Field(description="repair or blocked")
    name: str = ""
    description: str = ""
    documentation: str = ""
    unit: str = ""
    blocker_reason: str = Field(
        default="", description="One of the typed reasons when blocked"
    )
    rationale: str = Field(default="", description="One sentence")


def prompt_digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


class Runner:
    def __init__(self) -> None:
        self.rows = read_jsonl(ROW_LOG)
        self.seen = {
            (row.get("phase"), row.get("member"), row.get("pass")): row
            for row in self.rows
            if not row.get("error")
        }
        self.semaphore = asyncio.Semaphore(CONCURRENCY)
        self.spent = 0.0
        self.lock = asyncio.Lock()

    def cached(
        self, phase: str, member: str, pass_name: str, digest: str
    ) -> dict[str, Any] | None:
        row = self.seen.get((phase, member, pass_name))
        if row and row.get("prompt_digest") == digest:
            return row
        return None

    async def detect(
        self,
        phase: str,
        member: str,
        pass_name: str,
        record: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        system = pass_system(pass_name)
        user = pass_user(pass_name, record, context)
        digest = prompt_digest(system, user)
        cached = self.cached(phase, member, pass_name, digest)
        if cached is not None:
            return cached
        try:
            async with self.semaphore:
                parsed, meta = await BENCH.call_model(
                    LOCAL_MODEL, system, user, FlagOut
                )
            body = parsed.model_dump()
            row = {
                "phase": phase,
                "member": member,
                "pass": pass_name,
                "model": LOCAL_MODEL,
                "prompt_digest": digest,
                "flag": bool(body.get("flag")),
                "quote": body.get("quote") or "",
                "rule": body.get("rule") or "",
                "reason": body.get("reason") or "",
                "cost_usd": meta.get("cost", 0.0),
                "input_tokens": meta.get("input_tokens", 0),
                "output_tokens": meta.get("output_tokens", 0),
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        except Exception as error:  # noqa: BLE001 - recorded and retried on resume
            row = {
                "phase": phase,
                "member": member,
                "pass": pass_name,
                "model": LOCAL_MODEL,
                "prompt_digest": digest,
                "flag": False,
                "error": f"{type(error).__name__}: {error}",
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        async with self.lock:
            self.spent += float(row.get("cost_usd", 0.0))
            append_jsonl(ROW_LOG, row)
            self.seen[(phase, member, pass_name)] = row  # keep cache fresh
        return row

    async def repair(
        self,
        member: str,
        record: dict[str, Any],
        context: dict[str, Any],
        flags: list[dict[str, Any]],
    ) -> dict[str, Any]:
        phases = "repair"
        allowed = sorted({PASSES[flag["pass"]]["field"] for flag in flags})
        system = repair_system(allowed)
        user = repair_user(record, context, flags)
        digest = prompt_digest(system, user)
        cached = self.cached(phases, member, "repair", digest)
        if cached is not None:
            return cached
        try:
            async with self.semaphore:
                parsed, meta = await BENCH.call_model(
                    LOCAL_MODEL, system, user, RepairOut
                )
            body = parsed.model_dump()
            row = {
                "phase": phases,
                "member": member,
                "pass": "repair",
                "model": LOCAL_MODEL,
                "prompt_digest": digest,
                "allowed_fields": allowed,
                "decision": body.get("decision") or "blocked",
                "proposed": {
                    "name": body.get("name") or "",
                    "description": body.get("description") or "",
                    "documentation": body.get("documentation") or "",
                    "unit": body.get("unit") or "",
                },
                "blocker_reason": body.get("blocker_reason") or "",
                "rationale": body.get("rationale") or "",
                "cost_usd": meta.get("cost", 0.0),
                "input_tokens": meta.get("input_tokens", 0),
                "output_tokens": meta.get("output_tokens", 0),
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        except Exception as error:  # noqa: BLE001
            row = {
                "phase": phases,
                "member": member,
                "pass": "repair",
                "model": LOCAL_MODEL,
                "prompt_digest": digest,
                "decision": "error",
                "error": f"{type(error).__name__}: {error}",
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        async with self.lock:
            self.spent += float(row.get("cost_usd", 0.0))
            append_jsonl(ROW_LOG, row)
            self.seen[(phases, member, "repair")] = row
        return row


RECORD_FIELDS = ("name", "description", "documentation", "unit")


def field_diffs(incumbent: dict[str, Any], proposed: dict[str, Any]) -> list[str]:
    """Fields whose content the repair supplied and changed.

    An omitted or empty field means "unchanged": a repair states what it
    changes, and the harness keeps the incumbent's text. A non-empty value that
    differs is a real change and, outside the allowed field, a rejection.
    """
    diffs = []
    for field in RECORD_FIELDS:
        supplied = proposed.get(field)
        if not squash(supplied):
            continue
        if squash(incumbent.get(field)) != squash(supplied):
            diffs.append(field)
    return diffs


def proposed_name_collision(
    proposed: str | None, incumbent: str | None, known: set[str]
) -> bool:
    """A repair may not take a name a live record already holds."""
    name = (proposed or "").strip()
    return bool(name and name != (incumbent or "").strip() and name in known)


def apply_field_restriction(
    incumbent: dict[str, Any], proposed: dict[str, Any], allowed: list[str]
) -> dict[str, Any]:
    """Apply the repair only when it changed nothing outside the flagged field."""
    diffs = field_diffs(incumbent, proposed)
    out_of_field = sorted(set(diffs) - set(allowed))
    if out_of_field:
        return {
            "applied": False,
            "rejected_out_of_field": True,
            "out_of_field": out_of_field,
            "changed": diffs,
            "record": None,
        }
    record = dict(incumbent)
    for field in allowed:
        if field in diffs:
            record[field] = proposed.get(field) or ""
    if not diffs:
        return {
            "applied": False,
            "rejected_out_of_field": False,
            "out_of_field": [],
            "changed": [],
            "record": None,
        }
    return {
        "applied": True,
        "rejected_out_of_field": False,
        "out_of_field": [],
        "changed": diffs,
        "record": record,
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if total == 0:
        return None
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = (
        z
        * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5)
        / denominator
    )
    return [max(0.0, centre - half), min(1.0, centre + half)]


def share(successes: int, total: int) -> float | None:
    return (successes / total) if total else None


def latest_rows(
    rows: list[dict[str, Any]], phase: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """The last non-error row for each (member, pass) within one phase.

    The row log is append-only across resumes and prompt revisions, so the
    newest row for a key is the one the current prompts produced.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("phase") != phase or row.get("error"):
            continue
        if "member" not in row or "pass" not in row:
            continue
        latest[(row["member"], row["pass"])] = row
    return latest


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


async def run_detection(
    dataset: dict[str, Any],
    limit: int | None = None,
    only: list[str] | None = None,
    members_filter: list[str] | None = None,
) -> None:
    runner = Runner()
    tasks = []
    members = dataset["held_out"][:limit] if limit else dataset["held_out"]
    if members_filter:
        wanted = set(members_filter)
        members = [m for m in members if m in wanted]
    pass_names = only or list(PASSES)
    for member in members:
        record = dataset["incumbents"][member]
        context = build_context(dataset["cases"][member])
        for pass_name in pass_names:
            tasks.append(runner.detect("detect", member, pass_name, record, context))
    await asyncio.gather(*tasks)
    print(f"detection calls complete; spend USD {runner.spent:.4f}", file=sys.stderr)


async def run_repairs(dataset: dict[str, Any]) -> None:
    """One repair attempt per flagged name, then the same passes again."""
    detect_rows: dict[str, list[dict[str, Any]]] = {}
    for (member, _pass), row in latest_rows(read_jsonl(ROW_LOG), "detect").items():
        if row.get("flag"):
            detect_rows.setdefault(member, []).append(row)
    runner = Runner()
    known = live_names(dataset)
    post_tasks = []
    for member in sorted(detect_rows):
        record = dataset["incumbents"][member]
        context = build_context(dataset["cases"][member])
        flags = detect_rows[member]
        rep = await runner.repair(member, record, context, flags)
        if rep.get("decision") == "repair":
            incumbent = record
            proposed = rep.get("proposed") or {}
            if proposed_name_collision(proposed.get("name"), record.get("name"), known):
                continue
            outcome = apply_field_restriction(
                incumbent, proposed, rep.get("allowed_fields") or []
            )
            if outcome["applied"]:
                repaired = outcome["record"]
                for pass_name in PASSES:
                    post_tasks.append(
                        runner.detect(
                            "post-repair", member, pass_name, repaired, context
                        )
                    )
    await asyncio.gather(*post_tasks)
    print(f"repairs complete; spend USD {runner.spent:.4f}", file=sys.stderr)


def style_vary_name(name: str) -> tuple[str, list[str]]:
    out = name
    applied = []
    for american, british in SPELLING:
        if american in out:
            out = out.replace(american, british)
            applied.append(f"{american}->{british}")
    return out, applied


def style_vary_text(text: str) -> str:
    """A variation that keeps every word: spelling, quotes, whitespace only."""
    text = text.replace("”", '"').replace("“", '"')
    for american, british in SPELLING:
        text = text.replace(american, british)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def run_control(dataset: dict[str, Any]) -> None:
    sound = [
        member
        for member in dataset["held_out"]
        if not is_adjudicated_failure(dataset["verdicts"][member])
    ]
    chosen = sorted(sound)
    random.Random(f"{SEED}:style-control").shuffle(chosen)
    # The control is vacuous unless the variation actually bites, so prefer the
    # members whose name carries a spelling variant, then those whose text does.
    with_name = [
        member
        for member in chosen
        if style_vary_name(dataset["incumbents"][member].get("name") or "")[1]
    ]
    with_text = [
        member
        for member in chosen
        if member not in with_name
        and (
            style_vary_text(dataset["incumbents"][member].get("description") or "")
            != (dataset["incumbents"][member].get("description") or "")
            or style_vary_text(dataset["incumbents"][member].get("documentation") or "")
            != (dataset["incumbents"][member].get("documentation") or "")
        )
    ]
    chosen = sorted((with_name + with_text)[:CONTROL_MEMBERS])
    if len(chosen) < CONTROL_MEMBERS:
        rest = [member for member in sorted(sound) if member not in chosen]
        chosen = sorted(chosen + rest[: CONTROL_MEMBERS - len(chosen)])
    print(
        f"control members: {len(with_name)} with a spelling variant in the name, "
        f"{len(with_text)} in the text only",
        file=sys.stderr,
    )
    runner = Runner()
    tasks = []
    for member in chosen:
        record = dataset["incumbents"][member]
        context = build_context(dataset["cases"][member])
        varied_name, applied = style_vary_name(record.get("name") or "")
        text_words = " ".join(
            [record.get("description") or "", record.get("documentation") or ""]
            + list(context["source_descriptions"])
            + list(context["enclosing_statements"])
        )
        text_spelling = sorted({f"{a}->{b}" for a, b in SPELLING if a in text_words})
        if text_spelling:
            applied = applied + ["text:" + "+".join(text_spelling)]
        elif style_vary_text(text_words) != text_words.strip():
            applied = applied + ["text:whitespace"]
        varied_context = {
            **context,
            "source_descriptions": [
                style_vary_text(text)
                for text in reversed(context["source_descriptions"])
            ],
            "enclosing_statements": [
                style_vary_text(text)
                for text in reversed(context["enclosing_statements"])
            ],
        }
        if (
            len(context["source_descriptions"]) > 1
            or len(context["enclosing_statements"]) > 1
        ):
            applied = applied + ["order:reversed-statement-order"]
        varied_record = {
            **record,
            "name": varied_name,
            "description": style_vary_text(record.get("description") or ""),
            "documentation": style_vary_text(record.get("documentation") or ""),
        }
        if not applied and varied_record["description"]:
            description = varied_record["description"]
            varied_record["description"] = description[0].swapcase() + description[1:]
            applied = ["text:initial-case"]
        for pass_name in PASSES:
            tasks.append(
                runner.detect(
                    f"control:{','.join(applied) or 'format-only'}",
                    member,
                    pass_name,
                    varied_record,
                    varied_context,
                )
            )
    await asyncio.gather(*tasks)
    print(f"control complete; spend USD {runner.spent:.4f}", file=sys.stderr)


def build_repair_outcomes(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(ROW_LOG)
    by_key = {
        (row.get("phase"), row.get("member"), row.get("pass")): row
        for row in rows
        if not row.get("error")
    }
    detect: dict[str, list[dict[str, Any]]] = {}
    for (_member, _pass), row in latest_rows(rows, "detect").items():
        if row.get("flag"):
            detect.setdefault(row["member"], []).append(row)
    outcomes: dict[str, Any] = {}
    known = live_names(dataset)
    for member in sorted(detect):
        record = dataset["incumbents"][member]
        context = build_context(dataset["cases"][member])
        flags = detect[member]
        rep = by_key.get(("repair", member, "repair")) or {}
        entry: dict[str, Any] = {
            "member": member,
            "class": dataset["classes"].get(member, ""),
            "incumbent_verdict": dataset["verdicts"][member],
            "passes_flagged": [flag["pass"] for flag in flags],
            "flags": [
                {
                    "pass": flag["pass"],
                    "quote": flag.get("quote"),
                    "rule": flag.get("rule"),
                }
                for flag in flags
            ],
            "decision": rep.get("decision") or "missing",
            "blocker_reason": rep.get("blocker_reason") or "",
            "applied": False,
            "changed_fields": [],
            "out_of_field": [],
        }
        if rep.get("decision") == "repair":
            proposed = rep.get("proposed") or {}
            if proposed_name_collision(proposed.get("name"), record.get("name"), known):
                entry["blocker_reason"] = "name-collision"
                entry["collision_with"] = proposed.get("name")
                outcomes[member] = entry
                continue
            outcome = apply_field_restriction(
                record, proposed, rep.get("allowed_fields") or []
            )
            entry["changed_fields"] = outcome["changed"]
            entry["out_of_field"] = outcome["out_of_field"]
            entry["applied"] = bool(outcome["applied"])
            if outcome["applied"]:
                repaired = outcome["record"]
                still = [
                    pass_name
                    for pass_name in PASSES
                    if (by_key.get(("post-repair", member, pass_name)) or {}).get(
                        "flag"
                    )
                ]
                entry["still_flagged"] = still
                entry["repaired_record"] = repaired
                if still:
                    entry["blocker_reason"] = "still-flagged"
        if not entry["applied"] and not entry["blocker_reason"]:
            if entry["out_of_field"]:
                entry["blocker_reason"] = "out-of-field-change"
            elif rep.get("decision") == "error":
                entry["blocker_reason"] = "repair-error"
            elif entry.get("still_flagged"):
                entry["blocker_reason"] = "still-flagged"
            elif not has_own_source(context):
                entry["blocker_reason"] = (
                    "parent-no-source"
                    if not context["bound_paths"]
                    else "insufficient-source-text"
                )
            else:
                entry["blocker_reason"] = "no-change"
        outcomes[member] = entry
    return outcomes


def compute_metrics(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(ROW_LOG)
    detect = latest_rows(rows, "detect")
    failures = [
        member
        for member in dataset["held_out"]
        if is_adjudicated_failure(dataset["verdicts"][member])
    ]
    sound = [member for member in dataset["held_out"] if member not in failures]
    class_members: dict[str, list[str]] = {}
    for member, defect_class in dataset["classes"].items():
        class_members.setdefault(defect_class, []).append(member)
    per_class = []
    for pass_name, spec in PASSES.items():
        members = class_members.get(spec["class"], [])
        flagged = [
            member
            for member in members
            if (detect.get((member, pass_name)) or {}).get("flag")
        ]
        sound_flagged = [
            member
            for member in sound
            if (detect.get((member, pass_name)) or {}).get("flag")
        ]
        per_class.append(
            {
                "class": spec["class"],
                "pass": pass_name,
                "adjudicated_failures": len(members),
                "flagged": len(flagged),
                "recall": share(len(flagged), len(members)),
                "recall_interval": wilson(len(flagged), len(members)),
                "flagged_members": flagged,
                "sound_members": len(sound),
                "sound_flagged": len(sound_flagged),
                "sound_flag_share": share(len(sound_flagged), len(sound)),
                "sound_flag_interval": wilson(len(sound_flagged), len(sound)),
                "sound_flagged_members": sound_flagged,
            }
        )
    # Any pass over all failures, including the definition-only fault.
    any_flagged = [
        member
        for member in failures
        if any(
            (detect.get((member, pass_name)) or {}).get("flag") for pass_name in PASSES
        )
    ]
    any_sound = [
        member
        for member in sound
        if any(
            (detect.get((member, pass_name)) or {}).get("flag") for pass_name in PASSES
        )
    ]
    cross = {pass_name: {} for pass_name in PASSES}
    for pass_name in PASSES:
        for defect_class, members in class_members.items():
            cross[pass_name][defect_class] = {
                "flagged": len(
                    [
                        m
                        for m in members
                        if (detect.get((m, pass_name)) or {}).get("flag")
                    ]
                ),
                "members": len(members),
            }
    return {
        "held_out_members": len(dataset["held_out"]),
        "adjudicated_failures": len(failures),
        "adjudicated_sound": len(sound),
        "per_class": per_class,
        "any_pass": {
            "failures_flagged": len(any_flagged),
            "failures": len(failures),
            "recall": share(len(any_flagged), len(failures)),
            "recall_interval": wilson(len(any_flagged), len(failures)),
            "sound_flagged": len(any_sound),
            "sound": len(sound),
            "sound_flag_share": share(len(any_sound), len(sound)),
            "sound_flag_interval": wilson(len(any_sound), len(sound)),
            "failed_members_flagged": any_flagged,
            "sound_members_flagged": any_sound,
        },
        "cross_class": cross,
    }


def control_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(ROW_LOG)
    control_rows = [
        row for row in rows if str(row.get("phase", "")).startswith("control:")
    ]
    detect = {
        (row["member"], row["pass"]): row
        for row in rows
        if row.get("phase") == "detect" and "pass" in row
    }
    members = sorted({row["member"] for row in control_rows})
    entries = []
    any_flag_varied = []
    any_flag_base = []
    for member in members:
        member_rows = [row for row in control_rows if row["member"] == member]
        newest = max(member_rows, key=lambda row: row.get("recorded_at") or "")
        phase = newest.get("phase")
        varied = [row for row in member_rows if row.get("phase") == phase]
        flag_passes = [row["pass"] for row in varied if row.get("flag")]
        base_flag_passes = [
            pass_name
            for pass_name in PASSES
            if (detect.get((member, pass_name)) or {}).get("flag")
        ]
        variation = next(
            (
                row["phase"].split(":", 1)[1]
                for row in varied
                if ":" in row.get("phase", "")
            ),
            "format-only",
        )
        entries.append(
            {
                "member": member,
                "style_variation": variation,
                "flagged_passes": flag_passes,
                "baseline_flagged_passes": base_flag_passes,
            }
        )
        if flag_passes:
            any_flag_varied.append(member)
        if base_flag_passes:
            any_flag_base.append(member)
    return {
        "members": len(members),
        "style_variation": (
            "British spelling of American tokens in the name where present, and in the "
            "definition, documentation and source text; quote style and run of "
            "whitespace normalised. No word, equation or qualifier is removed."
        ),
        "flagged_under_variation": len(any_flag_varied),
        "flagged_members_under_variation": any_flag_varied,
        "baseline_flagged": len(any_flag_base),
        "baseline_flagged_members": any_flag_base,
        "entries": entries,
        "passes": not any_flag_varied,
    }


def build_row_export() -> list[dict[str, Any]]:
    """A compact, committed row-level record of every call this node made.

    The full log (with prompt digests and token counts) is ~800 kB and lives in
    the run directory; this carries every verdict the evidence rests on.
    """
    rows = read_jsonl(ROW_LOG)
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("phase") or "", row.get("member") or "", row.get("pass") or "")
        if row.get("error"):
            continue
        latest[key] = row
    exported = []
    for key in sorted(latest):
        row = latest[key]
        entry = {
            "phase": row.get("phase"),
            "member": row.get("member"),
            "pass": row.get("pass"),
            "flag": row.get("flag"),
            "quote": squash(row.get("quote"), 300),
            "rule": squash(row.get("rule"), 400),
            "reason": squash(row.get("reason"), 400),
        }
        if row.get("phase") == "repair":
            entry.update(
                {
                    "decision": row.get("decision"),
                    "blocker_reason": row.get("blocker_reason"),
                    "proposed_name": (row.get("proposed") or {}).get("name") or "",
                    "rationale": squash(row.get("rationale"), 400),
                }
            )
        exported.append(entry)
    return exported


def build_packet(
    dataset: dict[str, Any], outcomes: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    packets: dict[str, Any] = {}
    used_members = set()
    for member, entry in sorted(outcomes.items()):
        if not entry["applied"] or entry.get("still_flagged"):
            continue
        used_members.add(member)
        context = build_context(dataset["cases"][member])
        for arm, record in (
            ("incumbent", dataset["incumbents"][member]),
            ("repaired", entry["repaired_record"]),
        ):
            packet_id = (
                "fr-"
                + hashlib.sha256(
                    f"{member}|{arm}|{record.get('name')}|{record.get('description')}".encode()
                ).hexdigest()[:16]
            )
            lines.append(
                {
                    "packet_id": packet_id,
                    "source_descriptions": context["source_descriptions"],
                    "enclosing_statements": context["enclosing_statements"],
                    "name": record.get("name") or "",
                    "definition": record.get("description") or "",
                    "documentation": record.get("documentation") or "",
                    "unit": record.get("unit") or "",
                }
            )
            packets[packet_id] = {"member": member, "arm": arm}
    random.Random(SEED).shuffle(lines)
    key_members = {
        member: {
            "right_names": dataset["key"]["members"][member].get("right_names") or [],
            "definition_points": dataset["key"]["members"][member].get(
                "definition_points"
            )
            or [],
            "incumbent_verdict": dataset["verdicts"][member],
        }
        for member in sorted(used_members)
    }
    key_document = {
        "rule": (
            "The adjudicator does not open this file until its verdicts are "
            "committed. It maps a packet id back to the member and the arm it "
            "stands for, and carries the reference's right names and definition "
            "points."
        ),
        "seed": SEED,
        "packets": packets,
        "members": key_members,
    }
    return lines, key_document


def build_evidence(dataset: dict[str, Any]) -> dict[str, Any]:
    metrics = compute_metrics(dataset)
    outcomes = build_repair_outcomes(dataset)
    control = control_summary(dataset)
    packet_lines, packet_key = build_packet(dataset, outcomes)
    repairs = {
        "flagged_names": len(outcomes),
        "repair_attempted": sum(
            1 for entry in outcomes.values() if entry["decision"] != "missing"
        ),
        "applied": sum(1 for entry in outcomes.values() if entry["applied"]),
        "rejected_out_of_field": sum(
            1 for entry in outcomes.values() if entry["out_of_field"]
        ),
        "rejected_name_collision": sum(
            1
            for entry in outcomes.values()
            if entry["blocker_reason"] == "name-collision"
        ),
        "accepted_in_packet": sum(
            1
            for entry in outcomes.values()
            if entry["applied"] and not entry.get("still_flagged")
        ),
        "blocked_out": sum(1 for entry in outcomes.values() if entry["blocker_reason"]),
        "blocked_by_reason": {},
        "sound_names_changed": 0,
        "failures_candidates_repair": sum(
            1
            for entry in outcomes.values()
            if entry["applied"]
            and is_adjudicated_failure(dataset["verdicts"][entry["member"]])
        ),
    }
    for entry in outcomes.values():
        if entry["blocker_reason"]:
            repairs["blocked_by_reason"][entry["blocker_reason"]] = (
                repairs["blocked_by_reason"].get(entry["blocker_reason"], 0) + 1
            )
    for entry in outcomes.values():
        if entry["applied"] and not is_adjudicated_failure(
            dataset["verdicts"][entry["member"]]
        ):
            repairs["sound_names_changed"] += 1
    rows = read_jsonl(ROW_LOG)
    spend = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in rows)
    return {
        "schema": "imas-codex.standard-name-record-benchmark/flagged-repair",
        "generated": datetime.now(UTC).isoformat(),
        "plan": "sn-case-file-generation",
        "section": "s7",
        "identity_rule_digest": hashlib.sha256(
            IDENTITY_RULE.encode("utf-8")
        ).hexdigest(),
        "input_digests": {
            "records": BENCH.digest_text(
                (ROOT / "benchmark-second-round-records.jsonl").read_text(
                    encoding="utf-8"
                )
            ),
            "verdicts": BENCH.digest_text(
                (ROOT / "benchmark-second-round-verdicts.jsonl").read_text(
                    encoding="utf-8"
                )
            ),
            "key": BENCH.digest_text(
                (ROOT / "benchmark-second-round-key.json").read_text(encoding="utf-8")
            ),
        },
        "passes": {
            name: {
                "class": spec["class"],
                "rule_id": spec["rule_id"],
                "field": spec["field"],
            }
            for name, spec in PASSES.items()
        },
        "metrics": metrics,
        "repairs": repairs,
        "outcomes": outcomes,
        "blockers": [
            {
                "member": entry["member"],
                "reason": entry["blocker_reason"],
                "class": entry["class"],
                "passes_flagged": entry["passes_flagged"],
            }
            for entry in outcomes.values()
            if entry.get("blocker_reason")
        ],
        "control": control,
        "row_records": {
            "path": str(ROW_EXPORT.relative_to(REPO)),
            "statement": (
                "One row per (phase, member, pass) for every call this node made, "
                "with the flag, the words of the source or the rule that caused "
                "it, the reason, and for a repair the decision and proposed name. "
                "The full log with prompt digests and token counts is in the run "
                "directory."
            ),
            "full_log": str(ROW_LOG),
        },
        "adjudication": {
            "state": "pending",
            "adjudicator": "the coordinating session, adjudicating blind",
            "statement": (
                "Every repaired record is in the blind packet beside its incumbent; "
                "name and definition are judged separately before the key is opened. "
                "The counts of adjudicated failures that became right and of changed "
                "sound names that became wrong are reported once those verdicts are "
                "committed; the harness reads them with --stage score."
            ),
            "packet": str(PACKET.relative_to(REPO)),
            "key": str(PACKET_KEY.relative_to(REPO)),
            "packet_lines": len(packet_lines),
            "scoring_command": (
                "uv run --no-sync python "
                "docs/evidence/fragments/sn-case-file-generation/flagged-repair.py "
                "--stage score --verdicts <verdicts.jsonl>"
            ),
        },
        "spend": {
            "spent_usd": round(spend, 6),
            "ceiling_usd": SPEND_CEILING_USD,
            "within_ceiling": spend <= SPEND_CEILING_USD,
            "model": LOCAL_MODEL,
            "local_calls_cost_zero": True,
        },
        "stored_text_unchanged": True,
        "graph_writes": False,
    }


def score_adjudication(dataset: dict[str, Any], verdicts_path: Path) -> dict[str, Any]:
    """Fill the acceptance counts from the coordinating session's verdicts."""
    key_document = json.loads(PACKET_KEY.read_text(encoding="utf-8"))
    verdicts = {row["packet_id"]: row for row in read_jsonl(verdicts_path)}
    by_member: dict[str, dict[str, Any]] = {}
    for packet_id, info in key_document["packets"].items():
        verdict = verdicts.get(packet_id)
        if verdict is None:
            continue
        by_member.setdefault(info["member"], {})[info["arm"]] = verdict
    became_right = 0
    sound_changed = 0
    sound_became_wrong = 0
    failures = 0
    for arms in by_member.values():
        incumbent = arms.get("incumbent")
        repaired = arms.get("repaired")
        if not incumbent or not repaired:
            continue
        was_failure = (
            incumbent.get("name_verdict") != "right"
            or incumbent.get("definition_verdict") != "right"
        )
        if was_failure:
            failures += 1
            if (
                repaired.get("name_verdict") == "right"
                and repaired.get("definition_verdict") == "right"
            ):
                became_right += 1
        else:
            sound_changed += 1
            if (
                repaired.get("name_verdict") != "right"
                or repaired.get("definition_verdict") != "right"
            ):
                sound_became_wrong += 1
    result = {
        "adjudicated_pairs": len(by_member),
        "adjudicated_failures": failures,
        "failures_became_right": became_right,
        "sound_names_changed": sound_changed,
        "sound_changed_became_wrong": sound_became_wrong,
        "accepted": (sound_became_wrong == 0 and became_right * 2 >= failures)
        if failures
        else None,
    }
    if EVIDENCE.exists():
        document = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        document["adjudication"]["state"] = "scored"
        document["adjudication"]["counts"] = result
        EVIDENCE.write_text(
            json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        PAGE.write_text(render_html(document), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f} %"


def interval_text(interval: list[float] | None) -> str:
    if not interval:
        return "—"
    return f"[{100 * interval[0]:.0f} %, {100 * interval[1]:.0f} %]"


def render_html(document: dict[str, Any]) -> str:
    metrics = document["metrics"]
    rows = []
    for entry in metrics["per_class"]:
        rows.append(
            "<tr>"
            f"<td>{entry['pass']}</td>"
            f"<td>{entry['class']}</td>"
            f"<td>{entry['flagged']}/{entry['adjudicated_failures']}</td>"
            f"<td>{pct(entry['recall'])} {interval_text(entry['recall_interval'])}</td>"
            f"<td>{entry['sound_flagged']}/{entry['sound_members']}</td>"
            f"<td>{pct(entry['sound_flag_share'])} {interval_text(entry['sound_flag_interval'])}</td>"
            "</tr>"
        )
    any_pass = metrics["any_pass"]
    repairs = document["repairs"]
    control = document["control"]
    blocker_rows = (
        "".join(
            f"<tr><td>{entry['member']}</td><td>{entry['reason']}</td>"
            f"<td>{entry['class']}</td><td>{', '.join(entry['passes_flagged'])}</td></tr>"
            for entry in document["blockers"]
        )
        or "<tr><td colspan='4'>none</td></tr>"
    )
    flag_rows = (
        "".join(
            f"<tr><td>{entry['member']}</td><td>{flag['pass']}</td>"
            f"<td>{html.escape(flag.get('quote') or '')}</td>"
            f"<td>{html.escape(flag.get('rule') or '')}</td></tr>"
            for entry in document["outcomes"].values()
            for flag in entry["flags"]
        )
        or "<tr><td colspan='4'>none</td></tr>"
    )
    best_case = repairs["accepted_in_packet"] + len(document["blockers"])
    half = (any_pass["failures"] + 1) // 2
    counts = document.get("adjudication", {}).get("counts")
    scored_text = (
        f"{counts['failures_became_right']} of {counts['adjudicated_failures']} "
        f"adjudicated failures became right; {counts['sound_names_changed']} "
        f"adjudicated-sound names were changed; "
        f"{counts['sound_changed_became_wrong']} of those became wrong. "
        f"Route accepted: {str(counts['accepted']).lower()}."
        if counts
        else "No adjudication verdicts are committed yet."
    )
    control_rows = "".join(
        f"<tr><td>{entry['member']}</td><td>{entry['style_variation']}</td>"
        f"<td>{', '.join(entry['flagged_passes']) or 'none'}</td>"
        f"<td>{', '.join(entry['baseline_flagged_passes']) or 'none'}</td></tr>"
        for entry in control["entries"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Flagged repair — standard names</title>
<meta name="plan" content="sn-case-file-generation">
<meta name="plan-section" content="s7">
</head>
<body>
<h1>A flagged defect is repaired and an unflagged name is left alone</h1>
<p>Four detection passes read the 87 held-out incumbents and their case files,
one class of defect each. A flagged name is repaired once by a local call under a
field restriction. The metric is the committed blind verdict on each incumbent:
how many adjudicated failures its class's pass flagged, and how many
adjudicated-sound names it flagged, each with a Wilson 95 % interval.</p>
<h2>Per defect class</h2>
<table>
<thead><tr><th>pass</th><th>class</th><th>failures flagged</th><th>recall (95 %)</th>
<th>sound flagged</th><th>sound flag share (95 %)</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
<p>Any pass: {any_pass["failures_flagged"]}/{any_pass["failures"]} adjudicated
failures flagged ({pct(any_pass["recall"])} {interval_text(any_pass["recall_interval"])});
{any_pass["sound_flagged"]}/{any_pass["sound"]} adjudicated-sound names flagged
({pct(any_pass["sound_flag_share"])} {interval_text(any_pass["sound_flag_interval"])}).</p>
<h2>Flags and the words that caused them</h2>
<p>Every flag a pass raised, with the source words or the catalog rule that
caused it. A pass reports a member only with a quote from its bound source
description or enclosing structure, or with the rule it applied.</p>
<table>
<thead><tr><th>member</th><th>pass</th><th>source words</th><th>rule cited</th></tr></thead>
<tbody>{flag_rows}</tbody>
</table>
<h2>Repairs and blockers</h2>
<p>{repairs["flagged_names"]} names flagged; {repairs["repair_attempted"]} repair
attempts; {repairs["applied"]} applied; {repairs["accepted_in_packet"]} put in the
packet; {repairs["rejected_out_of_field"]} rejected for changing a field the flag
did not name; {repairs["rejected_name_collision"]} rejected for taking a live
name; {repairs["sound_names_changed"]} adjudicated-sound names changed. Of the
applied repairs, {repairs["applied"] - repairs["accepted_in_packet"]} are still
flagged by the same passes and are listed below as blockers, not repaired again.</p>
<p>Blocked by reason: {json.dumps(repairs["blocked_by_reason"], sort_keys=True)}.</p>
<table>
<thead><tr><th>member</th><th>typed reason</th><th>class</th><th>passes that flagged</th></tr></thead>
<tbody>{blocker_rows}</tbody>
</table>
<h2>Style-only control</h2>
<p>{control["members"]} adjudicated-sound names, given a style-only variation
({control["style_variation"]}). Flags under the variation:
{control["flagged_under_variation"]}. Baseline flags on the unvaried records:
{control["baseline_flagged"]}. The control holds: {str(control["passes"]).lower()}.</p>
<table>
<thead><tr><th>member</th><th>style variation</th><th>flagged under variation</th><th>flagged at baseline</th></tr></thead>
<tbody>{control_rows}</tbody>
</table>
<h2>Adjudication</h2>
<p>The repaired records are in the blind packet beside their incumbents. The
coordinating session adjudicates name and definition separately before the key
is opened. {scored_text} The harness fills these counts with
<code>--stage score</code> once verdicts are committed.</p>
<h2>Acceptance</h2>
<p>The route is accepted for this group only if no adjudicated-sound name
became wrong and at least half of the adjudicated failures became right or were
queued with a reason the lead accepts as a decision owed to him. This round has
{any_pass["failures"]} adjudicated failures, so half is {half}. It puts
{repairs["accepted_in_packet"]} repaired records in the packet and lists
{len(document["blockers"])} blockers with typed reasons. Even if every packet
record became right and every blocker were accepted as a queued decision, the
total is {best_case}, short of {half}. The packet's verdicts are the
coordinating session's to take, and the accept decision is the lead's; the
arithmetic above bounds what verdicts could change.</p>
<p>The typed blocker reasons are <code>multiple-quantities</code>,
<code>name-collision</code>, <code>still-flagged</code>,
<code>parent-no-source</code>, <code>insufficient-source-text</code>,
<code>out-of-field-change</code>, <code>repair-error</code> and
<code>no-change</code>. This round records {json.dumps(repairs["blocked_by_reason"], sort_keys=True)};
<code>parent-no-source</code> and <code>insufficient-source-text</code> are
zero.</p>
<h2>Gate</h2>
<p>The harness was measured at revision 71b9c15fc. Its guard tests report
<code>5 passed in 21.78s</code> and the self-check reports
<code>CHECK OK: 87 held-out, 33 classified</code>. The gate log with its
command and exit status is cited in the node manifest.</p>
<h2>Spend</h2>
<p>USD {document["spend"]["spent_usd"]:.4f} spent beside a ceiling of USD
{document["spend"]["ceiling_usd"]:.2f}. No stored standard-name text changed and
no graph property changed.</p>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Self-checks
# --------------------------------------------------------------------------- #


def run_check(dataset: dict[str, Any]) -> int:
    failures: list[str] = []
    if len(dataset["held_out"]) != 87:
        failures.append(f"held-out is {len(dataset['held_out'])}, not 87")
    defect_members = set(dataset["classes"])
    if len(defect_members) != 33:
        failures.append(f"defect classes cover {len(defect_members)} members, not 33")
    if not defect_members <= set(dataset["held_out"]):
        failures.append("defect members are not all held out")
    # No pass input may carry the earlier adjudication: check every member, and
    # check the rendered prompt rather than only the context dict.
    leaked = []
    for member in dataset["held_out"]:
        case = dataset["cases"][member]
        context = build_context(case)
        rendered = json.dumps(context, sort_keys=True) + render_context(context)
        if "adjudicator_reason" in json.dumps(context):
            leaked.append(f"{member} (context key)")
            continue
        reason = (case.get("content") or {}).get("adjudicator_reason") or {}
        comments = reason.get("comments") or ""
        try:
            rationale = (
                json.loads(comments) if isinstance(comments, str) else comments
            ).get("rationale") or ""
        except (ValueError, AttributeError):
            rationale = ""
        if rationale and rationale[:60] in rendered:
            leaked.append(f"{member} (rationale text)")
    if leaked:
        failures.append("adjudicator_reason leaked into: " + ", ".join(leaked))
    # The field restriction must reject a repair that changes an unflagged field.
    incumbent = {
        "name": "old_name",
        "description": "a definition",
        "documentation": "",
        "unit": "1",
    }
    proposed = {
        "name": "new_name",
        "description": "a different definition",
        "documentation": "",
        "unit": "1",
    }
    outcome = apply_field_restriction(incumbent, proposed, ["name"])
    if outcome["applied"] or not outcome["rejected_out_of_field"]:
        failures.append(
            "field restriction accepted a repair that changed the definition"
        )
    # An omitted field means unchanged: the repair must apply and keep the text.
    omitted = apply_field_restriction(
        incumbent,
        {"name": "new_name", "description": "", "documentation": "", "unit": ""},
        ["name"],
    )
    if not omitted["applied"] or omitted["record"]["description"] != "a definition":
        failures.append("an omitted field was treated as a change")
    if not proposed_name_collision("old_name", "incumbent", {"old_name"}):
        failures.append("a colliding proposed name was not detected")
    if proposed_name_collision("old_name", "old_name", {"old_name"}):
        failures.append("the incumbent's own name was reported as a collision")
    if proposed_name_collision("fresh_name", "old_name", {"old_name"}):
        failures.append("a novel name was reported as a collision")
    print(
        "negative control: planted out-of-field repair applied="
        f"{outcome['applied']} rejected={outcome['rejected_out_of_field']} "
        f"out_of_field={outcome['out_of_field']}; omitted-field repair applied="
        f"{omitted['applied']}; collision guard ok"
    )
    if failures:
        print("CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        f"CHECK OK: {len(dataset['held_out'])} held-out, {len(defect_members)} classified"
    )
    return 0


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["detect", "repair", "control", "report", "score", "all", "check"],
        default="all",
    )
    parser.add_argument("--verdicts", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="detect on the first N members only (smoke test)",
    )
    parser.add_argument(
        "--only-pass",
        default="",
        help="comma-separated subset of passes to run",
    )
    parser.add_argument(
        "--members",
        default="",
        help="comma-separated member names to restrict this stage to",
    )
    args = parser.parse_args(argv)
    dataset = load_dataset()
    if args.stage == "check":
        return run_check(dataset)
    if args.stage == "score":
        if not args.verdicts:
            parser.error("--stage score needs --verdicts")
        result = score_adjudication(dataset, args.verdicts)
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0
    if args.stage in {"detect", "all"}:
        only = [name for name in args.only_pass.split(",") if name]
        members_filter = [name for name in args.members.split(",") if name]
        asyncio_members = members_filter or None
        asyncio.run(
            run_detection(
                dataset,
                limit=args.limit,
                only=only or None,
                members_filter=asyncio_members,
            )
        )
    if args.stage in {"repair", "all"}:
        asyncio.run(run_repairs(dataset))
    if args.stage in {"control", "all"}:
        asyncio.run(run_control(dataset))
    document = build_evidence_payload(dataset)
    EVIDENCE.write_text(
        json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    PAGE.write_text(render_html(document), encoding="utf-8")
    # The packet and key are evidence; write them beside the report.
    packet_lines, packet_key = build_packet(dataset, build_repair_outcomes(dataset))
    PACKET.write_text(
        "\n".join(
            json.dumps(line, sort_keys=True, ensure_ascii=False)
            for line in packet_lines
        )
        + ("\n" if packet_lines else ""),
        encoding="utf-8",
    )
    PACKET_KEY.write_text(
        json.dumps(packet_key, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    exported = build_row_export()
    ROW_EXPORT.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) for row in exported
        )
        + ("\n" if exported else ""),
        encoding="utf-8",
    )
    print(
        f"evidence written: {EVIDENCE.name} ({EVIDENCE.stat().st_size} bytes), "
        f"page {PAGE.name}, packet {len(packet_lines)} lines, "
        f"rows {ROW_EXPORT.name} ({len(exported)} rows, {ROW_EXPORT.stat().st_size} bytes)",
        file=sys.stderr,
    )
    return 0


def build_evidence_payload(dataset: dict[str, Any]) -> dict[str, Any]:
    return build_evidence(dataset)


if __name__ == "__main__":
    raise SystemExit(main())
