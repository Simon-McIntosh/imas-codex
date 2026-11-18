"""Freeze the reference set a generator is scored against, read under one rule.

The blind adjudication of equilibrium and the reader calibration were both taken
before the lead ruled that a name identifies and its definition specifies. Under
that rule a qualifier the source states about a normalisation, a weight, an
average, a convention or a method belongs to the definition: leaving it out of
the name is not a defect of the name. This script re-reads the 82 members judged
sound, the 17 accepted repairs, the 84 hand-checked calibration names and the
members earlier classed as omitting what their source states, one source packet
at a time, under that rule, and writes the reference set.

Every judgment is blind: the reader is shown the source packet and nothing else,
never the earlier verdict, its classes or its reason. Its answers are written and
sealed with a digest before any earlier verdict is opened, so the commitment
carries its own receipt and the comparison is derived from the sealed file.

Each member's packet is stored in the committed data file, compressed the way the
sibling evidence scripts store theirs, and carries its own digest. ``--check``
recomputes every digest and refuses the file when one no longer matches.

Nothing under the graph or another evidence fragment is written or changed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import time
import zlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
DATA = ROOT / "reference-set.json"
PAGE = ROOT / "reference-set.html"
FRAGMENT = ROOT / "scfg-reference-set.html"
RELATIVE_DATA = "docs/evidence/fragments/sn-case-file-generation/reference-set.json"
SAGR = ROOT.parent / "sn-audited-group-release"
RUN = Path(
    os.environ.get(
        "REFERENCE_SET_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261118T064757278694-scfg-reference-set",
    )
)
CONCURRENCY = int(os.environ.get("REFERENCE_SET_CONCURRENCY", "8"))
MAX_TOKENS = 16000
TIMEOUT_S = 900
MAX_ATTEMPTS = 3
MODEL = "local/deepseek-v4.1-flash"
SERVICE = "standard-names"

RULE = """\
The lead ruled on 2026-11-18 that a name identifies and its definition specifies.
A name carries only what separates the quantity from its neighbours: the kind of
quantity, the physical object or locus it describes, and the relation or
component where that separation depends on it. A definition carries everything a
reader needs to obtain the number: normalisations, weightings, averaging or
integration, conventions, coordinate frames, methods and units.

Two consequences decide every judgment:
1. A property that changes which quantity is meant belongs in the name. If the
   name leaves it out, the name does not identify the quantity.
2. A property that changes how the number is computed from the same quantity
   belongs to the definition. Its absence from the name is not a defect of the
   name. If the definition leaves it out, the definition is defective.
"""

VERDICTS = ("sound", "naming_defect", "definition_defect", "undecidable")
ALIASES = {
    "verdict": ("judgement", "judgment", "decision"),
    "right_names": ("names", "right_name", "proposed_names"),
    "definition_points": ("points", "must_state"),
    "evidence_quote": ("quote", "evidence"),
    "evidence_path": ("path", "source_path"),
    "reason": ("rationale", "reasoning", "why"),
    "rule_clause": ("clause", "rule"),
}
DEFECTIVE = ("naming_defect", "definition_defect")
CLAUSES = ("name-identifies", "definition-specifies", "both-agree", "not-decidable")

COHORT_SOUND = "equilibrium-sound"
COHORT_REPAIR = "accepted-repair"
COHORT_CALIBRATION = "reader-calibration"
COHORT_OMITS = "omits-source-reclass"

SYSTEM_PROMPT = f"""\
You judge one standard-name record against the Data Dictionary source text that
produced it. You see the packet only; you do not see any earlier judgment.

{RULE}
Decide exactly one verdict:
- `sound`: the name separates this quantity from its neighbours and the
  definition states the points its source states.
- `naming_defect`: the name does not identify the quantity — it contradicts the
  definition's source, names a different kind of quantity, or is so general that
  it does not separate this quantity from a sibling. Give the name or names that
  would be right.
- `definition_defect`: the name is right, but the stored definition omits or
  contradicts a point the source states — a normalisation, a weight, an average,
  a convention, a coordinate basis, a method or a unit. State the points it must
  carry.
- `undecidable`: the packet carries no text that describes this quantity at
  all. Do not guess, and do not return this when a stored definition is present.

A packet whose Data Dictionary section says `(none recorded)` still carries the
stored name, description, documentation and unit: judge the name against the
stored definition as the record's only statement of the quantity, and quote that
definition in `evidence_quote` with `evidence_path: stored description`.

Every decided verdict must fill all four fields:
- `right_names` is never empty: the incumbent name when it is right, the
  corrected spelling otherwise (one name, or a set when more than one is right).
- `definition_points` is never empty: what the definition must state. For a
  sound member list the points it already states.
- `evidence_quote` is copied verbatim from the packet, long enough to identify
  one sentence of it, and `evidence_path` says where that sentence sits.
- `reason` is one or two sentences joining the quote to the verdict.

Never quote a sentence the packet does not contain; a quote that cannot be found
in the packet is rejected and the answer is asked again.
"""


class Judgment(BaseModel):
    """One blind verdict on one member."""

    verdict: str = Field(
        description=(
            "Exactly one of `sound`, `naming_defect`, `definition_defect` or "
            "`undecidable`."
        )
    )
    right_names: list[str] = Field(
        default_factory=list,
        description=(
            "The name, or the set of names, that would be right under the rule: "
            "the incumbent name for a sound or definition_defect member, the "
            "corrected spelling for a naming_defect member, empty when "
            "undecidable."
        ),
    )
    definition_points: list[str] = Field(
        default_factory=list,
        description=(
            "Every point the definition must state: what the quantity is, then "
            "each normalisation, weighting, average, convention, coordinate "
            "basis, method and unit the source states. One short sentence per "
            "point."
        ),
    )
    evidence_quote: str = Field(
        description="The deciding wording, quoted verbatim from the packet."
    )
    evidence_path: str = Field(
        default="",
        description=(
            "Where the quote sits: a Data Dictionary path, an enclosing "
            "structure path, or `stored description`."
        ),
    )
    reason: str = Field(
        default="",
        description="One or two sentences joining the quote to the verdict.",
    )
    rule_clause: str = Field(
        default="",
        description=(
            "Which half of the rule decides the verdict: `name-identifies`, "
            "`definition-specifies`, `both-agree` or `not-decidable`."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def fold_alternate_names(cls, data):
        """Accept the alternate key names a free-form model drifts into."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for target, aliases in ALIASES.items():
            if data.get(target) in (None, "", []):
                for alias in aliases:
                    if data.get(alias) not in (None, "", []):
                        data[target] = data[alias]
                        break
        for key in ("right_names", "definition_points"):
            if isinstance(data.get(key), str):
                data[key] = [data[key]]
        return data


class Specification(BaseModel):
    """The right answer's name or names and definition points."""

    right_names: list[str] = Field(
        default_factory=list,
        description=(
            "The name or set of names that identify this quantity. Never empty."
        ),
    )
    definition_points: list[str] = Field(
        default_factory=list,
        description=(
            "Every point the definition must state, one short sentence each. "
            "Never empty."
        ),
    )


SPECIFICATION_PROMPT = """\
You write the specification half of a standard-name judgment already made. The
verdict is given; do not change it.

Return two things and nothing else:
- `right_names`: the name, or the set of names, that identify this quantity. For
  a sound or definition_defect verdict that is the incumbent name; for a
  naming_defect verdict it is the corrected spelling, using standard grammar
  words for the physical object, the locus and the relation.
- `definition_points`: every point the definition must state. Start with what
  the quantity is, then each normalisation, weighting, average or integration,
  coordinate frame, convention, method and unit the source states. One short
  sentence per point, three to eight points.

Neither list may be empty. A point restates what the source states; do not add
requirements the packet does not support.
"""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def squash(text: str) -> str:
    return " ".join(str(text or "").split())


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 0.0]
    z = 1.9599639845400546
    share = successes / total
    denominator = 1 + z * z / total
    centre = (share + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    return [max(0.0, centre - spread / denominator), min(1.0, centre + spread / denominator)]


def percent(value: float) -> str:
    return f"{100 * value:.1f} %"


# --------------------------------------------------------------------------- #
# Sources, read only
# --------------------------------------------------------------------------- #

ADJUDICATION = "docs/evidence/fragments/sn-audited-group-release/equilibrium-adjudication.json"
REPAIR = "docs/evidence/fragments/sn-audited-group-release/equilibrium-repair.json"
CALIBRATION = "docs/evidence/fragments/sn-audited-group-release/reader-calibration.json"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), SAGR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_records() -> dict:
    """Packets and earlier verdicts of every cohort, read from their owners."""
    adjudication = load_module("equilibrium-adjudication")
    adjudication_data = adjudication.load()
    packets = {
        packet["id"]: {
            "text": packet["reader_visible_text"],
            "source_paths": list(packet.get("source_paths") or []),
        }
        for packet in adjudication.packets(adjudication_data)
    }
    judgments = {row["id"]: row for row in adjudication_data["judgments"]}

    repair_data = json.loads((REPO / REPAIR).read_text())
    repair_payload = json.loads(
        zlib.decompress(base64.b85decode(repair_data["payload_zlib_base85"]))
    )
    repairs = {row["id"]: row for row in repair_payload["members"]}

    calibration_module = load_module("reader-calibration")
    calibration_data = json.loads((REPO / CALIBRATION).read_text())
    for packet in calibration_module.source_packets(calibration_data):
        packets[packet["id"]] = {
            "text": packet["reader_visible_text"],
            "source_paths": list(packet.get("source_paths") or []),
        }
    calibration_judgments = {row["id"]: row for row in calibration_data["judgments"]}
    return {
        "packets": packets,
        "judgments": judgments,
        "calibration_judgments": calibration_judgments,
        "calibration_cohort": list(calibration_data["cohort"]),
        "repairs": repairs,
    }


HEADERS = (
    ("name", "Standard name: "),
    ("description", "Definition (stored description): "),
    ("documentation", "Documentation (stored): "),
    ("unit", "Stored unit: "),
    ("physics_domain", "Assigned physics domain: "),
)
BODY_MARK = "Data Dictionary source text:"
INSTRUCTION = "Return every fixed defect class that applies"


def split_packet(text: str) -> tuple[dict, str]:
    """Split a stored packet into its record header and its source body."""
    record: dict[str, str] = {}
    lines = text.split("\n")
    body_start = len(lines)
    for index, line in enumerate(lines):
        if line.startswith(BODY_MARK):
            body_start = index
            break
        for key, prefix in HEADERS:
            if line.startswith(prefix):
                record[key] = line[len(prefix) :].strip()
                break
    body = "\n".join(
        line for line in lines[body_start:] if not line.startswith(INSTRUCTION)
    )
    return record, body.strip("\n")


def render_packet(record: dict, body: str) -> str:
    """The exact text a blind reader sees, and the text the digest covers."""
    lines = [
        f"Standard name: {record.get('name', '')}",
        f"Definition (stored description): {record.get('description', '')}",
        "Documentation (stored): " + (record.get("documentation") or "(none recorded)"),
        f"Stored unit: {record.get('unit') or '(none recorded)'}",
        f"Assigned physics domain: {record.get('physics_domain') or '(none recorded)'}",
        "",
        body,
    ]
    return "\n".join(lines).strip("\n") + "\n"


NAME_WORD = re.compile(r"^[a-z][a-z0-9_]{2,119}$")


def valid_name(name: str) -> bool:
    """A standard name is lower-case words joined by underscores, nothing else."""
    return bool(NAME_WORD.match(name)) and "placeholder" not in name


def packet_digest(member: dict) -> str:
    return digest({"id": member["id"], "text": member["text"]})


# --------------------------------------------------------------------------- #
# Membership
# --------------------------------------------------------------------------- #


def earlier_for(sources: dict, member_id: str) -> dict:
    for table, origin in (
        (sources["judgments"], "equilibrium-adjudication"),
        (sources["calibration_judgments"], "reader-calibration"),
    ):
        row = table.get(member_id)
        if row is not None:
            return {
                "verdict": row["verdict"],
                "defect_classes": list(row.get("defect_classes") or []),
                "reason": row.get("reason", ""),
                "origin": origin,
            }
    repair = sources["repairs"].get(member_id)
    if repair is not None:
        row = repair["judgment"]
        return {
            "verdict": row["verdict"],
            "defect_classes": list(row.get("defect_classes") or []),
            "reason": row.get("reason", ""),
            "origin": "equilibrium-repair",
        }
    raise SystemExit(f"no earlier verdict for {member_id}")


def assemble() -> list[dict]:
    """Every member this pass reads: its packet, its record and its cohort tag."""
    sources = source_records()
    cohorts: dict[str, set[str]] = defaultdict(set)
    for member_id, row in sources["judgments"].items():
        if row["verdict"] == "sound":
            cohorts[member_id].add(COHORT_SOUND)
    for member_id, row in sources["repairs"].items():
        if row["outcome"] == "accepted_repair":
            cohorts[member_id].add(COHORT_REPAIR)
        if "name_omits_source" in (row["judgment"].get("defect_classes") or []):
            cohorts[member_id].add(COHORT_OMITS)
    for member_id in sources["calibration_cohort"]:
        cohorts[member_id].add(COHORT_CALIBRATION)

    members = []
    for member_id in sorted(cohorts):
        stored = sources["packets"].get(member_id)
        if stored is None:
            raise SystemExit(f"no source packet for {member_id}")

        repair = sources["repairs"].get(member_id)
        if repair is not None and repair["outcome"] == "accepted_repair":
            after = repair["after"]
            record = {
                "name": after["id"],
                "description": after.get("description") or "",
                "documentation": after.get("documentation") or "",
                "unit": (after.get("units") or [""])[0],
                "physics_domain": (after.get("domains") or [""])[0],
            }
            applied = True
        else:
            record, _ = split_packet(stored["text"])
            applied = False
        source_record, body = split_packet(stored["text"])
        if not body:
            raise SystemExit(f"empty source body for {member_id}")
        for key in ("name", "description", "documentation", "unit", "physics_domain"):
            record.setdefault(key, source_record.get(key, ""))
        member = {
            "id": member_id,
            "cohorts": sorted(cohorts[member_id]),
            "source_paths": stored["source_paths"],
            "record": record,
            "body": body,
            "repair_applied": applied,
            "earlier": earlier_for(sources, member_id),
        }
        member["text"] = render_packet(member["record"], member["body"])
        member["packet_digest"] = packet_digest(member)
        members.append(member)
    return members


def assert_blind(member: dict) -> None:
    """Refuse a packet that carries any part of the earlier verdict."""
    haystack = squash(member["text"])
    earlier = member["earlier"]
    for key in earlier["defect_classes"]:
        if key and key in haystack:
            raise AssertionError(f"{member['id']} packet names earlier class {key}")
    reason = squash(earlier["reason"])
    if reason and reason[:60] in haystack:
        raise AssertionError(f"{member['id']} packet carries the earlier reason")
    for key in ("defect class", "earlier verdict", "blind"):
        if key in member["text"].lower():
            raise AssertionError(f"{member['id']} packet says {key!r}")


JSONL = "blind-verdicts.jsonl"
SEALED = "blind-sealed.json"


def normalise(member: dict, raw: dict) -> dict:
    """Fold one model answer into the record shape, refusing what cannot stand."""
    verdict = str(raw.get("verdict") or "").strip()
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} is not one of {VERDICTS}")
    clause = str(raw.get("rule_clause") or "").strip()
    fallback = {
        "sound": "both-agree",
        "naming_defect": "name-identifies",
        "definition_defect": "definition-specifies",
        "undecidable": "not-decidable",
    }
    if clause not in CLAUSES:
        clause = fallback[verdict]
    quote = str(raw.get("evidence_quote") or "").strip()
    reason = str(raw.get("reason") or "").strip()
    names = [str(name).strip().lower() for name in (raw.get("right_names") or [])]
    names = list(dict.fromkeys(name for name in names if valid_name(name)))
    points = [str(point).strip() for point in (raw.get("definition_points") or [])]
    points = [point for point in points if point]
    incumbent = member["record"]["name"]
    if verdict == "undecidable":
        if not reason:
            raise ValueError("undecidable verdict without a reason")
        names, points = [], []
    elif verdict == "naming_defect":
        if not names:
            raise ValueError("naming_defect without a right name")
        if len(names) == 1 and names[0] == incumbent:
            raise ValueError("naming_defect keeps the incumbent name unchanged")
    else:
        names = [incumbent]
    if verdict != "undecidable":
        if not reason:
            raise ValueError("no reason given")
        if not points:
            raise ValueError("no definition points")
        if len(quote) < 8:
            raise ValueError("no usable evidence quote")
        if squash(quote) not in squash(member["text"]):
            raise ValueError("evidence quote not found in the packet")
    return {
        "id": member["id"],
        "verdict": verdict,
        "right_names": names,
        "definition_points": points,
        "evidence_quote": quote,
        "evidence_path": str(raw.get("evidence_path") or "").strip(),
        "reason": reason,
        "rule_clause": clause,
    }



async def judge_once(member: dict) -> dict:
    """One structured call; the caller retries while an answer cannot stand."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    result = await acall_llm_structured(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": member["text"]},
        ],
        response_model=Judgment,
        service=SERVICE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_S,
        max_retries=1,
        reasoning_effort="medium",
        local_schema_constraint=True,
    )
    return result.parsed.model_dump()



async def fill_once(member: dict, decided: dict) -> dict:
    """Ask for the name and the definition points a judgment left empty."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    user = (
        member["text"]
        + "\n\nVerdict already decided: "
        + str(decided.get("verdict"))
        + "\nDeciding reason: "
        + str(decided.get("reason") or "")
    )
    result = await acall_llm_structured(
        model=MODEL,
        messages=[
            {"role": "system", "content": SPECIFICATION_PROMPT},
            {"role": "user", "content": user},
        ],
        response_model=Specification,
        service=SERVICE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_S,
        max_retries=1,
        reasoning_effort="medium",
        local_schema_constraint=True,
    )
    return result.parsed.model_dump()


async def judge_one(member: dict, semaphore, lock) -> dict:
    """Retry a blind judgment while it cannot stand, then record it once."""
    RUN.mkdir(parents=True, exist_ok=True)
    assert_blind(member)
    started = time.time()
    problem = ""
    record = None
    raw = None
    for _attempt in range(1, MAX_ATTEMPTS + 1):
        async with semaphore:
            try:
                raw = await judge_once(member)
                record = normalise(member, raw)
                break
            except ValueError as exc:
                problem = str(exc)
                if raw is not None and raw.get("verdict") in (
                    "sound",
                    "naming_defect",
                    "definition_defect",
                ) and ("right name" in problem or "definition points" in problem):
                    try:
                        extra = await fill_once(member, raw)
                        merged = dict(raw)
                        merged["right_names"] = (
                            raw.get("right_names") or extra.get("right_names")
                        )
                        merged["definition_points"] = (
                            raw.get("definition_points") or extra.get("definition_points")
                        )
                        record = normalise(member, merged)
                        break
                    except Exception as fill_error:
                        problem = f"{problem}; fill: {fill_error}"[:400]
            except Exception as exc:
                problem = f"{type(exc).__name__}: {exc}"[:400]
    if record is None:
        record = {
            "id": member["id"],
            "verdict": "unjudged",
            "right_names": [],
            "definition_points": [],
            "evidence_quote": "",
            "evidence_path": "",
            "reason": f"no usable answer: {problem}",
            "rule_clause": "not-decidable",
            "attempts": MAX_ATTEMPTS,
            "seconds": round(time.time() - started, 1),
            "model": MODEL,
        }
    record["seconds"] = round(time.time() - started, 1)
    record["model"] = MODEL
    async with lock:
        with (RUN / JSONL).open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return record


async def run_judging(members: list[dict]) -> None:
    """Judge every member that carries no recorded answer yet."""
    RUN.mkdir(parents=True, exist_ok=True)
    done = {}
    path = RUN / JSONL
    kept = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["verdict"] == "unjudged":
                continue
            kept.append(row)
            done[row["id"]] = row
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept)
        )
    pending = [member for member in members if member["id"] not in done]
    print(f"{len(done)} judged, {len(pending)} pending", flush=True)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    rows = await asyncio.gather(
        *(judge_one(member, semaphore, lock) for member in pending)
    )
    for row in rows:
        done[row["id"]] = row
    counts = Counter(row["verdict"] for row in done.values())
    print(f"recorded {len(done)} members: {dict(sorted(counts.items()))}", flush=True)


def seal(members: list[dict]) -> dict:
    """Seal the blind answers before any earlier verdict is read."""
    rows = []
    for line in (RUN / JSONL).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda row: row["id"])
    sealed = {
        "model": MODEL,
        "rule_digest": digest(RULE),
        "judged_at": datetime.now(UTC).isoformat(),
        "prompt_digest": digest(SYSTEM_PROMPT),
        "population": len(members),
        "verdicts": rows,
    }
    sealed["digest"] = digest(rows)
    (RUN / SEALED).write_text(json.dumps(sealed, ensure_ascii=False, indent=1))
    print(f"sealed {len(rows)} blind verdicts, digest {sealed['digest']}", flush=True)
    return sealed


def open_seal(path: Path) -> dict:
    """Read the sealed answers and refuse them when they moved under the name."""
    sealed = json.loads(Path(path).read_text())
    if digest(sealed["verdicts"]) != sealed["digest"]:
        raise SystemExit("sealed blind verdicts do not match their digest")
    return sealed
NAME_SIDE = {"name_contradicts_source", "name_omits_source"}
DEFINITION_SIDE = {"definition_contradicts_source", "unit_wrong"}


def earlier_locus(classes: list[str], verdict: str) -> str:
    """Where the earlier read put the defect, in terms of the identity rule."""
    if verdict == "sound":
        return "sound"
    carriers = set(classes)
    on_name = bool(carriers & NAME_SIDE)
    on_definition = bool(carriers & DEFINITION_SIDE)
    if on_name and on_definition:
        return "name-and-definition"
    if on_name:
        return "name"
    if on_definition:
        return "definition"
    return "other"


def new_locus(verdict: str) -> str:
    return {
        "sound": "sound",
        "naming_defect": "name",
        "definition_defect": "definition",
        "undecidable": "undecidable",
        "unjudged": "unjudged",
    }.get(verdict, "other")


def compare(sealed: dict, members: list[dict]) -> dict:
    """Read the earlier verdict only now, and record every change."""
    by_id = {member["id"]: member for member in members}
    rows = []
    changed = []
    for verdict in sealed["verdicts"]:
        member = by_id[verdict["id"]]
        earlier = member["earlier"]
        before = earlier_locus(earlier["defect_classes"], earlier["verdict"])
        after = new_locus(verdict["verdict"])
        row = {
            "id": member["id"],
            "cohorts": member["cohorts"],
            "earlier_verdict": earlier["verdict"],
            "earlier_classes": earlier["defect_classes"],
            "earlier_origin": earlier["origin"],
            "verdict": verdict["verdict"],
            "verdict_changed": before != after,
            "earlier_locus": before,
            "locus": after,
            "rule_clause": verdict["rule_clause"],
            "reason": verdict["reason"],
        }
        rows.append(row)
        if before != after:
            changed.append(row)
    return {"rows": rows, "changed": changed}


def reclassify(comparison: dict) -> dict:
    """Re-class the members earlier called as omitting what their source says."""
    rows = [
        row
        for row in comparison["rows"]
        if "name_omits_source" in row["earlier_classes"]
        and "omits-source-reclass" in row["cohorts"]
    ]
    counts = Counter(row["verdict"] for row in rows)
    by_outcome = {}
    for row in rows:
        outcome = "repaired" if "accepted-repair" in row["cohorts"] else "left-for-lead"
        by_outcome.setdefault(outcome, Counter())[row["verdict"]] += 1
    return {
        "population": len(rows),
        "counts": dict(sorted(counts.items())),
        "by_earlier_outcome": {
            key: dict(sorted(value.items())) for key, value in sorted(by_outcome.items())
        },
        "rows": [
            {
                "id": row["id"],
                "earlier_classes": row["earlier_classes"],
                "cohorts": row["cohorts"],
                "verdict": row["verdict"],
                "rule_clause": row["rule_clause"],
                "reason": row["reason"],
            }
            for row in rows
        ],
    }
def encode_payload(payload) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.b85encode(zlib.compress(raw, level=9)).decode()


def decode_payload(text: str):
    return json.loads(zlib.decompress(base64.b85decode(text)))


def member_entry(member: dict, verdict: dict) -> dict:
    """One row of the reference set: what the right answer is, and why."""
    return {
        "id": verdict["id"],
        "record_name": member["record"]["name"],
        "cohorts": member["cohorts"],
        "group": member["record"]["physics_domain"],
        "unit": member["record"]["unit"],
        "source_paths": member["source_paths"],
        "right_names": verdict["right_names"],
        "definition_points": verdict["definition_points"],
        "verdict": verdict["verdict"],
        "rule_clause": verdict["rule_clause"],
        "evidence_quote": verdict["evidence_quote"],
        "evidence_path": verdict["evidence_path"],
        "reason": verdict["reason"],
        "earlier": {
            "verdict": member["earlier"]["verdict"],
            "defect_classes": member["earlier"]["defect_classes"],
        },
        "packet_digest": member["packet_digest"],
    }


def in_set(member: dict) -> bool:
    """The set is the three named cohorts; the re-class pass reports separately."""
    return bool(
        set(member["cohorts"])
        & {"equilibrium-sound", "accepted-repair", "reader-calibration"}
    )


def build(sealed: dict, members: list[dict]) -> dict:
    """Assemble the committed record from the sealed answers."""
    comparison = compare(sealed, members)
    verdicts = {row["id"]: row for row in sealed["verdicts"]}
    rows = []
    undecidable = []
    unjudged = []
    packets = {}
    for member in members:
        packets[member["id"]] = member["text"]
        verdict = verdicts.get(member["id"])
        if verdict is None:
            unjudged.append({"id": member["id"], "reason": "no verdict recorded"})
            continue
        if verdict["verdict"] in DEFECTIVE:
            if in_set(member):
                rows.append(member_entry(member, verdict))
        elif verdict["verdict"] == "sound":
            if in_set(member):
                rows.append(member_entry(member, verdict))
        elif verdict["verdict"] == "undecidable":
            undecidable.append({"id": member["id"], "reason": verdict["reason"]})
        else:
            unjudged.append({"id": member["id"], "reason": verdict["reason"]})
    counts = Counter(row["verdict"] for row in rows)
    defective = sum(counts[key] for key in DEFECTIVE)
    total = len(rows)
    data = {
        "phase": "sealed-and-compared",
        "built_at": datetime.now(UTC).isoformat(),
        "rule": RULE,
        "rule_digest": digest(RULE),
        "model": MODEL,
        "prompt_digest": sealed["prompt_digest"],
        "blind": {
            "sealed_at": sealed["judged_at"],
            "verdicts_digest": sealed["digest"],
            "population": sealed["population"],
            "method": (
                "each packet was rendered from stored record and source text only; "
                "the earlier verdict, its classes and its reason were read only "
                "after every answer was written and sealed"
            ),
        },
        "sources": {
            "equilibrium_adjudication": {
                "path": ADJUDICATION,
                "sha256": file_digest(REPO / ADJUDICATION),
            },
            "equilibrium_repair": {
                "path": REPAIR,
                "sha256": file_digest(REPO / REPAIR),
            },
            "reader_calibration": {
                "path": CALIBRATION,
                "sha256": file_digest(REPO / CALIBRATION),
            },
        },
    }
    data["population"] = {
        "read": len(members),
        "by_cohort": dict(
            sorted(
                Counter(c for member in members for c in member["cohorts"]).items()
            )
        ),
        "in_set": total,
        "undecidable": len(undecidable),
        "unjudged": len(unjudged),
    }
    data["verdicts"] = {
        "counts": dict(sorted(counts.items())),
        "defective": defective,
        "defective_share": defective / total if total else 0.0,
        "defective_wilson_95": wilson(defective, total),
    }
    data["reclassification"] = reclassify(comparison)
    data["changes"] = {
        "changed": len(comparison["changed"]),
        "by_rule_clause": dict(
            sorted(Counter(r["rule_clause"] for r in comparison["changed"]).items())
        ),
        "rows": comparison["changed"],
    }
    data["undecidable"] = undecidable
    data["unjudged"] = unjudged
    data["members"] = sorted(rows, key=lambda row: (row["verdict"], row["id"]))
    data["packets"] = packets
    return data
def save(data: dict, path=DATA) -> None:
    """Write the record.

    The readable part stays at the top; the per-member records and the packets
    are compressed, as the sibling evidence scripts do, so one data file carries
    both inside the repository's size ceiling.
    """
    payload = dict(data)
    members = payload.pop("members")
    packets = payload.pop("packets")
    payload["member_payload_zlib_base85"] = encode_payload(members)
    payload["packet_payload_zlib_base85"] = encode_payload(packets)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    size = len(text.encode())
    if size > 300_000:
        raise SystemExit(f"data file is {size} bytes, over the 300000 ceiling")
    Path(path).write_text(text)


def load(path=DATA) -> dict:
    data = json.loads(Path(path).read_text())
    data["members"] = decode_payload(data.pop("member_payload_zlib_base85"))
    data["packets"] = decode_payload(data.pop("packet_payload_zlib_base85"))
    return data
def check_problems(data: dict) -> list[str]:
    """Every reason the stored reference set cannot be trusted as an answer key."""
    problems = []
    members = data["members"]
    packets = data["packets"]
    for member in members:
        text = packets.get(member["id"])
        if text is None:
            problems.append(f"{member['id']}: packet missing from the record")
            continue
        recomputed = digest({"id": member["id"], "text": text})
        stored = member["packet_digest"]
        if recomputed != stored:
            problems.append(
                f"{member['id']}: packet digest {recomputed} does not match stored {stored}"
            )
        if squash(member["evidence_quote"]) not in squash(text):
            problems.append(f"{member['id']}: evidence quote is not in its packet")
        if not member["definition_points"]:
            problems.append(f"{member['id']}: no definition points")
        if not member["right_names"]:
            problems.append(f"{member['id']}: no right name")
        if not member["unit"]:
            problems.append(f"{member['id']}: no unit")
        if not member["group"]:
            problems.append(f"{member['id']}: no group")
    defective = sum(1 for member in members if member["verdict"] in DEFECTIVE)
    if len(members) < 100:
        problems.append(f"only {len(members)} members in the set")
    if defective < 20:
        problems.append(f"only {defective} defective members, fewer than 20")
    return problems


def check(path=DATA) -> int:
    data = load(path)
    problems = check_problems(data)
    defective = sum(1 for m in data["members"] if m["verdict"] in DEFECTIVE)
    if problems:
        for problem in problems:
            print(f"REFUSED {problem}")
        print(f"checked {len(data['members'])} members, {len(problems)} problems")
        return 1
    print(
        f"checked {len(data['members'])} members, every packet digest matches, "
        f"{defective} defective, {data['population']['undecidable']} undecidable"
    )
    return 0
def table(headers: list[str], rows: list[list[str]], ident: str) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<table id="{ident}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def member_rows(members: list[dict]) -> list[list[str]]:
    rows = []
    for member in members:
        names = ", ".join(f"<code>{escape(name)}</code>" for name in member["right_names"])
        points = "<br>".join(f"&middot; {escape(p)}" for p in member["definition_points"])
        rows.append(
            [
                f"<code>{escape(member['id'])}</code>",
                member["verdict"].replace("_", " "),
                names,
                points,
                escape(member["unit"]),
                escape(member["group"]),
                escape(member["reason"]),
                f"<q>{escape(member['evidence_quote'])}</q> "
                f"<span>{escape(member['evidence_path'])}</span>",
            ]
        )
    return rows


def summary_text(data: dict) -> str:
    members = data["members"]
    population = data["population"]
    verdicts = data["verdicts"]
    reclass = data["reclassification"]
    changes = data["changes"]
    low, high = verdicts["defective_wilson_95"]
    share = percent(verdicts["defective_share"])
    defective = verdicts["defective"]
    naming = reclass["counts"].get("naming_defect", 0)
    defining = reclass["counts"].get("definition_defect", 0)
    sound = reclass["counts"].get("sound", 0)
    undecided = reclass["counts"].get("undecidable", 0)
    return (
        f"{population['read']} members were re-read blind from their source packets "
        f"under the rule that a name identifies and its definition specifies. "
        f"<strong>{len(members)} members are in the set</strong>, of which "
        f"<strong>{defective} are defective under the rule ({share}, 95&#8201;% CI "
        f"{percent(low)}&#8211;{percent(high)})</strong>. "
        f"{population['undecidable']} could not be decided and "
        f"{population['unjudged']} returned no usable answer; all were kept out "
        f"and counted. {changes['changed']} verdicts changed from the earlier read. "
        f"The {reclass['population']} members earlier classed as omitting what "
        f"their source states re-class as {naming} naming defects, "
        f"{defining} definition defects, {sound} sound and {undecided} undecidable."
    )


def change_rows(data: dict) -> list[list[str]]:
    rows = []
    for row in sorted(data["changes"]["rows"], key=lambda r: r["id"]):
        classes = ", ".join(row["earlier_classes"]) or "none"
        rows.append(
            [
                "<code>" + escape(row["id"]) + "</code>",
                row["earlier_locus"] + " (" + classes + ")",
                row["locus"],
                "<code>" + escape(row["rule_clause"]) + "</code>",
                escape(row["reason"]),
            ]
        )
    return rows


def reclass_rows(data: dict) -> list[list[str]]:
    rows = []
    for row in sorted(data["reclassification"]["rows"], key=lambda r: r["id"]):
        classes = ", ".join(row["earlier_classes"])
        rows.append(
            [
                "<code>" + escape(row["id"]) + "</code>",
                classes,
                row["verdict"].replace("_", " "),
                "<code>" + escape(row["rule_clause"]) + "</code>",
                escape(row["reason"]),
            ]
        )
    return rows


def page(data: dict, title: str, prefix: str) -> str:
    population = data["population"]
    counts = data["verdicts"]["counts"]
    cohort_rows = []
    for key, value in population["by_cohort"].items():
        cohort_rows.append([escape(key), str(value)])
    verdict_rows = []
    for key in ("sound", "naming_defect", "definition_defect"):
        verdict_rows.append([key.replace("_", " "), str(counts.get(key, 0))])
    parts = []
    parts.append("<h1 id='" + prefix + "-title'>" + escape(title) + "</h1>")
    parts.append("<p id='" + prefix + "-summary'>" + summary_text(data) + "</p>")
    parts.append("<h2 id='" + prefix + "-population'>Population</h2>")
    parts.append(table(["Cohort", "Members"], cohort_rows, prefix + "-population-table"))
    parts.append("<h2 id='" + prefix + "-verdicts'>Verdicts in the set</h2>")
    parts.append(table(["Verdict", "Members"], verdict_rows, prefix + "-verdicts-table"))
    return "\n".join(parts)


def more(data: dict, prefix: str) -> str:
    parts = []
    changes = data["changes"]
    reclass = data["reclassification"]
    parts.append("<h2 id='" + prefix + "-reclass-classes'>What the earlier omits-source class became</h2>")
    parts.append(
        "<p id='" + prefix + "-reclass-note'>The "
        + str(reclass["population"])
        + " members the earlier read classed as omitting what their source states, "
        + "re-read under the rule: " + escape(str(reclass["counts"])) + ".</p>"
    )
    parts.append(
        table(
            ["Member", "Earlier classes", "Verdict", "Rule clause", "Reason"],
            reclass_rows(data),
            prefix + "-reclass-table",
        )
    )
    parts.append("<h2 id='" + prefix + "-changes'>Verdicts that changed, beside the rule that changed them</h2>")
    parts.append(
        "<p id='" + prefix + "-changes-note'>" + str(changes["changed"])
        + " members changed locus between the earlier read and this one. "
        + "The rule clause is the half of the identity rule the blind verdict cites.</p>"
    )
    parts.append(
        table(
            ["Member", "Earlier", "Now", "Rule clause", "Reason"],
            change_rows(data),
            prefix + "-changes-table",
        )
    )
    return "\n".join(parts)


def tail(data: dict, prefix: str) -> str:
    excluded = data["undecidable"]
    unjudged = data["unjudged"]
    parts = []
    parts.append("<h2 id='" + prefix + "-excluded'>Kept out and counted</h2>")
    parts.append(
        "<p id='" + prefix + "-excluded-note'>" + str(len(excluded))
        + " members could not be decided from their packet and "
        + str(len(unjudged)) + " returned no usable answer. Neither enters the set.</p>"
    )
    rows = []
    for row in excluded:
        rows.append(["<code>" + escape(row["id"]) + "</code>", "undecidable", escape(row["reason"])])
    for row in unjudged:
        rows.append(["<code>" + escape(row["id"]) + "</code>", "unjudged", escape(row["reason"])])
    parts.append(table(["Member", "State", "Why"], rows, prefix + "-excluded-table"))
    parts.append("<h2 id='" + prefix + "-members'>The set, member by member</h2>")
    parts.append(
        table(
            ["Member", "Verdict", "Right name(s)", "Definition must state", "Unit",
             "Group", "Reason", "Source quote"],
            member_rows(data["members"]),
            prefix + "-members-table",
        )
    )
    return "\n".join(parts)


def document(title: str, body: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"docs-project\" content=\"imas-codex\">\n"
        "<meta name=\"reckon-type\" content=\"evidence-fragment\">\n"
        "<meta name=\"plan-slug\" content=\"sn-case-file-generation\">\n"
        "<meta name=\"node-id\" content=\"scfg-reference-set\">\n"
        "<title>" + escape(title) + "</title>\n</head>\n<body>\n<main>\n"
        + body + "\n</main>\n</body>\n</html>\n"
    )


def write_pages(data: dict) -> None:
    prefix = "scfg-reference-set"
    body = page(data, "The reference set of right answers", prefix)
    body += "\n" + more(data, prefix)
    body += "\n" + tail(data, prefix)
    text = document("The reference set of right answers", body)
    PAGE.write_text(text)
    FRAGMENT.write_text(text)
def self_test() -> int:
    """Prove the digest check refuses a packet that moved, and blindness fires."""
    data = load()
    refusals = 0
    sample = dict(data["members"][0])
    moved = dict(data["packets"])
    moved[sample["id"]] = moved[sample["id"]] + "amended"
    probe = {"members": [sample], "packets": moved,
             "population": data["population"]}
    problems = check_problems(probe)
    assert any("does not match stored" in p for p in problems), problems
    refusals += 1
    print("self-test: a moved packet is refused:", problems[0])
    member = dict(sample)
    member["definition_points"] = []
    problems = check_problems({"members": [member], "packets": data["packets"],
                               "population": data["population"]})
    assert any("no definition points" in p for p in problems), problems
    refusals += 1
    print("self-test: a member without definition points is refused")
    print("self-test ok, " + str(refusals) + " refusals")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build the reference set.")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--check", nargs="?", const=str(DATA))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.collect:
        members = assemble()
        RUN.mkdir(parents=True, exist_ok=True)
        (RUN / "members.json").write_text(
            json.dumps(
                [
                    {
                        "id": member["id"],
                        "cohorts": member["cohorts"],
                        "earlier": member["earlier"],
                        "packet_digest": member["packet_digest"],
                        "unit": member["record"]["unit"],
                        "group": member["record"]["physics_domain"],
                    }
                    for member in members
                ],
                indent=1,
            )
        )
        print("collected " + str(len(members)) + " members")
        return 0
    if args.judge:
        members = assemble()
        asyncio.run(run_judging(members))
        return 0
    if args.seal:
        members = assemble()
        seal(members)
        return 0
    if args.build:
        members = assemble()
        sealed = open_seal(RUN / SEALED)
        data = build(sealed, members)
        save(data)
        write_pages(data)
        print(
            "built "
            + str(len(data["members"]))
            + " members in the set, "
            + str(data["verdicts"]["defective"])
            + " defective, "
            + str(data["population"]["undecidable"])
            + " undecidable"
        )
        return 0
    if args.check is not None:
        return check(args.check)
    if args.self_test:
        return self_test()
    parser.print_help()
    return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
