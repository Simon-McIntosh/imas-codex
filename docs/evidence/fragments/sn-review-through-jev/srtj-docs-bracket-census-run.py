#!/usr/bin/env python3
"""Classify the brackets that make accepted documents fail the prose-link rule.

The instrument under measurement is the ``link_hygiene`` gate in
``imas_codex.standard_names.docs_gates`` as the prose-link node left it: it
fails a bracketed prose phrase that is neither a markdown link nor inside
mathematics, and a ``name:`` link that targets the documented name itself.
That revision is not yet merged, so the gate is imported from that node's own
worktree and the absolute path it resolved from is recorded in the output.

Over every accepted document in the live graph (read-only, bounded), each
offending construct is classified by its exact syntax and context: whether it
is a placeholder in the future-name (``dd:``) link convention, a markdown
link whose target is a DD path, a bare bracketed phrase carrying no target, a
bracketed phrase that is notation the rule should have excluded, a self-link,
or something else. Bare phrases are further resolved against the catalogue of
standard name identities, descriptions and DD-path documentation.

Nothing here writes to the graph and nothing calls a model.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

FRAGMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]

# The prose-link gate revision this node measures is unmerged; it lives in
# that node's own worktree. Importing it from the source tree under test keeps
# the measurement honest about which gate revision produced the number.
GATE_WORKTREE = Path(
    "/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/"
    "4f554c2d-3a28-4c55-8645-67a1e8a30dea/srtj-docs-prose-link-check"
)
sys.path.insert(0, str(GATE_WORKTREE))

from imas_codex.standard_names.docs_gates import (  # noqa: E402
    DocumentationGateOutcome,
    score_documentation,
)
from imas_codex.standard_names.doc_links import (  # noqa: E402
    find_prose_bracket_phrases,
)
import imas_codex.standard_names.docs_gates as _gate_module  # noqa: E402
from imas_codex.graph.client import GraphClient  # noqa: E402

OUTPUT_PATH = FRAGMENT_DIR / "srtj-docs-bracket-census.json"

PROSE_REASON = "bracketed phrase names no standard name: ["
BARE_REASON = "bare bracket names a standard name but carries no link: ["
SELF_REASON = "targets the documented name itself:"

DOCUMENT_LIMIT = 4000
NODE_LIMIT = 40000

_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
_DD_TARGET_RE = re.compile(r"^dd:[a-z0-9_/]+$")
_NAME_TARGET_RE = re.compile(r"^name:[a-z][a-z0-9_]+$")
_UNIT_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9^\-\*\./]*")
_NUMERIC_INTERVAL_RE = re.compile(r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?")
_CITATION_RE = re.compile(r"\[\d+\]")


def snake(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", phrase.strip().lower()).strip("_")


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.strip().lower()).strip()


def _load_catalogue(gc) -> dict:
    """Read the identity, description and DD-path corpora the phrases resolve against."""

    names = gc.query(
        "MATCH (s:StandardName) "
        "RETURN s.id AS id, s.description AS descr, s.name_stage AS stage"
    )
    node_total = gc.query("MATCH (n:IMASNode) RETURN count(*) AS n")[0]["n"]
    node_docs = gc.query(
        "MATCH (n:IMASNode) WHERE n.documentation IS NOT NULL "
        "RETURN n.id AS id, n.documentation AS docs ORDER BY n.id LIMIT $limit",
        limit=NODE_LIMIT,
    )
    node_ids = gc.query("MATCH (n:IMASNode) RETURN n.id AS id")
    return {
        "names": names,
        "node_total": node_total,
        "node_docs": node_docs,
        "node_ids": {r["id"] for r in node_ids},
    }


def _notation_class(phrase: str) -> str | None:
    """Return a notation class when the rule should have excluded the bracket."""

    stripped = phrase.strip()
    if not re.search(r"[A-Za-z]", stripped):
        return "non_alphabetic"
    if _NUMERIC_INTERVAL_RE.fullmatch(stripped):
        return "numeric_interval"
    tokens = stripped.split()
    if all(_UNIT_TOKEN_RE.fullmatch(t) for t in tokens) and (
        len(tokens) == 1 or re.search(r"[0-9^/*]", stripped)
    ):
        return "unit_expression"
    if _CITATION_RE.fullmatch(phrase):
        return "citation"
    return None


def main() -> int:
    started = time.monotonic()
    if not GATE_WORKTREE.is_dir():
        raise SystemExit(f"gate worktree absent: {GATE_WORKTREE}")

    with GraphClient() as gc:
        docs = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.name_stage IN ['accepted', 'approved'] "
            "AND sn.documentation IS NOT NULL "
            "RETURN sn.id AS id, sn.documentation AS docs ORDER BY sn.id LIMIT $limit",
            limit=DOCUMENT_LIMIT,
        )
        catalogue = _load_catalogue(gc)
        dd_link_docs = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.name_stage IN ['accepted', 'approved'] "
            "AND sn.documentation CONTAINS '(dd:' RETURN count(*) AS n"
        )[0]["n"]
        dd_any_docs = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.name_stage IN ['accepted', 'approved'] "
            "AND sn.documentation CONTAINS 'dd:' RETURN count(*) AS n"
        )[0]["n"]

    name_ids = {r["id"] for r in catalogue["names"]}
    by_stage: Counter = Counter()
    descriptions: dict[str, list[str]] = {}
    for row in catalogue["names"]:
        by_stage[row.get("stage") or "none"] += 1
        key = normalise(row.get("descr") or "")
        if key:
            descriptions.setdefault(key, []).append(row["id"])

    node_docs_by_id = {r["id"]: (r.get("docs") or "") for r in catalogue["node_docs"]}

    offences: list[dict] = []
    for record in docs:
        text = (record.get("docs") or "").strip()
        if not text:
            continue
        score = score_documentation(text, documenting_name=record.get("id") or None)
        hygiene = score.gate_vector["link_hygiene"]
        if hygiene.outcome is not DocumentationGateOutcome.FAIL:
            continue
        reason = hygiene.reason
        entry = {"document_id": record["id"], "reason": reason}
        if reason.startswith(PROSE_REASON):
            phrase = reason[len(PROSE_REASON) : -1]
            entry["construct"] = "prose_bracket"
            entry["phrase"] = phrase
            entry["phrase_has_space"] = " " in phrase.strip()
            entry["word_count"] = len(phrase.split())
            entry["in_a_dd_target_link"] = f"[{phrase}](dd:" in text
            entry["in_a_name_target_link"] = f"[{phrase}](name:" in text
            entry["notation_suspect"] = _notation_class(phrase)
            entry["linked_anywhere_in_document"] = f"[{phrase}](" in text
            key = snake(phrase)
            entry["snake"] = key
            if key in name_ids:
                entry["resolves_to"] = "existing_name_id"
            elif normalise(phrase) in descriptions:
                entry["resolves_to"] = "existing_description"
            else:
                entry["resolves_to"] = None
            hits = [
                node_id
                for node_id, node_text in node_docs_by_id.items()
                if phrase and phrase.lower() in node_text.lower()
            ]
            entry["dd_doc_matches"] = len(hits)
            entry["dd_doc_match_sample"] = hits[:3]
            if entry["resolves_to"] is None and hits:
                entry["resolves_to"] = "dd_path_documentation"
        elif reason.startswith(BARE_REASON):
            entry["construct"] = "bare_snake_bracket"
            entry["phrase"] = reason[len(BARE_REASON) : -1]
        elif SELF_REASON in reason:
            entry["construct"] = "self_link"
            match = re.search(r"name:([a-z0-9_]+)", reason)
            entry["target"] = match.group(1) if match else None
            entry["asserted_equal_document"] = entry["target"] == record["id"]
        else:
            entry["construct"] = "other"
        offences.append(entry)

    counts = Counter(entry["construct"] for entry in offences)
    prose = [entry for entry in offences if entry["construct"] == "prose_bracket"]
    resolve_counts = Counter(entry["resolves_to"] or "none_of_these" for entry in prose)

    def examples(predicate, n: int = 1) -> list[dict]:
        return [entry for entry in offences if predicate(entry)][:n]

    classes = {
        "future_name_dd_placeholder": [
            entry for entry in offences if entry.get("in_a_dd_target_link")
        ],
        "markdown_link_to_dd_path": [
            entry
            for entry in offences
            if _DD_TARGET_RE.match(entry.get("target") or "")
            or entry.get("in_a_dd_target_link")
        ],
        "bare_prose_bracket": prose,
        "notation_should_be_excluded": [
            entry for entry in prose if entry.get("notation_suspect")
        ],
        "self_link": [entry for entry in offences if entry["construct"] == "self_link"],
        "other": [entry for entry in offences if entry["construct"] == "other"],
    }

    # Positive controls: show the instrument sees known-present instances of
    # each construct before any count of zero is reported as an absence.
    controls = {
        "prose_scanner_sees_a_phrase": {
            "purpose": "the bracket scanner must return a prose phrase given one",
            "found": list(
                find_prose_bracket_phrases("differs from the [ion density].")
            ),
            "passed": list(
                find_prose_bracket_phrases("differs from the [ion density].")
            )
            == ["ion density"],
        },
        "gate_fails_a_prose_bracket": {
            "purpose": "the gate must fail a document holding a bare prose bracket",
            "reason": score_documentation("differs from the [ion density].")
            .gate_vector["link_hygiene"]
            .reason,
            "passed": score_documentation("differs from the [ion density].")
            .gate_vector["link_hygiene"]
            .reason.startswith(PROSE_REASON),
        },
        "gate_fails_a_self_link": {
            "purpose": "the gate must fail a link to the documented name",
            "reason": score_documentation(
                "see [x](name:lithium_density_at_limiter).",
                documenting_name="lithium_density_at_limiter",
            )
            .gate_vector["link_hygiene"]
            .reason,
            "passed": SELF_REASON
            in score_documentation(
                "see [x](name:lithium_density_at_limiter).",
                documenting_name="lithium_density_at_limiter",
            )
            .gate_vector["link_hygiene"]
            .reason,
        },
        "catalogue_description_lookup_works": {
            "purpose": "the description index must hold descriptions, so a zero match on a phrase is interpretable",
            "descriptions_indexed": len(descriptions),
            "sample_ids": sorted(descriptions.values())[0][:1] if descriptions else [],
            "passed": len(descriptions) > 0,
        },
        "dd_path_documentation_index_works": {
            "purpose": "the DD-path documentation corpus must be non-empty, so a zero phrase match is interpretable",
            "nodes_with_documentation": len(node_docs_by_id),
            "nodes_total": catalogue["node_total"],
            "passed": len(node_docs_by_id) > 0,
        },
        "dd_link_syntax_positive_control": {
            "purpose": "the dd: target syntax the future-name convention uses must be a valid link target in the gate",
            "target_valid": bool(
                _gate_module._VALID_LINK_TARGET_RE.fullmatch(
                    "dd:equilibrium/time_slice/profiles_1d/electrons/density"
                )
            ),
            "gate_expression": "_VALID_LINK_TARGET_RE.fullmatch('dd:equilibrium/...')",
        },
    }

    payload = {
        "schema": "srtj-docs-bracket-census",
        "created_at": datetime.now(UTC).isoformat(),
        "gate_module_path": str(Path(_gate_module.__file__).resolve()),
        "gate_worktree": str(GATE_WORKTREE),
        "working_directory": str(Path.cwd()),
        "documents_read": len(docs),
        "document_limit": DOCUMENT_LIMIT,
        "catalogue": {
            "standard_names": len(name_ids),
            "by_stage": dict(sorted(by_stage.items())),
            "descriptions_indexed": len(descriptions),
            "imas_nodes_total": catalogue["node_total"],
            "imas_nodes_with_documentation": len(catalogue["node_docs"]),
        },
        "future_name_convention": {
            "accepted_documents_containing_any_dd_token": dd_any_docs,
            "accepted_documents_containing_a_dd_link": dd_link_docs,
        },
        "offences_total": len(offences),
        "offence_counts": dict(sorted(counts.items())),
        "class_counts": {k: len(v) for k, v in classes.items()},
        "prose_resolution_counts": dict(sorted(resolve_counts.items())),
        "prose_phrase_word_counts": dict(
            sorted(Counter(entry["word_count"] for entry in prose).items())
        ),
        "prose_phrases_all_contain_a_space": all(
            entry["phrase_has_space"] for entry in prose
        ),
        "prose_phrases_linked_elsewhere_in_same_document": sum(
            1 for entry in prose if entry["linked_anywhere_in_document"]
        ),
        "prose_phrases_in_a_dd_target_link": sum(
            1 for entry in prose if entry["in_a_dd_target_link"]
        ),
        "bare_snake_bracket_count": counts["bare_snake_bracket"],
        "examples": {
            "bare_prose_bracket_resolving_to_existing_name_id": examples(
                lambda e: (
                    e["construct"] == "prose_bracket"
                    and e["resolves_to"] == "existing_name_id"
                ),
                5,
            ),
            "bare_prose_bracket_none_of_these": examples(
                lambda e: (
                    e["construct"] == "prose_bracket" and e["resolves_to"] is None
                ),
                5,
            ),
            "bare_prose_bracket_resolving_to_dd_documentation": examples(
                lambda e: (
                    e["construct"] == "prose_bracket"
                    and e["resolves_to"] == "dd_path_documentation"
                ),
                5,
            ),
            "self_link": classes["self_link"][:5],
            "future_name_dd_placeholder": classes["future_name_dd_placeholder"][:5],
            "notation_should_be_excluded": classes["notation_should_be_excluded"][:5],
        },
        "positive_controls": controls,
        "offences_per_document": dict(
            sorted(Counter(entry["document_id"] for entry in offences).items())
        ),
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    payload["classification_judgement"] = {
        "note": (
            "Syntax classes (future_name_dd_placeholder, markdown_link_to_dd_path, "
            "bare_prose_bracket, self_link) are read from the text by regex. The "
            "resolve_to sub-class of a bare phrase is a judgement: it says the phrase's "
            "snake-cased form or normalised text coincides with an existing catalogue "
            "string, which is evidence of what the writer meant, not proof."
        )
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print("documents_read:", len(docs), "offences:", len(offences))
    print("offence_counts:", dict(counts))
    print("class_counts:", payload["class_counts"])
    print("prose_resolution_counts:", payload["prose_resolution_counts"])
    print("dd in accepted docs (any token):", dd_any_docs, "as link:", dd_link_docs)
    for name, control in controls.items():
        print("control", name, "passed=", control.get("passed"), control)
    print("wrote", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
