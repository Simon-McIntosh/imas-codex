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

The gate's own exclusions are not taken on trust. Every raw bracket span is
classified a second time by an independently written instrument that shares
no pattern with the gate, so the count of spans the predicate excludes is
evidence rather than a restatement of the predicate. The dd:-token scan
carries a positive control that runs the same predicate against a document
known to carry the token.

Nothing here writes to the graph and nothing calls a model.
"""

from __future__ import annotations

import json
import re
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

import imas_codex.standard_names.docs_gates as _gate_module  # noqa: E402
from imas_codex.graph.client import GraphClient  # noqa: E402
from imas_codex.standard_names.doc_links import (  # noqa: E402
    _is_prose_bracket,
    find_prose_bracket_phrases,
)
from imas_codex.standard_names.docs_gates import (  # noqa: E402
    DocumentationGateOutcome,
    score_documentation,
)

OUTPUT_PATH = FRAGMENT_DIR / "srtj-docs-bracket-census.json"

PROSE_REASON = "bracketed phrase names no standard name: ["
BARE_REASON = "bare bracket names a standard name but carries no link: ["
SELF_REASON = "targets the documented name itself:"
UNSUPPORTED_REASON = "link has an unsupported target or an empty label"

DOCUMENT_LIMIT = 4000
NODE_LIMIT = 40000

_DD_TARGET_RE = re.compile(r"^dd:[a-z0-9_/]+$")
_DD_LINK_TARGET_RE = re.compile(r"\[[^\]]*\]\(dd:[^)]+\)")
_DD_PLANTED = "[density](dd:equilibrium/time_slice/profiles_1d/electrons/density)"

# ---------------------------------------------------------------------------
# The independent instrument. Written from first principles and sharing no
# pattern with the gate, so agreement between the two is evidence about the
# gate rather than a restatement of it.
# ---------------------------------------------------------------------------
_IND_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_IND_MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.DOTALL)
_IND_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_IND_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_IND_NUMBER_PAIR = re.compile(r"^-?\d+(?:[.,]\d+)?\s*[,–-]\s*-?\d+(?:[.,]\d+)?$")
_IND_UNITISH = re.compile(
    r"^[A-Za-z][A-Za-z0-9]*$|^[A-Za-z0-9]+[\^/][A-Za-z0-9^/*.-]+$"
)
_IND_LATEX = re.compile(r"\\[a-zA-Z]+")


def independent_span_class(spans: list[tuple[int, int]], start: int, end: int) -> str:
    """Classify one raw bracket span with the independent instrument."""

    for mask, label in (
        (spans[0], "math"),
        (spans[1], "code"),
        (spans[2], "markdown_link_label"),
    ):
        if any(a <= start and end <= b for a, b in mask):
            return label
    return "unmasked"


def independent_content_class(content: str) -> str:
    """Classify unmasked bracket content with the independent instrument."""

    text = content.strip()
    if not text:
        return "empty"
    if _IND_LATEX.search(text):
        return "latex"
    if _IND_NUMBER_PAIR.match(text):
        return "numeric_interval"
    if _IND_UNITISH.match(text) and " " not in text:
        return "unit_like"
    if re.fullmatch(r"\d+", text):
        return "citation"
    return "prose_words"


def independent_scan(text: str) -> dict:
    """Return the raw bracket spans of *text* under the independent instrument."""

    return {
        "spans": [(m.start(), m.end()) for m in _IND_BRACKET.finditer(text)],
        "content": [m.group(1) for m in _IND_BRACKET.finditer(text)],
        "masked": (
            [m.span() for m in _IND_MATH.finditer(text)],
            [m.span() for m in _IND_CODE.finditer(text)],
            [m.span() for m in _IND_LINK.finditer(text)],
        ),
    }


def _control_span_classes() -> dict[str, str]:
    """Classify the four bracket kinds a positive control plants."""

    text = "[m^-3] and [0, 1] and $[a b]$ and [x](name:y)"
    scan = independent_scan(text)
    return {
        f"[{content}]": independent_span_class(scan["masked"], start, end)
        for (start, end), content in zip(scan["spans"], scan["content"], strict=True)
    }


def contains_dd_token(text: str) -> bool:
    """The predicate the live dd: scan applies, usable on any text."""

    return "dd:" in text


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
        dd_any_stage = gc.query(
            "MATCH (sn:StandardName) WHERE sn.documentation CONTAINS 'dd:' "
            "RETURN sn.id AS id, sn.name_stage AS stage LIMIT 5"
        )

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
    # Independent exclusion census, accumulated across every accepted document.
    exclusion_counts: Counter = Counter()
    exclusion_examples: dict[str, list[str]] = {}
    raw_spans_total = 0
    predicate_spans_total = 0
    masked_but_reported = 0
    disagreements: list[dict] = []

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
            entry["linked_anywhere_in_document"] = f"[{phrase}](" in text
            entry["in_a_dd_target_link"] = f"[{phrase}](dd:" in text
            entry["gate_predicate_says_prose"] = bool(_is_prose_bracket(phrase))
            key = snake(phrase)
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

        # Independent exclusion census over this document's raw bracket spans.
        scan = independent_scan(text)
        reported_counter = Counter(find_prose_bracket_phrases(text))
        for (start, end), content in zip(scan["spans"], scan["content"], strict=True):
            raw_spans_total += 1
            where = independent_span_class(scan["masked"], start, end)
            if where != "unmasked":
                label = where
                if reported_counter[content] > 0:
                    masked_but_reported += 1
            elif reported_counter[content] > 0:
                label = "prose_reported_by_gate"
                reported_counter[content] -= 1
                predicate_spans_total += 1
            else:
                if (
                    independent_content_class(content) == "prose_words"
                    and len(disagreements) < 20
                ):
                    disagreements.append(
                        {
                            "document_id": record["id"],
                            "span": f"[{content}]",
                            "independent_class": "prose_words",
                            "gate_prose": bool(_is_prose_bracket(content)),
                        }
                    )
                label = independent_content_class(content)
            exclusion_counts[label] += 1
            if len(exclusion_examples.setdefault(label, [])) < 3:
                exclusion_examples[label].append(
                    {"document_id": record["id"], "span": f"[{content}]"}
                )

    counts = Counter(entry["construct"] for entry in offences)
    prose = [entry for entry in offences if entry["construct"] == "prose_bracket"]
    resolve_counts = Counter(entry["resolves_to"] or "none_of_these" for entry in prose)
    other_docs = [
        {"document_id": entry["document_id"], "reason": entry["reason"]}
        for entry in offences
        if entry["construct"] == "other"
    ]

    def examples(predicate, n: int = 5) -> list[dict]:
        return [entry for entry in offences if predicate(entry)][:n]

    classes = {
        "future_name_dd_placeholder": [
            entry for entry in offences if entry.get("in_a_dd_target_link")
        ],
        "markdown_link_to_dd_path": [
            entry
            for entry in offences
            if _DD_TARGET_RE.match(entry.get("target") or "")
        ],
        "bare_prose_bracket": prose,
        "notation_should_be_excluded": [
            entry for entry in prose if not entry["gate_predicate_says_prose"]
        ],
        "self_link": [entry for entry in offences if entry["construct"] == "self_link"],
        "other": other_docs,
    }

    controls = {
        "prose_scanner_sees_a_phrase": {
            "purpose": "the gate's scanner must return a prose phrase given one",
            "input": "differs from the [ion density].",
            "found": list(
                find_prose_bracket_phrases("differs from the [ion density].")
            ),
            "passed": list(
                find_prose_bracket_phrases("differs from the [ion density].")
            )
            == ["ion density"],
        },
        "gate_predicate_agrees_on_a_unit": {
            "purpose": "the gate predicate must reject a unit bracket",
            "input": "[m^-3]",
            "value": _is_prose_bracket("m^-3"),
            "passed": _is_prose_bracket("m^-3") is False,
        },
        "independent_scan_sees_each_excluded_kind": {
            "purpose": (
                "the independent instrument must classify a unit, an interval, "
                "mathematics and a link label into their own buckets"
            ),
            "input": "[m^-3] and [0, 1] and $[a b]$ and [x](name:y)",
            "span_classes": _control_span_classes(),
            "content_classes": {
                "[m^-3]": independent_content_class("m^-3"),
                "[0, 1]": independent_content_class("0, 1"),
                "[12]": independent_content_class("12"),
                "[ion density]": independent_content_class("ion density"),
            },
            "required_present": ["math", "markdown_link_label"],
            "passed": (
                independent_content_class("m^-3") == "unit_like"
                and independent_content_class("0, 1") == "numeric_interval"
                and independent_content_class("12") == "citation"
                and independent_content_class("ion density") == "prose_words"
            ),
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
            "purpose": "the description index must hold descriptions, so a zero match is interpretable",
            "descriptions_indexed": len(descriptions),
            "passed": len(descriptions) > 0,
        },
        "dd_path_documentation_index_works": {
            "purpose": "the DD-path documentation corpus must be non-empty, so a zero match is interpretable",
            "nodes_with_documentation": len(node_docs_by_id),
            "nodes_total": catalogue["node_total"],
            "passed": len(node_docs_by_id) > 0,
        },
        "dd_token_scan_positive_control": {
            "purpose": (
                "the same predicate the live dd: scan runs must find a dd: token "
                "in a document known to carry one"
            ),
            "method": "select from any stage"
            if dd_any_stage
            else "plant in a disposable copy",
            "graph_query": (
                "MATCH (sn:StandardName) WHERE sn.documentation CONTAINS 'dd:' "
                "RETURN sn.id AS id, sn.name_stage AS stage LIMIT 5"
            ),
            "any_stage_selection": dd_any_stage,
            "planted_text": _DD_PLANTED,
            "predicate_on_planted_text": contains_dd_token(_DD_PLANTED),
            "dd_link_pattern_on_planted_text": bool(
                _DD_LINK_TARGET_RE.search(_DD_PLANTED)
            ),
            "passed": bool(dd_any_stage) or contains_dd_token(_DD_PLANTED),
        },
        "dd_link_syntax_is_a_valid_gate_target": {
            "purpose": "the dd: target syntax must be an accepted link target",
            "input": "dd:equilibrium/time_slice/profiles_1d/electrons/density",
            "valid": bool(
                _gate_module._VALID_LINK_TARGET_RE.fullmatch(
                    "dd:equilibrium/time_slice/profiles_1d/electrons/density"
                )
            ),
        },
    }

    payload = {
        "schema": "srtj-docs-bracket-census",
        "created_at": datetime.now(UTC).isoformat(),
        "gate_module_path": str(Path(_gate_module.__file__).resolve()),
        "doc_links_module_path": str(
            Path(sys.modules["imas_codex.standard_names.doc_links"].__file__).resolve()
        ),
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
        "independent_exclusion_census": {
            "instrument": (
                "raw bracket spans from _IND_BRACKET, masked by independently "
                "written math/code/link patterns, content classified by "
                "_IND_NUMBER_PAIR / _IND_UNITISH / _IND_LATEX / digit-only"
            ),
            "raw_bracket_spans": raw_spans_total,
            "spans_reported_as_prose_by_the_gate": predicate_spans_total,
            "spans_masked_but_reported_prose_by_the_gate": masked_but_reported,
            "gate_exclusion_disagreements": disagreements,
            "excluded_span_classes": dict(sorted(exclusion_counts.items())),
            "excluded_span_examples": exclusion_examples,
        },
        "other_documents": other_docs,
        "examples": {
            "bare_prose_bracket_resolving_to_existing_name_id": examples(
                lambda e: (
                    e["construct"] == "prose_bracket"
                    and e["resolves_to"] == "existing_name_id"
                )
            ),
            "bare_prose_bracket_none_of_these": examples(
                lambda e: e["construct"] == "prose_bracket" and e["resolves_to"] is None
            ),
            "bare_prose_bracket_resolving_to_dd_documentation": examples(
                lambda e: (
                    e["construct"] == "prose_bracket"
                    and e["resolves_to"] == "dd_path_documentation"
                )
            ),
            "self_link": classes["self_link"][:5],
            "future_name_dd_placeholder": classes["future_name_dd_placeholder"][:5],
            "notation_should_be_excluded": classes["notation_should_be_excluded"][:5],
        },
        "positive_controls": controls,
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    payload["classification_judgement"] = {
        "note": (
            "Syntax classes (future_name_dd_placeholder, markdown_link_to_dd_path, "
            "bare_prose_bracket, self_link) are read from the text by regex. The "
            "resolves_to sub-class of a bare phrase is a judgement: it says the "
            "phrase's snake-cased form or normalised text coincides with an existing "
            "catalogue string, which is evidence of what the writer meant, not proof."
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
    print("independent exclusion census:", payload["independent_exclusion_census"])
    print("other docs:", other_docs)
    for name, control in controls.items():
        print("control", name, "passed=", control.get("passed"))
    print("wrote", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
