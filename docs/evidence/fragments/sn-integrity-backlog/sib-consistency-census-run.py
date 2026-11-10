"""Read-only consistency census over the live standard-name catalog.

Groups live names (``accepted`` and ``drafted``) two ways and reports the
families whose members disagree with each other:

* by ``physical_base`` — the grammar base token, so the family is the shared
  physical quantity.  Within a family the census compares the opening each
  member's description uses, the structural template its documentation
  follows, and the ``physics_domain`` its members are filed under.
* by shared concept — a definitional block inside documentation, of the form
  "the <concept> is defined by" followed by the documentation's formula.  A
  concept carried by several members is one idea the catalog states several
  times, so it is where a divergent definition is a defect rather than a
  difference between distinct quantities.

Both axes are needed.  The flux-surface average is the witness: the catalog
writes that one operator five ways, spread across many ``physical_base``
families because each name's base token is the averaged quantity, not the
average.  A ``physical_base`` grouping alone cannot see it.

Everything here is read-only against the live graph.  ``--planted`` starts a
disposable Neo4j, writes a synthetic family with two divergent definitions and
a second family whose two members are identical, and runs the same census code
over both, so the detector is shown to fire on a planted divergence and to stay
silent on a family that carries none.

Usage::

    uv run --no-sync python docs/evidence/fragments/sn-integrity-backlog/\\
        sib-consistency-census-run.py --live --json <out.json> [--planted]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LIVE_NAME_STAGES = ("accepted", "drafted")

NAME_QUERY = """
MATCH (n:StandardName)
WHERE n.name_stage IN $stages
RETURN n.id AS id,
       n.name_stage AS name_stage,
       n.docs_stage AS docs_stage,
       n.physical_base AS physical_base,
       n.physics_domain AS physics_domain,
       n.description AS description,
       n.documentation AS documentation,
       n.reviewer_score_docs AS reviewer_score_docs
ORDER BY n.id
"""

#: A documentation concept block: a sentence introducing one shared idea,
#: optionally followed by the display-math definition that sentence sets up.
_CONCEPT_RE = re.compile(
    r"(?m)^(?:The |This )?(?P<lemma>[A-Za-z][A-Za-z\-‐‑‒– ]{1,60}?)"
    r"\s+is\s+(?:defined|given)\b[^\n]*"
)
_MATH_RE = re.compile(r"\$\$(?P<body>.{0,1200}?)\$\$", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

#: Documentation template features, each one structural rather than lexical.
_TEMPLATE_FEATURES = (
    ("opening", re.compile(r"^\S")),
    ("math", re.compile(r"\$\$", re.S)),
    ("where", re.compile(r"\bwhere\b")),
    ("clarification", re.compile(r"(?:^|\n\n)(?:This|It) ")),
    ("named_link", re.compile(r"\(name:[^)]+\)")),
    ("headings", re.compile(r"(?m)^#{1,6} ")),
    ("bullets", re.compile(r"(?m)^[-*] ")),
)

#: Words that carry no concept identity when a lemma is normalised.
_LEMMA_STOPWORDS = {"the", "this", "a", "an"}

_TEXT_LIMIT = 320


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalise_lemma(lemma: str) -> str:
    """Canonical form of a concept lemma: lowercase tokens, no punctuation."""
    lowered = lemma.lower().replace("‐", "-").replace("‑", "-")
    lowered = lowered.replace("‒", "-").replace("–", "-")
    tokens = [t for t in re.split(r"[^a-z0-9]+", lowered) if t]
    while tokens and tokens[0] in _LEMMA_STOPWORDS:
        tokens = tokens[1:]
    return " ".join(tokens)


#: Words at which a description stops naming the shared base concept and starts
#: naming this member's own qualifiers ("Number density **of** carbon ions").
_CONCEPT_CUTS = {
    "of",
    "over",
    "on",
    "at",
    "in",
    "per",
    "for",
    "that",
    "which",
    "due",
    "divided",
    "defined",
    "with",
    "within",
    "along",
    "from",
    "relative",
    "averaged",
    "measured",
    "evaluated",
    "obtained",
    "normalised",
    "normalized",
    "computed",
    "expressed",
    "equal",
    "averaging",
}


def base_phrase(description: str) -> str:
    """The phrase a member's description uses for its family's shared concept.

    Two members of one ``physical_base`` family state the same quantity, so
    they should name it with the same words.  The phrase runs from the start of
    the first sentence to the first member-specific qualifier: "Number density
    of carbon ions..." yields ``number density``.  Divergence here is a
    definition disagreement inside one family, not the ordinary fact that two
    members describe different quantities.
    """
    first = _SENTENCE_END.split(_collapse(description))[0] if description else ""
    tokens = [t for t in re.split(r"[^a-z0-9]+", first.lower()) if t]
    while tokens and tokens[0] in _LEMMA_STOPWORDS:
        tokens = tokens[1:]
    cut = next(
        (index for index, token in enumerate(tokens) if token in _CONCEPT_CUTS),
        min(len(tokens), 4),
    )
    return " ".join(tokens[:cut])


def template_signature(documentation: str) -> str:
    """Structural skeleton of one documentation body.

    Each present feature contributes its letter, in a fixed order, so two
    bodies with the same skeleton have the same signature whatever their
    wording.  ``-`` marks a body with none of the features.
    """
    text = documentation or ""
    letters = "".join(
        name[0] for name, pattern in _TEMPLATE_FEATURES if pattern.search(text)
    )
    return letters or "-"


def concept_blocks(documentation: str) -> list[dict[str, str]]:
    """Definitional concept blocks in one documentation body.

    Each block carries the lemma (the shared idea being defined) and the text
    that defines it — the sentence plus the formula it introduces, whitespace
    collapsed so only wording differences remain.
    """
    text = documentation or ""
    blocks: list[dict[str, str]] = []
    for match in _CONCEPT_RE.finditer(text):
        lemma = _normalise_lemma(match.group("lemma"))
        if not lemma:
            continue
        tail = text[match.end() : match.end() + 1400]
        formula = _MATH_RE.search(tail)
        body = match.group(0)
        if formula:
            body = f"{body} $$ {_collapse(formula.group('body'))} $$"
        blocks.append({"lemma": lemma, "definition": _collapse(body)[:_TEXT_LIMIT]})
    return blocks


def _fold_lemmas(counts: Counter[str]) -> dict[str, str]:
    """Map each lemma to the lemma that carries it when both name one concept.

    A bare lemma ("average") is folded into the longest lemma of which it is a
    token suffix and that is carried by at least two members, so a document
    that says only "the average is defined by" joins the concept its siblings
    name in full.  The fold is recorded in the output as ``lemma_variants``,
    so a reader can always see which lemmas were merged.
    """
    ordered = sorted(counts, key=lambda lemma: (-len(lemma.split()), lemma))
    canonical: dict[str, str] = {}
    for lemma in counts:
        tokens = lemma.split()
        match = lemma
        for longer in ordered:
            if longer == lemma:
                continue
            longer_tokens = longer.split()
            if (
                len(longer_tokens) > len(tokens)
                and longer_tokens[-len(tokens) :] == tokens
                and counts[longer] >= 2
            ):
                match = longer
                break
        canonical[lemma] = match
    return canonical


def _member_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name_stage": row.get("name_stage"),
        "docs_stage": row.get("docs_stage"),
        "physics_domain": row.get("physics_domain"),
        "description": _collapse(row.get("description") or "")[:_TEXT_LIMIT],
        "base_phrase": base_phrase(row.get("description") or ""),
        "template_signature": template_signature(row.get("documentation") or ""),
        "reviewer_score_docs": row.get("reviewer_score_docs"),
    }


def _anchor(members: list[dict[str, Any]]) -> dict[str, Any]:
    """The member whose text and template the family should converge on.

    Follows the family anchor rule used by the docs pipeline: an accepted
    member with real documentation and the highest docs score wins, with ties
    and missing scores settled by id so the choice is deterministic.
    """
    candidates = [
        member
        for member in members
        if member.get("docs_stage") == "accepted" and member.get("description")
    ] or [member for member in members if member.get("description")]
    if not candidates:
        return {}

    def sort_key(member: dict[str, Any]) -> tuple[float, str]:
        score = member.get("reviewer_score_docs")
        return (-(score if isinstance(score, (int, float)) else -1.0), member["id"])

    return sorted(candidates, key=sort_key)[0]


def census_families(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group rows by ``physical_base`` and report the divergent families."""
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base = (row.get("physical_base") or "").strip()
        if base:
            families[base].append(_member_row(row))

    reported: list[dict[str, Any]] = []
    for base in sorted(families):
        members = sorted(families[base], key=lambda member: member["id"])
        if len(members) < 2:
            continue

        phrases = Counter(
            member["base_phrase"] for member in members if member["base_phrase"]
        )
        modal_phrase, modal_phrase_count = (
            phrases.most_common(1)[0] if phrases else ("", 0)
        )
        definition_divergent = len(phrases) > 1

        descriptions: dict[str, str] = {}
        anchor_descriptions: dict[str, str] = {}
        for member in members:
            text = member.pop("description", "")
            anchor_descriptions[member["id"]] = text
            descriptions.setdefault(member["base_phrase"], text)

        accepted = [member for member in members if member["docs_stage"] == "accepted"]
        signatures = Counter(member["template_signature"] for member in accepted)
        modal_signature, modal_signature_count = (
            signatures.most_common(1)[0] if signatures else ("", 0)
        )
        template_divergent = len(signatures) > 1

        domains = Counter(
            member["physics_domain"] for member in members if member["physics_domain"]
        )
        grouping_inconsistent = len(domains) > 1 or (
            bool(domains) and any(not member["physics_domain"] for member in members)
        )

        if not (definition_divergent or template_divergent or grouping_inconsistent):
            continue

        anchor = _anchor(members)
        entry = {
            "physical_base": base,
            "member_count": len(members),
            "member_template_signatures": dict(
                Counter(
                    f"{member['id']}={member['template_signature']}"
                    for member in members
                    if member["docs_stage"] == "accepted"
                )
            ),
            "divergence": {
                "definition": {
                    "divergent": definition_divergent,
                    "phrase_variants": [
                        {
                            "phrase": phrase,
                            "members": count,
                            "example": next(
                                member["id"]
                                for member in members
                                if member["base_phrase"] == phrase
                            ),
                            "description": descriptions[phrase][:240],
                        }
                        for phrase, count in phrases.most_common(6)
                    ],
                    "variants_omitted": max(0, len(phrases) - 6),
                },
                "template": {
                    "divergent": template_divergent,
                    "signature_variants": [
                        {
                            "signature": signature,
                            "members": count,
                            "example": next(
                                (
                                    member["id"]
                                    for member in accepted
                                    if member["template_signature"] == signature
                                ),
                                None,
                            ),
                        }
                        for signature, count in signatures.most_common()
                    ],
                },
                "grouping": {
                    "inconsistent": grouping_inconsistent,
                    "domain_variants": dict(domains.most_common()),
                    "members_without_domain": sum(
                        1 for member in members if not member["physics_domain"]
                    ),
                },
            },
            "proposed": {
                "definition_from": anchor.get("id"),
                "definition": anchor_descriptions.get(anchor.get("id")),
                "definition_phrase": modal_phrase,
                "definition_phrase_share": (
                    round(modal_phrase_count / len(members), 3) if members else None
                ),
                "template_signature": modal_signature,
                "template_from": next(
                    (
                        member["id"]
                        for member in accepted
                        if member["template_signature"] == modal_signature
                    ),
                    None,
                ),
                "template_share": (
                    round(modal_signature_count / len(accepted), 3)
                    if accepted
                    else None
                ),
                "domain": (domains.most_common(1)[0][0] if domains else None),
            },
            "members": [member["id"] for member in members],
        }
        reported.append(entry)

    return {
        "families_total": sum(1 for members in families.values() if len(members) > 1),
        "families_divergent": len(reported),
        "families": reported,
    }


def census_concepts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group rows by the shared concepts their documentation defines."""
    per_lemma: dict[str, list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(
        list
    )
    for row in rows:
        for block in concept_blocks(row.get("documentation") or ""):
            per_lemma[block["lemma"]].append((row, block))

    canonical = _fold_lemmas(Counter({lemma: len(v) for lemma, v in per_lemma.items()}))

    groups: dict[str, list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(list)
    for lemma, entries in per_lemma.items():
        groups[canonical[lemma]].extend(entries)

    reported: list[dict[str, Any]] = []
    for concept in sorted(groups):
        entries = groups[concept]
        member_ids = sorted({row["id"] for row, _ in entries})
        if len(member_ids) < 2:
            continue
        variants: dict[str, list[str]] = defaultdict(list)
        for row, block in entries:
            variants[block["definition"]].append(row["id"])
        declared_text = _modal_definition(entries)
        declared_members = sorted(set(variants[declared_text]))
        entry = {
            "concept": concept,
            "lemma_variants": sorted(
                {lemma for lemma in per_lemma if canonical[lemma] == concept}
            ),
            "member_count": len(member_ids),
            "definition_variants": len(variants),
            "divergent": len(variants) > 1,
            "variants": [
                {
                    "definition": definition,
                    "member_count": len(sorted(set(ids))),
                    "members": sorted(set(ids))[:12],
                    "members_total": len(sorted(set(ids))),
                }
                for definition, ids in sorted(
                    variants.items(), key=lambda item: (-len(set(item[1])), item[0])
                )
            ],
            "proposed": {
                "definition": declared_text,
                "definition_from": declared_members[0],
                "members_using_it": len(declared_members),
            },
        }
        if entry["divergent"]:
            reported.append(entry)

    return {
        "concepts_total": len(groups),
        "concepts_divergent": len(reported),
        "concepts": reported,
    }


def _modal_definition(entries: list[tuple[dict[str, Any], dict[str, str]]]) -> str:
    """The definition text most members of a concept group use."""
    counts: Counter[str] = Counter()
    for _row, block in entries:
        counts[block["definition"]] += 1
    return counts.most_common(1)[0][0]


def census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The whole census over one set of names."""
    universe = {
        "name_stages": list(LIVE_NAME_STAGES),
        "names": len(rows),
        "by_name_stage": dict(
            Counter(row.get("name_stage") for row in rows).most_common()
        ),
        "by_docs_stage": dict(
            Counter(row.get("docs_stage") for row in rows).most_common()
        ),
        "by_physical_base": len(
            {row.get("physical_base") for row in rows if row.get("physical_base")}
        ),
        "named_on_any_base": sum(1 for row in rows if not row.get("physical_base")),
    }
    families = census_families(rows)
    concepts = census_concepts(rows)

    witness = next(
        (
            entry
            for entry in concepts["concepts"]
            if "flux surface averag" in entry["concept"]
            or "flux surface average" in " ".join(entry["lemma_variants"])
        ),
        None,
    )
    return {
        "universe": universe,
        "families": families,
        "concepts": concepts,
        "witness": {
            "concept": (witness or {}).get("concept"),
            "found": witness is not None,
            "definition_variants": (witness or {}).get("definition_variants"),
            "member_count": (witness or {}).get("member_count"),
            "entry": witness,
        },
    }


def read_live() -> list[dict[str, Any]]:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as graph:
        return graph.query(NAME_QUERY, stages=list(LIVE_NAME_STAGES))


def planted_control() -> dict[str, Any]:
    """Plant families in a disposable graph and census them with the same code.

    Two divergent definitions in one family must be counted; two identical
    members of another must not.  The graph is started by the module that owns
    the disposable-Neo4j lifecycle and is torn down here.
    """
    from imas_codex.graph import temp_neo4j

    scratch = tempfile.mkdtemp(
        prefix="sib-consistency-census-", dir=tempfile.gettempdir()
    )
    directory = Path(scratch)
    bolt_port, http_port = _free_port_pair()
    proc = None
    try:
        proc, _log = temp_neo4j.start_temp_neo4j(
            directory, bolt_port, http_port, load_dump=False
        )
        uri = f"bolt://127.0.0.1:{bolt_port}"
        from imas_codex.graph.client import GraphClient

        planted = [
            {
                "id": "planted_alpha_quantity",
                "name_stage": "accepted",
                "docs_stage": "accepted",
                "physical_base": "planted_quantity",
                "physics_domain": "transport",
                "description": "Radial flux of the planted quantity per unit area.",
                "documentation": (
                    "Radial flux of the planted quantity per unit area.\n\n"
                    "The planted operator is defined by\n\n$$\\langle Q\\rangle = "
                    "\\oint Q\\,dS/V$$\n\nwhere $Q$ is the local quantity."
                ),
            },
            {
                "id": "planted_beta_quantity",
                "name_stage": "accepted",
                "docs_stage": "accepted",
                "physical_base": "planted_quantity",
                "physics_domain": "transport",
                "description": (
                    "Volumetric average of the planted quantity over the plasma volume."
                ),
                "documentation": (
                    "The planted operator is given componentwise by\n\n"
                    "$$\\langle Q\\rangle_V = \\int Q\\,dV/V$$\n\n"
                    "This is a volume average.\n\n## Notes\n\n- one"
                ),
            },
            {
                "id": "planted_identical_one",
                "name_stage": "accepted",
                "docs_stage": "accepted",
                "physical_base": "identical_quantity",
                "physics_domain": "transport",
                "description": "Density of the identical species.",
                "documentation": "Density of the identical species.\n\n$$n = n_0$$",
            },
            {
                "id": "planted_identical_two",
                "name_stage": "accepted",
                "docs_stage": "accepted",
                "physical_base": "identical_quantity",
                "physics_domain": "transport",
                "description": "Density of the identical species.",
                "documentation": "Density of the identical species.\n\n$$n = n_0$$",
            },
        ]
        with GraphClient(uri=uri) as graph:
            graph.query(
                """
                UNWIND $rows AS row
                CREATE (n:StandardName)
                SET n = row
                """,
                rows=planted,
            )
            rows = graph.query(NAME_QUERY, stages=list(LIVE_NAME_STAGES))

        result = census(rows)
        bases = {
            entry["physical_base"]: entry for entry in result["families"]["families"]
        }
        concepts = {entry["concept"]: entry for entry in result["concepts"]["concepts"]}
        divergent_family = bases.get("planted_quantity")
        identical_family = bases.get("identical_quantity")
        planted_concept = concepts.get("planted operator")

        checks = {
            "divergent_family_detected": divergent_family is not None,
            "divergent_family_phrase_variants": (
                len(
                    (divergent_family or {})
                    .get("divergence", {})
                    .get("definition", {})
                    .get("phrase_variants", [])
                )
            ),
            "divergent_family_template_variants": (
                len(
                    (divergent_family or {})
                    .get("divergence", {})
                    .get("template", {})
                    .get("signature_variants", [])
                )
            ),
            "identical_family_not_reported": identical_family is None,
            "planted_concept_detected": planted_concept is not None,
            "planted_concept_definition_variants": (
                (planted_concept or {}).get("definition_variants")
            ),
        }
        checks["passed"] = (
            checks["divergent_family_detected"]
            and checks["divergent_family_phrase_variants"] == 2
            and checks["divergent_family_template_variants"] >= 2
            and checks["identical_family_not_reported"]
            and checks["planted_concept_detected"]
            and checks["planted_concept_definition_variants"] == 2
        )
        return {
            "graph": "disposable Neo4j, empty at start, planted rows only",
            "planted_rows": [row["id"] for row in planted],
            "checks": checks,
            "census_over_planted": {
                "families_total": result["families"]["families_total"],
                "families_divergent": result["families"]["families_divergent"],
                "concepts_divergent": result["concepts"]["concepts_divergent"],
            },
        }
    finally:
        if proc is not None:
            temp_neo4j.stop_temp_neo4j(proc)


def _free_port_pair() -> tuple[int, int]:
    import socket

    socks = []
    try:
        for _ in range(2):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            socks.append(sock)
        return socks[0].getsockname()[1], socks[1].getsockname()[1]
    finally:
        for sock in socks:
            sock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the census JSON here")
    parser.add_argument(
        "--planted",
        action="store_true",
        help="also plant families in a disposable graph and census them",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=True,
        help="census the live graph (default)",
    )
    args = parser.parse_args(argv)

    rows = read_live()
    result = census(rows)
    result["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    result["source"] = "live Neo4j graph, read-only; no writes, no LLM spend"

    if args.planted:
        result["control"] = planted_control()
        if not result["control"]["checks"]["passed"]:
            print("PLANTED CONTROL FAILED", file=sys.stderr)
            return 1

    families = result["families"]
    concepts = result["concepts"]
    print(
        f"names={result['universe']['names']} "
        f"families={families['families_total']} divergent={families['families_divergent']} "
        f"concepts={concepts['concepts_total']} divergent={concepts['concepts_divergent']} "
        f"witness={result['witness']['concept']!r} "
        f"witness_variants={result['witness']['definition_variants']}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")
        print(f"wrote {args.json} ({args.json.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
