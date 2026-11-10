"""Read-only consistency census over the live standard-name catalog.

Groups live names (``accepted`` and ``drafted``) two ways and reports the
families whose members disagree with each other:

* by ``physical_base`` — the grammar base token, so the family is the shared
  physical quantity.  Within a family the census compares the phrase each
  member's first sentence uses for the family's base concept (qualifiers
  before or after the concept's own words do not make a different phrase), the
  structural template its documentation follows, and the ``physics_domain``
  its members are filed under.
* by shared concept — a definitional block inside documentation, of the form
  "the <concept> is defined by" followed by the documentation's formula.  A
  concept carried by several members is one idea the catalog states several
  times, so it is where a divergent definition is a defect rather than a
  difference between distinct quantities.

Both axes are needed.  The flux-surface average is the witness: the catalog
writes that one operator in many distinct texts, spread across many
``physical_base`` families because each name's base token is the averaged
quantity, not the average.  A ``physical_base`` grouping alone cannot see it.

``--html`` renders the summary fragment from the same census result, so the
fragment carries only figures the JSON beside it also carries.

Everything here is read-only against the live graph.  ``--planted`` starts a
disposable Neo4j, writes a synthetic family with two divergent definitions and
a second family whose two members are identical, and runs the same census code
over both, so the detector is shown to fire on a planted divergence and to stay
silent on a family that carries none.

Usage::

    uv run --no-sync python docs/evidence/fragments/sn-integrity-backlog/\\
        sib-consistency-census-run.py --live --json <out.json> \\
        [--html <out.html>] [--planted]
"""

from __future__ import annotations

import argparse
import html
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
       n.reviewer_score_docs AS reviewer_score_docs,
       EXISTS { MATCH (n)<-[:PRODUCED_NAME]-(dd:StandardNameSource {source_type: 'dd'})
                WHERE coalesce(dd.status, '') <> 'stale' } AS has_dd_source
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
#: A feature that every body carries discriminates nothing, so only features
#: that divide real documentation are listed.
_TEMPLATE_FEATURES = (
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


def _first_sentence_tokens(description: str) -> list[str]:
    first = _SENTENCE_END.split(_collapse(description))[0] if description else ""
    tokens = [t for t in re.split(r"[^a-z0-9]+", first.lower()) if t]
    while tokens and tokens[0] in _LEMMA_STOPWORDS:
        tokens = tokens[1:]
    return tokens


def base_phrase(description: str, base: str) -> str:
    """The words a member's description uses to name its family's base concept.

    Members of one ``physical_base`` family state the same quantity, so they
    should name it with the same words; what differs between them is their own
    qualifiers, which must not be read as a definition disagreement.  Those
    qualifiers sit on either side of the concept: "Toroidal magnetic field
    magnitude ..." carries ``toroidal`` before it and ``magnitude`` after it,
    while "the magnetic field" carries neither.  So the phrase is the base
    concept's own token span wherever it appears in the first sentence.  A
    member whose description never names the base concept falls back to its
    leading noun phrase, and that fallback is the real divergence: the member
    states a quantity other than the family's.
    """
    tokens = _first_sentence_tokens(description)
    base_tokens = [t for t in re.split(r"[^a-z0-9]+", base.lower()) if t]
    if base_tokens:
        for start in range(len(tokens) - len(base_tokens) + 1):
            if tokens[start : start + len(base_tokens)] == base_tokens:
                return " ".join(base_tokens)
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
        "template_signature": template_signature(row.get("documentation") or ""),
        "reviewer_score_docs": row.get("reviewer_score_docs"),
        "has_dd_source": bool(row.get("has_dd_source")),
    }


def family_anchor(members: list[dict[str, Any]]) -> str | None:
    """The member the family should converge on, by the shared anchor rule.

    Delegates to :func:`imas_codex.standard_names.families.select_anchor`, the
    one rule the docs pipeline and the harmonize command both use, so this
    census cannot name a different anchor from the pipeline that rewrites the
    family.  A census family has no parent row of its own, so the parent arm of
    the rule is left off and the anchor is the highest-scoring DD-sourced
    accepted member, or its deterministic fallbacks.
    """
    from imas_codex.standard_names.families import select_anchor

    return select_anchor(
        None,
        None,
        None,
        members,
        parent_has_dd_source=None,
    )


def census_families(
    rows: list[dict[str, Any]], default_template: str = ""
) -> dict[str, Any]:
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

        descriptions: dict[str, str] = {}
        phrases_of: dict[str, str] = {}
        for member in members:
            text = member["description"]
            member["base_phrase"] = base_phrase(text, base)
            phrases_of[member["id"]] = member["base_phrase"]
            descriptions.setdefault(member["base_phrase"], text)

        phrases = Counter(phrases_of.values())
        modal_phrase, modal_phrase_count = (
            phrases.most_common(1)[0] if phrases else ("", 0)
        )
        definition_divergent = len(phrases) > 1

        accepted = [member for member in members if member["docs_stage"] == "accepted"]
        signatures = Counter(member["template_signature"] for member in accepted)
        modal_signature, modal_signature_count = (
            signatures.most_common(1)[0] if signatures else (default_template, 0)
        )
        # A family with no accepted member to take a template from is written
        # to the catalog-wide modal template; the proposal says which of the
        # two it is rather than presenting a borrowed template as the family's.
        template_source = "family_modal" if signatures else "catalog_modal"
        template_divergent = len(signatures) > 1

        domains = Counter(
            member["physics_domain"] for member in members if member["physics_domain"]
        )
        grouping_inconsistent = len(domains) > 1 or (
            bool(domains) and any(not member["physics_domain"] for member in members)
        )

        if not (definition_divergent or template_divergent or grouping_inconsistent):
            continue

        descriptions_of_member = {
            member["id"]: member["description"] for member in members
        }
        anchor_id = family_anchor(members)
        if anchor_id is None:
            # The shared rule refuses a family with no Data-Dictionary-sourced
            # member.  The census must still propose something, so it names the
            # modal accepted member and records that the shared rule was
            # unavailable rather than presenting the choice as its verdict.
            anchor_id = (
                min(accepted, key=lambda member: member["id"])["id"]
                if accepted
                else None
            )
            anchor_rule = "fallback: no DD-sourced eligible member"
        else:
            anchor_rule = "select_anchor"

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
                "definition_from": anchor_id,
                "definition": descriptions_of_member.get(anchor_id),
                "anchor_rule": anchor_rule,
                "definition_phrase": modal_phrase,
                "definition_phrase_share": (
                    round(modal_phrase_count / len(members), 3) if members else None
                ),
                "template_signature": modal_signature,
                "template_source": template_source,
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
        rows_by_member: dict[str, dict[str, Any]] = {}
        text_by_member: dict[str, str] = {}
        for row, block in entries:
            rows_by_member.setdefault(row["id"], row)
            text_by_member.setdefault(row["id"], block["definition"])
        anchor = family_anchor([rows_by_member[mid] for mid in member_ids])
        if anchor is not None:
            proposed_text = text_by_member[anchor]
            anchor_rule = "select_anchor"
        else:
            anchor = declared_members[0]
            proposed_text = declared_text
            anchor_rule = "fallback: modal definition"
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
                "definition": proposed_text,
                "definition_from": anchor,
                "anchor_rule": anchor_rule,
                "members_using_it": len(set(variants[proposed_text])),
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
    catalog_templates = Counter(
        template_signature(row.get("documentation") or "")
        for row in rows
        if row.get("docs_stage") == "accepted" and row.get("documentation")
    )
    default_template = catalog_templates.most_common(1)[0][0]
    families = census_families(rows, default_template)
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
    gaps = [
        {
            "physical_base": entry["physical_base"],
            "definition": entry["proposed"].get("definition"),
            "template_signature": entry["proposed"].get("template_signature"),
        }
        for entry in families["families"]
        if not entry["proposed"].get("definition")
        or not entry["proposed"].get("template_signature")
    ]
    return {
        "universe": universe,
        "families": families,
        "concepts": concepts,
        "proposals": {
            "families": len(families["families"]),
            "with_definition": sum(
                1
                for entry in families["families"]
                if entry["proposed"].get("definition")
            ),
            "with_template": sum(
                1
                for entry in families["families"]
                if entry["proposed"].get("template_signature")
            ),
            "gaps": gaps,
            "complete": not gaps,
        },
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
                    "Volumetric average of the deposited quantity over the "
                    "plasma volume."
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
            # The shared anchor rule only considers a member entailed by a
            # live DD source, so the planted names get one; without it the
            # control would exercise the census's declared fallback instead of
            # the rule the pipeline actually uses.
            graph.query(
                """
                CREATE (s:StandardNameSource {id: 'planted_dd', source_type: 'dd', status: 'live'})
                WITH s
                MATCH (n:StandardName) WHERE n.id STARTS WITH 'planted_'
                CREATE (s)-[:PRODUCED_NAME]->(n)
                """
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
            "planted_proposal_definition": (
                (divergent_family or {}).get("proposed", {}).get("definition")
            ),
            "planted_proposal_template": (
                (divergent_family or {}).get("proposed", {}).get("template_signature")
            ),
            "proposals_complete": result["proposals"]["complete"],
        }
        checks["passed"] = (
            checks["divergent_family_detected"]
            and checks["divergent_family_phrase_variants"] == 2
            and checks["divergent_family_template_variants"] >= 2
            and checks["identical_family_not_reported"]
            and checks["planted_concept_detected"]
            and checks["planted_concept_definition_variants"] == 2
            and bool(checks["planted_proposal_definition"])
            and bool(checks["planted_proposal_template"])
            and checks["proposals_complete"]
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


_FRAGMENT_PREFIX = "sib-consistency-census-"
_SUMMARY_TOP = 10


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _examples(members: list[str], count: int = 2) -> str:
    return ", ".join(f"<code>{_esc(name)}</code>" for name in members[:count])


def _family_table(families: list[dict[str, Any]]) -> str:
    rows = []
    for family in families:
        divergence = family["divergence"]
        phrases = divergence["definition"]["phrase_variants"]
        variants = len(phrases) + divergence["definition"].get("variants_omitted", 0)
        templates = divergence["template"]["signature_variants"]
        rows.append(
            "<tr>"
            f"<td>{_esc(family['physical_base'])}</td>"
            f"<td>{family['member_count']}</td>"
            f"<td>{variants}</td>"
            f"<td>{len(templates)}</td>"
            f"<td>{'yes' if divergence['grouping']['inconsistent'] else 'no'}</td>"
            f"<td>{_esc(family['proposed']['definition_phrase'])}</td>"
            f"<td>{_esc(family['proposed']['template_signature'])}</td>"
            f"<td>{_examples(family['members'])}</td>"
            "</tr>"
        )
    return (
        f'<table id="{_FRAGMENT_PREFIX}top-families"><thead><tr>'
        "<th>physical_base</th><th>members</th><th>definition variants</th>"
        "<th>template variants</th><th>grouping inconsistent</th>"
        "<th>proposed definition phrase</th><th>proposed template</th>"
        "<th>example members</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _family_examples(families: list[dict[str, Any]]) -> str:
    blocks = []
    for family in families:
        phrases = family["divergence"]["definition"]["phrase_variants"]
        rows = [
            "<tr>"
            f"<td>{_esc(variant['phrase'])}</td>"
            f"<td>{variant['members']}</td>"
            f"<td><code>{_esc(variant['example'])}</code></td>"
            f"<td>{_esc(variant['description'])}</td>"
            "</tr>"
            for variant in phrases
        ]
        omitted = family["divergence"]["definition"].get("variants_omitted", 0)
        more = f" ({omitted} further variants omitted)" if omitted else ""
        proposed = family["proposed"]
        anchor = _esc(family["physical_base"]).replace("_", "-")
        blocks.append(
            f'<h5 id="{_FRAGMENT_PREFIX}examples-{anchor}">'
            f"{_esc(family['physical_base'])}: {len(phrases)} definition variants shown"
            f"{_esc(more)}</h5>"
            "<table><thead><tr><th>phrase</th><th>members</th>"
            "<th>example member</th><th>definition text</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            f"<p>Proposed definition from <code>{_esc(proposed['definition_from'])}</code> "
            f"(anchor rule <code>{_esc(proposed['anchor_rule'])}</code>, phrase "
            f"<em>{_esc(proposed['definition_phrase'])}</em> carried by "
            f"{proposed['definition_phrase_share']:.1%} of members): "
            f"{_esc(proposed['definition'])}</p>"
            f"<p>Proposed template <code>{_esc(proposed['template_signature'])}</code> "
            f"from <code>{_esc(proposed['template_from'])}</code> "
            f"({_esc(proposed['template_source'])}, "
            f"{proposed['template_share']:.1%} of members); domain "
            f"<code>{_esc(proposed['domain'])}</code>.</p>"
        )
    return "".join(blocks)


def render_fragment(result: dict[str, Any]) -> str:
    """Render the summary HTML fragment from the census result alone."""
    universe = result["universe"]
    families = result["families"]
    concepts = result["concepts"]
    proposals = result["proposals"]
    entry = result["witness"]["entry"]
    control = result["control"]
    all_families = families["families"]
    counted = {
        "definition": sum(
            1 for f in all_families if f["divergence"]["definition"]["divergent"]
        ),
        "template": sum(
            1 for f in all_families if f["divergence"]["template"]["divergent"]
        ),
        "grouping": sum(
            1 for f in all_families if f["divergence"]["grouping"]["inconsistent"]
        ),
    }
    largest = sorted(
        all_families, key=lambda f: (-f["member_count"], f["physical_base"])
    )[:_SUMMARY_TOP]
    proposed = entry["proposed"]
    top_concepts = sorted(
        (concept for concept in concepts["concepts"] if concept["divergent"]),
        key=lambda c: (-c["member_count"], c["concept"]),
    )[:_SUMMARY_TOP]
    checks = control["checks"]

    parts = [f'<div id="{_FRAGMENT_PREFIX}census">']
    parts.append(
        f'<h3 id="{_FRAGMENT_PREFIX}headline">{universe["names"]} live names, '
        f"{families['families_total']} families, {families['families_divergent']} "
        "divergent</h3>"
    )
    parts.append(
        "<p>Read-only pass over the live graph: names at <code>accepted</code> or "
        f"<code>drafted</code> ({_esc(json.dumps(universe['by_name_stage']))}), grouped "
        f"by <code>physical_base</code> ({universe['by_physical_base']} bases) and by the "
        "shared concepts their documentation defines. Three divergences are counted per "
        f"family: the phrase each member uses for the shared base concept, the "
        f"documentation template signature its accepted members follow, and the "
        f"<code>physics_domain</code> it is filed under. "
        f"{families['families_divergent']} of {families['families_total']} multi-member "
        f"families diverge on at least one ({counted['definition']} on definition phrase, "
        f"{counted['template']} on documentation template, {counted['grouping']} on "
        f"grouping); {concepts['concepts_divergent']} of {concepts['concepts_total']} "
        "shared-concept groups define the concept more than one way.</p>"
    )
    parts.append(_family_table(largest))
    parts.append(
        f'<h4 id="{_FRAGMENT_PREFIX}family-examples">Definition variants and texts of '
        "the largest families</h4>"
    )
    parts.append(
        "<p>Each row is one phrase the members use for the family's shared concept, with "
        "the number of members using it, one example member, and an example definition "
        "text. Divergence is judged on where the family's base phrase begins and ends in "
        "the first sentence: qualifiers before or after the base concept's own words do "
        "not count as a different definition.</p>"
    )
    parts.append(_family_examples(largest))
    parts.append(
        f'<h4 id="{_FRAGMENT_PREFIX}witness">Named witness: '
        f"{_esc(result['witness']['concept'])} written "
        f"{result['witness']['definition_variants']} ways across "
        f"{result['witness']['member_count']} members</h4>"
    )
    parts.append(
        "<p>The flux-surface average is one operator the catalog defines repeatedly, so "
        "its definitions are directly comparable. Each row is one distinct definition "
        "text and the members carrying it. The proposed single definition is the text "
        f"carried by the shared anchor <code>{_esc(proposed['definition_from'])}</code> "
        f"(anchor rule <code>{_esc(proposed['anchor_rule'])}</code>); "
        f"{proposed['members_using_it']} of {entry['member_count']} members already use "
        "that exact text.</p>"
    )
    parts.append(
        f'<table id="{_FRAGMENT_PREFIX}witness-variants"><thead><tr><th>members</th>'
        "<th>examples</th><th>definition text</th></tr></thead><tbody>"
        + "".join(
            "<tr>"
            f"<td>{variant['member_count']}</td>"
            f"<td>{_examples(variant['members'])}</td>"
            f"<td><code>{_esc(variant['definition'])}</code></td>"
            "</tr>"
            for variant in entry["variants"]
        )
        + "</tbody></table>"
    )
    parts.append(
        f'<pre id="{_FRAGMENT_PREFIX}witness-proposed">{_esc(proposed["definition"])}</pre>'
    )
    parts.append(
        f'<h4 id="{_FRAGMENT_PREFIX}top-concepts">Largest divergent shared concepts</h4>'
    )
    parts.append(
        f'<table id="{_FRAGMENT_PREFIX}top-concepts-table"><thead><tr><th>concept</th>'
        "<th>members</th><th>definition variants</th><th>lemma variants</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            f"<td>{_esc(concept['concept'])}</td>"
            f"<td>{concept['member_count']}</td>"
            f"<td>{concept['definition_variants']}</td>"
            f"<td>{_esc(', '.join(concept['lemma_variants']))}</td>"
            "</tr>"
            for concept in top_concepts
        )
        + "</tbody></table>"
    )
    parts.append(
        f'<p id="{_FRAGMENT_PREFIX}control">Planted control: a disposable graph holds a '
        "family with two divergent definitions and a second family whose two members are "
        "identical. The census counted "
        f"{checks['divergent_family_phrase_variants']} definition variants and "
        f"{checks['divergent_family_template_variants']} template variants in the "
        f"divergent family, found {checks['planted_concept_definition_variants']} "
        "definitions of the planted shared concept, did not report the identical family, "
        f"and proposed <code>{_esc(checks['planted_proposal_template'])}</code> with a "
        f"non-null definition for it: "
        f"<code>{'passed' if checks['passed'] else 'failed'}</code>. Every divergent "
        f"family in the live pass carries a proposed definition "
        f"({proposals['with_definition']} of {proposals['families']}) and template "
        f"({proposals['with_template']} of {proposals['families']}); proposals complete: "
        f"<code>{str(proposals['complete']).lower()}</code>. The live pass re-planted "
        "that graph and re-read the live catalog in one run; the log records "
        "<code>EXIT=0</code>.</p>"
    )
    parts.append("</div>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the census JSON here")
    parser.add_argument(
        "--html",
        type=Path,
        help="write the summary HTML fragment here, rendered from the same result",
    )
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

    if not result["proposals"]["complete"]:
        print(
            f"PROPOSALS INCOMPLETE: {len(result['proposals']['gaps'])}", file=sys.stderr
        )
        print(json.dumps(result["proposals"]["gaps"][:5], indent=1), file=sys.stderr)
        return 1

    if args.planted:
        result["control"] = planted_control()
        if not result["control"]["checks"]["passed"]:
            print("PLANTED CONTROL FAILED", file=sys.stderr)
            return 1

    families = result["families"]
    concepts = result["concepts"]
    proposals = result["proposals"]
    summary = (
        f"names={result['universe']['names']} "
        f"families={families['families_total']} divergent={families['families_divergent']} "
        f"concepts={concepts['concepts_total']} divergent={concepts['concepts_divergent']} "
        f"witness={result['witness']['concept']!r} "
        f"witness_variants={result['witness']['definition_variants']} "
        f"proposals={proposals['with_definition']}/{proposals['families']} "
        f"templates={proposals['with_template']}/{proposals['families']}"
    )
    print(summary)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")
        print(f"wrote {args.json} ({args.json.stat().st_size} bytes)")
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_fragment(result))
        print(f"wrote {args.html} ({args.html.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
