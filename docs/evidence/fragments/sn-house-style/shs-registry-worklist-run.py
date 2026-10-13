#!/usr/bin/env python
"""Census the convention entries the frozen WEST roster resolves to but the
conventions registry lacks.

The conventions registry lives in ISN (imas-standard-names) beside the
vocabularies it is keyed by. This driver reads the frozen roster, parses every
name with the public grammar parser, and reports the tokens and patterns the
parse produces for which the registry carries no entry. It writes the four
worklists the plan's section-2 table asks for — concepts, notation, templates —
and lists every name the parser refuses. It never decides which kinds truly
need an entry; that is the coordinator's call.

Run against the live registry (writes the results JSON beside this script):

    uv run --no-sync python docs/evidence/fragments/sn-house-style/shs-registry-worklist-run.py

Point it at a scratch copy of the registry to prove the census can report a gap:

    uv run --no-sync python .../shs-registry-worklist-run.py \\
        --conventions-dir /scratch/conventions

The token-to-segment-kind split mirrors the accessor's own reading of a parse
(``imas_standard_names.grammar.terms``), so the census and the accessor agree
about which token fills which segment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from imas_standard_names.grammar import parse
from imas_standard_names.grammar.terms import _qualifier_segment_kind
from imas_standard_names.grammar.vocab_loaders import load_conventions

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]
ROSTER = (
    REPO_ROOT
    / "imas_codex/standard_names/manifests/reviews"
    / "v0.4.0rc7+west-task-2e.sn_names.yaml"
)
DEFAULT_RESULTS = SCRIPT.with_name("shs-registry-worklist-results.json")

# Segment kinds each registry file is keyed by, following the plan's section-2
# table. A token's kind decides which worklist it can appear in.
CONCEPT_KINDS = frozenset(
    {
        "transformation",  # operators
        "physical_base",
        "geometric_base",
        "coordinate",
        "object",  # locus, by LocusType
        "position",
        "geometry",
        "region",
    }
)
NOTATION_KINDS = frozenset(
    {"physical_base", "geometric_base", "subject", "component", "coordinate"}
)


def _locus_segment_kind(locus_type) -> str:
    return {
        "ENTITY": "object",
        "POSITION": "position",
        "GEOMETRY": "geometry",
        "REGION": "region",
    }[locus_type.name]


def _iter_operators(operators):
    for operator in operators:
        yield operator
        for argument in operator.args:
            yield from _iter_operators(argument.operators)


def _walk(ir):
    """Yield ``(role, segment_kind, token)`` for every grammar token the parse
    of a name names. Roles distinguish a locus token from a locus qualifier so
    the census can report each separately."""
    for operator in _iter_operators(ir.operators):
        yield ("operator", "transformation", operator.op)
    for qualifier in ir.qualifiers:
        yield ("qualifier", _qualifier_segment_kind(qualifier.token), qualifier.token)
    base_kind = "physical_base" if ir.base.kind.name == "QUANTITY" else "geometric_base"
    yield ("base", base_kind, ir.base.token)
    if ir.projection is not None:
        proj_kind = (
            "component" if ir.projection.shape.name == "COMPONENT" else "coordinate"
        )
        yield ("projection", proj_kind, ir.projection.axis)
    if ir.locus is not None:
        yield ("locus", _locus_segment_kind(ir.locus.type), ir.locus.token)
        for qualifier in ir.locus.qualifiers:
            yield ("locus_qualifier", _qualifier_segment_kind(qualifier), qualifier)
    if ir.mechanism is not None:
        yield ("process", "process", ir.mechanism.token)


def _filled_segments(ir) -> frozenset[str]:
    """The segment kinds the parse of a name fills — the pattern key that
    selects a template, matching ``terms._ir_segments``."""
    kinds: set[str] = set()
    if ir.operators:
        kinds.add("transformation")
    if ir.projection is not None:
        kinds.add(
            "component" if ir.projection.shape.name == "COMPONENT" else "coordinate"
        )
    kinds.update(_qualifier_segment_kind(q.token) for q in ir.qualifiers)
    kinds.add("physical_base" if ir.base.kind.name == "QUANTITY" else "geometric_base")
    if ir.locus is not None:
        kinds.add(_locus_segment_kind(ir.locus.type))
    if ir.mechanism is not None:
        kinds.add("process")
    return frozenset(kinds)


def _template_matches(ir, templates) -> bool:
    return any(
        entry.base == ir.base.token and set(entry.segments) <= _filled_segments(ir)
        for entry in templates
    )


def _token_in_concepts(reg, token: str) -> bool:
    return token in reg.concepts.concepts


def _token_in_notation(reg, token: str) -> bool:
    return token in reg.notation.notation


def census(registry_dir: Path | None) -> dict:
    roster = yaml.safe_load(ROSTER.read_text())
    names = roster["names"]
    reg = load_conventions(registry_dir)

    concepts_use: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "names": set(), "kind": ""}
    )
    notation_use: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "names": set(), "kind": ""}
    )
    kind_concepts: dict[str, Counter] = defaultdict(Counter)
    kind_notation: dict[str, Counter] = defaultdict(Counter)
    concept_hits: dict[str, set[str]] = defaultdict(set)
    notation_hits: dict[str, set[str]] = defaultdict(set)
    patterns_use: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "names": set()})
    template_hit_patterns: set[tuple] = set()
    unparsed: list[str] = []
    parsed = 0

    for name in names:
        try:
            ir = parse(name, strict=True).ir
        except Exception as exc:  # noqa: BLE001 - a name the grammar refuses
            unparsed.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        parsed += 1
        for _role, kind, token in _walk(ir):
            if kind in CONCEPT_KINDS:
                kind_concepts[kind][token] += 1
                if _token_in_concepts(reg, token):
                    concept_hits[kind].add(token)
                else:
                    concepts_use[token]["count"] += 1
                    concepts_use[token]["names"].add(name)
                    concepts_use[token]["kind"] = kind
            if kind in NOTATION_KINDS:
                kind_notation[kind][token] += 1
                if _token_in_notation(reg, token):
                    notation_hits[kind].add(token)
                else:
                    notation_use[token]["count"] += 1
                    notation_use[token]["names"].add(name)
                    notation_use[token]["kind"] = kind
        pattern = (ir.base.token, tuple(sorted(_filled_segments(ir))))
        patterns_use[pattern]["count"] += 1
        patterns_use[pattern]["names"].add(name)
        if _template_matches(ir, reg.templates.templates):
            template_hit_patterns.add(pattern)

    def _pattern_row(pattern: str, info: dict) -> dict:
        return {
            "base": pattern[0],
            "segments": list(pattern[1]),
            "count": info["count"],
            "names": sorted(info["names"])[:5],
        }

    templates_gap = [
        _pattern_row(p, info)
        for p, info in patterns_use.items()
        if p not in template_hit_patterns
    ]
    templates_gap.sort(key=lambda r: (-r["count"], r["base"], tuple(r["segments"])))
    templates_hit = [
        _pattern_row(p, info)
        for p, info in patterns_use.items()
        if p in template_hit_patterns
    ]
    templates_hit.sort(key=lambda r: (-r["count"], r["base"], tuple(r["segments"])))

    concepts_rows = [
        {
            "token": t,
            "kind": i["kind"],
            "count": i["count"],
            "names": sorted(i["names"])[:5],
        }
        for t, i in concepts_use.items()
    ]
    concepts_rows.sort(key=lambda r: (-r["count"], r["token"]))
    notation_rows = [
        {
            "token": t,
            "kind": i["kind"],
            "count": i["count"],
            "names": sorted(i["names"])[:5],
        }
        for t, i in notation_use.items()
    ]
    notation_rows.sort(key=lambda r: (-r["count"], r["token"]))

    def _kind_totals(
        kind_counts: dict[str, Counter], hits: dict[str, set[str]]
    ) -> dict:
        return {
            kind: {
                "distinct_tokens": len(counter),
                "with_entry": len(hits.get(kind, set())),
                "missing": len(counter) - len(hits.get(kind, set())),
            }
            for kind, counter in sorted(kind_counts.items())
        }

    distinct_base = len({p[0] for p in patterns_use})
    return {
        "framework": "conventions-registry-gap-census",
        "registry_dir": str(registry_dir) if registry_dir else "(live: ISN default)",
        "roster": str(ROSTER.relative_to(REPO_ROOT)),
        "totals": {
            "roster_names": len(names),
            "parsed": parsed,
            "unparsed": len(unparsed),
            "concepts": {
                "distinct_tokens": len(concepts_use),
                "kinds": _kind_totals(kind_concepts, concept_hits),
            },
            "notation": {
                "distinct_tokens": len(notation_use),
                "kinds": _kind_totals(kind_notation, notation_hits),
            },
            "templates": {
                "distinct_patterns": len(patterns_use),
                "distinct_bases": distinct_base,
                "with_template": len(template_hit_patterns),
                "missing": len(patterns_use) - len(template_hit_patterns),
            },
        },
        "concepts": concepts_rows,
        "notation": notation_rows,
        "templates": templates_gap,
        "templates_satisfied": templates_hit,
        "unparsed": unparsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conventions-dir",
        type=Path,
        default=None,
        help="Scratch copy of the conventions directory to read instead of ISN's.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    result = census(args.conventions_dir)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    totals = result["totals"]
    print(
        f"roster names: {totals['roster_names']} parsed: {totals['parsed']} unparsed: {totals['unparsed']}"
    )
    print(f"concepts missing: {totals['concepts']['distinct_tokens']}")
    print(f"notation missing: {totals['notation']['distinct_tokens']}")
    print(
        f"templates missing: {totals['templates']['missing']} of {totals['templates']['distinct_patterns']} patterns"
    )
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
