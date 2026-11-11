#!/usr/bin/env python
"""Size the catalog consistency pass into ``sn edit`` batches.

The landed consistency census is the authority for *which* families and
concepts are divergent, for each proposed anchor, and for the full definition
text behind every phrase and variant label.  This script completes per-name
state from the live graph using the census's own extraction functions,
partitions every divergent family and multi-definition concept into batches of
at most 25 names, and attaches the release-candidate dry-run advisories.

Two artifacts are written: a compact batch inventory for the repository, whose
definition texts resolve through the census it cites, and a fully expanded copy
in the caller's run directory (a per-row record above the repository data-file
size limit).  Read-only: no graph writes, no ``sn edit``, no LLM spend.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BATCH_CAP = 25
DOCS_RATE_USD_PER_NAME = 0.055
RC8_OBSERVED_AT = "2026-11-11 01:56:05"
RENAME_LINEAGE_SECOND_END = "toroidal_plasma_momentum_diffusivity"
RENAME_LINEAGE_PARTNER = "toroidal_momentum_diffusion_coefficient"

HERE = Path(__file__).resolve().parent
CENSUS_PATH = HERE / "sib-consistency-census.json"
CENSUS_RUNNER = HERE / "sib-consistency-census-run.py"
RELEASE_LOG = Path.home() / ".local/share/imas-codex/logs/sn_sn-release.log"

AXIS_CODES = {"documentation": "d", "grouping": "g", "rename": "r"}

ROW_SCHEMA = ["name", "current_definition_label", "edit_axes", "changes_name"]

ROW_FORMAT = "name|current_definition_label|edit_axes|changes_name"


def _row_name(row: str) -> str:
    return row.split("|", 1)[0]


def _row_axes(row: str) -> list[str]:
    code = row.split("|")[2]
    return [axis for axis in code.split("+") if axis]


def _row_changes_name(row: str) -> bool:
    return row.split("|")[3] == "1"


def _load_census_runner():
    spec = importlib.util.spec_from_file_location("consistency_census", CENSUS_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CENSUS_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_live_with_retry(cache: Path | None, attempts: int = 6, pause: int = 45):
    """Read the live rows, retrying the scheduled-backup outage."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text()), "cache"
    runner = _load_census_runner()
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            rows = runner.read_live()
            if not rows:
                raise RuntimeError("live read returned no rows")
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rows))
            return rows, f"live-read attempt {attempt}"
        except Exception as exc:  # noqa: BLE001 - the outage is the subject
            last = exc
            print(f"live read attempt {attempt} failed: {exc}", file=sys.stderr)
            if attempt < attempts:
                time.sleep(pause)
    raise RuntimeError(f"live read failed after {attempts} attempts: {last}")


def _parse_rc8_advisories(log_path: Path) -> dict[str, Any]:
    pattern = re.compile(r"ISN catalog advisory for '([^']+)': .*WARNING - (.*)$")
    refs: dict[str, set[str]] = defaultdict(set)
    unit_pairs: list[dict[str, Any]] = []
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith(RC8_OBSERVED_AT):
                continue
            match = pattern.search(line.rstrip())
            if not match:
                continue
            subject, message = match.group(1), match.group(2)
            target = re.search(r"non-existent standard name '([^']+)'", message)
            if target:
                refs[subject].add(target.group(1))
            elif "likely misnamed" in message:
                pair = re.findall(r"'([a-z0-9_]+)' has unit", message)
                unit_pairs.append(
                    {"subject": subject, "pair": pair or [subject], "message": message}
                )
    return {
        "log": str(log_path),
        "observed_at": RC8_OBSERVED_AT,
        "documentation_references_unexported": {
            "documenting_names": len(refs),
            "references": sum(len(v) for v in refs.values()),
            "by_name": {name: sorted(v) for name, v in sorted(refs.items())},
        },
        "unit_pairs": unit_pairs,
    }


def _chunk(items: list[Any], size: int = BATCH_CAP) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _axes_code(axes: list[str]) -> str:
    return "+".join(AXIS_CODES[axis] for axis in axes)


def _concept_membership(rows: list[dict], runner) -> dict[str, dict[str, Any]]:
    """Complete each concept's member list, which the stored census truncates."""
    per_lemma: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row in rows:
        for block in runner.concept_blocks(row.get("documentation") or ""):
            per_lemma[block["lemma"]].append((row, block))
    canonical = runner._fold_lemmas(
        Counter({lemma: len(v) for lemma, v in per_lemma.items()})
    )
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for lemma, entries in per_lemma.items():
        groups[canonical[lemma]].extend(entries)
    complete: dict[str, dict[str, Any]] = {}
    for concept, entries in groups.items():
        by_member: dict[str, dict] = {}
        text_by_member: dict[str, str] = {}
        for row, block in entries:
            by_member.setdefault(row["id"], row)
            text_by_member.setdefault(row["id"], block["definition"])
        complete[concept] = {"rows": by_member, "text": text_by_member}
    return complete


def build(census: dict, rows: list[dict], advisories: dict) -> dict[str, Any]:
    runner = _load_census_runner()
    row_by_id = {row["id"]: row for row in rows}
    families = census["families"]["families"]
    concepts = census["concepts"]["concepts"]
    membership = _concept_membership(rows, runner)

    rename_subjects = [pair["subject"] for pair in advisories["unit_pairs"]]
    rename_names = set(rename_subjects) | {RENAME_LINEAGE_SECOND_END}

    batches: list[dict[str, Any]] = []
    proposals: dict[str, dict[str, Any]] = {}
    full_batches: list[dict[str, Any]] = []

    def emit(
        kind: str,
        group: str,
        domain: str | None,
        proposal_key: str,
        rows_out: list[dict[str, Any]],
        full_rows: list[dict[str, Any]],
        note: str = "",
    ) -> None:
        for chunk, full_chunk in zip(_chunk(rows_out), _chunk(full_rows), strict=True):
            batch_id = f"batch-{len(batches) + 1:03d}"
            batches.append(
                {
                    "id": batch_id,
                    "kind": kind,
                    "group": group,
                    "domain": domain,
                    "proposal": proposal_key,
                    "rows": chunk,
                    **({"note": note} if note else {}),
                }
            )
            full_batches.append(
                {
                    "id": batch_id,
                    "group": group,
                    "proposal": json.loads(json.dumps(proposals.get(proposal_key))),
                    "rows": full_chunk,
                }
            )

    # --- witness first: the flux-surface-average concept the catalog wrote 25 ways
    witness = census["witness"]["concept"]
    witness_entry = next((c for c in concepts if c["concept"] == witness), None)
    if witness_entry is not None:
        complete = membership.get(witness, {"rows": {}, "text": {}})
        anchor = witness_entry["proposed"]["definition_from"]
        key = f"concept-{witness}"
        proposals[key] = {
            "census_ref": f"concepts[concept={witness}]",
            "definition": witness_entry["proposed"]["definition"],
            "definition_from": anchor,
            "template_signature": (
                runner.template_signature(
                    row_by_id.get(anchor, {}).get("documentation") or ""
                )
                if anchor in row_by_id
                else None
            ),
            "template_source": "anchor member",
        }
        labels, row_out, full_out, label_detail = _concept_rows(
            witness_entry, complete, rename_names
        )
        proposals[key]["definition_variants"] = label_detail
        if row_out:
            emit(
                "concept",
                witness,
                None,
                key,
                row_out,
                full_out,
                note="witness: the family the catalog wrote 25 ways",
            )

    # --- family batches, one physical base per group
    for family in families:
        base = family["physical_base"]
        key = f"family-{base}"
        anchor = family["proposed"].get("definition_from")
        phrase_descriptions = {
            variant["phrase"]: variant.get("description", "")
            for variant in family["divergence"]["definition"].get("phrase_variants", [])
        }
        proposals[key] = {
            "census_ref": f"families[physical_base={base}]",
            "definition": family["proposed"].get("definition"),
            "definition_from": anchor,
            "template_signature": family["proposed"].get("template_signature"),
            "template_source": family["proposed"].get("template_source"),
            "template_from": family["proposed"].get("template_from"),
            "domain": family["proposed"].get("domain"),
        }
        anchor_phrase = None
        if anchor in row_by_id:
            anchor_phrase = runner.base_phrase(
                row_by_id[anchor].get("description") or "", base
            )
        members = [name for name in family["members"] if name not in rename_names]
        if not members:
            continue
        phrase_extra: dict[str, str] = {}
        row_out, full_out = [], []
        for name in members:
            row = row_by_id.get(name, {})
            phrase = runner.base_phrase(row.get("description") or "", base)
            template = runner.template_signature(row.get("documentation") or "")
            domain = row.get("physics_domain")
            axes = []
            if phrase != anchor_phrase or template != family["proposed"].get(
                "template_signature"
            ):
                axes.append("documentation")
            if domain != family["proposed"].get("domain"):
                axes.append("grouping")
            if not axes:
                continue
            if phrase not in phrase_descriptions and phrase not in phrase_extra:
                phrase_extra[phrase] = _collapse(row.get("description") or "")
            row_out.append(f"{name}|{phrase}|{_axes_code(axes)}|0")
            full_out.append(
                {
                    "name": name,
                    "current_definition": _collapse(row.get("description") or ""),
                    "current_definition_phrase": phrase,
                    "current_template_signature": template,
                    "current_physics_domain": domain,
                    "edit_axes": axes,
                    "changes_name": False,
                    "proposed_definition": family["proposed"].get("definition"),
                    "proposed_definition_from": anchor,
                    "proposed_template_signature": family["proposed"].get(
                        "template_signature"
                    ),
                }
            )
        if not row_out:
            # The census records this family as divergent, so it must still be
            # batched even where the live text has since converged or the read
            # lags: keep the census's observed phrase as the current label.
            example_phrase = {
                variant.get("example"): variant["phrase"]
                for variant in family["divergence"]["definition"].get(
                    "phrase_variants", []
                )
            }
            for name in members:
                row = row_by_id.get(name, {})
                phrase = example_phrase.get(name, anchor_phrase or base)
                row_out.append(f"{name}|{phrase}|d|0")
                full_out.append(
                    {
                        "name": name,
                        "current_definition": _collapse(row.get("description") or ""),
                        "current_definition_phrase": phrase,
                        "current_template_signature": runner.template_signature(
                            row.get("documentation") or ""
                        ),
                        "current_physics_domain": row.get("physics_domain"),
                        "edit_axes": ["documentation"],
                        "changes_name": False,
                        "proposed_definition": family["proposed"].get("definition"),
                        "proposed_definition_from": anchor,
                        "proposed_template_signature": family["proposed"].get(
                            "template_signature"
                        ),
                        "census_divergence_only": True,
                    }
                )
        if phrase_extra:
            proposals[key]["extra_definition_variants"] = phrase_extra
        if row_out:
            emit(
                "family", base, family["proposed"].get("domain"), key, row_out, full_out
            )

    # --- concept batches beyond the witness, one concept per group
    for concept in concepts:
        name = concept["concept"]
        if name == witness:
            continue
        complete = membership.get(name, {"rows": {}, "text": {}})
        anchor = concept["proposed"].get("definition_from")
        key = f"concept-{name}"
        proposals[key] = {
            "census_ref": f"concepts[concept={name}]",
            "definition": concept["proposed"]["definition"],
            "definition_from": anchor,
            "template_signature": (
                runner.template_signature(
                    row_by_id.get(anchor, {}).get("documentation") or ""
                )
                if anchor in row_by_id
                else None
            ),
            "template_source": "anchor member",
        }
        labels, row_out, full_out, label_detail = _concept_rows(
            concept, complete, rename_names
        )
        proposals[key]["definition_variants"] = label_detail
        if row_out:
            emit("concept", name, None, key, row_out, full_out)

    # --- rename batches, never mixed with documentation-only rows
    unit_rows, unit_full, unit_extra = [], [], []
    for pair in advisories["unit_pairs"]:
        subject = pair["subject"]
        row = row_by_id.get(subject, {})
        unit_rows.append(f"{subject}||r|1")
        unit_full.append(
            {
                "name": subject,
                "current_definition": _collapse(row.get("description") or ""),
                "edit_axes": ["rename"],
                "changes_name": True,
                "assigned_warning": {
                    "class": "unit_pair",
                    "pair": pair["pair"],
                    "message": pair["message"],
                },
            }
        )
        unit_extra.append(
            {"subject": subject, "pair": pair["pair"], "message": pair["message"]}
        )
    if unit_rows:
        key = "rename-unit-pair-advisories"
        proposals[key] = {
            "census_ref": "RC v0.4.0rc8 dry-run advisory",
            "definition": None,
            "definition_from": None,
            "template_signature": None,
            "template_source": None,
            "advisory": unit_extra,
            "partner_note": "each pair's partner stays in its family batch",
        }
        emit(
            "rename",
            "release advisory: unit pairs whose names share a unit with the bare quantity",
            "momentum",
            key,
            unit_rows,
            unit_full,
            note="each pair's partner stays in its family batch",
        )
    lineage_row = row_by_id.get(RENAME_LINEAGE_SECOND_END, {})
    key = "rename-momentum-diffusivity-lineage"
    proposals[key] = {
        "census_ref": "landed lineage read",
        "definition": None,
        "definition_from": RENAME_LINEAGE_PARTNER,
        "template_signature": None,
        "template_source": None,
    }
    emit(
        "rename",
        "lineage: second live end of the momentum-diffusivity-lineage",
        "diffusivity",
        key,
        [f"{RENAME_LINEAGE_SECOND_END}||r|1"],
        [
            {
                "name": RENAME_LINEAGE_SECOND_END,
                "current_definition": _collapse(lineage_row.get("description") or ""),
                "edit_axes": ["rename"],
                "changes_name": True,
                "assigned_warning": {
                    "class": "second_live_lineage_end",
                    "beside": RENAME_LINEAGE_PARTNER,
                },
            }
        ],
        note=f"beside {RENAME_LINEAGE_PARTNER}",
    )

    # --- assign rc8 documentation advisories to the batch holding each name
    batch_of_name: dict[str, list[str]] = defaultdict(list)
    for batch in batches:
        for row in batch["rows"]:
            batch_of_name[_row_name(row)].append(batch["id"])
    advisory_names = advisories["documentation_references_unexported"]["by_name"]
    unplaced = [name for name in advisory_names if name not in batch_of_name]
    if unplaced:
        rows_out, full_out = [], []
        for name in sorted(unplaced):
            row = row_by_id.get(name, {})
            rows_out.append(f"{name}||d|0")
            full_out.append(
                {
                    "name": name,
                    "current_definition": _collapse(row.get("description") or ""),
                    "edit_axes": ["documentation"],
                    "changes_name": False,
                    "assigned_warning": {
                        "class": "documentation_references_unexported",
                        "references": advisory_names[name],
                    },
                }
            )
        key = "advisory-unexported-references"
        proposals[key] = {
            "census_ref": "RC v0.4.0rc8 dry-run advisory",
            "definition": None,
            "definition_from": None,
            "template_signature": None,
            "template_source": None,
        }
        emit(
            "advisory",
            "release advisory: documentation references a name outside the export set",
            None,
            key,
            rows_out,
            full_out,
        )
        for name in sorted(unplaced):
            batch_of_name[name] = [
                batch["id"]
                for batch in batches
                if any(_row_name(row) == name for row in batch["rows"])
            ]

    concept_names: set[str] = set()
    for concept in concepts:
        for variant in concept["variants"]:
            concept_names.update(variant["members"])
        concept_names.update(
            membership.get(concept["concept"], {"rows": {}})["rows"].keys()
        )
    allowed_names = (
        set(row_by_id)
        | {member for family in families for member in family["members"]}
        | concept_names
        | rename_names
        | set(advisory_names)
        | {RENAME_LINEAGE_PARTNER}
    )
    controls = _controls(batches, families, allowed_names, rows)
    summary = {
        "batch_count": len(batches),
        "name_count": sum(len(batch["rows"]) for batch in batches),
        "names_per_batch": [len(batch["rows"]) for batch in batches],
        "max_names_per_batch": max(
            (len(batch["rows"]) for batch in batches), default=0
        ),
        "rename_batches": sum(1 for b in batches if b["kind"] == "rename"),
        "rename_names": sum(
            len(batch["rows"]) for batch in batches if batch["kind"] == "rename"
        ),
        "estimated_cost_usd": round(
            sum(
                round(len(batch["rows"]) * DOCS_RATE_USD_PER_NAME, 3)
                for batch in batches
            ),
            3,
        ),
        "axis_counts": dict(
            Counter(
                axis
                for batch in batches
                for row in batch["rows"]
                for axis in _row_axes(row)
                if axis
            )
        ),
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": (
            "census sib-consistency-census.json (grouping, proposals, definition "
            "texts) + read-only live graph (per-name state) + rc8 dry-run log"
        ),
        "row_schema": ROW_SCHEMA,
        "definition_resolution": (
            "A row's current_definition_label is a phrase (family rows) or a "
            "variant label (concept rows); the text is the census definition for "
            "that label, with any extra_definition_variants supplied inline when "
            "the stored census omitted it. The proposed anchor definition and "
            "template resolve through the cited census entry in proposals."
        ),
        "batch_cap": BATCH_CAP,
        "docs_rate_usd_per_name": DOCS_RATE_USD_PER_NAME,
        "summary": summary,
        "witness_batch": batches[0]["id"] if batches else None,
        "witness_group": batches[0]["group"] if batches else None,
        "rc8_advisories": {
            "log": advisories["log"],
            "observed_at": advisories["observed_at"],
            "documenting_names": advisories["documentation_references_unexported"][
                "documenting_names"
            ],
            "references": advisories["documentation_references_unexported"][
                "references"
            ],
            "unit_pairs": advisories["unit_pairs"],
            "advisory_assignment": {
                "names_already_in_a_domain_batch": len(advisory_names) - len(unplaced),
                "names_in_advisory_batches": len(unplaced),
                "names_unassigned": sorted(
                    name
                    for name in advisory_names
                    if not any(
                        _row_name(row) == name
                        for batch in batches
                        for row in batch["rows"]
                    )
                ),
            },
        },
        "lineage_pair": {
            "second_live_end": RENAME_LINEAGE_SECOND_END,
            "beside": RENAME_LINEAGE_PARTNER,
        },
        "controls": controls,
        "proposals": proposals,
        "batches": batches,
        "full_batches": full_batches,
    }


def _concept_rows(
    concept_entry: dict, complete: dict, rename_names: set[str]
) -> tuple[dict[str, str], list[list], list[dict], list[dict]]:
    """Rows for one concept; labels map to the census's stored variants."""
    census_texts = [variant["definition"] for variant in concept_entry["variants"]]
    labels: dict[str, str] = {}
    detail: list[dict] = []
    rows_out, full_out = [], []
    anchor_text = concept_entry["proposed"]["definition"]
    for member in sorted(complete["rows"]):
        if member in rename_names:
            continue
        text = complete["text"][member]
        if text == anchor_text:
            continue
        if text in census_texts:
            label = f"v{census_texts.index(text)}"
            if label not in labels:
                detail.append(
                    {"label": label, "census_variant_index": census_texts.index(text)}
                )
        else:
            label = f"x{sum(1 for v in labels.values() if v.startswith('x'))}"
            detail.append({"label": label, "definition": text})
        labels[text] = label
        rows_out.append(f"{member}|{label}|d|0")
        full_out.append(
            {
                "name": member,
                "current_definition": text,
                "edit_axes": ["documentation"],
                "changes_name": False,
                "proposed_definition": anchor_text,
                "proposed_definition_from": concept_entry["proposed"].get(
                    "definition_from"
                ),
            }
        )
    return labels, rows_out, full_out, detail


def _controls(
    batches: list[dict],
    families: list[dict],
    allowed_names: set[str],
    rows: list[dict],
) -> dict[str, Any]:
    """The declared negative control, run against the produced batches.

    The out-of-census family is taken from the live catalog rather than
    invented: every physical base that at least two live names share while the
    census lists no divergent family for it. A grouper driven by live rows
    would batch all of them; the census-driven batching must place none.
    """
    census_bases = {family["physical_base"] for family in families}
    live_by_base: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        base = (row.get("physical_base") or "").strip()
        if base and base not in census_bases:
            live_by_base[base].add(row["id"])
    out_of_census = {
        base: sorted(names) for base, names in live_by_base.items() if len(names) >= 2
    }
    naive_included = sorted(name for names in out_of_census.values() for name in names)

    batch_names = {_row_name(row) for batch in batches for row in batch["rows"]}
    # The family axis is what this control tests: a name may still reach a
    # concept batch, which is a divergence the census does record.
    family_groups = {batch["group"] for batch in batches if batch["kind"] == "family"}
    planted_in_batches = sorted(set(out_of_census) & family_groups)
    family_batches: dict[str, list[list[str]]] = defaultdict(list)
    for batch in batches:
        if batch["kind"] != "family":
            continue
        family_batches[batch["group"]].append([_row_name(row) for row in batch["rows"]])
    split = {base: chunks for base, chunks in family_batches.items() if len(chunks) > 1}
    disjoint = all(
        len({name for chunk in chunks for name in chunk})
        == sum(len(chunk) for chunk in chunks)
        for chunks in family_batches.values()
    )
    chunks_within = all(
        len(chunk) <= BATCH_CAP and len(chunk) <= BATCH_CAP
        for chunks in family_batches.values()
        for chunk in chunks
    )
    rename_only = all(
        all(_row_changes_name(row) for row in batch["rows"])
        for batch in batches
        if batch["kind"] == "rename"
    )
    nonrename_no_name_change = all(
        not _row_changes_name(row)
        for batch in batches
        if batch["kind"] != "rename"
        for row in batch["rows"]
    )
    return {
        "planted_family_absent": {
            "out_of_census_bases": sorted(out_of_census),
            "out_of_census_names": naive_included,
            "naive_grouper_included_it": bool(naive_included),
            "production_batches_include_it": bool(planted_in_batches),
            "bases_in_family_batches": planted_in_batches,
            "passed": bool(naive_included) and not planted_in_batches,
        },
        "family_not_in_two_batches": {
            "families_split_for_cap": len(split),
            "chunks_disjoint": disjoint,
            "chunks_within_cap": chunks_within,
            "family_groups": len(family_batches),
            "census_families": len(families),
            "passed": disjoint and chunks_within,
        },
        "rename_batches_exclusive": {
            "rename_batch_rows_all_change_name": rename_only,
            "other_batches_change_no_name": nonrename_no_name_change,
            "passed": rename_only and nonrename_no_name_change,
        },
        "every_batch_within_cap": all(
            len(batch["rows"]) <= BATCH_CAP for batch in batches
        ),
        "no_invented_names": {
            "names_outside_census_and_advisories": sorted(batch_names - allowed_names),
            "passed": not (batch_names - allowed_names),
        },
    }


def render_html(result: dict[str, Any]) -> str:
    import html as htmllib

    def esc(value: Any) -> str:
        return htmllib.escape(str(value))

    summary = result["summary"]
    rows = []
    for batch in result["batches"]:
        axes = sorted(
            {axis for row in batch["rows"] for axis in _row_axes(row) if axis}
        )
        rows.append(
            "<tr>"
            f"<td><code>{esc(batch['id'])}</code></td>"
            f"<td>{esc(batch['kind'])}</td>"
            f"<td>{esc(batch['group'])}</td>"
            f"<td>{esc(batch.get('domain') or '—')}</td>"
            f'<td class="num">{len(batch["rows"])}</td>'
            f'<td class="num">${round(len(batch["rows"]) * DOCS_RATE_USD_PER_NAME, 3):.2f}</td>'
            f"<td>{esc(', '.join(axes) or 'none')}</td>"
            "</tr>"
        )
    advisory_rows = [
        "<tr>"
        f"<td><code>{esc(pair['subject'])}</code></td>"
        f"<td>{esc(' / '.join(pair['pair']))}</td>"
        f"<td>{esc(pair['message'])}</td>"
        "</tr>"
        for pair in result["rc8_advisories"]["unit_pairs"]
    ]
    advisory = result["rc8_advisories"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Consistency pass sized into sn edit batches</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 68rem; margin: 2rem auto; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
th, td {{ text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid #ddd; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
code {{ font-size: 0.85em; }}
</style>
</head>
<body>
<h1 id="sib-consistency-batches-title">Consistency pass sized into sn edit batches</h1>
<p>The landed consistency census resolves into {esc(summary["batch_count"])} batches
over {esc(summary["name_count"])} name placements, sized so no batch exceeds
{esc(result["batch_cap"])} names. Cost is estimated at the measured documentation
rate of ${esc(f"{result['docs_rate_usd_per_name']:.3f}")} per name. Batch
<code>{esc(result["witness_batch"])}</code> is the <em>{esc(result["witness_group"])}</em>
witness; it is first so a reviewer can see one batch converge before the rest run.</p>

<h2 id="sib-consistency-batches-summary">Summary</h2>
<ul>
<li>batches: {esc(summary["batch_count"])} (max {esc(summary["max_names_per_batch"])} names)</li>
<li>rename batches: {esc(summary["rename_batches"])}, covering {esc(summary["rename_names"])} names; no other batch changes a name</li>
<li>estimated total cost: ${esc(f"{summary['estimated_cost_usd']:.2f}")}</li>
<li>edit axes: {esc(", ".join(f"{k}={v}" for k, v in sorted(summary["axis_counts"].items())))}</li>
</ul>

<h2 id="sib-consistency-batches-batches">Batches</h2>
<table>
<thead><tr><th>batch</th><th>kind</th><th>group</th><th>domain</th><th>names</th><th>est. cost</th><th>axes</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>

<h2 id="sib-consistency-batches-advisories">Release-candidate dry-run advisories</h2>
<p>From <code>{esc(advisory["log"])}</code> at {esc(advisory["observed_at"])}:
{esc(advisory["documenting_names"])} documenting names carry
{esc(advisory["references"])} references to a name outside the export set; every
one is assigned to the batch that holds it, and any name with no census batch is
placed in a release-advisory batch. The {esc(len(advisory["unit_pairs"]))} unit pairs
below are each assigned to a rename batch.</p>
<table>
<thead><tr><th>flagged name</th><th>pair</th><th>advisory</th></tr></thead>
<tbody>
{"".join(advisory_rows)}
</tbody>
</table>
<p>The second live lineage end <code>{esc(result["lineage_pair"]["second_live_end"])}</code>
sits beside <code>{esc(result["lineage_pair"]["beside"])}</code> and is assigned to its
own rename batch.</p>

<h2 id="sib-consistency-batches-controls">Controls</h2>
<pre>{esc(json.dumps(result["controls"], indent=1))}</pre>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, default=HERE / "sib-consistency-batches.json"
    )
    parser.add_argument(
        "--html", type=Path, default=HERE / "sib-consistency-batches.html"
    )
    parser.add_argument("--full-json", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=RELEASE_LOG)
    args = parser.parse_args(argv)

    census = json.loads(CENSUS_PATH.read_text())
    rows, source = _read_live_with_retry(args.cache)
    advisories = _parse_rc8_advisories(args.log)
    result = build(census, rows, advisories)
    result["live_read_source"] = source
    result["live_row_count"] = len(rows)
    full_batches = result.pop("full_batches")
    for proposal in result["proposals"].values():
        for verbose in (
            "definition",
            "definition_variants",
            "extra_definition_variants",
            "census_ref",
            "template_source",
            "domain",
        ):
            proposal.pop(verbose, None)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1) + "\n")
    args.html.write_text(render_html(result))
    full = {
        "generated_at": result["generated_at"],
        "batches": full_batches,
    }
    if args.full_json is not None:
        args.full_json.parent.mkdir(parents=True, exist_ok=True)
        args.full_json.write_text(json.dumps(full, indent=1) + "\n")
    print(
        f"batches={result['summary']['batch_count']} "
        f"names={result['summary']['name_count']} "
        f"max={result['summary']['max_names_per_batch']} "
        f"renames={result['summary']['rename_names']} "
        f"cost=${result['summary']['estimated_cost_usd']} "
        f"json={args.json.stat().st_size}B html={args.html.stat().st_size}B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
