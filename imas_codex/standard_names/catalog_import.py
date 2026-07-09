"""Catalog check — compare reviewed YAML entries against the graph (read-only).

``check_catalog`` reports which names are only in the catalog, only in the
graph, or present in both but with differing editorial fields.  It never
writes to the graph.

Two other paths own the write side and do NOT live here:

- restoring / bootstrapping the graph from a published catalog (which replays
  each entry's ``sources:`` block and rebuilds provenance) is the diff-by-id
  reconciler ``catalog_reconcile.reconcile_catalog``;
- folding reviewed curator edits from a merged catalog PR back into the ledger
  is ``sn merge``.

This module only compares; it never recreates nodes and never rebuilds
provenance.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Computed fields — re-derived from graph edges (HAS_PARENT / HAS_ERROR) on
#: export.  A curator edit to one has no effect (it is overwritten on the next
#: export), so ``check_catalog`` warns about it and strips it before ISN model
#: validation; it is never compared against the graph.
COMPUTED_FIELDS: frozenset[str] = frozenset({"arguments", "error_variants"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Summary of a catalog-vs-graph sync check."""

    only_in_catalog: list[str] = field(default_factory=list)
    only_in_graph: list[str] = field(default_factory=list)
    diverged: list[dict[str, Any]] = field(default_factory=list)
    in_sync: int = 0
    catalog_commit_sha: str | None = None
    graph_commit_sha: str | None = None


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _resolve_catalog_sha(catalog_dir: Path) -> str | None:
    """Resolve the git HEAD SHA of the catalog directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(catalog_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            logger.debug("Catalog commit SHA: %s", sha)
            return sha
        logger.debug("git rev-parse failed: %s", result.stderr.strip())
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("Could not resolve catalog SHA: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Domain-from-path derivation
# ---------------------------------------------------------------------------

#: Pattern: standard_names/<domain>.yml or standard_names/<domain>.yaml
#: (domain = file basename without extension; per-domain layout)
_DOMAIN_PATH_RE = re.compile(r"standard_names/([^/]+)\.ya?ml$")


def _derive_domain_from_path(yaml_path: Path) -> str | None:
    """Derive physics_domain from the file path convention.

    Expects ``<root>/standard_names/<domain>.yml`` (per-domain layout).
    Returns the domain string or None if the path doesn't match.
    Rejects names containing ``/``.
    """
    path_str = str(yaml_path).replace("\\", "/")
    m = _DOMAIN_PATH_RE.search(path_str)
    if m:
        domain = m.group(1)
        if "/" in domain:
            return None
        return domain
    return None


# ---------------------------------------------------------------------------
# Grammar decomposition (re-uses graph_ops helper)
# ---------------------------------------------------------------------------


def _grammar_decomposition(name: str) -> dict[str, str | None]:
    """Parse name via ISN grammar.

    Returns the bare-name segment columns plus ``grammar_parse_version``
    and ``validation_diagnostics_json`` — the same dict the persist path
    writes (single extraction authority).
    """
    from imas_codex.standard_names.graph_ops import _parse_grammar

    return _parse_grammar(name)


# ---------------------------------------------------------------------------
# Entry conversion
# ---------------------------------------------------------------------------


def _entry_to_graph_dict(
    entry: Any,
    *,
    physics_domain: list[str] | str | None,
) -> dict[str, Any]:
    """Convert a validated ISN entry to a graph-comparison dict.

    Does NOT include graph-only fields (source_paths, etc.).
    Grammar fields are derived from the entry name.

    ``physics_domain`` is the scalar primary domain and ``source_domains``
    carries the full list.  Accepts either a list (legacy catalog form) or a
    scalar.
    """
    grammar = _grammar_decomposition(entry.name)

    links = [str(lnk) for lnk in entry.links] if entry.links else []
    raw_constraints = getattr(entry, "constraints", None)
    constraints = list(raw_constraints) if raw_constraints else []

    if isinstance(physics_domain, list):
        source_domains = list(physics_domain)
        primary_domain = source_domains[0] if source_domains else None
    elif isinstance(physics_domain, str) and physics_domain:
        primary_domain = physics_domain
        source_domains = [physics_domain]
    else:
        primary_domain = None
        source_domains = []

    # Descriptions are plain Unicode text — strip any LaTeX/math markup that a
    # stale catalog YAML carries so it is not compared verbatim against the
    # graph. documentation keeps LaTeX.
    from imas_codex.standard_names.workers import normalize_description_text

    result: dict[str, Any] = {
        "id": entry.name,
        "description": normalize_description_text(entry.description)
        if entry.description
        else None,
        "documentation": entry.documentation or None,
        "kind": str(entry.kind) if hasattr(entry, "kind") and entry.kind else None,
        "unit": str(entry.unit) if hasattr(entry, "unit") and entry.unit else None,
        "links": links or None,
        "validity_domain": getattr(entry, "validity_domain", None) or None,
        "constraints": constraints or None,
        "physics_domain": primary_domain,
        "source_domains": source_domains,
        "status": str(entry.status) if entry.status else "draft",
        "deprecates": str(entry.deprecates) if entry.deprecates else None,
        "superseded_by": str(entry.superseded_by) if entry.superseded_by else None,
    }

    # Merge grammar fields
    result.update(grammar)

    return result


# ---------------------------------------------------------------------------
# Check mode (catalog-vs-graph comparison)
# ---------------------------------------------------------------------------

_CHECK_FIELDS = (
    "description",
    "documentation",
    "kind",
    "unit",
    "validity_domain",
    "constraints",
    "physics_domain",
)

# Fields that are graph-only extensions (not in the ISN catalog model)
# but may appear in YAML files. Strip before model validation.
_GRAPH_ONLY_FIELDS = {
    "dd_paths",
    "physics_domain",
    "cocos_transformation_type",
    "cocos",
}

# Fields removed from the ISN Pydantic model in a prior ISN release.
# Strip before model validation to avoid ``extra_forbidden`` errors;
# read back with ``getattr`` + default to avoid AttributeError.
_ISN_REMOVED_FIELDS: frozenset[str] = frozenset({"constraints", "validity_domain"})


def check_catalog(
    catalog_dir: Path,
) -> CheckResult:
    """Compare catalog entries against graph without importing.

    Returns a :class:`CheckResult` describing which entries are only in
    the catalog, only in the graph, or present in both but with differing
    field values.

    Parameters
    ----------
    catalog_dir:
        Path to directory containing YAML catalog entries.

    Returns
    -------
    CheckResult with sync status details.
    """
    import yaml
    from imas_standard_names.models import StandardNameEntry
    from pydantic import TypeAdapter

    from imas_codex.graph.client import GraphClient

    ta = TypeAdapter(StandardNameEntry)
    catalog_sha = _resolve_catalog_sha(catalog_dir)
    result = CheckResult(catalog_commit_sha=catalog_sha)

    # Parse catalog entries
    yaml_files = sorted(
        p
        for p in catalog_dir.rglob("*")
        if p.suffix in (".yml", ".yaml") and p.is_file()
    )

    catalog_entries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for yaml_path in yaml_files:
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            if isinstance(data, dict):
                # Legacy per-file layout — skip with warning
                continue

            if not isinstance(data, list):
                continue

            path_domain = _derive_domain_from_path(yaml_path) or "unscoped"

            for entry_data in data:
                if not isinstance(entry_data, dict):
                    continue

                # Warn about computed-field edits (curator warning)
                for cf in COMPUTED_FIELDS:
                    if cf in entry_data:
                        warnings.append(
                            f"{cf} is computed from HAS_PARENT / HAS_ERROR "
                            f"graph edges and will be overwritten on next "
                            f"export — edit has no effect.  See "
                            f"COMPUTED_FIELDS."
                        )
                        logger.warning(
                            "%s is computed from HAS_PARENT / HAS_ERROR "
                            "graph edges and will be overwritten on next "
                            "export — edit has no effect.  See "
                            "COMPUTED_FIELDS.",
                            cf,
                        )

                # Strip computed + graph-only fields before ISN model validation
                model_data = {
                    k: v
                    for k, v in entry_data.items()
                    if k not in _GRAPH_ONLY_FIELDS
                    and k not in COMPUTED_FIELDS
                    and k not in _ISN_REMOVED_FIELDS
                }
                entry = ta.validate_python(model_data)

                raw_pd = entry_data.get("physics_domain")
                if isinstance(raw_pd, list):
                    pd_value: list[str] | str = raw_pd
                elif isinstance(raw_pd, str) and raw_pd:
                    pd_value = raw_pd
                else:
                    pd_value = [path_domain]
                graph_dict = _entry_to_graph_dict(entry, physics_domain=pd_value)
                catalog_entries[graph_dict["id"]] = graph_dict

        except Exception:
            continue

    if not catalog_entries:
        return result

    # Fetch graph entries
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.name_stage = 'accepted'
            RETURN sn.id AS id,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   sn.unit AS unit,
                   sn.source_paths AS source_paths,
                   sn.validity_domain AS validity_domain,
                   sn.constraints AS constraints,
                   sn.physics_domain AS physics_domain,
                   sn.catalog_commit_sha AS catalog_commit_sha
            """
        )

    graph_entries: dict[str, dict[str, Any]] = {}
    graph_sha: str | None = None
    for row in rows:
        graph_entries[row["id"]] = dict(row)
        if row.get("catalog_commit_sha") and not graph_sha:
            graph_sha = row["catalog_commit_sha"]

    result.graph_commit_sha = graph_sha

    # Compare
    catalog_names = set(catalog_entries.keys())
    graph_names = set(graph_entries.keys())

    result.only_in_catalog = sorted(catalog_names - graph_names)
    result.only_in_graph = sorted(graph_names - catalog_names)

    for name in sorted(catalog_names & graph_names):
        cat = catalog_entries[name]
        graph = graph_entries[name]

        diffs: dict[str, Any] = {}
        for fld in _CHECK_FIELDS:
            cat_val = _normalize_field(cat.get(fld))
            graph_val = _normalize_field(graph.get(fld))
            if cat_val != graph_val:
                diffs[fld] = {"catalog": cat_val, "graph": graph_val}

        if diffs:
            result.diverged.append({"name": name, "fields": diffs})
        else:
            result.in_sync += 1

    return result


def _normalize_field(val: Any) -> Any:
    """Normalize a field value for comparison."""
    if val is None:
        return None
    if isinstance(val, list):
        return tuple(sorted(str(v) for v in val)) if val else None
    if isinstance(val, str):
        return val.strip() if val.strip() else None
    return val
