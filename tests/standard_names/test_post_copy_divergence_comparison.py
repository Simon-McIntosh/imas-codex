"""The post-copy guard compares meaning, not rendering.

The byte-level comparison refused every publish whose prose was folded by the
catalog YAML writer.  That is the guard reading the wrong side of what it
asserts: prose lives in the graph flat with single spaces while the writer
folds lines.

A domain is a stored decision, not a resolution rule.  The node's
``physics_domain`` is the value the one writer set, so the guard compares it
directly and never re-derives one from the producing sources; a node the
writer has not decided stores none, and compares as undecided.

Four directions are pinned, because a guard that stops refusing is worse than
one that refuses too much — it is the last check between the graph and a
published catalog:

* prose that differs only by line folding PASSES;
* prose that genuinely differs STILL REFUSES;
* a node storing no domain STILL REFUSES against a published one, rather than
  resolving one from its sources — a name the writer has not decided cannot
  back a published bucket;
* a genuine domain disagreement STILL REFUSES.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from imas_codex.standard_names.catalog_import import check_catalog
from imas_codex.standard_names.export import CATALOG_EDGE_MODEL_VERSION

#: Same text on both sides of the render boundary: the published entry is
#: folded at the writer's line width, the graph stores it flat.
_FOLDED = "Electron temperature\nin the core plasma\nmeasured by Thomson scattering"
_FLAT = "Electron temperature in the core plasma measured by Thomson scattering"
#: A genuinely different wording (not a render difference at any width).
_OTHER = "Electron temperature resolved from the Thomson spectrum"


def _entry(
    *,
    name: str = "electron_temperature",
    description: str = _FLAT,
    documentation: str = "A temperature measurement.",
) -> dict:
    """One published reviewable entry in the catalog's per-domain shape."""
    return {
        "name": name,
        "description": description,
        "documentation": documentation,
        "kind": "scalar",
        "unit": "eV",
        "links": [],
        "status": "draft",
    }


def _write_catalog(
    tmp_path: Path,
    *,
    domain: str,
    entries: list[dict],
    sidecar_domain: str | None,
) -> Path:
    """Write ``<root>/standard_names/<domain>.yml`` plus the sidecar manifest."""
    sn_dir = tmp_path / "standard_names"
    sn_dir.mkdir(parents=True, exist_ok=True)
    (sn_dir / f"{domain}.yml").write_text(yaml.safe_dump(entries), encoding="utf-8")
    names: dict[str, dict] = {}
    for entry in entries:
        block: dict[str, object] = {
            "kind": "scalar",
            "status": "draft",
            "links": [],
            "sources": [],
        }
        if sidecar_domain is not None:
            block["physics_domain"] = sidecar_domain
        names[entry["name"]] = block
    manifest = {
        "catalog_name": "imas-standard-names-catalog",
        "cocos_convention": 17,
        "grammar_version": "0.7.0",
        "isn_model_version": "0.7.0",
        "dd_version_lineage": ["4.1.1"],
        "generated_by": "test",
        "generated_at": "2024-01-01T00:00:00Z",
        "candidate_count": len(entries),
        "published_count": len(entries),
        "excluded_below_score_count": 0,
        "excluded_unreviewed_count": 0,
        "edge_model_version": CATALOG_EDGE_MODEL_VERSION,
        "domains_included": [domain],
        "names": names,
    }
    (tmp_path / "catalog.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return tmp_path


def _graph_rows(
    *,
    description: str,
    documentation: str = "A temperature measurement.",
    physics_domain: str | None = "equilibrium",
    producing_source_domains: list[str] | None = None,
) -> list[dict]:
    """Graph rows shaped like the guard's fetch query.

    ``physics_domain`` is the node's stored decision.  ``producing_source_domains``
    is a decoy: the sources' own domains reach the row, and the guard must not
    read them, because a name's domain is the stored decision and not something
    re-derived from where the name came from.
    """
    return [
        {
            "id": "electron_temperature",
            "description": description,
            "documentation": documentation,
            "kind": "scalar",
            "unit": "eV",
            "source_paths": None,
            "validity_domain": None,
            "constraints": None,
            "physics_domain": physics_domain,
            "catalog_commit_sha": None,
            "resolved_physics_domains": producing_source_domains or [],
        }
    ]


def _mock_graph(rows: list[dict]):
    """Patch the GraphClient the guard opens so no live connection is made."""
    inst = MagicMock()
    inst.query.return_value = rows
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=inst)
    ctx.__exit__ = MagicMock(return_value=False)
    # catalog_import imports GraphClient function-locally from its source
    # module, so the patch targets the binding the import resolves at call time.
    return patch("imas_codex.graph.client.GraphClient", return_value=ctx)


def test_folded_prose_passes_while_content_is_identical(tmp_path: Path) -> None:
    """Half 1: prose that differs only by line folding agrees."""
    catalog = _write_catalog(
        tmp_path,
        domain="equilibrium",
        entries=[_entry(description=_FOLDED)],
        sidecar_domain="equilibrium",
    )
    with _mock_graph(_graph_rows(description=_FLAT)):
        result = check_catalog(catalog)

    assert result.diverged == []
    assert result.in_sync == 1
    assert result.describe_divergence() is None


def test_genuinely_different_prose_still_refuses(tmp_path: Path) -> None:
    """Half 2: a real wording difference survives the whitespace collapse."""
    catalog = _write_catalog(
        tmp_path,
        domain="equilibrium",
        entries=[_entry(description=_FOLDED)],
        sidecar_domain="equilibrium",
    )
    with _mock_graph(_graph_rows(description=_OTHER)):
        result = check_catalog(catalog)

    assert len(result.diverged) == 1
    fields = result.diverged[0]["fields"]
    assert "description" in fields
    assert _FOLDED.split() != _OTHER.split()
    assert result.describe_divergence() is not None


def test_node_without_stored_domain_refuses_a_published_one(tmp_path: Path) -> None:
    """Half 3: a node the writer has not decided cannot back a published bucket.

    The graph node stores no ``physics_domain`` while its producing sources
    carry one, and the catalog publishes that source's domain.  The guard must
    read the node's stored decision — none — and report the divergence, rather
    than resolve the source's domain for it.  A resolution here would be a
    second writer, and would publish a bucket the graph does not hold.
    """
    catalog = _write_catalog(
        tmp_path,
        domain="equilibrium",
        entries=[_entry()],
        sidecar_domain="equilibrium",
    )
    with _mock_graph(
        _graph_rows(
            description=_FLAT,
            physics_domain=None,
            producing_source_domains=["equilibrium"],
        )
    ):
        result = check_catalog(catalog)

    assert len(result.diverged) == 1
    assert result.diverged[0]["fields"]["physics_domain"] == {
        "catalog": "equilibrium",
        "graph": None,
    }
    assert result.describe_divergence() is not None


def test_genuine_domain_disagreement_still_refuses(tmp_path: Path) -> None:
    """Half 4: a stored domain that contradicts the published one refuses.

    The stored field is authoritative wherever it is set: a published domain
    that disagrees with the node's decision still refuses, even when every
    other field agrees.
    """
    catalog = _write_catalog(
        tmp_path,
        domain="equilibrium",
        entries=[_entry()],
        sidecar_domain="equilibrium",
    )
    with _mock_graph(_graph_rows(description=_FLAT, physics_domain="transport")):
        result = check_catalog(catalog)

    assert len(result.diverged) == 1
    fields = result.diverged[0]["fields"]
    assert "physics_domain" in fields
    assert result.describe_divergence() is not None
