"""The post-copy guard compares meaning, not rendering.

The byte-level comparison refused every publish whose prose was folded by the
catalog YAML writer and every identity whose domain was derived from its
producing sources at export time.  Both are the guard reading the wrong side
of what it asserts: prose lives in the graph flat with single spaces while
the writer folds lines, and a domain is a resolution rule (stored value,
else producing-source value), not a stored scalar.

Four directions are pinned, because a guard that stops refusing is worse than
one that refuses too much — it is the last check between the graph and a
published catalog:

* prose that differs only by line folding PASSES;
* prose that genuinely differs STILL REFUSES;
* a node storing no domain whose producing source carries one PASSES;
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
    (sn_dir / f"{domain}.yml").write_text(
        yaml.safe_dump(entries), encoding="utf-8"
    )
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
    (tmp_path / "catalog.yml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )
    return tmp_path


def _graph_rows(
    *,
    description: str,
    documentation: str = "A temperature measurement.",
    resolved_physics_domains: list[str] | None = None,
) -> list[dict]:
    """Graph rows shaped like the guard's fetch query after domain resolution."""
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
            "catalog_commit_sha": None,
            "resolved_physics_domains": resolved_physics_domains
            or ["equilibrium"],
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


def test_node_without_domain_but_sourced_domain_passes(tmp_path: Path) -> None:
    """Half 3: the graph resolves a missing stored domain from its source.

    The published tree carries the derived domain (the exporter's resolution
    written into the sidecar); the graph node stores none, so the guard must
    resolve the domain the way the exporter does — from the producing
    sources — rather than compare a stored scalar against a resolved value.
    """
    catalog = _write_catalog(
        tmp_path,
        domain="equilibrium",
        entries=[_entry()],
        sidecar_domain="equilibrium",
    )
    with _mock_graph(
        _graph_rows(description=_FLAT, resolved_physics_domains=["equilibrium"])
    ):
        result = check_catalog(catalog)

    assert result.diverged == []
    assert result.in_sync == 1
    assert result.describe_divergence() is None


def test_genuine_domain_disagreement_still_refuses(tmp_path: Path) -> None:
    """Half 4: an entry whose stored-only domain contradicts the resolution.

    The stored field stays authoritative wherever it is set: a published
    domain that disagrees with the node-or-source resolution still refuses,
    even when every other field agrees.
    """
    catalog = _write_catalog(
        tmp_path,
        domain="equilibrium",
        entries=[_entry()],
        sidecar_domain="equilibrium",
    )
    with _mock_graph(
        _graph_rows(description=_FLAT, resolved_physics_domains=["transport"])
    ):
        result = check_catalog(catalog)

    assert len(result.diverged) == 1
    fields = result.diverged[0]["fields"]
    assert "physics_domain" in fields
    assert result.describe_divergence() is not None
