"""Loader for sn-sources manifest files consumed by ``sn run --focus``.

A focus token that is a file on disk is treated as a ``schema_version: 1``
sn-sources manifest (see ``config/sn_sources.schema.json``): it is validated
against the committed JSON Schema and its ``sources`` mapping flattened to
``<ids>/<path>`` focus paths. Any non-compliance raises ``SourcesManifestError``
so a malformed manifest fails fast rather than silently seeding garbage.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import yaml


class SourcesManifestError(ValueError):
    """A focus file is missing, unreadable, or not schema-compliant."""


def _load_schema() -> dict:
    ref = resources.files("imas_codex.standard_names.config").joinpath(
        "sn_sources.schema.json"
    )
    with ref.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def is_sources_file(token: str) -> bool:
    """True if *token* refers to an existing file (vs a bare DD path)."""
    try:
        return Path(token).is_file()
    except OSError:
        return False


def load_sources_file(path: str | Path) -> list[str]:
    """Load and validate an sn-sources manifest; return flattened focus paths.

    Returns the ``sources`` mapping flattened to a de-duplicated, order-preserving
    list of ``<ids>/<path>`` strings.

    Raises:
        SourcesManifestError: file missing/unreadable, invalid YAML, or the
            document fails the ``sn_sources`` JSON Schema.
    """
    p = Path(path)
    if not p.is_file():
        raise SourcesManifestError(f"sn-sources file not found: {path}")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SourcesManifestError(f"{path}: not valid YAML ({exc})") from exc
    if not isinstance(doc, dict):
        raise SourcesManifestError(f"{path}: top level must be a mapping")

    try:
        import jsonschema

        try:
            jsonschema.validate(doc, _load_schema())
        except jsonschema.ValidationError as exc:
            raise SourcesManifestError(
                f"{path}: not schema-compliant — {exc.message} "
                f"(at {'/'.join(str(k) for k in exc.absolute_path) or '<root>'})"
            ) from exc
    except ImportError:
        # Minimal structural fallback when jsonschema is unavailable.
        if doc.get("schema_version") != 1:
            raise SourcesManifestError(
                f"{path}: unsupported or missing schema_version (expected 1)"
            ) from None
        if not isinstance(doc.get("sources"), dict) or not doc["sources"]:
            raise SourcesManifestError(
                f"{path}: 'sources' must be a non-empty mapping"
            ) from None

    flat: list[str] = []
    seen: set[str] = set()
    for ids, paths in doc["sources"].items():
        for rel in paths:
            full = f"{ids}/{rel}"
            if full not in seen:
                seen.add(full)
                flat.append(full)
    if not flat:
        raise SourcesManifestError(f"{path}: manifest resolved to zero sources")
    return flat


def expand_focus_tokens(tokens: list[str]) -> list[str]:
    """Expand any sn-sources file tokens in *tokens* to their flattened paths.

    File tokens are replaced by the manifest's ``<ids>/<path>`` entries; bare DD
    paths pass through unchanged. Order is preserved and duplicates dropped.
    """
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        expanded = load_sources_file(tok) if is_sources_file(tok) else [tok]
        for item in expanded:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out
