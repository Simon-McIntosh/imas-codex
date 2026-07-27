"""The ``export_scope`` stamp must be a value the pinned ISN manifest model accepts.

``_write_manifest`` validates the manifest through the ISN
``StandardNameCatalogManifest`` model but only *warns* when validation fails —
so a scope string the model rejects is written unvalidated and silently. These
tests pin every scope the exporter can emit against the model's own literal set,
so the two cannot drift.
"""

from __future__ import annotations

import typing

import pytest
import yaml

from imas_codex.standard_names.export import _write_manifest, resolve_export_scope


def _allowed_scopes() -> set[str]:
    """The literal set the pinned ISN manifest model accepts, read from the model."""
    from imas_standard_names.models import StandardNameCatalogManifest

    annotation = StandardNameCatalogManifest.model_fields["export_scope"].annotation
    return {
        value
        for arg in typing.get_args(annotation)
        for value in typing.get_args(arg)
        if isinstance(value, str)
    }


def _manifest_kwargs() -> dict:
    return {
        "cocos_convention": 11,
        "candidate_count": 1,
        "published_count": 1,
        "excluded_below_score_count": 0,
        "excluded_unreviewed_count": 0,
        "min_score_applied": 0.65,
        "min_description_score_applied": None,
        "include_unreviewed": False,
        "source_commit_sha": None,
        "domains_included": ["equilibrium"],
    }


class TestResolveExportScope:
    def test_review_batch_wins(self) -> None:
        assert (
            resolve_export_scope(review_batch=["a"], domain="equilibrium") == "review"
        )

    def test_empty_batch_is_still_a_review_export(self) -> None:
        """An empty list is a batch of zero, not the absence of a batch."""
        assert resolve_export_scope(review_batch=[], domain=None) == "review"

    def test_domain_scoped(self) -> None:
        assert resolve_export_scope(review_batch=None, domain="equilibrium") == "domain"

    def test_unscoped_is_full(self) -> None:
        assert resolve_export_scope(review_batch=None, domain=None) == "full"

    @pytest.mark.parametrize(
        ("review_batch", "domain"),
        [(None, None), (None, "equilibrium"), (["a"], None)],
    )
    def test_every_resolved_scope_is_accepted_by_the_isn_model(
        self, review_batch: list[str] | None, domain: str | None
    ) -> None:
        scope = resolve_export_scope(review_batch=review_batch, domain=domain)
        assert scope in _allowed_scopes()


class TestManifestValidates:
    @pytest.mark.parametrize(
        ("review_batch", "domain"),
        [(None, None), (None, "equilibrium"), (["b_name", "a_name"], None)],
    )
    def test_written_manifest_round_trips_through_the_isn_model(
        self, tmp_path, review_batch: list[str] | None, domain: str | None
    ) -> None:
        """A written catalog.yml must validate — not merely be written."""
        from imas_standard_names.models import StandardNameCatalogManifest

        _write_manifest(
            tmp_path,
            export_scope=resolve_export_scope(review_batch=review_batch, domain=domain),
            review_batch=review_batch,
            **_manifest_kwargs(),
        )
        data = yaml.safe_load((tmp_path / "catalog.yml").read_text())
        StandardNameCatalogManifest.model_validate(data)
