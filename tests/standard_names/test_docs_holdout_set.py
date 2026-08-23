"""Integrity checks for the catalog-documentation evaluation holdout."""

import pytest

from imas_codex.standard_names.benchmark_reference import (
    DOCS_HOLDOUT_FIELDS,
    curated_example_names,
    curated_example_split_keys,
    load_docs_holdout,
    load_docs_holdout_authority,
)


def test_docs_holdout_has_statistically_useful_sample() -> None:
    rows = load_docs_holdout()

    assert len(rows) >= 30


def test_docs_holdout_pairs_dd_paths_with_catalog_documentation() -> None:
    rows = load_docs_holdout()
    catalog_commits = {row["catalog_commit"] for row in rows}

    for row in rows:
        assert set(row) == DOCS_HOLDOUT_FIELDS
        assert row["split_key"] == row["dd_path"]
        assert row["catalog_description"].strip()
        assert row["catalog_documentation"].strip()
        assert row["catalog_source"].startswith(
            "imas_standard_names/resources/standard_name_examples/"
        )

    assert len({row["split_key"] for row in rows}) == len(rows)
    assert len(catalog_commits) == 1
    catalog_commit = next(iter(catalog_commits))
    assert len(catalog_commit) == 40
    assert all(character in "0123456789abcdef" for character in catalog_commit)


@pytest.mark.requires_graph
def test_docs_holdout_physics_authority_matches_dd_path_bindings() -> None:
    rows = load_docs_holdout()
    authority = load_docs_holdout_authority(row["dd_path"] for row in rows)

    assert set(authority) == {row["dd_path"] for row in rows}
    assert sum(row["declared_unit"] is not None for row in rows) == 85
    assert sum(row["cocos_transformation_type"] is not None for row in rows) == 24
    for row in rows:
        bound = authority[row["dd_path"]]
        assert row["declared_unit"] == bound["declared_unit"]
        assert row["cocos_transformation_type"] == bound["cocos_transformation_type"]


def test_docs_holdout_is_disjoint_from_curated_prompt_examples() -> None:
    holdout_split_keys = {row["split_key"] for row in load_docs_holdout()}
    holdout_names = {row["catalog_name"] for row in load_docs_holdout()}
    prompt_example_split_keys = curated_example_split_keys()
    prompt_example_names = curated_example_names()

    assert prompt_example_split_keys
    assert prompt_example_names
    assert holdout_names.isdisjoint(prompt_example_names)
    assert holdout_split_keys.isdisjoint(prompt_example_split_keys)
