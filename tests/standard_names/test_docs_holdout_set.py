"""Integrity checks for the catalog-documentation evaluation holdout."""

from imas_codex.standard_names.benchmark_reference import (
    REFERENCE_NAMES,
    curated_example_split_keys,
    load_docs_holdout,
)


def test_docs_holdout_is_non_empty() -> None:
    rows = load_docs_holdout()

    assert rows


def test_docs_holdout_pairs_dd_paths_with_catalog_documentation() -> None:
    rows = load_docs_holdout()

    for row in rows:
        assert row["split_key"] == row["dd_path"]
        assert row["dd_path"] in REFERENCE_NAMES
        assert row["catalog_name"] == REFERENCE_NAMES[row["dd_path"]]["name"]
        assert row["catalog_description"].strip()
        assert row["catalog_documentation"].strip()
        assert row["catalog_source"].startswith(
            "imas_standard_names/resources/standard_name_examples/"
        )
        assert len(row["catalog_commit"]) == 40


def test_docs_holdout_is_disjoint_from_curated_prompt_examples() -> None:
    holdout_split_keys = {row["split_key"] for row in load_docs_holdout()}
    prompt_example_split_keys = curated_example_split_keys()

    assert prompt_example_split_keys
    assert holdout_split_keys.isdisjoint(prompt_example_split_keys)
