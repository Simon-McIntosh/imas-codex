"""Selecting convention entries for a name through the grammar accessor.

The shared conventions registry (concepts, notation, templates) lives beside
the grammar vocabularies it is keyed by and is reached through the grammar
package's selecting accessor. The entries that apply to a name are chosen from
the parse of that name, so the accessor is the only path and a text search can
never be the mechanism under test.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from imas_standard_names.grammar import vocab_loaders
from imas_standard_names.grammar.terms import AppliedConventions, select_conventions

NAME = "flux_surface_averaged_argon_density_at_plasma_boundary"

# Exactly the entries the parse of NAME selects, identified by their keys.
EXPECTED_KEYS = {
    "concepts": ("flux_surface_averaged", "plasma_boundary"),
    "notation": ("density",),
    "templates": (("density", ("subject",)),),
}

REMOVED_CONCEPT = "flux_surface_averaged"


def _entry_keys(applied: AppliedConventions) -> dict[str, tuple]:
    """The entries an accessor result carries, named by key alone."""
    return {
        "concepts": tuple(concept.token for concept in applied.concepts),
        "notation": tuple(entry.token for entry in applied.notation),
        "templates": tuple(
            (template.base, template.segments) for template in applied.templates
        ),
    }


def _conventions_source() -> Path:
    """The shipped conventions directory the loaders read by default."""
    return (
        Path(vocab_loaders.__file__).resolve().parent / "vocabularies" / "conventions"
    )


def _registry_without_concept(tmp_path: Path):
    """Load the shipped registry from a scratch copy missing one concept entry."""
    scratch = tmp_path / "conventions"
    shutil.copytree(_conventions_source(), scratch)
    concepts_file = scratch / "concepts.yml"
    data = yaml.safe_load(concepts_file.read_text(encoding="utf-8"))
    del data["concepts"][REMOVED_CONCEPT]
    concepts_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return vocab_loaders.load_conventions(directory=scratch)


def _target_registry(tmp_path: Path):
    """The registry the positive assertion runs against.

    ``None`` selects the shipped registry (``load_conventions()``). Point it at
    a scratch copy instead and the same assertion must fail.
    """
    return None


def test_selects_exactly_the_expected_convention_entries(tmp_path: Path) -> None:
    """The accessor returns the four applicable entries and nothing else."""
    applied = select_conventions(NAME, registry=_target_registry(tmp_path))

    assert _entry_keys(applied) == EXPECTED_KEYS


def test_scratch_registry_omits_the_flux_surface_average_concept(
    tmp_path: Path,
) -> None:
    """The scratch copy really drops the entry, so the control above can fail."""
    reduced = _registry_without_concept(tmp_path)

    assert REMOVED_CONCEPT not in reduced.concepts.concepts
    applied = select_conventions(NAME, registry=reduced)
    assert _entry_keys(applied)["concepts"] == ("plasma_boundary",)
