"""Shared eligibility rules for signal description retrieval and embedding."""

from __future__ import annotations


def description_is_name(description: str, name: str, accessor: str) -> bool:
    """A full identifier or its colon suffix supplies no quantity description."""
    text = description.strip()
    identities = {
        value.strip()
        for identity in (name, accessor)
        for value in (identity, identity.split(":")[-1])
        if value.strip()
    }
    return bool(text) and text in identities


def searchable_description_predicate(node: str = "s", text: str | None = None) -> str:
    """Cypher equivalent of the document-side eligibility rule."""
    text = text or f"{node}.description"
    identities = ", ".join(
        expression
        for field in ("name", "accessor")
        for expression in (
            f"trim(coalesce({node}.{field}, ''))",
            f"trim(last(split(coalesce({node}.{field}, ''), ':')))",
        )
    )
    return (
        f"(trim(coalesce({text}, '')) <> '' AND NOT (trim({text}) IN [{identities}]))"
    )
