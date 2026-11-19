"""Replacement rules for the project rename.

The old name is held as data here rather than spelled out in prose throughout
the tree: this module is the one place that knows what the old name looks like,
and :mod:`scripts.rename.check` uses the same table to search for it.
"""

from __future__ import annotations

import re

# Ordered longest-token-first per spelling family.  Every rule is a literal
# substring replacement: `imas_codex` appears inside longer identifiers,
# environment variables and paths, so a word-boundary match would miss most of
# the real consumers.
REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("IMAS_CODEX", "ITER_CODEX"),
    ("IMAS Codex", "ITER Codex"),
    ("imas_codex", "iter_codex"),
    ("imas-codex", "iter-codex"),
    ("imas-cx", "iter-cx"),
    ("imas_cx", "iter_cx"),
)

# What the closing search looks for.  One pattern covers the full name in
# either case and either separator (`imas_codex`, `imas-codex`, `IMAS_CODEX`,
# `IMAS Codex`) and the server key spelling `imas-cx`, without matching the
# kept `imas_standard_names`, `imas_dd` or `imas-dd` spellings.
OLD_NAME_PATTERN = re.compile(r"imas[-_ ](?:codex|cx)", re.IGNORECASE)

# Basenames that carry the project name and therefore move with the rename.
# `imas_dd` and `imas_standard_names` are deliberately absent: they name the
# IMAS data dictionary and the standard-names grammar, which keep their names.
PATH_TOKENS: tuple[str, ...] = ("imas_codex", "imas-codex")

# Directories never walked or rewritten.  `.git` is version control, the rest
# are local build and environment outputs that are either regenerated or not
# part of the tracked tree.
SKIP_DIRS: frozenset[str] = frozenset(
    {".git", ".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__"}
)
