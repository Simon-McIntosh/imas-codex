"""Regression: Gate B fails loudly when ISN cannot be imported.

Previously the grammar parse gate swallowed ``ImportError`` and logged a
warning, letting the export proceed to emit an *unvalidated* catalog — and
``_validate_entry`` would then crash on the same missing import anyway.
A missing ISN toolchain must block the export at the gate with a clear
``isn_unavailable`` issue, for both RC and final releases.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from imas_codex.standard_names.export import GATE_B, _run_gate_b


def _isn_grammar_unimportable():
    # Mapping a submodule to None in sys.modules makes
    # ``from imas_standard_names.grammar import parse_name`` raise ImportError.
    return patch.dict(sys.modules, {"imas_standard_names.grammar": None})


def test_gate_b_fails_when_isn_grammar_unimportable() -> None:
    candidates = [{"id": "electron_temperature", "cocos": None}]
    with _isn_grammar_unimportable():
        result = _run_gate_b(candidates, cocos_convention=17, final=False)

    assert result.gate == GATE_B
    assert not result.passed, "Gate B must fail when ISN grammar is unavailable"
    assert any(i["type"] == "isn_unavailable" for i in result.issues), (
        f"expected an isn_unavailable issue; got {[i['type'] for i in result.issues]}"
    )


def test_isn_unavailable_blocks_final_release_too() -> None:
    candidates = [{"id": "electron_temperature", "cocos": None}]
    with _isn_grammar_unimportable():
        result = _run_gate_b(candidates, cocos_convention=17, final=True)
    assert not result.passed
    assert any(i["type"] == "isn_unavailable" for i in result.issues)


def test_gate_b_passes_when_isn_available_and_name_parses() -> None:
    """Positive control: with ISN importable and a valid name, no isn_unavailable."""
    candidates = [{"id": "electron_temperature", "cocos": None}]
    result = _run_gate_b(candidates, cocos_convention=17, final=False)
    assert not any(i["type"] == "isn_unavailable" for i in result.issues)
