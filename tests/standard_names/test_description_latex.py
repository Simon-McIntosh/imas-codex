"""Tests that LaTeX/math markup can never survive in a `description`.

Descriptions are plain Unicode text (the ISN convention); LaTeX belongs in
`documentation`. Three layers enforce this:

1. NORMALIZE — ``normalize_description_text`` strips ``$…$``/``$$…$$``
   delimiters, converts ``\\phi \\theta \\rho \\pi`` (and the half-converted
   ``\\φ`` corruption) to Unicode, and flattens other backslash-commands /
   braces to plain text. Idempotent; DD tokens (``phi_tor``) untouched;
   ``documentation`` never touched.
2. PROJECT — ``_entry_to_graph_dict`` normalizes ``description`` when it
   projects a catalog entry for the graph comparison, so a stale YAML
   carrying ``$…$`` is compared as clean text rather than as markup.
3. GATE — a ``$`` or backslash in a ``description`` is a critical audit
   failure (``description_notation_check``), so such a name is quarantined
   even if the normalizer missed an edge case. ``documentation`` is exempt.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Normalizer strips LaTeX
# ---------------------------------------------------------------------------


class TestNormalizerStripsLatex:
    def _norm(self, text: str) -> str:
        from imas_codex.standard_names.workers import normalize_description_text

        return normalize_description_text(text)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # $…$ delimiters removed, \phi → φ
            ("In the $(R, \\phi, Z)$ frame.", "In the (R, φ, Z) frame."),
            # already-corrupted \φ (stranded backslash) → φ
            ("In the $(R, \\φ, Z)$ frame.", "In the (R, φ, Z) frame."),
            # \pi → π
            ("The period is $2\\pi$ radians.", "The period is 2π radians."),
        ],
    )
    def test_exact_plain_text(self, raw: str, expected: str) -> None:
        assert self._norm(raw) == expected

    @pytest.mark.parametrize(
        "raw,must_contain",
        [
            ("Density $n_\\phi$ profile.", "n_φ"),
            ("Magnitude $|\\mathbf{k}|$ of the wavevector.", "|k|"),
            ("Wavenumber $k_R$ radial component.", "k_R"),
            ("Charge states W$^{1+}$ through W$^{74+}$.", "W1+"),
            ("Charge states W$^{1+}$ through W$^{74+}$.", "W74+"),
        ],
    )
    def test_no_markup_remains(self, raw: str, must_contain: str) -> None:
        out = self._norm(raw)
        assert "$" not in out, f"'$' survived in {out!r}"
        assert "\\" not in out, f"backslash survived in {out!r}"
        assert "{" not in out and "}" not in out, f"brace survived in {out!r}"
        assert must_contain in out, f"{must_contain!r} missing from {out!r}"

    @pytest.mark.parametrize(
        "raw",
        [
            "In the $(R, \\phi, Z)$ frame.",
            "In the $(R, \\φ, Z)$ frame.",
            "The period is $2\\pi$ radians.",
            "Density $n_\\phi$ profile.",
            "Magnitude $|\\mathbf{k}|$ of the wavevector.",
            "Charge states W$^{1+}$ through W$^{74+}$.",
            "in the right-handed cylindrical (R, phi, Z) frame",
        ],
    )
    def test_idempotent(self, raw: str) -> None:
        once = self._norm(raw)
        assert self._norm(once) == once

    def test_dd_tokens_untouched(self) -> None:
        out = self._norm("stored at phi_tor and b_field_phi with k_phi")
        assert "phi_tor" in out
        assert "b_field_phi" in out
        assert "k_phi" in out
        assert "φ" not in out

    def test_plain_description_unchanged(self) -> None:
        # A clean plain-text description must pass through verbatim.
        clean = "Toroidal magnetic field on the plasma boundary."
        assert self._norm(clean) == clean


# ---------------------------------------------------------------------------
# 2. Projection normalizes description (documentation left alone)
# ---------------------------------------------------------------------------


class TestGraphDictNormalizesDescription:
    def test_entry_to_graph_dict_normalizes_description(self) -> None:
        from imas_codex.standard_names import catalog_import

        entry = MagicMock()
        entry.name = "toroidal_angle"
        entry.description = "In the $(R, \\phi, Z)$ frame."
        entry.documentation = "doc $\\phi$ keeps latex"
        entry.kind = None
        entry.unit = None
        entry.links = None
        entry.validity_domain = None
        entry.constraints = None
        entry.status = None
        entry.deprecates = None
        entry.superseded_by = None

        # Grammar decomposition needs the ISN grammar; stub to isolate the
        # description-normalization behaviour under test.
        with patch.object(catalog_import, "_grammar_decomposition", return_value={}):
            result = catalog_import._entry_to_graph_dict(entry, physics_domain=None)

        assert result["description"] == "In the (R, φ, Z) frame."
        # documentation must NOT be normalized.
        assert result["documentation"] == "doc $\\phi$ keeps latex"


# ---------------------------------------------------------------------------
# 3. Gate — $ / backslash in a description quarantines
# ---------------------------------------------------------------------------


class TestDescriptionLatexGate:
    def test_check_is_registered_critical(self) -> None:
        from imas_codex.standard_names.audits import CRITICAL_CHECKS

        assert "description_notation_check" in CRITICAL_CHECKS

    def test_dollar_description_flagged_critical(self) -> None:
        from imas_codex.standard_names.audits import (
            description_notation_check,
            has_critical_audit_failure,
        )

        issues = description_notation_check(
            {"description": "In the $(R, \\phi, Z)$ frame."}
        )
        assert issues, "a $-markup description must raise an issue"
        assert has_critical_audit_failure(issues) is True

    def test_stranded_backslash_description_flagged(self) -> None:
        from imas_codex.standard_names.audits import description_notation_check

        # No '$', but a stranded backslash (the \φ corruption) must still fire.
        issues = description_notation_check({"description": "coordinate (R, \\φ, Z)"})
        assert issues

    def test_clean_description_not_flagged(self) -> None:
        from imas_codex.standard_names.audits import (
            description_notation_check,
            has_critical_audit_failure,
        )

        issues = description_notation_check(
            {"description": "Toroidal angle in the (R, φ, Z) frame."}
        )
        assert issues == []
        assert has_critical_audit_failure(issues) is False

    def test_documentation_latex_not_flagged(self) -> None:
        from imas_codex.standard_names.audits import description_notation_check

        # LaTeX in documentation is legitimate; the description is clean.
        issues = description_notation_check(
            {
                "description": "Electron temperature profile.",
                "documentation": "The temperature $T_e$ with $\\phi$ notation.",
            }
        )
        assert issues == []

    def test_run_audits_wires_the_check(self) -> None:
        from imas_codex.standard_names.audits import run_audits

        issues = run_audits(
            {
                "id": "toroidal_angle",
                "description": "In the $(R, \\phi, Z)$ frame.",
                "documentation": "",
                "unit": "rad",
            }
        )
        assert any("audit:description_notation_check:" in i for i in issues)

    def test_is_quarantined_on_latex_description(self) -> None:
        from imas_codex.standard_names.workers import _is_quarantined

        issue = "audit:description_notation_check: description contains math markup"
        assert _is_quarantined([issue], {}) is True
        # A clean name with a passing pydantic layer is not quarantined.
        assert _is_quarantined([], {"pydantic": {"passed": True}}) is False
