"""Tests for Greek word→symbol normalization of description prose."""

from imas_codex.standard_names.workers import (
    normalize_description_notation,
    normalize_description_text,
)


class TestNormalizeDescriptionNotation:
    def test_coordinate_frame_tuple(self):
        assert (
            normalize_description_notation(
                "in the right-handed cylindrical (R, phi, Z) frame"
            )
            == "in the right-handed cylindrical (R, φ, Z) frame"
        )

    def test_toroidal_angle_word(self):
        assert (
            normalize_description_notation("increasing toroidal angle phi.")
            == "increasing toroidal angle φ."
        )

    def test_theta_and_rho(self):
        assert (
            normalize_description_notation("poloidal angle theta at radius rho")
            == "poloidal angle θ at radius ρ"
        )

    def test_dd_tokens_untouched(self):
        # Underscore is a word character — no boundary, no substitution.
        assert (
            normalize_description_notation("stored at k_phi and momentum_phi")
            == "stored at k_phi and momentum_phi"
        )

    def test_capitalized_phi_untouched(self):
        # Capital Phi may be the flux Φ or a sentence-start angle — left alone.
        assert (
            normalize_description_notation("Phi denotes the toroidal flux")
            == "Phi denotes the toroidal flux"
        )

    def test_words_containing_greek_untouched(self):
        assert (
            normalize_description_notation("the sapphire photodiode")
            == "the sapphire photodiode"
        )

    def test_empty(self):
        assert normalize_description_notation("") == ""


class TestNormalizeDescriptionText:
    def test_spelling_then_symbols(self):
        assert (
            normalize_description_text("centre of the (R, phi, Z) frame")
            == "center of the (R, φ, Z) frame"
        )
