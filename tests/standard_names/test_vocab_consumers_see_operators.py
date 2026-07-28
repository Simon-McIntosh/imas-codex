"""Every vocabulary consumer reads the grammar through one accessor.

``SEGMENT_TOKEN_MAP`` is not the whole grammar vocabulary — operators are a
separate mechanism occupying no segment slot. A consumer reading only that map
cannot see 51 legal tokens, and each such consumer reproduces the same class of
error in its own way: a plural-dedup check calls ``squares`` novel, a promotion
check calls a registered operator unregistered.

These tests pin every consumer to :func:`grammar_tokens_by_segment` so the two
halves of the vocabulary cannot come apart again one module at a time.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.segments import (
    OPERATOR_SEGMENT,
    grammar_tokens_by_segment,
)


def _isn_available() -> bool:
    try:
        import imas_standard_names.grammar.constants  # noqa: F401
    except ImportError:
        return False
    return True


requires_isn = pytest.mark.skipif(
    not _isn_available(), reason="imas-standard-names not installed"
)

#: Operators whose presence in a consumer's vocabulary is asserted directly.
#: Sampled across attachment kinds so a consumer that somehow carried only one
#: group would still fail.
SAMPLE_OPERATORS = ("square", "inverse", "flux_surface_averaged", "magnitude", "ratio")


@requires_isn
class TestTokenFilterSeesOperators:
    """The plural-dedup check compares against operators as well as segments."""

    def test_every_operator_is_in_the_comparison_set(self):
        from imas_codex.standard_names.vocab_token_filter import _load_existing_tokens

        tokens = _load_existing_tokens()
        operators = set(grammar_tokens_by_segment()[OPERATOR_SEGMENT])
        missing = sorted(operators - tokens)
        assert not missing, f"dedup cannot see operators: {missing}"

    def test_segment_tokens_are_still_present(self):
        from imas_codex.standard_names.vocab_token_filter import _load_existing_tokens

        tokens = _load_existing_tokens()
        for token in ("electron", "toroidal", "density"):
            assert token in tokens

    def test_plural_of_an_operator_is_rejected_as_a_duplicate(self):
        """``squares`` duplicates the operator ``square``, so it is not novel."""
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        verdict = classify_vocab_token("squares", "qualifier")
        assert verdict.action == "reject"
        assert "square" in verdict.reason

    def test_plural_of_a_segment_token_still_rejected(self):
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        verdict = classify_vocab_token("electrons", "subject")
        assert verdict.action == "reject"

    def test_a_genuinely_novel_token_is_still_accepted(self):
        """Widening the comparison set must not start rejecting real proposals."""
        from imas_codex.standard_names.vocab_token_filter import classify_vocab_token

        verdict = classify_vocab_token("wibble_frobnicator", "physical_base")
        assert verdict.action == "accept"

    def test_degrades_to_empty_without_the_grammar(self, monkeypatch):
        """Absent ISN the check turns off rather than dedupe against a stale list."""
        import imas_codex.standard_names.vocab_token_filter as filter_mod
        from imas_codex.standard_names import segments as seg_mod

        monkeypatch.setattr(seg_mod, "_load_segment_token_map", lambda: None)
        monkeypatch.setattr(seg_mod, "_operator_tokens", lambda: frozenset())
        grammar_tokens_by_segment.cache_clear()
        try:
            assert filter_mod._load_existing_tokens() == frozenset()
        finally:
            grammar_tokens_by_segment.cache_clear()


@requires_isn
class TestPromotionSeesOperators:
    """The promotion check's vocabulary carries the operator class."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        import imas_codex.standard_names.vocab_promotion as promo

        promo._SEGMENT_VOCAB_CACHE = None
        yield
        promo._SEGMENT_VOCAB_CACHE = None

    def test_operator_class_is_present(self):
        from imas_codex.standard_names.vocab_promotion import _load_isn_segment_vocab

        vocab = _load_isn_segment_vocab()
        assert OPERATOR_SEGMENT in vocab
        for operator in SAMPLE_OPERATORS:
            assert operator in vocab[OPERATOR_SEGMENT]

    def test_segment_classes_are_still_present(self):
        from imas_codex.standard_names.vocab_promotion import _load_isn_segment_vocab

        vocab = _load_isn_segment_vocab()
        assert "physical_base" in vocab
        assert "qualifier" in vocab
        assert "density" in vocab["physical_base"]


@requires_isn
class TestNoConsumerReadsTheSegmentMapAlone:
    """A new consumer bypassing the accessor is the recurrence to catch.

    Reading ``SEGMENT_TOKEN_MAP`` is legitimate only where the segment enums are
    genuinely the whole answer. Every such site is listed with its reason, so
    adding an unlisted one fails here rather than silently reintroducing
    operator-blindness in a sixth module.
    """

    #: path -> why reading the segment map directly is CORRECT there.
    LEGITIMATE_READERS = {
        "imas_codex/standard_names/segments.py": (
            "owns the accessor; reads the map to build it"
        ),
        "imas_codex/standard_names/context.py": (
            "renders the closed-vocab prompt block; injects operators separately "
            "via _load_operators_full, and the drift test asserts both arrive"
        ),
    }

    #: path -> the operator-blindness this reader still has. NOT an exemption:
    #: each entry is a known defect awaiting a routing change, listed so it stays
    #: visible instead of being rediscovered as the next instance of this bug.
    KNOWN_OPERATOR_BLIND: dict[str, str] = {}

    def _readers(self) -> set[str]:
        """Modules that actually IMPORT the segment map, not merely mention it."""
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            [
                "git",
                "grep",
                "-l",
                "-E",
                r"^\s*from imas_standard_names\.grammar\.constants import "
                r".*SEGMENT_TOKEN_MAP",
                "--",
                "imas_codex/",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        return {line for line in out.stdout.split("\n") if line.strip()}

    def test_every_segment_map_reader_is_accounted_for(self):
        accounted = set(self.LEGITIMATE_READERS) | set(self.KNOWN_OPERATOR_BLIND)
        unlisted = sorted(self._readers() - accounted)
        assert not unlisted, (
            "these modules import SEGMENT_TOKEN_MAP directly and so cannot see "
            "operators; route them through grammar_tokens_by_segment(), or list "
            "them with a reason if the segment enums really are the whole answer "
            f"there: {unlisted}"
        )

    def test_the_routed_consumers_no_longer_read_it(self):
        """The modules routed through the accessor must not regress."""
        readers = self._readers()
        for path in (
            "imas_codex/standard_names/vocab_token_filter.py",
            "imas_codex/standard_names/vocab_promotion.py",
            "imas_codex/standard_names/vocab_semantic_dedup.py",
            # model-facing: a composer asking this tool what operators exist was
            # told the class does not exist, so it kept offering them as
            # qualifiers — upstream of every gap the classifier then had to sort.
            "imas_codex/llm/sn_tools.py",
            "imas_codex/standard_names/workers.py",
        ):
            assert path not in readers, (
                f"{path} reads SEGMENT_TOKEN_MAP again — it must go through "
                f"grammar_tokens_by_segment()"
            )

    def test_listed_paths_still_exist(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        for path in (*self.LEGITIMATE_READERS, *self.KNOWN_OPERATOR_BLIND):
            assert (root / path).exists(), f"{path} no longer exists — drop the entry"

    def test_known_blind_readers_are_still_blind(self):
        """When one is fixed, move it out — a stale defect list misleads."""
        readers = self._readers()
        for path, defect in self.KNOWN_OPERATOR_BLIND.items():
            assert path in readers, (
                f"{path} no longer imports SEGMENT_TOKEN_MAP — the defect is "
                f"fixed, so drop it from KNOWN_OPERATOR_BLIND ({defect})"
            )
