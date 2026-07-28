"""Every ISN vocabulary reaches every prompt seat that claims to teach the grammar.

Two axes have to be discovered rather than listed, because a hand-written list is
what lets a vocabulary slip through unnoticed: ``operators`` and the segment
enums each went missing from a consumer that enumerated what it knew about.

- **Seats** are found by walking the prompt tree for templates that include the
  shared grammar reference. A new seat that embeds it is covered the day it is
  added.
- **Vocabularies** are found by introspecting ``load_*`` in ISN's
  ``vocab_loaders``. A vocabulary ISN adds shows up here as a failure until it is
  either injected or justified in :data:`UNREACHED`.

A (seat, vocabulary) pair may only be exempt through :data:`UNREACHED`, which
carries the reason and the owner of the fix.
"""

from __future__ import annotations

import inspect

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt
from imas_codex.standard_names.context import build_compose_context

#: Template that carries the grammar vocabulary. A seat embedding it is asserting
#: that its model can see the whole grammar.
GRAMMAR_REFERENCE = "sn/_grammar_reference.md"

#: Tokens too short to assert on: a single letter or pair occurs inside unrelated
#: prose, so a substring hit proves nothing either way.
MIN_TOKEN_LEN = 3

#: Tokens known not to reach the rendered prompt, per vocabulary, with the reason
#: and where the fix belongs.  Exemptions are per TOKEN, never per vocabulary: a
#: whole-vocabulary exemption would hide real drift in the other 171 tokens, which
#: is exactly the failure this test exists to catch.
UNREACHED: dict[str, tuple[frozenset[str], str]] = {
    # ISN-side. These loci are in the locus registry but in no SEGMENT_TOKEN_MAP
    # segment, so codex's injection — which walks the segment map — has nothing to
    # emit them from. Same shape as the `normalizing_qualifiers` token
    # `gyrocenter`. Fix belongs in imas-standard-names: a registry entry reachable
    # by the parser should be reachable by a consumer enumerating the vocabulary.
    "locus_registry": (
        frozenset({"active_wall_point", "beam_path", "primary"}),
        "locus registry entries absent from every SEGMENT_TOKEN_MAP segment",
    ),
}


def _isn_available() -> bool:
    try:
        import imas_standard_names.grammar.vocab_loaders  # noqa: F401
    except ImportError:
        return False
    return True


requires_isn = pytest.mark.skipif(
    not _isn_available(), reason="imas-standard-names not installed"
)


# ---------------------------------------------------------------------------
# Axis discovery
# ---------------------------------------------------------------------------


def _grammar_seats() -> list[str]:
    """Prompt names that embed the shared grammar reference.

    Walks the prompt tree rather than naming seats, so a seat added later is
    covered without editing this file. Returns loader-style names
    (``sn/generate_name_system``) for the *system* templates, which are the ones
    ``render_prompt`` can render standalone.
    """
    stem = GRAMMAR_REFERENCE.rsplit("/", 1)[-1].removesuffix(".md")
    seats: list[str] = []
    for path in sorted(PROMPTS_DIR.rglob("*.md")):
        if path.name.startswith("_"):
            continue  # a partial, not a seat
        if stem not in path.read_text(encoding="utf-8"):
            continue
        seats.append(str(path.relative_to(PROMPTS_DIR).with_suffix("")))
    return seats


def _vocabulary_tokens() -> dict[str, frozenset[str]]:
    """Every ISN vocabulary, by loader name, flattened to its token strings.

    Introspects ``load_*`` in ``vocab_loaders`` so a vocabulary ISN adds is
    discovered. Loaders that need arguments, or whose result holds no token
    strings, are skipped — they carry no vocabulary to assert on.
    """
    from imas_standard_names.grammar import vocab_loaders

    out: dict[str, frozenset[str]] = {}
    for name, fn in inspect.getmembers(vocab_loaders, inspect.isfunction):
        if not name.startswith("load_"):
            continue
        if fn.__module__ != vocab_loaders.__name__:
            continue  # re-exported from elsewhere
        if inspect.signature(fn).parameters:
            continue  # needs arguments — not a plain vocabulary accessor
        try:
            loaded = fn()
        except Exception:
            continue
        tokens = _tokens_of(loaded)
        if tokens:
            out[name.removeprefix("load_")] = tokens
    return out


def _tokens_of(loaded: object) -> frozenset[str]:
    """Extract token strings from whatever shape a loader returned.

    Loaders return a flat set/tuple of tokens, a mapping keyed by token, or a
    registry object holding one such mapping attribute. Only string keys that
    look like grammar tokens are kept.
    """
    if isinstance(loaded, str):
        return frozenset()
    if isinstance(loaded, dict):
        return frozenset(k for k in loaded if isinstance(k, str))
    if isinstance(loaded, (set, frozenset, tuple, list)):
        return frozenset(t for t in loaded if isinstance(t, str))
    # A registry dataclass: find its token-keyed mapping.
    collected: set[str] = set()
    for attr, value in vars(loaded).items():
        if attr.startswith("_"):
            continue
        if isinstance(value, dict):
            collected.update(k for k in value if isinstance(k, str))
        elif isinstance(value, (set, frozenset, tuple, list)):
            collected.update(t for t in value if isinstance(t, str))
    return frozenset(collected)


@pytest.fixture(scope="module")
def seats() -> list[str]:
    return _grammar_seats()


@pytest.fixture(scope="module")
def vocabularies() -> dict[str, frozenset[str]]:
    return _vocabulary_tokens()


@pytest.fixture(scope="module")
def rendered(seats: list[str]) -> dict[str, str]:
    """Render every discovered seat once with the production context."""
    ctx = build_compose_context()
    out: dict[str, str] = {}
    for seat in seats:
        try:
            out[seat] = render_prompt(seat, context=ctx)
        except Exception:
            # A user-side template needing per-call variables is not a
            # standalone-renderable seat; the system template carries the
            # grammar reference and is asserted on.
            continue
    return out


# ---------------------------------------------------------------------------
# The axes must actually be discovered
# ---------------------------------------------------------------------------


@requires_isn
class TestAxisDiscovery:
    """A silently empty axis would make every assertion below vacuous."""

    def test_seats_discovered(self, seats):
        assert seats, f"no prompt embeds {GRAMMAR_REFERENCE}"

    def test_the_name_bearing_seats_are_among_them(self, seats):
        for seat in ("sn/generate_name_system", "sn/refine_name_system"):
            assert seat in seats

    def test_vocabularies_discovered(self, vocabularies):
        assert len(vocabularies) >= 10, (
            f"expected ISN to expose ≥10 vocabularies, found {sorted(vocabularies)}"
        )

    def test_operators_are_one_of_the_discovered_vocabularies(self, vocabularies):
        assert "operators" in vocabularies
        assert len(vocabularies["operators"]) >= 40

    def test_rendered_seats_are_nonempty(self, rendered):
        assert rendered, "no seat rendered"
        for seat, text in rendered.items():
            assert text.strip(), f"{seat} rendered empty"

    def test_exemptions_name_a_real_vocabulary(self, vocabularies):
        """An exemption for a vocabulary that no longer exists is dead weight."""
        from imas_standard_names.grammar import vocab_loaders

        for vocabulary, (exempt, _reason) in UNREACHED.items():
            assert hasattr(vocab_loaders, f"load_{vocabulary}"), (
                f"UNREACHED names '{vocabulary}' but ISN has no "
                f"load_{vocabulary}() — drop the entry"
            )
            unknown = sorted(exempt - vocabularies.get(vocabulary, frozenset()))
            assert not unknown, (
                f"UNREACHED exempts {unknown} from {vocabulary}, but ISN no longer "
                f"declares those tokens — drop them"
            )


# ---------------------------------------------------------------------------
# The drift assertion
# ---------------------------------------------------------------------------


def _missing(tokens: frozenset[str], text: str, *, vocabulary: str = "") -> list[str]:
    """Tokens of ``vocabulary`` absent from ``text``, minus its known exemptions."""
    exempt = UNREACHED.get(vocabulary, (frozenset(), ""))[0]
    return sorted(
        t
        for t in tokens
        if len(t) >= MIN_TOKEN_LEN and t not in exempt and t not in text
    )


@requires_isn
class TestEveryVocabularyReachesEverySeat:
    """The assertion both partial predecessors of this test were missing."""

    def test_no_vocabulary_is_silently_absent(self, rendered, vocabularies):
        failures: list[str] = []
        for vocabulary, tokens in sorted(vocabularies.items()):
            for seat, text in sorted(rendered.items()):
                gone = _missing(tokens, text, vocabulary=vocabulary)
                if gone:
                    failures.append(
                        f"{seat}: {vocabulary} missing {len(gone)}/{len(tokens)} "
                        f"tokens, e.g. {gone[:8]}"
                    )
        assert not failures, (
            "grammar vocabulary absent from a seat that embeds the grammar "
            "reference; inject it or justify the tokens in UNREACHED:\n"
            + "\n".join(failures)
        )

    def test_every_operator_token_reaches_every_seat(self, rendered, vocabularies):
        """Pinned separately because it is the vocabulary that keeps going missing."""
        operators = vocabularies["operators"]
        for seat, text in sorted(rendered.items()):
            gone = _missing(operators, text)
            assert not gone, f"{seat} is missing operators: {gone}"

    def test_exempted_tokens_are_genuinely_unreachable(self, rendered, vocabularies):
        """An exemption that has since been fixed must be removed, not left to rot."""
        for vocabulary, (exempt, reason) in UNREACHED.items():
            assert vocabulary in vocabularies, (
                f"UNREACHED names vocabulary '{vocabulary}' which is no longer "
                f"discovered — drop the entry"
            )
            for token in sorted(exempt):
                assert any(token not in text for text in rendered.values()), (
                    f"{vocabulary} token '{token}' now reaches every seat — drop it "
                    f"from UNREACHED ({reason})"
                )
