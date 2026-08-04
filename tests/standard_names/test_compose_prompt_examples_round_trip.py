"""Drift guard: every ✓/✅-endorsed example NAME in the compose prompt and its
shared includes must round-trip through the public ISN parser.

The compose model is taught by example. If the prompt endorses a name
(``✓ `name``` / ``✅ `name```) that the public ISN grammar cannot strictly
parse and re-compose unchanged, the model is being trained on non-ISN
vocabulary — the exact drift this guard prevents. The lossless public
parser/composer is the oracle:

    compose(parse(name, strict=True).ir) == name

Extraction is deliberately high-precision: only a backtick token IMMEDIATELY
following a ✓/✅ marker counts as an *endorsed name*. This excludes vocabulary
TOKEN mentions (table cells, "use X" lists, ``_of_<token>`` fragments),
operator tokens, and ``(not `X`)`` negatives — none of which are standalone
standard names.

The ALLOWLIST holds endorsed examples that are genuinely unrepresentable today
because they reference a base/locus token not yet in the closed ISN vocabulary
(``UnknownBaseTokenError``). These are tracked under the separate vocab-
templating work that registers the missing tokens (or rewrites the example);
each must keep FAILING until then, so a stale allowlist entry is itself a test
failure (it means the vocab gap was closed and the entry should be removed).
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("imas_standard_names")

from imas_standard_names import compose, parse  # noqa: E402

from imas_codex.llm.prompt_loader import PROMPTS_DIR  # noqa: E402

# Compose system prompt + the shared includes it renders (see the
# ``{% include %}`` directives in generate_name_system.md).
_PROMPT_FILES = [
    "sn/generate_name_system.md",
    "shared/sn/_grammar_reference.md",
    "shared/sn/_exemplars.md",
    "shared/sn/_exemplars_name_only.md",
    "shared/sn/_coordinate_conventions.md",
    "shared/sn/_nc_rules.md",
]

# A backtick token immediately after a ✓ / ✅ marker is an endorsed example.
_ENDORSED = re.compile(r"[✓✅]\s*`([^`]+)`")
# Standard-name shape: lowercase snake_case with >= 2 segments.
_NAME = re.compile(r"^[a-z][a-z0-9]+(?:_[a-z0-9]+)+$")

# Endorsed examples that reference a base/locus token not yet registered in the
# closed ISN vocabulary. They cannot round-trip until the token is added (or the
# example is rewritten) — tracked under the vocab-templating follow-up. Keep the
# reason specific so closing a gap is obvious.
_VOCAB_GAP_ALLOWLIST: dict[str, str] = {}


def _round_trips(name: str) -> bool:
    try:
        return compose(parse(name, strict=True).ir) == name
    except Exception:
        return False


def _endorsed_names() -> dict[str, str]:
    """Map each endorsed example name -> the file it appears in (first hit)."""
    found: dict[str, str] = {}
    for rel in _PROMPT_FILES:
        text = (PROMPTS_DIR / rel).read_text(encoding="utf-8")
        for match in _ENDORSED.finditer(text):
            tok = match.group(1)
            if _NAME.match(tok) and "=" not in tok and "<" not in tok:
                found.setdefault(tok, rel)
    return found


_ENDORSED_NAMES = _endorsed_names()


def test_some_endorsed_names_were_extracted() -> None:
    """Guard the guard: a parser change must not silently empty the corpus."""
    assert len(_ENDORSED_NAMES) >= 30, (
        f"only {len(_ENDORSED_NAMES)} endorsed names extracted — extractor broke?"
    )


@pytest.mark.parametrize("name", sorted(_ENDORSED_NAMES))
def test_endorsed_prompt_name_round_trips(name: str) -> None:
    """Every ✓/✅-endorsed example round-trips, unless a known vocab gap."""
    if name in _VOCAB_GAP_ALLOWLIST:
        pytest.skip(f"vocab-gap allowlist: {_VOCAB_GAP_ALLOWLIST[name]}")
    assert _round_trips(name), (
        f"endorsed example {name!r} (in {_ENDORSED_NAMES[name]}) does not "
        f"round-trip through the public ISN parser. Fix the example to a "
        f"canonical form, or add it to _VOCAB_GAP_ALLOWLIST with a reason."
    )


@pytest.mark.parametrize("name", sorted(_VOCAB_GAP_ALLOWLIST))
def test_allowlist_has_no_stale_entries(name: str) -> None:
    """An allowlisted name must STILL fail; if it now round-trips, the vocab
    gap was closed and the entry must be removed from the allowlist."""
    assert not _round_trips(name), (
        f"{name!r} now round-trips — remove it from _VOCAB_GAP_ALLOWLIST"
    )


# ---------------------------------------------------------------------------
# NC composition-rule examples (imas_codex/llm/config/sn_composition_rules.yaml)
# ---------------------------------------------------------------------------
# The compose system prompt renders every rule's ``examples_good`` verbatim
# inside a ✓-marked code span (see shared/sn/_nc_rules.md).  The endorsed-name
# extractor above cannot reach them: it reads the UNRENDERED Jinja template
# (only ``{% for %}`` tags, no literal names), so the YAML was never checked.
# Load it directly and hold every endorsed NC example to the same round-trip
# contract — an ``examples_good`` entry the public ISN grammar cannot parse and
# re-compose unchanged is non-ISN vocabulary the compose model learns by
# example.  Each entry is a pure canonical name; any teaching gloss lives in the
# rule ``rule:`` prose, never the example list.


def _nc_good_examples() -> list[tuple[str, str]]:
    """(rule_id, name) for every ``examples_good`` entry, loaded like the pipeline."""
    from imas_codex.llm.prompt_loader import load_prompt_config

    cfg = load_prompt_config("sn_composition_rules")
    out: list[tuple[str, str]] = []
    for rule in cfg.get("composition_rules", []) or []:
        rid = rule.get("id", "?")
        for ex in rule.get("examples_good", []) or []:
            out.append((rid, ex))
    return out


_NC_GOOD_EXAMPLES = _nc_good_examples()


def test_nc_good_examples_were_extracted() -> None:
    """Guard the guard: a loader/YAML change must not silently empty the corpus."""
    assert len(_NC_GOOD_EXAMPLES) >= 30, (
        f"only {len(_NC_GOOD_EXAMPLES)} NC examples_good entries loaded — "
        "loader or YAML broke?"
    )


@pytest.mark.parametrize("rule_id,name", _NC_GOOD_EXAMPLES)
def test_nc_rule_good_example_round_trips(rule_id: str, name: str) -> None:
    """Every ``examples_good`` name round-trips through the public ISN parser."""
    assert _round_trips(name), (
        f"NC rule {rule_id} examples_good entry {name!r} does not round-trip "
        f"through the public ISN parser. Rewrite it to a canonical form (the "
        f"rule prose carries any teaching gloss)."
    )


def test_flux_surface_area_forms_round_trip_without_semantic_collapse() -> None:
    """The public grammar preserves cross-sectional and swept-surface forms."""
    cross_section = "poloidal_cross_sectional_area_of_flux_surface"
    swept_surface = "surface_area_of_flux_surface"

    cross_ir = parse(cross_section, strict=True).ir
    surface_ir = parse(swept_surface, strict=True).ir

    assert compose(cross_ir) == cross_section
    assert compose(surface_ir) == swept_surface
    assert cross_ir != surface_ir
    assert cross_ir.projection is not None
    assert cross_ir.projection.axis == "poloidal"
    assert {qualifier.token for qualifier in cross_ir.qualifiers} == {"cross_sectional"}
    assert surface_ir.projection is None
    assert {qualifier.token for qualifier in surface_ir.qualifiers} == {"surface"}


def test_consistency_rule_forbids_generic_flux_surface_area_umbrella() -> None:
    """Consistency cannot erase the DD distinction between two surface kinds."""
    from imas_codex.llm.prompt_loader import load_prompt_config

    rules = load_prompt_config("sn_composition_rules")["composition_rules"]
    synonym_rule = next(rule for rule in rules if rule["id"] == "NC-1")

    assert synonym_rule["severity"] == "hard"
    assert (
        "poloidal_cross_sectional_area_of_flux_surface" in synonym_rule["examples_good"]
    )
    assert "surface_area_of_flux_surface" in synonym_rule["examples_good"]
    assert "area_of_flux_surface" in synonym_rule["examples_bad"]
    assert "area_of_flux_surface" not in synonym_rule["examples_good"]


_SOURCE_AXIS_MARKERS = (
    "subject/object",
    "mechanism/cause",
    "locus/carrier",
    "projection/axis",
    "surface kind",
    "geometry representation",
    "coordinate kind",
    "aggregation",
    "process",
    "state",
    "unit",
    "dd-authoritative transformation/label semantics",
)


@pytest.mark.parametrize(
    "relative_path",
    ("sn/generate_name_system.md", "sn/generate_name_dd.md"),
)
def test_compose_prompts_carry_complete_source_axis_contract(
    relative_path: str,
) -> None:
    """Both compose seats fail closed on every authoritative semantic axis."""
    source = (PROMPTS_DIR / relative_path).read_text(encoding="utf-8")
    text = " ".join(source.lower().split())

    for marker in _SOURCE_AXIS_MARKERS:
        assert marker in text
    assert "vocab_gap" in text
    assert "area_of_flux_surface" in text
    assert "ambiguous umbrella" in text
    assert "cocos is fixed ddv4 catalog metadata" in text
    assert "psi_like" in text
    assert "ip_like" in text


@pytest.mark.parametrize(
    "relative_path",
    ("sn/generate_name_system.md", "sn/generate_name_dd.md"),
)
def test_ordinal_geometry_never_changes_the_physical_carrier(
    relative_path: str,
) -> None:
    """Removing an array ordinal cannot recast non-LOS geometry as a sight-line."""
    text = (PROMPTS_DIR / relative_path).read_text(encoding="utf-8").lower()

    for counterexample in (
        "thick_line",
        "pellet",
        "gas pipe",
        "shunt",
        "beam path",
        "interpolation",
        "other-object outline",
    ):
        assert counterexample in text
    assert "only genuine" in text or "only paths genuinely" in text
    assert "line_of_sight" in text
    assert "vocab_gap" in text


# A prefix transformation may coexist with a projection, and change_in is a
# bare-prefix operator. These forms MUST round-trip permanently — that
# co-existence is the grammar invariant this guard locks in.
_OPERATOR_PROJECTION_FORMS = [
    "tendency_of_toroidal_current_density",
    "time_derivative_of_radial_magnetic_field",
    "gradient_of_perpendicular_electron_pressure",
    "poloidal_change_in_ion_velocity",
    "change_in_electron_density",
    "toroidal_surface_integrated_current_density",
]


@pytest.mark.parametrize("name", _OPERATOR_PROJECTION_FORMS)
def test_operator_projection_forms_round_trip(name: str) -> None:
    assert _round_trips(name), (
        f"operator×projection form {name!r} must round-trip through ISN"
    )
