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

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt  # noqa: E402

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
    """True if *name* round-trips literally, or via its retired-spelling alias.

    A retired operator spelling (e.g. ``tendency_of_*``) no longer parses
    under its own text once the grammar demotes it to an advisory alias, but
    it still names a real, resolvable quantity — the same substitution the
    production audits apply before reading a name's semantics. A genuinely
    unrepresentable name has no alias to resolve through and still fails.
    """
    try:
        if compose(parse(name, strict=True).ir) == name:
            return True
    except Exception:
        pass

    from imas_codex.standard_names.audits import resolve_retired_operator_spellings

    resolved = resolve_retired_operator_spellings(name)
    if resolved == name:
        return False
    try:
        return compose(parse(resolved, strict=True).ir) == resolved
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


def _nc_bad_examples() -> list[tuple[str, str]]:
    """Return every rejected teaching example with its owning rule ID."""
    from imas_codex.llm.prompt_loader import load_prompt_config

    cfg = load_prompt_config("sn_composition_rules")
    out: list[tuple[str, str]] = []
    for rule in cfg.get("composition_rules", []) or []:
        rule_id = rule.get("id", "?")
        for example in rule.get("examples_bad", []) or []:
            out.append((rule_id, example))
    return out


_NC_BAD_EXAMPLES = _nc_bad_examples()

# These negative examples are valid public-grammar names whose rejection is
# semantic. Every other negative is expected to fail strict public parsing.
_SEMANTIC_NEGATIVE_EXAMPLES = {
    ("NC-1", "area_of_flux_surface"),
    ("NC-13", "radial_outline"),
    ("NC-13", "vertical_outline"),
    ("NC-28", "plasma_current_reference_waveform"),
    ("NC-30", "emissivity_of_infrared_camera"),
    ("NC-30", "radiance_of_visible_camera"),
}


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


@pytest.mark.parametrize("rule_id,name", _NC_BAD_EXAMPLES)
def test_nc_rule_bad_example_has_declared_public_oracle_disposition(
    rule_id: str,
    name: str,
) -> None:
    """Every negative is explicitly semantic or rejected by strict parsing."""
    is_semantic_negative = (rule_id, name) in _SEMANTIC_NEGATIVE_EXAMPLES
    assert _round_trips(name) is is_semantic_negative, (
        f"negative example {(rule_id, name)!r} changed public-grammar disposition; "
        "classify it explicitly as a semantic negative or a parser rejection"
    )


def test_flux_surface_area_forms_round_trip_without_semantic_collapse() -> None:
    """The public grammar preserves cross-sectional and swept-surface forms."""
    cross_section = "poloidal_plane_cross_sectional_area_of_flux_surface"
    swept_surface = "surface_area_of_flux_surface"

    cross_ir = parse(cross_section, strict=True).ir
    surface_ir = parse(swept_surface, strict=True).ir

    assert compose(cross_ir) == cross_section
    assert compose(surface_ir) == swept_surface
    assert cross_ir != surface_ir
    assert cross_ir.projection is None
    assert {
        (qualifier.category, qualifier.token) for qualifier in cross_ir.qualifiers
    } == {
        ("section_plane", "poloidal"),
        ("geometry", "cross_sectional"),
    }
    assert surface_ir.projection is None
    assert {qualifier.token for qualifier in surface_ir.qualifiers} == {"surface"}


@pytest.mark.parametrize(
    "name",
    (
        "radial_outline_of_wall",
        "vertical_outline_of_wall",
        "radial_outline_of_plasma_boundary",
        "vertical_outline_of_plasma_boundary",
    ),
)
def test_owner_qualified_outline_forms_round_trip(name: str) -> None:
    """The public grammar preserves the owner on each outline projection."""
    assert _round_trips(name)


def test_outline_owners_are_distinct_public_ir_identities() -> None:
    """Parsing cannot fold wall and plasma-boundary outlines together."""
    wall = parse("radial_outline_of_wall", strict=True).ir
    boundary = parse("radial_outline_of_plasma_boundary", strict=True).ir

    assert compose(wall) == "radial_outline_of_wall"
    assert compose(boundary) == "radial_outline_of_plasma_boundary"
    assert wall != boundary
    assert wall.locus is not None
    assert boundary.locus is not None
    assert wall.locus.token == "wall"
    assert boundary.locus.token == "plasma_boundary"


def test_outline_rule_matches_public_grammar_and_owner_semantics() -> None:
    """The registry endorses owned outlines and rejects only owner erasure."""
    from imas_codex.llm.prompt_loader import load_prompt_config

    rules = load_prompt_config("sn_composition_rules")["composition_rules"]
    outline_rule = next(rule for rule in rules if rule["id"] == "NC-13")

    assert outline_rule["severity"] == "hard"
    assert all(_round_trips(name) for name in outline_rule["examples_good"])
    assert set(outline_rule["examples_bad"]) == {"radial_outline", "vertical_outline"}
    assert all(_round_trips(name) for name in outline_rule["examples_bad"])
    assert "vertical_outline_of_plasma_boundary" in outline_rule["examples_good"]
    assert "vertical_outline_of_plasma_boundary" not in outline_rule["examples_bad"]


def test_consistency_rule_forbids_generic_flux_surface_area_umbrella() -> None:
    """Consistency cannot erase the DD distinction between two surface kinds."""
    from imas_codex.llm.prompt_loader import load_prompt_config

    rules = load_prompt_config("sn_composition_rules")["composition_rules"]
    synonym_rule = next(rule for rule in rules if rule["id"] == "NC-1")

    assert synonym_rule["severity"] == "hard"
    assert (
        "poloidal_plane_cross_sectional_area_of_flux_surface"
        in synonym_rule["examples_good"]
    )
    assert "surface_area_of_flux_surface" in synonym_rule["examples_good"]
    assert "area_of_flux_surface" in synonym_rule["examples_bad"]
    assert "area_of_flux_surface" not in synonym_rule["examples_good"]


# ---------------------------------------------------------------------------
# Compound-base atomicity — the oracle behind the decomposition audit
# ---------------------------------------------------------------------------
# The review rubrics tell the reviewer to decompose any registered token that a
# candidate absorbed into ``physical_base``, then exempt a short list of
# "lexicalised atomic" compounds from that audit. Which spellings qualify is
# decidable rather than editorial: a compound is atomic iff the closed
# ``physical_bases`` registry holds it as ONE token AND the public IR leaves its
# qualifier set empty. A spelling the IR splits into base + qualifier is
# decomposable however lexicalised it reads, and exempting it teaches the
# composer that absorbing a registered qualifier is acceptable — the same
# absorption the audit exists to catch. These tests pin the public oracle so a
# rubric may only exempt a compound the grammar actually treats as one token.


def _registered_bases() -> frozenset[str]:
    from imas_standard_names.grammar import get_grammar_context

    bases = get_grammar_context()["grammar"]["vocabularies"]["physical_bases"]
    return frozenset(str(token) for token in bases)


def _is_lexicalised_atomic(name: str) -> bool:
    """True iff ISN holds ``name`` as one registered base carrying no qualifier."""
    try:
        ir = parse(name, strict=True).ir
    except Exception:
        return False
    return name in _registered_bases() and not ir.qualifiers


@pytest.mark.parametrize(
    "name", ("safety_factor", "loop_voltage", "internal_inductance")
)
def test_atomic_compound_is_one_registered_base_without_qualifiers(name: str) -> None:
    """Only these may be exempted from the decomposition audit."""
    assert _is_lexicalised_atomic(name)
    assert _round_trips(name)


@pytest.mark.parametrize(
    "name,base,qualifiers",
    (
        ("minor_radius", "radius", {"minor"}),
        ("major_radius", "radius", {"major"}),
        (
            "poloidal_plane_cross_sectional_area",
            "area",
            {"poloidal", "cross_sectional"},
        ),
        ("surface_area", "area", {"surface"}),
        ("polarization_angle", "angle", {"polarization"}),
    ),
)
def test_qualifier_bearing_compound_is_not_atomic(
    name: str,
    base: str,
    qualifiers: set[str],
) -> None:
    """The IR splits these, so no rubric may exempt them from the audit."""
    ir = parse(name, strict=True).ir

    assert ir.base.token == base
    assert {token.token for token in ir.qualifiers} == qualifiers
    assert not _is_lexicalised_atomic(name)


@pytest.mark.parametrize("name", ("poloidal_flux", "ellipticity_angle"))
def test_unparseable_spelling_is_never_teachable_as_a_name(name: str) -> None:
    """A spelling the public parser rejects cannot be endorsed as valid."""
    from imas_standard_names.grammar import ParseError

    with pytest.raises(ParseError):
        parse(name, strict=True)
    assert not _round_trips(name)
    assert not _is_lexicalised_atomic(name)


# The three surfaces above state the exemption in prose or structured YAML;
# these tests hold each stated claim to the same oracle so the guidance cannot
# drift back to exempting a decomposable spelling or endorsing one the parser
# rejects. ``_grammar_reference.md`` renders into every compose, review, refine,
# enrich and docs system prompt, so a wrong claim there reaches every seat.


def _grammar_reference_exempt_tokens() -> list[str]:
    """Tokens the decomposition checklist exempts from the audit."""
    text = " ".join(
        (PROMPTS_DIR / "shared/sn/_grammar_reference.md")
        .read_text(encoding="utf-8")
        .split()
    )
    match = re.search(r"qualifier set empty:\s*((?:`[a-z_]+`(?:,\s*)?)+)", text)
    assert match, "exemption list not found — did the checklist wording change?"
    return re.findall(r"`([a-z_]+)`", match.group(1))


def test_grammar_reference_exempts_only_atomic_compounds() -> None:
    """Every compound the shared checklist exempts is one unqualified base."""
    exempt = _grammar_reference_exempt_tokens()

    assert exempt, "checklist exempts nothing — extractor broke?"
    for token in exempt:
        assert _is_lexicalised_atomic(token), (
            f"{token!r} is exempted from the decomposition audit but the public "
            f"grammar does not hold it as one unqualified base"
        )


def _decomposition_criterion() -> dict:
    from imas_codex.llm.prompt_loader import load_prompt_config

    criteria = load_prompt_config("sn_review_criteria")
    for value in _walk_mappings(criteria):
        if "I4.6_physical_base_decomposition" in value:
            return value["I4.6_physical_base_decomposition"]
    raise AssertionError("decomposition criterion not found in sn_review_criteria")


def _walk_mappings(node: object) -> list[dict]:
    """Every mapping reachable in a nested YAML structure."""
    found: list[dict] = []
    if isinstance(node, dict):
        found.append(node)
        for child in node.values():
            found.extend(_walk_mappings(child))
    elif isinstance(node, list):
        for child in node:
            found.extend(_walk_mappings(child))
    return found


def test_review_criterion_allows_only_atomic_compounds() -> None:
    """Each allowed compound in the rubric config is one unqualified base."""
    allowed = _decomposition_criterion()["examples_allowed"]

    assert allowed, "criterion allows nothing — loader or YAML broke?"
    for entry in allowed:
        name = entry.split()[0]
        assert _is_lexicalised_atomic(name), (
            f"{name!r} is listed as an allowed atomic compound but the public "
            f"grammar splits it or rejects it"
        )


def test_review_criterion_defect_examples_are_genuine_defects() -> None:
    """Each rubric defect example really is decomposable or unparseable."""
    defects = _decomposition_criterion()["examples_bad"]

    assert defects, "criterion lists no defects — loader or YAML broke?"
    for entry in defects:
        name = entry["name"]
        assert not _is_lexicalised_atomic(name), (
            f"{name!r} is taught as a decomposition defect but the public "
            f"grammar holds it as one unqualified base"
        )


def test_abbreviation_fix_targets_round_trip() -> None:
    """Every canonical spelling the exemplars prescribe is a real name."""
    text = (PROMPTS_DIR / "shared/sn/_exemplars_name_only.md").read_text(
        encoding="utf-8"
    )
    targets = {
        token
        for line in text.splitlines()
        if "*Fix:*" in line
        for token in re.findall(r"`([^`]+)`", line)
        if _NAME.match(token)
    }

    assert targets, "no fix targets extracted — markup changed?"
    for name in sorted(targets):
        assert _round_trips(name), (
            f"prescribed fix target {name!r} does not round-trip through the "
            f"public ISN parser"
        )


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
    "relative_path,prompt_name",
    (
        ("sn/generate_name_system.md", None),
        ("sn/generate_name_dd.md", "sn/generate_name_dd"),
        ("sn/generate_name_dd_names.md", "sn/generate_name_dd_names"),
    ),
)
def test_compose_prompts_carry_complete_source_axis_contract(
    relative_path: str,
    prompt_name: str | None,
) -> None:
    """Every compose path fails closed on every authoritative semantic axis."""
    if prompt_name is None:
        prompt = (PROMPTS_DIR / relative_path).read_text(encoding="utf-8")
    else:
        prompt = render_prompt(
            prompt_name,
            {"items": [], "nearby_existing_names": [], "reference_exemplars": []},
        )
    text = " ".join(prompt.lower().split())

    for marker in _SOURCE_AXIS_MARKERS:
        assert marker in text
    assert "vocab_gap" in text
    assert "area_of_flux_surface" in text
    assert "ambiguous umbrella" in text
    assert "cocos is fixed ddv4 catalog metadata" in text
    assert "psi_like" in text
    assert "ip_like" in text


@pytest.mark.parametrize(
    "relative_path,prompt_name",
    (
        ("sn/generate_name_system.md", None),
        ("sn/generate_name_dd.md", "sn/generate_name_dd"),
        ("sn/generate_name_dd_names.md", "sn/generate_name_dd_names"),
    ),
)
def test_ordinal_geometry_never_changes_the_physical_carrier(
    relative_path: str,
    prompt_name: str | None,
) -> None:
    """Removing an array ordinal cannot recast non-LOS geometry as a sight-line."""
    if prompt_name is None:
        prompt = (PROMPTS_DIR / relative_path).read_text(encoding="utf-8")
    else:
        prompt = render_prompt(
            prompt_name,
            {"items": [], "nearby_existing_names": [], "reference_exemplars": []},
        )
    text = prompt.lower()

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


@pytest.mark.parametrize(
    "relative_path,prompt_name",
    (
        ("sn/generate_name_system.md", None),
        ("sn/generate_name_dd.md", "sn/generate_name_dd"),
        ("sn/generate_name_dd_names.md", "sn/generate_name_dd_names"),
    ),
)
def test_compose_prompts_forbid_endpoint_position_identity(
    relative_path: str,
    prompt_name: str | None,
) -> None:
    """Every compose mode retains endpoint ordering only in source provenance."""
    if prompt_name is None:
        prompt = (PROMPTS_DIR / relative_path).read_text(encoding="utf-8")
    else:
        prompt = render_prompt(
            prompt_name,
            {"items": [], "nearby_existing_names": [], "reference_exemplars": []},
        )
    text = " ".join(prompt.lower().split())

    assert "first, second, third, start, end" in text
    assert "dd path and source description" in text
    assert "radial_coordinate_of_arc_of_circle_start_point" in text
    assert "same arc-of-circle carrier" in text
    assert "vocab_gap" in text
    assert "line_of_sight" in text


@pytest.mark.parametrize(
    "prompt_name",
    ("sn/generate_name_dd", "sn/generate_name_dd_names"),
)
def test_compose_modes_preserve_outline_and_unit_vector_owners(
    prompt_name: str,
) -> None:
    """Both user modes reject owner-erasing geometry consolidation."""
    text = render_prompt(
        prompt_name,
        {"items": [], "nearby_existing_names": [], "reference_exemplars": []},
    ).lower()

    assert "radial_outline_of_wall" in text
    assert "radial_outline_of_plasma_boundary" in text
    assert "z_direction_unit_vector_of_camera" in text
    assert "bare `radial_outline` / `vertical_outline`" in text


def _render_policy_surface(prompt_name: str) -> str:
    """Render one production prompt surface without calling external services."""
    if prompt_name == "sn/generate_name_system":
        from imas_codex.standard_names.context import build_compose_context

        return render_prompt(prompt_name, context=build_compose_context())
    return render_prompt(
        prompt_name,
        {
            "items": [],
            "nearby_existing_names": [],
            "reference_exemplars": [],
            "review_scored_examples": [],
            "batch_context": "",
        },
    )


@pytest.mark.parametrize(
    "prompt_name",
    (
        "sn/generate_name_system",
        "sn/generate_name_dd",
        "sn/generate_name_dd_names",
        "sn/review",
        "sn/review_names",
    ),
)
def test_every_policy_surface_preserves_owned_outline_identities(
    prompt_name: str,
) -> None:
    """Rendered compose and review prompts agree with the public outline IR."""
    rendered = _render_policy_surface(prompt_name)

    assert _round_trips("radial_outline_of_wall")
    assert _round_trips("radial_outline_of_plasma_boundary")
    assert "radial_outline_of_wall" in rendered
    assert "radial_outline_of_plasma_boundary" in rendered
    assert "`plasma_boundary_outline_r`" not in rendered


def test_locusless_unit_vector_is_a_semantic_negative_not_a_parser_failure() -> None:
    """The rich compose prompt states the public parser's actual disposition."""
    assert _round_trips("z_direction_unit_vector")
    rendered = _render_policy_surface("sn/generate_name_dd")

    assert "public parser accepts a locus-less unit-vector name" in rendered
    assert "semantically rejected for an owned device vector" in rendered
    assert "rejected by the grammar at error severity" not in rendered


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
