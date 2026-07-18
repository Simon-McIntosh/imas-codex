"""Post-generation audits for standard name candidates.

Six deterministic checks run after ISN validation to catch quality issues
that grammar/pydantic validation alone cannot detect. Each check returns
tagged issue strings (``"audit:<check_name>: <detail>"``) appended to the
candidate's ``validation_issues`` list.

Critical checks (quarantine on failure): latex_def_check, synonym_check,
multi_subject_check.
Non-critical (advisory only): provenance_verb_check, unit_dimension_check,
cocos_specificity_check.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _isn_process_tokens() -> frozenset[str]:
    """Return the canonical set of process tokens registered in ISN grammar.

    Queried from ``imas_standard_names.grammar.get_grammar_context()`` at runtime
    so the audit stays aligned with whichever ISN release is installed. Any token
    in this set is a legitimate ``due_to_<token>`` target and must not be flagged
    as an adjective by :func:`causal_due_to_check`.
    """
    try:
        from imas_standard_names.grammar import get_grammar_context

        ctx = get_grammar_context()
        for section in ctx.get("vocabulary_sections", []) or []:
            if section.get("segment") == "process":
                return frozenset(section.get("tokens") or ())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not load ISN process tokens: %s", exc)
    return frozenset()


@lru_cache(maxsize=1)
def _isn_locus_tokens() -> frozenset[str]:
    """Return the canonical set of locus tokens registered in installed ISN."""
    try:
        from imas_standard_names.grammar import vocab_loaders

        registry = vocab_loaders.load_locus_registry()
        return frozenset(registry.loci)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not load ISN locus registry: %s", exc)
    return frozenset()


# Checks whose failure demotes to quarantined
CRITICAL_CHECKS = frozenset(
    {
        "latex_def_check",
        "synonym_check",
        "multi_subject_check",
        "placeholder_check",
        "unit_validity_check",
        "generic_noun_check",
        "tautology_check",
        "spectral_suffix_check",
        "abbreviation_check",
        "name_description_consistency_check",
        "name_unit_consistency_check",
        "representation_artifact_check",
        "causal_due_to_check",
        "implicit_field_check",
        "density_unit_consistency_check",
        "position_coordinate_check",
        "vector_field_component_check",
        "segment_order_check",
        "aggregator_order_check",
        "diamagnetic_component_check",
        "amplitude_of_prefix_check",
        "mode_number_suffix_check",
        "cumulative_prefix_check",
        "pulse_schedule_reference_check",
        "ratio_binary_operator_check",
        "adjacent_duplicate_token_check",
        "semantic_similarity_check",
        "preposition_physical_base_check",
        "canonical_locus_check",
        "description_notation_check",
        "derived_parent_structure_check",
    }
)

# Map from head-noun tokens present in a standard name to the unit(s) they
# imply. Keys are tokens that appear in names; values are sets of acceptable
# units. When the name contains the token but the declared unit is not in the
# expected set, the audit raises a critical failure.
#
# These rules are deliberately conservative — only unambiguous head nouns are
# listed. Ambiguous words (``radiation``, ``field``) are left out because they
# appear across multiple physical dimensions.
_NAME_TOKEN_UNIT_EXPECTATIONS: dict[str, set[str]] = {
    # Energy head noun must be in energy units.
    "energy": {"J", "eV", "keV", "MeV", "GeV"},
    # Power implies a rate of energy delivery.
    "power": {"W", "MW", "kW"},
    # Temperature implies thermal units.
    "temperature": {"eV", "keV", "K"},
    # Pressure implies Pa (or dimensionally equivalent J/m^3).
    "pressure": {"Pa", "kPa", "MPa", "bar", "J.m^-3"},
    # Voltage implies V.
    "voltage": {"V", "kV", "mV"},
    # Angle / rotation implies rad.
    "angle": {"rad", "deg", "sr"},
    # Mass implies kg or u.
    "mass": {"kg", "u"},
    # Frequency implies Hz.
    "frequency": {"Hz", "kHz", "MHz", "GHz", "rad.s^-1", "s^-1"},
}

# Single-token names that are too generic to be self-describing standard names.
# A standard name must convey its meaning without requiring source-path context.
_GENERIC_NOUN_NAMES = frozenset(
    {
        "geometry",
        "data",
        "value",
        "quantity",
        "parameter",
        "coefficient",
        "coefficients",
        "element",
        "elements",
        "object",
        "objects",
        "node",
        "nodes",
        "index",
        "measure",
        "type",
        "name",
        "label",
        "status",
        "flag",
        "mode",
        "state",
        "version",
        "identifier",
        "metadata",
    }
)

# Regex patterns for tautological preposition chains.
# A name like "radial_position_of_reference_position" is wrong — "position_of_*_position"
# repeats the head noun. Similarly "component_of_*_component".
_TAUTOLOGY_HEADS = (
    "position",
    "component",
    "coordinate",
    "angle",
    "distance",
    "radius",
    "height",
    "width",
    "length",
)

# Tokens that indicate an unfilled prompt placeholder leaked through.
# Matches bracketed tokens containing these words anywhere in documentation/description.
_PLACEHOLDER_TOKENS = frozenset(
    {
        "condition",
        "specific condition",
        "specific physical condition",
        "quantity",
        "value",
        "unit",
        "placeholder",
        "todo",
        "fill in",
        "tbd",
    }
)

# Non-unit tokens that indicate an invalid unit expression — these are
# shape-related or semantic labels that should never appear in a unit string.
_INVALID_UNIT_TOKENS = frozenset(
    {
        "dimension",
        "rank",
        "fourier",
        "component",
        "coefficient",
        "shape",
        "index",
        "tbd",
        "n/a",
    }
)

# Provenance verbs that should not appear in standard names
_PROVENANCE_VERBS = frozenset(
    {"measured", "reconstructed", "fitted", "computed", "calculated"}
)

# Minimal unit → expected description noun mapping
_UNIT_NOUN_MAP: dict[str, set[str]] = {
    "m": {
        "position",
        "length",
        "distance",
        "radius",
        "height",
        "width",
        "displacement",
        "coordinate",
        "separation",
        "shift",
        "offset",
        "circumference",
        "perimeter",
        "major",
        "minor",
        "elongation",
    },
    "eV": {
        "temperature",
        "energy",
        "thermal",
        "potential",
        "ionization",
        "ionisation",
        "work function",
        "binding",
    },
    "K": {
        "temperature",
        "thermal",
    },
    "A": {
        "current",
    },
    "Pa": {
        "pressure",
    },
    "T": {
        "magnetic",
        "field",
    },
    "Wb": {
        "flux",
        "magnetic",
    },
    "V": {
        "voltage",
        "potential",
        "electric",
        "loop",
    },
    "W": {
        "power",
        "heating",
        "radiation",
        "radiated",
    },
    "m^-3": {
        "density",
        "concentration",
    },
    "s": {
        "time",
        "duration",
        "confinement",
        "period",
    },
    "rad": {
        "angle",
        "phase",
        "rotation",
        "toroidal",
        "poloidal",
    },
    "m^2": {
        "area",
        "cross",
        "section",
        "surface",
    },
    "m^3": {
        "volume",
    },
}

# Definition-indicator words for latex symbol definitions.
# Single words use \b word-boundary regex; multi-word phrases use substring match.
_DEFINITION_WORDS_SINGLE = frozenset(
    {"is", "are", "denotes", "represents", "where", "defined", "being"}
)
_DEFINITION_PHRASES = frozenset({"given by", "expressed as", "known as"})

# Pre-compiled word-boundary pattern for single definition words
_DEF_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _DEFINITION_WORDS_SINGLE) + r")\b"
)


def latex_def_check(candidate: dict[str, Any]) -> list[str]:
    """Check that every LaTeX symbol in documentation has a definition.

    Scans ``documentation`` for ``$...$`` groups and verifies each unique
    symbol has at least one definition sentence (heuristic: a sentence
    within 2 sentences of first occurrence containing the symbol, plus
    a definition-indicator word or a unit in brackets).
    """
    issues: list[str] = []
    doc = candidate.get("documentation") or ""
    if not doc:
        return issues

    # Find all inline math groups $...$  (not display $$...$$)
    # First remove display math to avoid double-matching
    doc_no_display = re.sub(r"\$\$[^$]+\$\$", " ", doc)
    symbols = set(re.findall(r"\$([^$]+)\$", doc_no_display))

    if not symbols:
        return issues

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", doc)

    for sym in symbols:
        # Skip very short fragments like single digits or operators
        sym_stripped = sym.strip()
        if len(sym_stripped) <= 1 and not sym_stripped.isalpha():
            continue

        # Skip pure numeric LaTeX like 10^{21}, 10^{-19}, 2.5\times10^{19}
        # — these are numeric magnitudes/exponents, not symbolic variables
        # requiring definition.
        if re.fullmatch(
            r"[\d.\-+]+(?:\s*\\times\s*)?1?0?\^?\{?-?\d+\}?|\d+(?:\.\d+)?|"
            r"1?0?\^\{?-?\d+\}?|\\times|\\cdot",
            sym_stripped,
        ):
            continue
        # Also skip if it's just a number followed by an exponent group
        if re.fullmatch(r"1?0?\s*\^\s*\{[-\d]+\}", sym_stripped):
            continue
        if re.fullmatch(r"\d+\^\{?-?\d+\}?", sym_stripped):
            continue

        # Skip universal physics / math constants that are self-evident:
        # \pi, 2\pi, \pi/2, \alpha, \beta, \gamma, \mu_0, \epsilon_0,
        # \hbar, c, e, k_B (optionally with a numeric scalar prefix or
        # simple rational factor). These need no definition sentence.
        if re.fullmatch(
            r"(?:\d+(?:\.\d+)?\s*)?"  # optional numeric coefficient
            r"(?:\\pi|\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\mu|"
            r"\\lambda|\\sigma|\\tau|\\omega|\\Omega|"
            r"\\mu_0|\\mu_\{0\}|\\epsilon_0|\\epsilon_\{0\}|\\hbar|"
            r"k_B|k_\{B\})"
            r"(?:\s*/\s*\d+)?",  # optional /N divisor
            sym_stripped,
        ):
            continue

        # Find first occurrence sentence index
        first_idx = None
        for i, sent in enumerate(sentences):
            if f"${sym}$" in sent or f"${sym_stripped}$" in sent:
                first_idx = i
                break

        if first_idx is None:
            continue

        # Check nearby sentences (within 2) for definition
        found_def = False
        window = sentences[max(0, first_idx - 1) : first_idx + 3]
        for sent in window:
            sent_lower = sent.lower()
            # Check for definition words near the symbol (word-boundary match)
            has_def_word = bool(_DEF_WORD_RE.search(sent_lower)) or any(
                p in sent_lower for p in _DEFINITION_PHRASES
            )
            # Check for unit in brackets/parentheses
            has_unit_bracket = bool(
                re.search(r"\([^)]*(?:eV|m|A|T|Pa|Wb|K|rad|s|W|V)[^)]*\)", sent)
            )
            if has_def_word or has_unit_bracket:
                found_def = True
                break

        if not found_def:
            issues.append(
                f"audit:latex_def_check: symbol ${sym}$ lacks a definition sentence"
            )

    return issues


def description_notation_check(candidate: dict[str, Any]) -> list[str]:
    """Descriptions must be plain Unicode text — no LaTeX/math markup.

    A ``$`` math delimiter or a backslash (a LaTeX command such as ``\\phi``,
    or a stranded backslash left by a half-converted Greek symbol ``\\φ``) in
    the ``description`` is a convention violation that must never reach the
    catalog. Firing here is a critical failure, so the name is quarantined
    even if the description normalizer missed an edge case. ``documentation``
    legitimately carries LaTeX and is not checked.
    """
    desc = candidate.get("description") or ""
    if "$" in desc or "\\" in desc:
        return [
            "audit:description_notation_check: description contains math "
            "markup ('$' or backslash); descriptions must be plain Unicode "
            "text (LaTeX belongs in the documentation field)"
        ]
    return []


# Time-derivative / rate-of-change markers that, when present in a
# description, require the name to carry an explicit tendency/change marker.
# Otherwise the name (a base quantity) contradicts the description (a rate).
_RATE_DESC_PATTERNS = (
    "instantaneous change",
    "instantaneous signed change",
    "rate of change",
    "time derivative",
    "time rate of change",
    "signed change in",
    "temporal derivative",
    "per unit time",
)

# Name prefixes/tokens that legitimately describe a rate/change quantity.
_RATE_NAME_MARKERS = (
    "tendency_of_",
    "change_in_",
    "rate_of_change_of_",
    "time_derivative_of_",
)


def description_verb_drift_check(candidate: dict[str, Any]) -> list[str]:
    """Flag name/description verb drift on rate-type paths.

    If a description claims the quantity is a time derivative or rate of
    change but the name lacks a rate marker (``tendency_of_``,
    ``change_in_``, ``rate_of_change_of_``), the name is mis-labelled as
    a base quantity. This is a critical mismatch that invites downstream
    misuse.

    Conversely, avoid the awkward literal prefix ``instant_change_`` in
    names — prefer ``change_in_`` or ``tendency_of_``. Names starting
    with ``instant_change_`` are also flagged.
    """
    issues: list[str] = []
    name = (
        str(candidate.get("id") or candidate.get("standard_name") or "").strip().lower()
    )
    description = str(candidate.get("description") or "").lower()

    if not name or not description:
        return issues

    # Guard: names starting with "instant_change_" or "instantaneous_change_"
    # should be replaced with "change_in_" or "tendency_of_".
    if name.startswith(("instant_change_", "instantaneous_change_")):
        issues.append(
            "audit:description_verb_drift_check: name begins with "
            f"'{name.split('_')[0]}_change_'; prefer 'change_in_' or "
            "'tendency_of_'"
        )
        return issues

    has_rate_desc = any(pat in description for pat in _RATE_DESC_PATTERNS)
    if not has_rate_desc:
        return issues

    # Flux quantities inherently describe flow rates — "per unit time" in
    # their description is definitional, not a verb drift indicator.
    # Only flag when strong rate language is used (time derivative, rate of change).
    _STRONG_RATE_PATTERNS = (
        "instantaneous change",
        "instantaneous signed change",
        "rate of change",
        "time derivative",
        "time rate of change",
        "temporal derivative",
    )
    has_strong_rate = any(pat in description for pat in _STRONG_RATE_PATTERNS)
    if not has_strong_rate and "_flux" in name:
        return issues  # "per unit time" in flux description is not verb drift

    has_rate_name = any(marker in name for marker in _RATE_NAME_MARKERS)
    if not has_rate_name:
        issues.append(
            "audit:description_verb_drift_check: description implies a "
            "rate/time-derivative but name lacks 'tendency_of_', "
            "'change_in_', or 'rate_of_change_of_' marker"
        )
    return issues


# Structural dimensionality tags leaked from DD data-type metadata that
# should not appear in human-readable descriptions.
_STRUCTURAL_DIM_RE = re.compile(r"\b([0-3])[dD]\b")

# Contexts where "2D"/"3D" is a valid physics descriptor, not a storage tag
_PHYSICS_DIM_CONTEXT_RE = re.compile(
    r"\b[0-3][dD]\s+"
    r"(?:poloidal|toroidal|equilibrium|magnetic|plasma|field|geometry|space"
    r"|grid|mesh|domain|cross[- ]section|plane|surface|volume|configuration)",
    re.IGNORECASE,
)


def structural_dim_tag_check(candidate: dict[str, Any]) -> list[str]:
    """Flag descriptions that echo DD data-type dimensionality tags.

    Tokens like ``1D``, ``2D``, ``3D`` in a description are a leak from
    the DD data type (``FLT_1D`` etc.) rather than a physically meaningful
    descriptor. However, physics-geometry usage like "2D poloidal plane"
    or "3D equilibrium" is valid and not flagged.
    """
    issues: list[str] = []
    description = str(candidate.get("description") or "")
    match = _STRUCTURAL_DIM_RE.search(description)
    if match:
        # Check if the dimensionality tag appears in a valid physics context
        if not _PHYSICS_DIM_CONTEXT_RE.search(description):
            issues.append(
                f"audit:structural_dim_tag_check: description contains "
                f"storage-shape tag '{match.group(0)}' (remove or rephrase "
                "in terms of the physical quantity)"
            )
    return issues


def provenance_verb_check(
    candidate: dict[str, Any], source_path: str | None = None
) -> list[str]:
    """Check that name contains no provenance verbs unless source path does too.

    Standard names should describe the physical quantity, not how it was
    obtained. Words like ``measured``, ``reconstructed`` are only allowed
    when the source DD path itself contains that word.
    """
    issues: list[str] = []
    name = candidate.get("id") or candidate.get("standard_name") or ""
    tokens = set(name.split("_"))
    source_tokens = set((source_path or "").replace("/", "_").split("_"))

    for verb in _PROVENANCE_VERBS:
        if verb in tokens and verb not in source_tokens:
            issues.append(
                f"audit:provenance_verb_check: name contains '{verb}' "
                f"but source path does not"
            )

    return issues


def synonym_check(
    candidate: dict[str, Any],
    existing_sns_in_domain: list[dict[str, Any]],
) -> list[str]:
    """Flag near-duplicate names with cosine similarity > 0.92.

    Compares the candidate's description embedding against precomputed
    embeddings of existing SNs in the same domain with the same unit.
    """
    issues: list[str] = []
    if not existing_sns_in_domain:
        return issues

    cand_name = candidate.get("id") or candidate.get("standard_name") or ""
    # User invariant: do NOT coerce missing units to "1"; treat them as
    # unknown so cross-name comparisons cannot be silently performed
    # against a fabricated dimensionless default.
    cand_unit = candidate.get("unit")

    # Get candidate embedding
    cand_embedding = candidate.get("description_embedding")
    if cand_embedding is None:
        return issues

    cand_vec = np.array(cand_embedding, dtype=np.float32)
    cand_norm = np.linalg.norm(cand_vec)
    if cand_norm == 0:
        return issues

    for existing in existing_sns_in_domain:
        ex_name = existing.get("name") or existing.get("id") or ""
        if ex_name == cand_name:
            continue
        ex_unit = existing.get("unit")
        # Skip the comparison if either side has no stored unit — we
        # cannot meaningfully assert duplicate-by-unit when the truth is
        # unknown, and fabricating "1" hides the gap.
        if cand_unit is None or ex_unit is None:
            continue
        if ex_unit != cand_unit:
            continue
        ex_embedding = existing.get("description_embedding")
        if ex_embedding is None:
            continue

        ex_vec = np.array(ex_embedding, dtype=np.float32)
        ex_norm = np.linalg.norm(ex_vec)
        if ex_norm == 0:
            continue

        cosine = float(np.dot(cand_vec, ex_vec) / (cand_norm * ex_norm))
        if cosine > 0.92:
            issues.append(
                f"audit:synonym_check: cosine={cosine:.3f} with existing "
                f"'{ex_name}' (same unit={cand_unit})"
            )

    return issues


def placeholder_check(candidate: dict[str, Any]) -> list[str]:
    """Detect unfilled prompt placeholders leaking into name/description/documentation.

    Flags bracketed tokens like ``[condition]``, ``[specific physical condition]``,
    ``[quantity]`` — these indicate the LLM copied the prompt's placeholder
    pattern verbatim instead of substituting a concrete value.
    """
    issues: list[str] = []
    name = candidate.get("id") or candidate.get("standard_name") or ""
    description = candidate.get("description") or ""
    documentation = candidate.get("documentation") or ""

    # Match [anything] where the inner text contains a placeholder token
    bracket_pattern = re.compile(r"\[([^\[\]]{1,80})\]")
    for field_name, text in (
        ("name", name),
        ("description", description),
        ("documentation", documentation),
    ):
        if not text:
            continue
        for match in bracket_pattern.finditer(text):
            inner = match.group(1).lower().strip()
            # Skip markdown links [text](url) — these have concrete link text.
            end = match.end()
            if end < len(text) and text[end] == "(":
                continue
            # Only flag when the bracketed content matches a known placeholder
            # token (single words or short phrases).
            for token in _PLACEHOLDER_TOKENS:
                if token in inner:
                    issues.append(
                        f"audit:placeholder_check: unfilled placeholder '[{match.group(1)}]' "
                        f"in {field_name}"
                    )
                    break

    return issues


def unit_validity_check(candidate: dict[str, Any]) -> list[str]:
    """Flag invented / malformed unit expressions.

    Catches units containing semantic labels (e.g. ``m^dimension``, ``Wb*fourier``)
    rather than valid SI symbols. Does not validate unit algebra (left to Pydantic /
    pint); this is a defensive sanity check against LLM hallucinations.

    Also flags DD-upstream quality issues: unit strings containing whitespace
    (prose unit names like ``"Elementary Charge Unit"``) and ``^dimension``
    placeholders that escaped the DD XML without resolution.

    Note:
        As of Phase B (DD unit overrides), the classes of DD-upstream defects
        enumerated in ``plans/research/standard-names/dd-unit-bugs.md`` are
        remapped at extraction time by
        ``imas_codex.standard_names.unit_overrides.resolve_unit`` — valid
        candidates should no longer reach this audit carrying prose units
        or ``m^dimension`` placeholders. This function is kept as a safety
        net: if either branch below fires in production, it means the
        override YAML (``standard_names/config/unit_overrides.yaml``)
        missed a case and needs a new rule.
    """
    issues: list[str] = []
    raw_unit = (candidate.get("unit") or "").strip()
    unit = raw_unit.lower()
    if not unit or unit in ("1", "dimensionless", "-", "none"):
        return issues

    # C.8: whitespace in unit string → dd_upstream severity
    if re.search(r"\s", raw_unit):
        issues.append(
            f"audit:unit_validity_check: unit '{raw_unit}' contains "
            f"whitespace — prose unit names are not valid SI expressions; "
            f"severity=dd_upstream"
        )
        return issues

    # C.8: ^dimension placeholder → dd_upstream severity
    if "^dimension" in unit:
        issues.append(
            f"audit:unit_validity_check: unit '{raw_unit}' contains "
            f"'^dimension' placeholder — unresolved DD variable; "
            f"severity=dd_upstream"
        )
        return issues

    # Split on unit algebra operators and check each token
    tokens = re.split(r"[\s*/.^()·×]+", unit)
    for tok in tokens:
        if not tok:
            continue
        if tok in _INVALID_UNIT_TOKENS:
            issues.append(
                f"audit:unit_validity_check: unit '{raw_unit}' "
                f"contains non-unit token '{tok}'"
            )
            break
    return issues


def unit_dimension_check(candidate: dict[str, Any]) -> list[str]:
    """Heuristic check that description nouns are consistent with unit.

    Uses a minimal unit→expected-noun-set map; flags when no noun from
    the expected set appears in the description.
    """
    issues: list[str] = []
    unit = candidate.get("unit") or ""
    description = (candidate.get("description") or "").lower()

    if not unit or unit in ("1", "dimensionless", "-") or not description:
        return issues

    # Check all unit keys for a match
    expected_nouns = _UNIT_NOUN_MAP.get(unit)
    if expected_nouns is None:
        return issues

    # Tokenize description
    desc_words = set(re.findall(r"[a-z]+", description))
    if not desc_words & expected_nouns:
        issues.append(
            f"audit:unit_dimension_check: unit='{unit}' but description "
            f"lacks expected terms {sorted(expected_nouns)[:5]}"
        )

    return issues


def name_unit_consistency_check(
    candidate: dict[str, Any], source_path: str | None = None
) -> list[str]:
    """Check that head-noun tokens in the *name* match the declared unit.

    Operates on the name alone (compose is always name-only per ADR-1).
    Catches cases like ``heating_power_due_to_ohmic``
    with unit ``J`` (should be ``W``) or ``neutral_beam_injection_unit_energy``
    with unit ``1`` (should be ``J`` or ``eV``).

    Ignores compound-unit decorations such as ``m^-3.W`` (power density) by
    also accepting units that *contain* an expected unit as a component. The
    failure is raised only when the name asserts a head dimension but the
    declared unit contains no compatible component.

    Normalized (gyrokinetic) quantities are exempt: when the DD source path
    contains ``_norm_`` or the name starts with / contains ``normalized``
    (or ``normalised``), a dimensionless unit is physically correct — the
    normalization divides out physical units.

    Args:
        candidate: Standard name candidate dict.
        source_path: Original DD path (used to detect ``_norm_`` segments).
    """
    issues: list[str] = []
    name = (candidate.get("id") or candidate.get("standard_name") or "").lower()
    unit = (candidate.get("unit") or "").strip()
    if not name or not unit:
        return issues
    if unit in ("1", "dimensionless", "-", "none"):
        dimensionless = True
    else:
        dimensionless = False

    # Bypass for normalized/gyrokinetic quantities: a dimensionless unit is
    # physically correct when the quantity has been normalized. Check the DD
    # source path first (authoritative), then fall back to name tokens.
    if dimensionless:
        # DD path contains _norm_ (e.g. moments_norm_gyrocenter/pressure)
        if source_path and "_norm_" in source_path.lower():
            return issues
        # Name starts with or contains normalization marker
        if (
            name.startswith("normalized_")
            or name.startswith("normalised_")
            or "_normalized_" in name
            or "_normalised_" in name
        ):
            return issues

    name_tokens = set(re.findall(r"[a-z]+", name))

    # Time-suffix compound names: the head noun is time, not the qualifier
    # token. ``energy_confinement_time``, ``particle_confinement_time``,
    # ``energy_decay_time``, ``current_diffusion_time`` etc. all have time
    # as the dimensional subject; the leading token classifies WHICH
    # characteristic time, not the unit. Unit should be seconds (or a time
    # decoration).
    _TIME_SUFFIX_MARKERS = (
        "_confinement_time",
        "_decay_time",
        "_relaxation_time",
        "_diffusion_time",
        "_lifetime",
        "_dwell_time",
        "_rise_time",
        "_fall_time",
        "_pulse_duration",
        "_pulse_length",
        "_persistence_time",
    )
    if any(marker in name for marker in _TIME_SUFFIX_MARKERS):
        return issues

    for token, expected_units in _NAME_TOKEN_UNIT_EXPECTATIONS.items():
        if token not in name_tokens:
            continue
        # Skip if name also contains a qualifier that shifts the head noun
        # (e.g. ``power_density`` has token ``power`` but density shifts the
        # unit to ``m^-3.W``).
        if token == "power" and "density" in name_tokens:
            continue
        if token == "energy" and "density" in name_tokens:
            continue
        if token == "angle" and ("offset" in name_tokens or "gradient" in name_tokens):
            continue
        # ``frequency`` as a modifier in compound nouns like
        # ``frequency_sweep_duration`` describes a different concept —
        # the head noun is ``duration`` (unit ``s``), not ``frequency``.
        if token == "frequency" and any(
            t in name_tokens
            for t in ("duration", "period", "sweep", "interval", "delay")
        ):
            continue
        # ``_flux`` shifts the head-noun dimension by time and area:
        # ``energy_flux`` → W.m^-2 (power per area), not energy.
        # ``mass_flux``   → kg.m^-2.s^-1, not mass.
        # ``charge_flux`` → A.m^-2, not charge.
        # The flux variant is too ambiguous (could be particle-flux of an
        # energy-bearing species) for a hard audit; defer to dimensional
        # consistency checks at the description layer.
        if "flux" in name_tokens and token in ("energy", "mass", "voltage"):
            continue
        # ``velocity`` shifts the head noun to a transport velocity:
        # ``energy_velocity_due_to_convection`` → m.s^-1 (convective
        # transport velocity of thermal energy), not energy. Same physics
        # as the ``_convection_velocity`` coefficient exemption below, but
        # covering the grammar-canonical ``velocity ... due_to_<process>``
        # ordering. Momentum is excluded: momentum per unit mass IS a
        # velocity, so that mismatch stays meaningful.
        if "velocity" in name_tokens and token in ("energy", "mass", "charge"):
            continue
        # ``_density`` qualifier already handled for power/energy above; also
        # exempt for mass (mass_density → kg.m^-3) and pressure (rare).
        if token == "mass" and "density" in name_tokens:
            continue
        # ``center_of_mass`` is a reference point (the barycentre), not a mass
        # quantity. The mass token is part of a compound location label, not a
        # dimensional subject. Same for similar location compounds.
        if token == "mass" and "center_of_mass" in name:
            continue
        # ``_source`` / ``_sink`` shift the head noun to a rate-per-volume:
        # ``energy_source`` → W/m^3 (volumetric power density), not energy.
        # ``particle_source`` → m^-3.s^-1, etc. Defer to description-layer
        # dimensional checks.
        if ("source" in name_tokens or "sink" in name_tokens) and token in (
            "energy",
            "power",
            "mass",
            "momentum",
        ):
            continue
        # ``_diffusivity`` / ``_conductivity`` / ``_resistivity`` describe
        # transport coefficients whose units depend on what is being
        # transported, not on the prefix token. ``ion_energy_diffusivity``
        # has m^2/s (kinematic diffusivity) regardless of the ``energy``
        # qualifier — same shape as thermal diffusivity.
        if any(
            coef in name
            for coef in (
                "_diffusivity",
                "_diffusion_coefficient",
                "_conductivity",
                "_conduction_coefficient",
                "_resistivity",
                "_viscosity",
                "_mobility",
                "_convection_coefficient",
                "_convection_velocity",
            )
        ):
            continue
        # _peaking_factor, _profile_factor, _ratio, _fraction names are
        # dimensionless by definition — the head noun (temperature, density)
        # describes the numerator quantity, not the declared unit.
        if any(
            suffix in name
            for suffix in (
                "_peaking_factor",
                "_profile_factor",
                "_ratio",
                "_fraction",
                "_normalized",
                "_normalised",
            )
        ):
            continue
        # Meta-descriptor names: constraint_weight_of_X, exact_flag_of_X,
        # convergence_count_of_X etc. describe META properties of a physical
        # quantity X (weight, flag, count) — the head noun is the meta token
        # (weight/flag/count), not X. Their unit is dimensionless by design.
        if any(
            marker in name
            for marker in (
                "_constraint_weight_of_",
                "_constraint_weight",
                "_exact_flag",
                "_iteration_count",
                "_convergence_count",
            )
        ):
            continue
        # Sensor/instrument qualifier names: ``temperature_sensor_signal_*``
        # describes a signal from a temperature sensor — the word
        # ``temperature`` classifies the sensor type, not the unit.
        # Rise/fall time, amplitude, etc. have time or voltage units.
        if f"{token}_sensor" in name or f"{token}_probe" in name:
            continue
        # ``mass_spectrometer`` is an instrument name — ``mass`` classifies
        # the spectrometer type (mass vs optical), not a mass quantity.
        if token == "mass" and "mass_spectrometer" in name:
            continue
        # ``energy_transport`` is a compound transport concept. The head noun
        # is the transport quantity, and ``energy`` classifies what is being
        # transported. Unit is a transport velocity (m/s) or diffusivity
        # (m^2/s), not energy.
        if token == "energy" and "energy_transport" in name:
            continue

        if dimensionless:
            issues.append(
                f"audit:name_unit_consistency_check: name contains '{token}' "
                f"but unit is dimensionless ('{unit}'); expected one of "
                f"{sorted(expected_units)}"
            )
            continue

        if unit in expected_units:
            continue
        # Accept compound units that list one expected unit as a factor or
        # exponentiated component (e.g. ``m^-3.W`` for power density is
        # already filtered above; ``N.m`` for torque is not a power issue).
        tokens_in_unit = set(re.findall(r"[A-Za-z]+", unit))
        if expected_units & tokens_in_unit:
            continue
        issues.append(
            f"audit:name_unit_consistency_check: name contains '{token}' "
            f"but unit='{unit}' is not in expected set "
            f"{sorted(expected_units)}"
        )

    return issues


def multi_subject_check(candidate: dict[str, Any]) -> list[str]:
    """Detect names combining two different subject segments.

    Uses ``parse_standard_name`` from ISN grammar to check whether
    multiple ``subject_*`` segments are detected in the name.
    """
    issues: list[str] = []
    name = candidate.get("id") or candidate.get("standard_name") or ""
    if not name:
        return issues

    try:
        from imas_standard_names.grammar import parse_standard_name

        parsed = parse_standard_name(name)
        # Binary operator implies two operands — this is legitimate
        if hasattr(parsed, "binary_operator") and parsed.binary_operator is not None:
            return issues
    except Exception:
        # Parse failure (incl. ISN ≥rc35 stacking / non-canonical-order
        # rejections) does NOT exempt the name from this audit — the greedy
        # heuristic below is purely lexical. Multi-subject names like
        # ``electron_deuterium_density`` raise at parse and must still be
        # flagged with the specific multi-subject reason.
        pass

    # Heuristic: check if name contains two subject enum values.
    # Use greedy longest-match so compound subjects like
    # ``deuterium_tritium`` consume their constituent tokens and prevent
    # ``deuterium`` + ``tritium`` from being counted as two subjects.
    try:
        from imas_standard_names.grammar import Subject

        # Sort longest-first for greedy matching
        all_subjects = sorted((s.value for s in Subject), key=len, reverse=True)
        remaining = name
        matched_subjects: list[str] = []
        for sv in all_subjects:
            # Match the subject value as a whole-word substring
            # (delimited by underscores or string boundaries)
            pattern = rf"(?:^|_){re.escape(sv)}(?:_|$)"
            if re.search(pattern, remaining):
                matched_subjects.append(sv)
                # Remove matched tokens so shorter subjects sharing the
                # same tokens (e.g. ``deuterium`` inside
                # ``deuterium_tritium``) are not double-counted.
                remaining = re.sub(pattern, "_", remaining).strip("_")

        # Exempt known unit-qualifier compounds where a subject token appears
        # as a modifier rather than a true subject. These are conventional
        # particle-count conversions in the DD, not dual subjects.
        #   - `*_electron_equivalent` — ionization-equivalent particle count
        #     (e.g. hydrogen molecule released → N electrons on full ionization)
        #   - `*_electron_temperature_equivalent` — temperature expressed as kT/e
        if name.endswith("_electron_equivalent"):
            matched_subjects = [s for s in matched_subjects if s != "electron"]

        # Exempt metadata/flag descriptors — names ending in ``_flag``,
        # ``_index``, or containing ``_state_`` reference classification
        # attributes rather than two physical subjects. E.g.
        # ``ion_state_neutral_flag`` describes a flag on the ion-state
        # enum; ``neutral`` is an enum value, not a second subject.
        _META_TOKENS = ("_flag", "_index", "_state_", "_type_flag")
        if any(tok in name for tok in _META_TOKENS):
            matched_subjects = []

        # Exempt compound physical_base patterns where a subject token
        # appears as part of a transport coefficient or diffusivity name.
        # E.g. ``ion_particle_diffusivity`` has subject=ion and
        # physical_base=particle_diffusivity — ``particle`` here is part
        # of the base, not a second subject.
        _COMPOUND_PB_TOKENS = (
            "particle_diffusivity",
            "particle_diffusion_coefficient",
            "particle_flux",
            "particle_source",
            "particle_source_rate",
            "particle_sink",
            "particle_confinement",
            "particle_radial_diffusivity",
            "particle_convection_velocity",
            "particle_convection_coefficient",
        )
        if any(cpb in name for cpb in _COMPOUND_PB_TOKENS):
            matched_subjects = [s for s in matched_subjects if s != "particle"]

        # Exempt orbit-classification and energy-classification subject
        # modifiers that form compound subjects with a following species.
        # E.g. ``trapped_electron`` is ONE compound subject, not two
        # separate subjects ``trapped`` + ``electron``.
        # ``co_passing_ion`` = co-passing ion (single compound subject).
        # ``fast_particles`` = fast particles (single compound subject).
        # ``total_thermal`` = total thermal (aggregator, not two subjects).
        _MODIFIER_SUBJECTS = frozenset(
            {
                "trapped",
                "co_passing",
                "counter_passing",
                "fast",
                "thermal",
                "total",
                "runaway",
            }
        )
        modifier_count = sum(1 for s in matched_subjects if s in _MODIFIER_SUBJECTS)
        if modifier_count > 0:
            if len(matched_subjects) - modifier_count >= 1:
                # Keep only the non-modifier subjects (the species they qualify)
                matched_subjects = [
                    s for s in matched_subjects if s not in _MODIFIER_SUBJECTS
                ]
            else:
                # ALL matched subjects are modifiers (e.g. trapped + fast) —
                # no true species subject present, so no dual-subject conflict
                matched_subjects = []

        # Exempt ``_to_{subject}_particles`` target descriptors in
        # collisional power transfer names. The ``_to_`` connector
        # separates source species from target population — only
        # the source subject counts.
        if "_to_" in name and "particles" in matched_subjects:
            matched_subjects = [s for s in matched_subjects if s != "particles"]

        # Exempt ``state`` in transfer patterns — it describes charge/
        # quantum state of the target species, not a separate subject.
        # E.g. ``…_to_thermal_ion_state`` has state=ion charge state.
        if "_to_" in name and "state" in matched_subjects:
            matched_subjects = [s for s in matched_subjects if s != "state"]

        # Exempt collisional target patterns: ``_with_{subject}``
        # E.g. ``torque_density_due_to_coulomb_collisions_with_ion``
        # has source species (fast_particle) and target species (ion).
        # The ``_with_`` connector marks the collision partner, not a
        # second primary subject.
        with_match = re.search(r"_with_(\w+)$", name)
        if with_match:
            target = with_match.group(1)
            # Remove the collision target from matched subjects
            matched_subjects = [s for s in matched_subjects if s != target]

        # Exempt ratio/comparison/transfer patterns:
        # ``{species1}_to_{species2}_…`` uses ``_to_`` as a conventional
        # connector between source and target species.  This check runs
        # AFTER modifier removal so that compound subjects like
        # ``fast_particle … _to_thermal_ion`` correctly reduce to 2
        # non-modifier subjects (particle, ion) and get exempted.
        if "_to_" in name and len(matched_subjects) == 2:
            matched_subjects = []

        if len(matched_subjects) >= 2:
            issues.append(
                f"audit:multi_subject_check: name contains multiple subjects: "
                f"{matched_subjects}"
            )
    except Exception:
        pass

    return issues


def cocos_specificity_check(
    candidate: dict[str, Any],
    source_cocos_type: str | None = None,
) -> list[str]:
    """If source path has COCOS transformation type, description must mention COCOS.

    Checks that the documentation contains the string ``COCOS`` and a digit
    when the source DD path is tagged with a ``cocos_transformation_type``.
    """
    issues: list[str] = []
    if not source_cocos_type:
        return issues

    doc = candidate.get("documentation") or ""
    desc = candidate.get("description") or ""
    combined = doc + " " + desc

    has_cocos = "COCOS" in combined or "cocos" in combined.lower()
    has_digit = bool(re.search(r"COCOS\s*\d", combined, re.IGNORECASE))

    if not (has_cocos and has_digit):
        issues.append(
            f"audit:cocos_specificity_check: source has cocos_transformation_type="
            f"'{source_cocos_type}' but documentation lacks 'COCOS <digit>'"
        )

    return issues


def generic_noun_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names that are bare generic nouns without physics-specific qualifiers.

    A standard name must be self-describing. Names like ``geometry``, ``data``,
    ``value``, or ``measure`` require source-path context to interpret and
    cannot be used as standalone identifiers across facilities.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip()
    if not name:
        return []
    tokens = [t for t in name.split("_") if t]
    if len(tokens) == 0:
        return []

    if len(tokens) == 1 and tokens[0] in _GENERIC_NOUN_NAMES:
        return [
            f"audit:generic_noun_check: name '{name}' is a bare generic noun; "
            "add a physics-specific qualifier (e.g. 'grid_object_geometry' "
            "instead of 'geometry')"
        ]

    if len(tokens) == 2:
        if tokens[-1] in _GENERIC_NOUN_NAMES and tokens[0] in {
            "raw",
            "input",
            "output",
            "generic",
            "basic",
        }:
            return [
                f"audit:generic_noun_check: name '{name}' uses a generic "
                "qualifier + generic noun; specify the physical quantity"
            ]

    return []


def tautology_check(candidate: dict[str, Any]) -> list[str]:
    """Flag tautological preposition chains like 'position_of_X_position'.

    Detects patterns where the same head noun (position, component, coordinate,
    etc.) appears on both sides of an ``_of_`` connector. These names are
    stylistically awkward and signal a missing physics-specific qualifier.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip()
    if "_of_" not in name:
        return []

    parts = name.split("_of_")
    if len(parts) < 2:
        return []

    issues: list[str] = []
    for i in range(len(parts) - 1):
        left_tokens = parts[i].split("_")
        right_tokens = parts[i + 1].split("_")
        if not left_tokens or not right_tokens:
            continue
        left_head = left_tokens[-1]
        right_head = right_tokens[-1]
        if left_head in _TAUTOLOGY_HEADS and left_head == right_head:
            issues.append(
                f"audit:tautology_check: name '{name}' repeats head noun "
                f"'{left_head}' across '_of_' (tautological chain); "
                f"replace the second occurrence with a specific qualifier"
            )
            break

    return issues


def spectral_suffix_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names ending in spectral-decomposition suffixes.

    Names like ``*_fourier_coefficients``, ``*_fourier_modes``, or
    ``*_harmonics`` place the decomposition type at the end as a generic
    suffix. The preferred pattern is ``mode_<n>_of_<quantity>`` or to name
    the decomposition component explicitly (e.g. ``fourier_amplitude_of_X``).
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    bad_suffixes = (
        "_fourier_coefficients",
        "_fourier_coefficient",
        "_fourier_modes",
        "_fourier_mode",
        "_harmonics",
        "_harmonic_coefficients",
    )
    for suf in bad_suffixes:
        if name.endswith(suf):
            return [
                f"audit:spectral_suffix_check: name '{name}' ends with "
                f"spectral suffix '{suf}'; use a mode-prefixed or "
                "amplitude-of-quantity pattern instead"
            ]
    return []


# Abbreviation prefixes/infixes forbidden by NC-5. A standard name must spell
# the concept out in full — no truncation or contraction.
_FORBIDDEN_ABBREVIATIONS = (
    ("norm_", "normalized_"),
    ("_norm_", "_normalized_"),
    ("perp_", "perpendicular_"),
    ("_perp_", "_perpendicular_"),
    ("par_", "parallel_"),
    ("_par_", "_parallel_"),
    ("temp_", "temperature_"),
    ("_temp_", "_temperature_"),
    ("pos_", "position_"),
    ("_pos_", "_position_"),
    ("max_", "maximum_"),
    ("_max_", "_maximum_"),
    ("min_", "minimum_"),
    ("_min_", "_minimum_"),
    ("avg_", "average_"),
    ("_avg_", "_average_"),
    ("sep_", "separatrix_"),
    ("_sep_", "_separatrix_"),
    ("ec_", "electron_cyclotron_"),
    ("_ec_", "_electron_cyclotron_"),
    ("ic_", "ion_cyclotron_"),
    ("_ic_", "_ion_cyclotron_"),
    ("nbi_", "neutral_beam_injector_"),
    ("_nbi_", "_neutral_beam_injector_"),
    ("lh_", "lower_hybrid_"),
    ("_lh_", "_lower_hybrid_"),
)


def abbreviation_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names that use truncated/abbreviated concept words.

    NC-5 mandates spelled-out concept words. Common offenders that slip past
    the LLM despite the prompt rule: ``norm_``, ``perp_``, ``par_``,
    ``temp_``, ``pos_``, ``max_``, ``min_``, ``sep_``. This audit catches
    them deterministically.

    False-positive safety: ``min``, ``max``, and ``avg`` are only flagged at
    token boundaries (prefix, suffix, or between underscores). Chemical
    element symbols and unit tokens are not affected since this audit only
    inspects the standard-name identifier, not units or documentation.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []
    tokens = set(name.split("_"))
    issues: list[str] = []
    seen_bare: set[str] = set()
    for abbrev, full in _FORBIDDEN_ABBREVIATIONS:
        # Strict token-boundary match: the abbreviation must appear as a
        # whole token in the name, never as a letter subsequence inside
        # another word (e.g. ``ic`` must not match ``ionic``).
        bare = abbrev.strip("_")
        if bare in seen_bare:
            continue
        if bare in tokens:
            issues.append(
                f"audit:abbreviation_check: name '{name}' contains "
                f"abbreviation '{bare}'; spell as '{full.strip('_')}'"
            )
            seen_bare.add(bare)
            break  # one report per name is sufficient
    return issues


# Description tokens that indicate spectral/decomposition semantics. If the
# description claims the quantity is a Fourier coefficient/spectral mode but
# the name carries none of these markers, the name-description pair is
# inconsistent.
_SPECTRAL_DESC_PATTERNS = (
    "fourier coefficient",
    "fourier mode",
    "spectral coefficient",
    "spectral mode",
    "harmonic amplitude",
    "harmonic coefficient",
)
_SPECTRAL_NAME_MARKERS = (
    "mode_",
    "_mode_",
    "_amplitude",
    "fourier",
    "harmonic",
    "spectral",
)


def _build_uk_to_us_mapping() -> dict[str, str]:
    """Build UK→US spelling map from breame's dictionary.

    Falls back to a minimal hardcoded set if breame is unavailable.
    """
    try:
        from breame.spelling import BRITISH_ENGLISH_SPELLINGS

        return {uk.lower(): us.lower() for uk, us in BRITISH_ENGLISH_SPELLINGS.items()}
    except (ImportError, AttributeError):
        return {
            "normalised": "normalized",
            "polarised": "polarized",
            "ionised": "ionized",
            "centre": "center",
            "fibre": "fiber",
            "metre": "meter",
            "colour": "color",
            "behaviour": "behavior",
            "modelled": "modeled",
            "catalogue": "catalog",
        }


_UK_TO_US_SPELLING = _build_uk_to_us_mapping()

_UK_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _UK_TO_US_SPELLING) + r")\b",
    re.IGNORECASE,
)


def american_spelling_check(candidate: dict[str, Any]) -> list[str]:
    """Flag British spellings in the name or any prose field.

    The ISN catalog uses American spelling throughout (``normalized`` not
    ``normalised``, ``polarized`` not ``polarised``). Names or
    descriptions containing British variants violate NC-17 and are
    quarantined so they can be regenerated with the canonical spelling.
    """
    issues: list[str] = []
    name = str(candidate.get("id") or candidate.get("name") or "").strip()
    fields: list[tuple[str, str]] = []
    if name:
        fields.append(("name", name.replace("_", " ")))
    for fld in ("description", "documentation", "validity_domain"):
        val = candidate.get(fld)
        if isinstance(val, str) and val.strip():
            fields.append((fld, val))
    constraints = candidate.get("constraints")
    if isinstance(constraints, list):
        for i, c in enumerate(constraints):
            if isinstance(c, str) and c.strip():
                fields.append((f"constraints[{i}]", c))

    seen: set[tuple[str, str]] = set()
    for field_name, text in fields:
        for m in _UK_WORD_RE.finditer(text):
            uk = m.group(0).lower()
            us = _UK_TO_US_SPELLING[uk]
            key = (field_name, uk)
            if key in seen:
                continue
            seen.add(key)
            issues.append(
                f"audit:american_spelling_check: field '{field_name}' "
                f"contains British spelling '{m.group(0)}'; use '{us}' "
                f"(ISN catalog follows American convention — NC-17)"
            )
    return issues


def name_description_consistency_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names whose description asserts a different concept than the name.

    Specifically detects the case where the description describes a
    Fourier/spectral decomposition but the name is simply the underlying
    quantity (e.g. ``normal_magnetic_field`` described as
    "Fourier coefficients of the normal component ..."). Either the name
    must mark the decomposition explicitly, or the description must be
    rewritten to describe the underlying quantity.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    description = str(candidate.get("description") or "").lower()
    if not name or not description:
        return []
    # Only flag when description strongly implies a decomposition
    if not any(pat in description for pat in _SPECTRAL_DESC_PATTERNS):
        return []
    # But the name carries no decomposition marker
    if any(marker in name for marker in _SPECTRAL_NAME_MARKERS):
        return []
    return [
        f"audit:name_description_consistency_check: description of '{name}' "
        "claims a spectral/Fourier decomposition but the name encodes only "
        "the underlying quantity; either add a decomposition marker to the "
        "name or rewrite the description"
    ]


_REPRESENTATION_NAME_RE = re.compile(
    r"(?:"
    # Generic coefficient / basis / spline / ggd / fourier suffixes
    r"_(?:coefficients|ggd|basis|spline|fourier_modes|harmonics_coefficients)"
    # GGD-coefficient variants
    r"|_ggd_coefficients"
    r"|_coefficient_on_ggd"
    # Interpolation coefficient variants (singular/plural, optional _on_ggd)
    r"|_interpolation_coefficients?(?:_on_ggd)?"
    # Finite-element coefficient variants (base + real/imaginary split)
    r"|_finite_element(?:_interpolation)?_coefficients?"
    r"|_finite_element_coefficients_(?:real|imaginary)_part"
    r")"
    r"(?:_|$)"
)

#: Heuristic regex: bare ``_on_ggd$`` suffix — flagged only when the
#: DD source path carries a GGD marker (``/ggd/`` or ``/grids_ggd/``).
#: Avoids false positives on legitimate ``_on_<other>`` suffixes.
_ON_GGD_SUFFIX_RE = re.compile(r"_on_ggd$")


def representation_artifact_check(
    candidate: dict[str, Any], source_path: str | None = None
) -> list[str]:
    """Flag names whose final tokens describe a basis-function representation.

    Names ending in ``_coefficients``, ``_ggd_coefficients``,
    ``_coefficient_on_ggd``, ``_interpolation_coefficient(s)(_on_ggd)``,
    ``_finite_element_coefficients_(real|imaginary)_part``, ``_basis``,
    ``_spline``, ``_fourier_modes`` etc. are storage representations of an
    underlying physical field, not standalone physics concepts.  They should
    be quarantined and the corresponding source path skipped at classification.

    A bare ``_on_ggd$`` suffix is flagged only when *source_path* carries a
    GGD marker (``/ggd/`` or ``/grids_ggd/``) — this avoids false positives
    on legitimate ``_on_<other>`` compound terms.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []
    if _REPRESENTATION_NAME_RE.search(name):
        return [
            f"audit:representation_artifact_check: name '{name}' encodes a "
            "basis-function or grid representation; the underlying physical "
            "quantity already has a standard name on the sibling path — "
            "this path should have been classified as skip"
        ]
    # Heuristic _on_ggd$ — only fire when DD source path carries a GGD marker
    if (
        _ON_GGD_SUFFIX_RE.search(name)
        and source_path
        and ("/ggd/" in source_path or "/grids_ggd/" in source_path)
    ):
        return [
            f"audit:representation_artifact_check: name '{name}' ends in "
            "'_on_ggd' and its DD source path contains a GGD marker — this "
            "is a grid-representation storage node, not a physics concept; "
            "the source path should have been classified as skip"
        ]
    return []


# Verbs/processes that are mis-used with the ``due_to_<X>`` template.  These
# fall into two classes:
#  - ``during_X`` would be more accurate (the X is a temporal event, not a
#    physical cause): disruption, breakdown, ramp_up, ramp_down, flat_top
#  - ``due_to_X_<verb>`` is required (X is an adjective, not a process noun):
#    ohmic → ohmic_dissipation/ohmic_heating
_DURE_TO_TEMPORAL = {
    "disruption",
    "breakdown",
    "ramp_up",
    "ramp_down",
    "flat_top",
    "shutdown",
    "startup",
}
_DURE_TO_ADJECTIVE = {
    "ohmic": "ohmic_dissipation or ohmic_heating",
    "neutral_beam": "neutral_beam_injection",
    "wave": "wave_heating",
    "halo": "halo_currents",
    "runaway": "runaway_electrons",
    "fast_ion": "fast_ions",
    "alpha": "alpha_particle_heating",
    "resistive": "resistive_dissipation or resistive_diffusion",
    "non_inductive": "non_inductive_drive or non_inductive_current_drive",
    "inductive": "inductive_drive",
    "turbulent": "turbulent_transport",
    "neoclassical": "neoclassical_transport",
    "anomalous": "anomalous_transport",
    "thermal": "thermal_fusion",
}


def causal_due_to_check(candidate: dict[str, Any]) -> list[str]:
    """Flag misuse of the ``due_to_<process>`` grammatical template.

    The ``due_to_`` template asserts a causal physical process.  It is wrong
    when the trailing token is a temporal event (use ``during_<event>``) or
    a bare adjective (use ``due_to_<adjective>_<process_noun>``).
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if "_due_to_" not in name:
        return []
    issues: list[str] = []
    suffix = name.split("_due_to_", 1)[1]

    isn_processes = _isn_process_tokens()
    # Exempt any ISN-registered process token — these are canonically valid
    # targets of the ``due_to_`` template regardless of English part-of-speech.
    if suffix in isn_processes or suffix.split("_", 1)[0] in isn_processes:
        return []

    for event in _DURE_TO_TEMPORAL:
        if event in isn_processes:
            continue
        if suffix == event or suffix.startswith(event + "_"):
            issues.append(
                f"audit:causal_due_to_check: name '{name}' uses 'due_to_{event}' — "
                f"'{event}' is a temporal event, not a physical process; use "
                f"'during_{event}' instead"
            )
            break
    for adj, suggestion in _DURE_TO_ADJECTIVE.items():
        if adj in isn_processes:
            continue
        if suffix == adj or suffix == adj + "_":
            issues.append(
                f"audit:causal_due_to_check: name '{name}' uses 'due_to_{adj}' — "
                f"'{adj}' is an adjective, not a process noun; "
                f"suggested_fix=due_to_{suggestion}"
            )
            break
    return issues


FIELD_DEVICE_WHITELIST = {
    "vacuum_toroidal_field_function",
    "vacuum_toroidal_field_flux_function",
    "resistance_of_poloidal_field_coil",
}

_FIELD_QUALIFIERS = (
    "magnetic",
    "electric",
    "radiation",
    "displacement",
    "velocity",
    "temperature",
    "pressure",
    "density",
    "flow",
    "vector",
)


def implicit_field_check(candidate: dict[str, Any]) -> list[str]:
    """Flag bare ``_field`` token without a qualifier (e.g. ``vacuum_toroidal_field``).

    The IMAS catalog and ISN convention require explicit ``magnetic_field``,
    ``electric_field``, etc. — never the colloquial bare ``field``.
    """
    name = candidate.get("id", "")
    if not name:
        return []
    # Device whitelist: names referring to physical devices (field coils,
    # toroidal-field machines) use "field" as a device qualifier, not a
    # physics-field concept.
    if name in FIELD_DEVICE_WHITELIST or "_field_coil" in name:
        return []
    # Constraint selectors (use_exact_*) reference the field they constrain
    # and legitimately use bare "_field" in the constraint target name.
    # Most are filtered by the extract-deny gate (W19A), but any that
    # survive should not be penalised for the constraint target's phrasing.
    if name.startswith("use_exact_"):
        return []
    # ``field_of_view`` is an optics term (viewing cone), not a physics
    # field.  Skip names containing this compound.
    if "field_of_view" in name:
        return []
    tokens = name.split("_")
    issues: list[str] = []
    for i, tok in enumerate(tokens):
        if tok != "field":
            continue
        prev = tokens[i - 1] if i > 0 else ""
        if prev not in _FIELD_QUALIFIERS:
            issues.append(
                f"audit:implicit_field_check: name '{name}' contains bare '_field' "
                f"after '{prev or '<start>'}'; qualify as 'magnetic_field', "
                f"'electric_field', etc."
            )
            break
    return issues


def density_unit_consistency_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``_density`` suffix when the declared unit lacks an inverse-length factor.

    A "density" in physics is a quantity per unit volume / area / length. The
    declared unit must therefore include ``m^-3`` (volumetric), ``m^-2``
    (areal), or ``m^-1`` (linear). Names ending in ``_density`` whose unit is
    a bare extensive quantity (e.g. ``kg.m.s^-1`` for momentum) are misnamed —
    drop ``_density`` or rename to reflect the actual quantity.

    Examples flagged:
    - ``toroidal_angular_momentum_density`` with unit ``kg.m.s^-1`` (linear
      momentum, not density).
    - ``electron_pressure_density`` with unit ``Pa`` (pressure already has
      energy-per-volume dimensions; ``_density`` is redundant).
    """
    name = candidate.get("id", "")
    unit = (candidate.get("unit") or "").strip()
    if not name or not unit:
        return []
    if "_density" not in name and not name.endswith("_density"):
        return []
    # Skip constraint-metadata suffixes: names like
    # ``toroidal_current_density_constraint_measurement_time`` carry
    # ``_density`` in the base quantity, not in the metadata suffix.
    # The unit refers to the suffix semantics (e.g. ``s`` for time).
    _CONSTRAINT_SUFFIXES = (
        "_constraint_measurement_time",
        "_constraint_weight",
        "_constraint_reconstructed",
        "_constraint_measured",
        "_constraint_time_measurement",
        "_constraint_position",
        "_constraint",
    )
    for suffix in _CONSTRAINT_SUFFIXES:
        if name.endswith(suffix):
            return []
    # Meta-prefix patterns: ``measurement_time_of_X_density_constraint``,
    # ``time_of_X_density_*`` describe a meta property of the constraint,
    # not the density itself. Unit refers to the meta property (s, weight).
    _META_PREFIXES = (
        "measurement_time_of_",
        "time_of_",
        "position_of_",
        "weight_of_",
        "exact_flag_of_",
    )
    for prefix in _META_PREFIXES:
        if name.startswith(prefix):
            return []
    # Acceptable density unit factors: any negative power of m.
    if "m^-" in unit or "m**-" in unit:
        return []
    # Special case: dimensionless density (rare but valid for fractions/probabilities)
    # is not flagged — declared unit "1" is allowed.
    if unit in {"1", ""}:
        return []
    # Spectral density: "density" in spectral context means per-frequency or
    # per-wavenumber, not per-volume. The unit refers to the extensive quantity.
    if "_spectrum" in name or "_spectral" in name:
        return []
    return [
        f"audit:density_unit_consistency_check: name '{name}' ends with "
        f"'_density' but declared unit '{unit}' has no inverse-length factor "
        f"(expected m^-1, m^-2, or m^-3). Either drop '_density' or correct "
        f"the unit."
    ]


def vector_field_component_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``_coordinate_of_<vector_field>`` and recommend short-form ``<axis>_<vector_field>``.

    The cylindrical-coordinate vocabulary (``radial``, ``vertical``, ``toroidal``,
    ``major_radius``, ``z_coordinate``) describes a *point in space*. When the
    target ``X`` is itself a vector field (surface normal, magnetic field vector,
    velocity vector), the correct usage is ``<axis>_<X>`` — you
    project the vector onto an axis, you do not extract a coordinate.

    Caught from equilibrium iteration:
    ``vertical_coordinate_of_surface_normal`` should be
    ``vertical_surface_normal``.
    """
    name = candidate.get("id", "")
    if not name:
        return []
    vector_field_tails = (
        "surface_normal",
        "magnetic_field_vector",
        "electric_field_vector",
        "velocity_vector",
        "current_density_vector",
        "poynting_vector",
    )
    issues: list[str] = []
    for axis in ("radial", "vertical", "toroidal"):
        bad = f"{axis}_coordinate_of_"
        if bad not in name:
            continue
        for tail in vector_field_tails:
            if name.endswith(bad + tail) or (bad + tail + "_") in name:
                issues.append(
                    f"audit:vector_field_component_check: name '{name}' applies "
                    f"'_coordinate_of_' to vector field '{tail}'; rename to "
                    f"'{axis}_{tail}' (vectors have components, "
                    f"points have coordinates)."
                )
    return issues


def position_coordinate_check(candidate: dict[str, Any]) -> list[str]:
    """Flag colloquial ``_position_of_X`` and recommend canonical coordinate vocabulary.

    Names like ``vertical_position_of_antenna``, ``radial_position_of_X`` and
    ``toroidal_position_of_X`` should use the canonical coordinate vocabulary
    that aligns with cylindrical (R, φ, Z) tokamak conventions:

    - ``radial_position_of_X`` → ``major_radius_of_X``.
    - ``toroidal_position_of_X`` → ``toroidal_angle_of_X``.
    - ``vertical_position_of_X`` → ``vertical_coordinate_of_X`` or
      ``z_coordinate_of_X``.

    The check fires unconditionally on the colloquial name pattern (it does not
    require a confirming description) because the canonical vocabulary already
    covers every R/Z/φ-coordinate use case in IMAS. Without this check,
    both forms would leak into the catalog as unintended synonyms.
    """
    name = candidate.get("id", "")
    if not name:
        return []
    issues: list[str] = []
    patterns = (
        ("radial_position_of_", "major_radius_of_<X>"),
        ("toroidal_position_of_", "toroidal_angle_of_<X>"),
        (
            "vertical_position_of_",
            "vertical_coordinate_of_<X> or z_coordinate_of_<X>",
        ),
    )
    for prefix, suggested in patterns:
        if prefix in name:
            issues.append(
                f"audit:position_coordinate_check: name '{name}' uses "
                f"colloquial '{prefix.rstrip('_')}_' form; rename to "
                f"{suggested} (cylindrical-coordinate canonical vocabulary)."
            )
    return issues


def segment_order_check(candidate: dict[str, Any]) -> list[str]:
    """Flag Component tokens appearing as a trailing suffix instead of a prefix.

    ISN grammar places Component segments (``toroidal``, ``poloidal``, ``radial``,
    ``parallel``, ``perpendicular``, ``vertical``, ``diamagnetic``) either as a
    leading prefix or via the ``<axis>_<quantity>`` short form. A
    trailing ``_<component>`` suffix after the quantity reverses segment order.

    Caught from transport iteration:
    ``ion_rotation_frequency_toroidal`` → ``toroidal_ion_rotation_frequency``.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []
    component_suffixes = (
        "toroidal",
        "poloidal",
        "radial",
        "parallel",
        "perpendicular",
        "vertical",
        "diamagnetic",
    )
    issues: list[str] = []
    for comp in component_suffixes:
        if not name.endswith(f"_{comp}"):
            continue
        stem = name[: -(len(comp) + 1)]
        # Only flag when the stem is a substantive quantity (has at least two
        # tokens AND contains no other component token as a prefix already).
        stem_tokens = stem.split("_")
        if len(stem_tokens) < 2:
            continue
        if stem_tokens[0] in component_suffixes:
            continue
        issues.append(
            f"audit:segment_order_check: name '{name}' ends with component "
            f"token '_{comp}'; Component segments must precede the Subject or "
            f"use '<axis>_<quantity>' short form. Rename to "
            f"'{comp}_{stem}'."
        )
    return issues


_AGGREGATOR_SUFFIXES = (
    "volume_averaged",
    "flux_surface_averaged",
    "surface_averaged",
    "line_averaged",
    "density_averaged",
    "time_averaged",
)


def aggregator_order_check(candidate: dict[str, Any]) -> list[str]:
    """Flag Aggregator tokens appearing as a trailing suffix instead of a prefix.

    ISN grammar places Aggregator segments (``volume_averaged``,
    ``flux_surface_averaged``, ``line_averaged``, ``time_averaged``, etc.) as a
    prefix before the physical base, not as a trailing suffix after it.

    Caught from transport iteration:
    ``ion_temperature_volume_averaged`` → ``volume_averaged_ion_temperature``
    (matches pattern already used by ``volume_averaged_electron_temperature``).
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []
    issues: list[str] = []
    for agg in _AGGREGATOR_SUFFIXES:
        suffix = f"_{agg}"
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        # Skip if the aggregator is immediately after another aggregator prefix
        # (defensive — unlikely in practice).
        if not stem:
            continue
        issues.append(
            f"audit:aggregator_order_check: name '{name}' ends with aggregator "
            f"token '_{agg}'; Aggregator segments must precede the Subject/Base. "
            f"Rename to '{agg}_{stem}'."
        )
    return issues


_DIAMAGNETIC_COMPONENT_PATTERN = "diamagnetic_component_of_"


def diamagnetic_component_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``diamagnetic_component_of_*`` and ``diamagnetic_<X>`` used as projection — diamagnetic is a drift, not a component.

    ``diamagnetic`` labels a specific drift velocity ``v_dia = B × ∇p / (qnB²)``,
    not a spatial projection axis like ``toroidal`` or ``poloidal``. Using
    ``diamagnetic_component_of_<X>`` or ``diamagnetic_<X>`` as a projection therefore either:

    - Makes no physical sense for scalars and projected fields (e.g.
      ``diamagnetic_electric_field``), or
    - Is redundant for a drift velocity (``v_dia`` IS the diamagnetic drift,
      not a component of something else).

    Canonical constructions:
    - For the drift velocity itself → ``diamagnetic_drift_velocity`` (no
      projection).
    - For a flux driven by the diamagnetic drift → ``diamagnetic_<base>`` or
      ``<base>_due_to_diamagnetic_drift``.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if _DIAMAGNETIC_COMPONENT_PATTERN not in name:
        return []
    tail = name.split(_DIAMAGNETIC_COMPONENT_PATTERN, 1)[1]
    return [
        f"audit:diamagnetic_component_check: name '{name}' uses "
        f"'diamagnetic_component_of_{tail}' — 'diamagnetic' labels a drift "
        f"(v_dia = B × ∇p / (qnB²)), not a spatial projection axis. Use "
        f"'diamagnetic_drift_velocity' for the drift itself, or "
        f"'<base>_due_to_diamagnetic_drift' for a flux driven by it."
    ]


def amplitude_of_prefix_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``amplitude_of_<X>`` / ``phase_of_<X>`` / ``magnitude_of_<X>`` prefix forms.

    For the amplitude, phase, and magnitude of a quantity ``<X>``, the
    canonical ISN form is the noun-suffix construction ``<X>_amplitude``,
    ``<X>_phase``, ``<X>_magnitude``. The prefix form ``amplitude_of_<X>``
    and siblings break the grammar when ``<X>`` contains a ``_of_`` or
    ``component_of_`` chain (e.g. ``amplitude_of_parallel_*``
    fails the vocabulary consistency check because ``amplitude_of_parallel``
    is not a Component token). Use the noun-suffix form consistently.

    ``real_part_of_`` and ``imaginary_part_of_`` are NOT flagged here —
    ISN grammar parses them correctly as ``transformation=real_part`` /
    ``transformation=imaginary_part`` prefix operators, producing proper
    HAS_PARENT derivation edges to the parent complex quantity.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    prefixes = (
        "amplitude_of_",
        "phase_of_",
        "magnitude_of_",
        "modulus_of_",
    )
    for prefix in prefixes:
        if name.startswith(prefix):
            noun = prefix[:-4]  # strip trailing "_of_"
            tail = name[len(prefix) :]
            return [
                f"audit:amplitude_of_prefix_check: name '{name}' uses "
                f"'{prefix}<X>' prefix — canonical ISN form is the "
                f"noun-suffix '{tail}_{noun}'. Prefix forms break grammar "
                f"when <X> contains '_of_' chains."
            ]
    return []


def mode_number_suffix_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``_per_<axis>_mode_number`` — canonical suffix drops ``_number``.

    The spectral qualifier is ``_per_toroidal_mode`` or ``_per_poloidal_mode``;
    the ``_number`` token is redundant because the mode index is implicit.
    Within a batch the spelling must be consistent: never emit both
    ``_per_toroidal_mode`` and ``_per_toroidal_mode_number``.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    bad_suffixes = (
        "_per_toroidal_mode_number",
        "_per_poloidal_mode_number",
    )
    for suffix in bad_suffixes:
        if name.endswith(suffix):
            canonical = suffix.rsplit("_number", 1)[0]
            return [
                f"audit:mode_number_suffix_check: name '{name}' ends with "
                f"'{suffix}' — canonical suffix is '{canonical}' (drop "
                f"'_number'; the mode index is implicit)."
            ]
    return []


def cumulative_prefix_check(candidate: dict[str, Any]) -> list[str]:
    """Flag spatial-integration misnomers like ``cumulative_``/``integrated_``/``running_``.

    DD leaf names ending in ``_inside`` (e.g. ``power_inside_thermal_n_tor``,
    ``current_tor_inside``) denote a quantity integrated inside the enclosing
    flux surface. The canonical ISN suffix is ``_inside_flux_surface`` placed
    directly after the quantity; ``cumulative_`` / ``integrated_`` / ``running_``
    lose the geometric meaning and are not part of the ISN grammar vocabulary.

    NOTE: ``accumulated_`` is NOT flagged here — DD gas-injection and coil-charge
    paths use ``accumulated_`` to denote a running total over time, which is a
    distinct physical concept from spatial flux-surface integration. Time
    accumulation is handled via the ISN process / transformation vocabulary.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    # Meta-descriptor prefixes refer to a property OF a named quantity, not to a
    # cumulative/integrated quantity itself. The "integrated_" inside is part of
    # the inner quantity's name (e.g. "line_integrated_electron_density"), so this
    # audit must not fire.
    _META_PREFIXES = (
        "measurement_time_of_",
        "time_of_",
        "position_of_",
        "weight_of_",
        "exact_flag_of_",
    )
    # Also skip meta-flag suffixes that wrap a quantity name (e.g. *_exact_flag,
    # *_iteration_count, *_convergence_count, *_constraint_weight*).
    _META_SUFFIXES = (
        "_exact_flag",
        "_iteration_count",
        "_convergence_count",
        "_constraint_weight",
    )
    if any(name.startswith(p) for p in _META_PREFIXES):
        return []
    if any(s in name for s in _META_SUFFIXES):
        return []
    bad_tokens = ("cumulative_", "integrated_", "running_")
    tokens = name.split("_")
    for bad in bad_tokens:
        stem = bad.rstrip("_")
        if stem in tokens:
            return [
                f"audit:cumulative_prefix_check: name '{name}' contains "
                f"'{stem}_' — for DD `_inside`-style quantities use the "
                f"suffix `_inside_flux_surface` placed after the quantity "
                f"instead of prefixing with `{stem}_`."
            ]
    return []


# ---- Regex for ad-hoc ratio patterns (C.7) --------------------------------
_ADHOC_RATIO_RE = re.compile(r"^(.+?)_to_(.+?)_ratio$")


def pulse_schedule_reference_check(
    candidate: dict[str, Any],
    source_path: str | None = None,
) -> list[str]:
    """Flag reference/reference-waveform sentinels from pulse_schedule IDS.

    Controller reference targets live under ``pulse_schedule/.../reference``
    or ``pulse_schedule/.../reference_waveform`` and are not physics standard
    name candidates.  Severity: critical.

    Triggers only when the NAME itself encodes a controller reference target —
    i.e. ends with ``_reference`` or ``_reference_waveform``.  A source_path
    under ``pulse_schedule/.../reference`` alone is NOT sufficient, because a
    physics standard name like ``plasma_current`` may legitimately have a
    pulse_schedule controller-reference source attached (the setpoint refers
    to the same physical quantity).  Only the name-suffix variant indicates
    that the SN itself is a control-layer sentinel rather than a physics
    quantity.  Severity: critical.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    issues: list[str] = []

    if name.endswith("_reference") or name.endswith("_reference_waveform"):
        issues.append(
            f"audit:pulse_schedule_reference_check: name '{name}' ends with "
            f"a reference/reference_waveform suffix — likely a controller "
            f"reference target, not a physics SN candidate; severity=critical"
        )

    return issues


def ratio_binary_operator_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ad-hoc ratio naming patterns; enforce ``ratio_of_<A>_to_<B>`` form.

    The ISN canonical form for ratios is ``ratio_of_<A>_to_<B>`` (ISN-10).
    Ad-hoc patterns like ``<A>_to_<B>_density_ratio`` or ``<A>_to_<B>_ratio``
    are rejected with a suggested rewrite.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()

    # Accept canonical form
    if name.startswith("ratio_of_") and "_to_" in name:
        return []

    # Detect ad-hoc ``<A>_to_<B>_ratio`` or ``<A>_to_<B>_<noun>_ratio``
    match = _ADHOC_RATIO_RE.match(name)
    if match:
        a_part = match.group(1)
        b_part = match.group(2)
        suggested = f"ratio_of_{a_part}_to_{b_part}"
        return [
            f"audit:ratio_binary_operator_check: name '{name}' uses ad-hoc "
            f"ratio form; canonical ISN form is 'ratio_of_<A>_to_<B>'; "
            f"suggested_fix={suggested}"
        ]

    return []


def instrument_owned_observable_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``<observable>_of_<instrument>`` anti-pattern (NC-30).

    Radiance, emissivity, temperature, brightness, and similar physical
    observables are properties of the emitting plasma or observed surface,
    NOT of the diagnostic detector that records them. A
    ``radiance_of_visible_camera`` is physically meaningless — the camera has
    responsivity, gain, and filters, but radiance belongs to the emitting
    column. Allow instrument-property nouns (responsivity, throughput,
    sensitivity, filter_bandwidth, field_of_view, integration_time, gain) but
    reject physical observables coupled to instrument tokens.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()

    _INSTRUMENTS = {
        "infrared_camera",
        "visible_camera",
        "xray_camera",
        "hard_xray_camera",
        "soft_xray_camera",
        "camera",
        "spectrometer",
        "bolometer",
        "interferometer",
        "reflectometer",
        "polarimeter",
        "thomson_scattering",
        "langmuir_probe",
        "mirnov_coil",
        "rogowski_coil",
        "flux_loop",
        "bpol_probe",
        "ece_receiver",
        "neutron_detector",
        "mse_diagnostic",
        "cxrs_diagnostic",
        "beam_emission",
    }
    _OBSERVABLES = {
        "radiance",
        "emissivity",
        "brightness",
        "temperature",
        "density",
        "pressure",
        "intensity",
        "spectral_intensity",
        "photon_flux",
        "irradiance",
        "luminance",
        "velocity",
    }
    for obs in _OBSERVABLES:
        for instr in _INSTRUMENTS:
            needle = f"{obs}_of_{instr}"
            if needle in name:
                return [
                    f"audit:instrument_owned_observable_check: name '{name}' "
                    f"attaches physical observable '{obs}' to instrument "
                    f"'{instr}'. Observables belong to the emitting source, "
                    f"not to the detector. Use "
                    f"'{obs}_observed_by_{instr}' or "
                    f"'<source>_{obs}_along_{instr}_line_of_sight'."
                ]
    return []


def profile_suffix_check(candidate: dict[str, Any]) -> list[str]:
    """Flag redundant ``_profile`` suffix on scalar quantities (NC-31).

    Every standard name denotes the scalar value at one coordinate; a
    "profile" is just the same quantity sampled on an axis. The suffix is
    redundant and should be stripped. Genuine profile-shape descriptors
    (peakedness, half_width, aspect_ratio) are permitted.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name.endswith("_profile"):
        return []
    # Exempt shape descriptors (where _profile modifies a shape noun)
    _SHAPE_PREFIXES_OK = (
        "peakedness_profile",
        "half_width_profile",
        "aspect_ratio_profile",
    )
    if any(name.endswith(tail) for tail in _SHAPE_PREFIXES_OK):
        return []
    suggested = name[: -len("_profile")]
    return [
        f"audit:profile_suffix_check: name '{name}' ends with redundant "
        f"'_profile' suffix. All spatial standard names are profiles by "
        f"convention — strip the suffix. suggested_fix={suggested}"
    ]


# Compound-subject pairs that look like token repetition but are legitimate
# fusion reactions or species identifiers.
_COMPOUND_SUBJECT_PAIRS = frozenset(
    {
        "deuterium_deuterium",
        "deuterium_tritium",
        "tritium_tritium",
        # Legitimate adjacent-duplicate compounds in fusion plasma physics
        "beam_beam",  # beam-beam reactions (NBI × NBI fusion)
    }
)

# Grammar connectives — these are structural glue, not content tokens,
# and must never trigger the repeated-token audit.
_GRAMMAR_CONNECTIVES = frozenset(
    {"of", "at", "per", "due", "to", "in", "by", "for", "along", "from"}
)


def adjacent_duplicate_token_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names where a token appears immediately adjacent to itself.

    Catches LLM composition bugs like ``toroidal_magnetic_magnetic_field_probe``
    or ``poloidal_magnetic_magnetic_field_probe_constraint_weight`` where the
    composer concatenates a subject and physical_base that share the same
    terminal/leading token, producing a tautological doubled word.

    This is narrower than :func:`repeated_token_check` — it only flags
    immediately-adjacent duplicates (``magnetic_magnetic``), never
    non-adjacent repetitions (``magnetic_field_at_magnetic_axis``).

    Legitimate adjacent compounds (``deuterium_deuterium``, ``beam_beam``)
    are collapsed via :data:`_COMPOUND_SUBJECT_PAIRS` before the scan.

    Severity: critical — quarantines the candidate.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []

    working = name
    for pair in _COMPOUND_SUBJECT_PAIRS:
        working = working.replace(pair, f"_compound_{pair.replace('_', '')}_")

    tokens = working.split("_")
    for i in range(1, len(tokens)):
        a, b = tokens[i - 1], tokens[i]
        if not a or not b:
            continue
        if a == b and a not in _GRAMMAR_CONNECTIVES and not a.startswith("compound"):
            return [
                f"audit:adjacent_duplicate_token_check: name '{name}' contains "
                f"adjacent duplicate token '{a}_{b}' — likely LLM concatenation "
                f"bug (e.g. subject ending in '{a}' + base starting with '{a}')"
            ]
    return []


def repeated_token_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names where a content token appears more than once.

    Splits the name on ``_`` and checks for duplicated tokens after
    filtering out grammar connectives (``of``, ``at``, ``per``, …).
    Compound-subject tokens like ``deuterium_deuterium`` are treated
    as atomic units and collapsed before the duplicate scan.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []

    # Collapse known compound-subject pairs into single placeholder tokens
    working = name
    for pair in _COMPOUND_SUBJECT_PAIRS:
        working = working.replace(pair, f"_compound_{pair.replace('_', '')}_")

    tokens = [t for t in working.split("_") if t and t not in _GRAMMAR_CONNECTIVES]
    # Also skip the placeholder tokens we inserted
    tokens = [t for t in tokens if not t.startswith("compound")]

    seen: set[str] = set()
    for tok in tokens:
        if tok in seen:
            return [
                f"audit:repeated_token_check: name '{name}' contains "
                f"duplicated content token '{tok}' — likely tautology"
            ]
        seen.add(tok)
    return []


def length_soft_cap_check(candidate: dict[str, Any]) -> list[str]:
    """Warn when a name exceeds 70 characters.

    The current corpus median is ~39 characters; names beyond 70 are
    usually over-qualified and should be simplified. This is advisory
    only — no quarantine.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip()
    n = len(name)
    if n > 70:
        return [
            f"audit:length_soft_cap_check: name length {n} chars exceeds "
            f"soft cap of 70 — consider simplification"
        ]
    return []


# Instrument tokens that indicate a diagnostic device
_STOKES_INSTRUMENTS = frozenset(
    {"polarimeter", "interferometer", "radiometer", "spectrometer"}
)

# Observable tokens that should not be bound to an instrument
_STOKES_OBSERVABLES = frozenset(
    {"stokes_vector", "stokes_parameter", "degree_of_polarization"}
)


def instrument_stokes_bind_check(candidate: dict[str, Any]) -> list[str]:
    """Flag Stokes observables coupled to an instrument token (NC-30 ext).

    Names like ``stokes_vector_of_polarimeter`` bind a measurement-frame
    quantity to a specific instrument. Prefer ``<quantity>_at_<locus>``
    or ``<quantity>_reconstructed`` instead.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []

    has_instrument = any(instr in name for instr in _STOKES_INSTRUMENTS)
    has_observable = any(obs in name for obs in _STOKES_OBSERVABLES)

    if has_instrument and has_observable:
        return [
            f"audit:instrument_stokes_bind_check: name '{name}' — "
            f"instrument-bound Stokes parameter — consider "
            f"<quantity>_at_<locus> form (NC-30 pattern)"
        ]
    return []


# Redundant position tokens: the ISN grammar has 'wall', not 'wall_surface'.
_REDUNDANT_POSITION_MAP: dict[str, str] = {
    "wall_surface": "wall",
}


def position_redundancy_check(candidate: dict[str, Any]) -> list[str]:
    """Flag redundant position tokens like ``at_wall_surface`` → ``at_wall``.

    The ISN Position vocabulary has ``wall`` but NOT ``wall_surface``.
    The ``_surface`` suffix is physically redundant — a wall IS a surface.
    Catches both ``_at_wall_surface`` and ``_on_wall_surface`` patterns.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []

    issues: list[str] = []
    for redundant, canonical in _REDUNDANT_POSITION_MAP.items():
        for prep in ("_at_", "_on_"):
            bad = f"{prep}{redundant}"
            good = f"{prep}{canonical}"
            if bad in name or name.endswith(f"{prep[1:]}{redundant}"):
                suggested = name.replace(bad, good)
                issues.append(
                    f"audit:position_redundancy_check: name '{name}' uses "
                    f"'{redundant}' as a position token, but ISN vocabulary "
                    f"has '{canonical}'. Rename to '{suggested}'."
                )
    return issues


def process_qualifier_check(candidate: dict[str, Any]) -> list[str]:
    """Flag over-qualified process tokens after ``due_to_``.

    Process tokens in ``due_to_<process>`` must be bare vocabulary entries.
    Appending spatial qualifiers (``_at_X``, ``_in_X``, ``_on_X``) to the
    process token produces an invalid compound that fails grammar validation.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []

    due_to_idx = name.find("_due_to_")
    if due_to_idx < 0:
        return []

    process_tail = name[due_to_idx + len("_due_to_") :]
    if not process_tail:
        return []

    # Check for spatial qualifiers embedded in the process token
    spatial_patterns = [
        ("_at_", "at"),
        ("_in_", "in"),
        ("_on_", "on"),
        ("_for_", "for"),
    ]
    issues: list[str] = []
    for pattern, prep in spatial_patterns:
        if pattern in process_tail:
            bare_process = process_tail.split(pattern, 1)[0]
            qualifier = process_tail.split(pattern, 1)[1]
            issues.append(
                f"audit:process_qualifier_check: name '{name}' appends "
                f"'{prep}_{qualifier}' to process token '{bare_process}' "
                f"after 'due_to_'. Process tokens must be bare vocabulary "
                f"entries — move the qualifier to the subject prefix or "
                f"Region segment."
            )
    return issues


def preposition_physical_base_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names whose ISN parse produces a ``physical_base`` starting with a preposition.

    When ISN grammar parses a name like ``normalized_of_particle_temperature``,
    it places ``normalized`` in the ``transformation`` slot and dumps
    ``of_particle_temperature`` into ``physical_base``.  A ``physical_base``
    starting with ``of_``, ``at_``, or ``due_to_`` is *always* a grammar
    defect — the preposition is a scope connector that leaked into the base
    because the name was mal-formed.  The correct form drops the preposition
    (e.g. ``normalized_particle_temperature``).

    Severity: critical — quarantines the candidate.
    """
    name = (candidate.get("id") or candidate.get("name") or "").strip()
    if not name:
        return []

    try:
        from imas_standard_names.grammar import parse_standard_name

        parsed = parse_standard_name(name)
    except Exception:
        # Parse failure is handled by other checks
        return []

    pb = getattr(parsed, "physical_base", None) or ""
    _BAD_PREFIXES = ("of_", "at_", "due_to_")
    for prefix in _BAD_PREFIXES:
        if pb.startswith(prefix):
            clean = pb[len(prefix) :]
            return [
                f"audit:preposition_physical_base_check: ISN parse of "
                f"'{name}' yields physical_base='{pb}' — a base must "
                f"never start with a preposition. The correct base is "
                f"'{clean}' (drop the '{prefix}' connector)."
            ]
    return []


# =============================================================================
# Canonical locus / field-at-region preposition checks
# =============================================================================

# Forbidden locus-token synonyms → canonical replacement.
# Canonical for the last-closed-flux-surface concept is ``plasma_boundary``
# (descriptive geometric noun rather than the physics-jargon ``separatrix``);
# this matches the project's catalog convention. The earlier short-lived
# inversion (which had ``separatrix`` canonical) was a regression and is
# explicitly avoided here.
_CANONICAL_LOCUS_SYNONYMS: dict[str, str] = {
    "separatrix": "plasma_boundary",
    "outboard_midplane_separatrix": "plasma_boundary",
    "last_closed_flux_surface": "plasma_boundary",
    "lcfs": "plasma_boundary",
    "divertor_plate": "divertor_target",
    "wall_surface": "wall",
    "first_wall_surface": "wall",
    "vacuum_vessel_wall": "wall",
    "pedestal_region": "pedestal",
    "edge_pedestal": "pedestal",
    "core_axis": "magnetic_axis",
}

# Synonym sources whose identity CHANGES under a geometric locus qualifier:
# the bare separatrix is the plasma boundary, but ``secondary_separatrix``
# (disconnected double-null) is a distinct surface that is NOT the boundary.
# Qualified forms of these tokens are never rewritten by the synonym audit.
_QUALIFIER_SENSITIVE_LOCUS_SYNONYMS: frozenset[str] = frozenset({"separatrix"})

# Substrings that, when found anywhere inside a compound locus token,
# indicate a canonical-locus violation. The exact-match map above
# handles known compounds; this scan catches future compounds the
# LLM might invent (e.g. ``upper_separatrix``, ``inner_divertor_plate``).
_CANONICAL_LOCUS_SUBSTRINGS: dict[str, str] = {
    "_separatrix": "_plasma_boundary",
    "_divertor_plate": "_divertor_target",
    "_lcfs": "_plasma_boundary",
}

# Bases that name an evaluated field (defined everywhere in the plasma
# and READ at a locus). When paired with a position/region locus, the
# relation MUST be ``_at_``. ``_of_`` is reserved for intrinsic
# geometric properties (area, radius, elongation, …).
_FIELD_BASES: frozenset[str] = frozenset(
    {
        "temperature",
        "density",
        "pressure",
        "magnetic_field",
        "electric_field",
        "magnetic_flux",
        "flux",
        "current",
        "current_density",
        "voltage",
        "loop_voltage",
        "velocity",
        "velocity_magnitude",
        "magnetic_shear",
        "safety_factor",
        "particle_flux",
        "energy_flux",
        "momentum_flux",
        "power",
        "power_density",
        "mass_density",
        "electric_potential",
        "electrostatic_potential",
        "kinetic_energy",
        "internal_energy",
        "enthalpy",
        "entropy",
        "radiation_density",
        "halo_current",
        "heat_flux",
    }
)


def canonical_locus_check(candidate: dict[str, Any]) -> list[str]:
    """Flag canonical-locus violations on the candidate name.

    Surfaces two anti-patterns that the compose prompt forbids but the
    LLM occasionally produces anyway:

    1. **Synonym locus token** — the candidate uses a deprecated
       synonym (``separatrix`` [bare], ``last_closed_flux_surface``,
       ``lcfs``, ``divertor_plate``, ``wall_surface``, …) instead of the
       canonical token (``plasma_boundary``, ``divertor_target``,
       ``wall``, …). The audit recommends the rewrite. (The canonical
       LCFS locus is ``plasma_boundary`` — the descriptive geometric
       noun — per ``_CANONICAL_LOCUS_SYNONYMS``; ``separatrix`` is
       qualifier-sensitive so ``secondary_separatrix`` stays distinct.)
    2. **Field-at-region preposition** — when the base is an evaluated
       field (flux, density, temperature, …) paired with a position or
       region locus, the relation MUST be ``_at_``. ``_of_<region>``
       reserves only intrinsic geometric properties (area, radius,
       elongation, …).

    Severity: critical — quarantines the candidate so the refine pool
    rewrites it under the same prompts.
    """
    name = (candidate.get("id") or candidate.get("name") or "").strip()
    if not name:
        return []

    issues: list[str] = []
    try:
        from imas_standard_names.grammar.parser import parse as ir_parse

        result = ir_parse(name)
        ir = getattr(result, "ir", None)
        if ir is None or ir.locus is None or ir.base is None:
            return []

        locus_token = ir.locus.token
        relation = (
            ir.locus.relation.value
            if hasattr(ir.locus.relation, "value")
            else str(ir.locus.relation)
        )
        locus_type = (
            ir.locus.type.value
            if hasattr(ir.locus.type, "value")
            else str(ir.locus.type)
        )
        base_token = ir.base.token

        locus_qualifiers = tuple(getattr(ir.locus, "qualifiers", ()) or ())
        canonical = _CANONICAL_LOCUS_SYNONYMS.get(locus_token)
        if canonical and (
            not locus_qualifiers
            or locus_token not in _QUALIFIER_SENSITIVE_LOCUS_SYNONYMS
        ):
            # Most synonym pairs survive geometric qualification
            # (``inner_divertor_plate`` IS the inner divertor target).
            # A qualifier-SENSITIVE pair names a distinct feature once
            # qualified — the secondary separatrix is not the plasma
            # boundary — so the rewrite must not apply there.
            issues.append(
                f"audit:canonical_locus_check: name '{name}' uses synonym "
                f"locus token '{locus_token}' — the canonical token is "
                f"'{canonical}'. Rewrite as "
                f"'{name.replace(locus_token, canonical)}'."
            )
        else:
            # Substring scan: catches UNREGISTERED compound forms the exact
            # map misses. A compound the ISN locus registry itself accepts
            # (compositional geometric qualifiers, e.g.
            # ``secondary_separatrix``) is a DISTINCT concept, not a synonym
            # spelling — rewriting it fabricates wrong physics
            # (``secondary_plasma_boundary``); leave those to the reviewer.
            registered_loci = _isn_locus_tokens()
            if locus_token not in registered_loci:
                for bad, good in _CANONICAL_LOCUS_SUBSTRINGS.items():
                    if bad in locus_token:
                        fixed_locus = locus_token.replace(bad, good)
                        # Only recommend rewrites that exist in the installed
                        # ISN locus registry; otherwise we fabricate bogus
                        # compounds.
                        if fixed_locus not in registered_loci:
                            continue
                        issues.append(
                            f"audit:canonical_locus_check: name '{name}' has "
                            f"locus token '{locus_token}' containing "
                            f"non-canonical substring '{bad}' — rewrite the "
                            f"locus as '{fixed_locus}'."
                        )
                        break

        if (
            relation == "of"
            and locus_type in {"position", "region"}
            and base_token in _FIELD_BASES
        ):
            corrected = name.replace(f"_of_{locus_token}", f"_at_{locus_token}")
            issues.append(
                f"audit:canonical_locus_check: field base '{base_token}' "
                f"with '_of_{locus_token}' is the field-at-region "
                f"anti-pattern — evaluated fields sample AT a position/"
                f"region. Rewrite as '{corrected}'."
            )
    except Exception:
        pass

    return issues


@lru_cache(maxsize=1)
def _decomposition_closed_vocab() -> dict[str, tuple[str, ...]]:
    """Closed-vocabulary token sets keyed by segment (all except open bases).

    Cached — the segment map is fixed for a given installed grammar.
    """
    from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

    aliases = {"coordinate", "object", "position"}
    return {
        seg: tuple(toks)
        for seg, toks in SEGMENT_TOKEN_MAP.items()
        if seg not in aliases and seg != "physical_base" and toks
    }


@lru_cache(maxsize=1)
def _registered_base_tokens() -> frozenset[str]:
    """Tokens the grammar accepts as atomic bases/carriers (cached).

    A ``physical_base`` in this set is a lexicalised compound the grammar owns
    (``convection_velocity``, ``diffusion_coefficient``, ``safety_factor``): a
    closed-vocab substring inside it is legitimate, not an absorption.
    """
    from imas_standard_names.grammar.parser import load_default_vocabularies

    vocabs = load_default_vocabularies()
    return frozenset(set(vocabs.bases) | set(vocabs.carriers))


def decomposition_audit_check(candidate: dict[str, Any]) -> list[str]:
    """Detect closed-vocabulary tokens genuinely absorbed into ``physical_base``.

    Parse-aware: the name is parsed under the current grammar and only the
    resulting ``physical_base`` is scanned for embedded closed-vocab tokens.
    A raw-name substring scan would flag a token even when the grammar
    correctly slots it (``ion_current_density`` → ``subject=ion``); parsing
    first means those are never reported. A ``physical_base`` the grammar
    registers as an atomic/lexicalised base is exempt — the embedded token is
    part of the base, not an absorption. Only a token left inside a
    ``physical_base`` the grammar does NOT accept as a base is a genuine
    decomposition failure and gets flagged.

    Names the grammar rejects outright return no issue here — the parse gate
    owns that failure, so it is not double-reported.

    This audit is deliberately **non-critical** (NOT in ``CRITICAL_CHECKS``):
    a surviving flag is a curation signal for the reviewer (rubric I4.6),
    not an auto-quarantine.

    Returns tagged issue strings of the form::

        "audit:decomposition_audit: name '<name>' contains closed-vocab token"
        " '<token>' (segment={<segments>}) absorbed into the name body. ..."
    """
    name = (candidate.get("id") or "").strip()
    if not name:
        return []

    try:
        from imas_standard_names.grammar import parse_standard_name

        from imas_codex.standard_names.decomposition import find_absorbed_closed_tokens
    except ImportError:
        return []

    # Parse under the current grammar. A name that does not parse (or is
    # non-canonical) is owned by the grammar gate, not this audit.
    try:
        model = parse_standard_name(name)
    except Exception:  # noqa: BLE001 — any grammar rejection is not our concern
        return []

    physical_base = (getattr(model, "physical_base", None) or "").strip()
    if not physical_base:
        return []

    # A grammar-registered atomic/lexicalised base owns any closed-vocab
    # substring it contains.
    if physical_base in _registered_base_tokens():
        return []

    closed_vocab = {
        seg: list(toks) for seg, toks in _decomposition_closed_vocab().items()
    }
    if not closed_vocab:
        return []

    absorbed = find_absorbed_closed_tokens(physical_base, closed_vocab)
    if not absorbed:
        return []

    # Group (token, segment) pairs by token so each token is reported once.
    grouped: dict[str, list[str]] = {}
    for tok, seg in absorbed:
        grouped.setdefault(tok, []).append(seg)

    issues: list[str] = []
    for tok in sorted(grouped):
        seg_str = ", ".join(sorted(grouped[tok]))
        issues.append(
            f"audit:decomposition_audit: name '{name}' contains closed-vocab "
            f"token '{tok}' (segment={{{seg_str}}}) absorbed into the name "
            f"body. Place it in its segment slot rather than letting it "
            f"leak into physical_base."
        )
    return issues


def ggd_implementation_leakage_check(candidate: dict[str, Any]) -> list[str]:
    """Flag GGD implementation details leaking into description or documentation.

    Detects patterns like "on the GGD edge grid", "GGD mesh", "GGD subgrid"
    that describe storage implementation rather than physics. Bare "GGD" in
    valid physics context (e.g. "general grid description") is NOT flagged.

    Non-critical audit — adds to validation_issues as a warning so the
    review/refine cycle can address it.
    """
    issues: list[str] = []
    _patterns = [
        r"\bon\s+(?:the|a|an)\s+GGD\b",
        r"\bGGD\s+(?:grid|mesh|element|subgrid|surface|cell|edge|node)\b",
        r"\bunstructured\s+GGD\b",
        r"\bGGD\s+(?:data\s+)?structure\b",
    ]
    _compiled = [re.compile(p, re.IGNORECASE) for p in _patterns]

    for field in ("description", "documentation"):
        text = candidate.get(field) or ""
        for pat in _compiled:
            m = pat.search(text)
            if m:
                issues.append(
                    f"audit:ggd_leakage: {field} contains GGD implementation "
                    f"detail '{m.group()}'. Descriptions and documentation "
                    f"should describe physics, not storage implementation."
                )
                break  # one issue per field is sufficient
    return issues


# ---------------------------------------------------------------------------
# Corpus-level (family) audit
# ---------------------------------------------------------------------------

# DD axis leaves and their frame-independent ISN axis token. The ``z`` leaf is
# frame-dependent (Cartesian ``z`` vs cylindrical ``vertical``) and resolved by
# :func:`_axis_token_for_leaf`; mirrors families._SUFFIX_TO_AXIS.
_LEAF_TO_AXIS_TOKEN: dict[str, str] = {
    "x": "x",
    "y": "y",
    "r": "radial",
    "phi": "toroidal",
}
# All DD axis leaves this audit recognises (adds the frame-dependent ``z``).
_AXIS_LEAVES: frozenset[str] = frozenset({*_LEAF_TO_AXIS_TOKEN, "z"})
# Axis prefixes a component name may lead with (longest-first for matching).
_FAMILY_AXIS_PREFIXES: tuple[str, ...] = (
    "perpendicular",
    "toroidal",
    "poloidal",
    "parallel",
    "vertical",
    "radial",
    "x",
    "y",
    "z",
)
# A machine-frame vector uses exactly one of these axis-token triples: Cartesian
# (x, y, z) or cylindrical (radial, toroidal, vertical). ``vertical`` is the
# cylindrical Z; ``z`` is the Cartesian third axis — the two frames never mix.
_CANONICAL_AXIS_TRIPLES: tuple[frozenset[str], ...] = (
    frozenset({"x", "y", "z"}),
    frozenset({"radial", "toroidal", "vertical"}),
)


def _node_frame(leaves: set[str]) -> str:
    """Infer a DD vector node's coordinate frame from its leaf set.

    A cylindrical member (``r``/``phi``) makes the node cylindrical (``z`` →
    ``vertical``); a purely Cartesian node (``x``/``y``) is Cartesian (``z`` →
    ``z``); a lone ``z`` defaults to cylindrical.
    """
    if leaves & {"r", "phi"}:
        return "cylindrical"
    if leaves & {"x", "y"}:
        return "cartesian"
    return "cylindrical"


def _axis_token_for_leaf(leaf: str, frame: str) -> str | None:
    """Canonical ISN axis token for a DD leaf given its node's frame."""
    if leaf == "z":
        return "z" if frame == "cartesian" else "vertical"
    return _LEAF_TO_AXIS_TOKEN.get(leaf)


def _split_axis_carrier_locus(
    name: str,
) -> tuple[str | None, str, str | None]:
    """Split a component name into (axis prefix, base carrier, locus token)."""
    axis: str | None = None
    rest = name
    for pfx in _FAMILY_AXIS_PREFIXES:
        if name.startswith(pfx + "_"):
            axis = pfx
            rest = name[len(pfx) + 1 :]
            break
    of_i = rest.rfind("_of_")
    at_i = rest.rfind("_at_")
    i = max(of_i, at_i)
    if i >= 0:
        return axis, rest[:i], (rest[i + 4 :] or None)
    return axis, rest, None


def _strip_source_scheme(path: str) -> str:
    """Drop a ``dd:`` / ``signals:`` provenance scheme prefix from a source path."""
    for scheme in ("dd:", "signals:"):
        if path.startswith(scheme):
            return path[len(scheme) :]
    return path


def vector_family_consistency_check(names: list[dict[str, Any]]) -> list[str]:
    """Corpus-level audit: the components of one DD vector node must agree.

    A DD *vector node* is a parent whose children are the axis leaves
    ``x``/``y``/``z`` or ``r``/``phi``/``z``. This audit groups the provided
    standard names by that node — derived from each name's ``source_paths``
    (``dd_paths`` as a fallback) — and, for each node, verifies its component
    names:

    1. use the canonical axis token for their leaf — a ``z`` leaf is ``z`` in a
       Cartesian node (x, y, z) and ``vertical`` in a cylindrical node
       (r, phi, z) — and never conflate two axes on one name;
    2. agree on the shared **base carrier** (the name minus the axis prefix and
       any ``_of_``/``_at_`` locus);
    3. agree on the **locus token** (all bare, or all the same ``_of_<device>``);
    4. agree on **physics_domain**;
    5. draw their axis tokens from a single canonical triple (``x, y, z`` or
       ``radial, toroidal, vertical``) — never a mix of frames;
    6. carry **documentation** consistently — when any sibling of a node has
       non-empty ``documentation``, every sibling must (a name minted as an
       ``attached`` merge onto a stub, rather than freshly composed, can be
       left with ``documentation=""`` forever — this is the exact shape of
       the z-axis documentation gap the systematic review found on 6/6
       affected coordinate/unit-vector triples, always the last-sorted axis).

    Each disagreement is one tagged issue string. Unlike the per-candidate
    audits this runs over a name corpus, mirroring
    :func:`semantic_similarity_check`'s standalone signature.
    """
    from collections import defaultdict

    # node -> {name: {"leaves": set[str], "domain": Any, "documented": bool}}
    nodes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in names:
        name = (entry.get("id") or entry.get("name") or "").strip()
        if not name:
            continue
        domain = entry.get("physics_domain")
        documented = bool((entry.get("documentation") or "").strip())
        paths = entry.get("source_paths") or entry.get("dd_paths") or []
        if not isinstance(paths, (list, tuple)):
            continue
        for raw in paths:
            if not isinstance(raw, str):
                continue
            path = _strip_source_scheme(raw)
            segs = path.split("/")
            if len(segs) < 2:
                continue
            leaf = segs[-1]
            if leaf not in _AXIS_LEAVES:
                continue
            node = "/".join(segs[:-1])
            member = nodes[node].setdefault(
                name, {"leaves": set(), "domain": domain, "documented": documented}
            )
            member["leaves"].add(leaf)

    issues: list[str] = []
    for node in sorted(nodes):
        members = nodes[node]
        # The node's frame (Cartesian vs cylindrical) is read from all its axis
        # leaves, so a ``z`` leaf resolves to ``z`` or ``vertical`` consistently
        # across the family.
        node_leaves: set[str] = set()
        for info in members.values():
            node_leaves |= info["leaves"]
        frame = _node_frame(node_leaves)
        carriers: set[str] = set()
        loci: set[str | None] = set()
        domains: set[Any] = set()
        axis_tokens: set[str] = set()
        for name in sorted(members):
            info = members[name]
            axis, carrier, locus = _split_axis_carrier_locus(name)
            carriers.add(carrier)
            loci.add(locus)
            if info["domain"] is not None:
                domains.add(info["domain"])
            if axis is not None:
                axis_tokens.add(axis)

            # (1) canonical axis token / no axis conflation on one name.
            expected = {_axis_token_for_leaf(le, frame) for le in info["leaves"]}
            if len(expected) > 1:
                issues.append(
                    f"audit:vector_family_consistency_check: name '{name}' "
                    f"covers multiple axis leaves {sorted(info['leaves'])} of "
                    f"vector node '{node}' — each axis is a distinct component."
                )
            elif axis is None or axis not in expected:
                exp = next(iter(expected))
                z_hint = (
                    " ('z' names the Cartesian third axis; 'vertical' the "
                    "cylindrical Z)"
                    if "z" in info["leaves"]
                    else ""
                )
                issues.append(
                    f"audit:vector_family_consistency_check: name '{name}' "
                    f"(leaf {sorted(info['leaves'])} of node '{node}') must "
                    f"lead with the canonical axis token '{exp}'{z_hint}."
                )

        if len(members) < 2:
            continue  # agreement checks need 2+ components

        member_list = ", ".join(sorted(members))
        if len(carriers) > 1:
            issues.append(
                f"audit:vector_family_consistency_check: vector node '{node}' "
                f"components disagree on base carrier {sorted(carriers)} "
                f"({member_list})."
            )
        if len(loci) > 1:
            printable = sorted("<none>" if lo is None else lo for lo in loci)
            issues.append(
                f"audit:vector_family_consistency_check: vector node '{node}' "
                f"components disagree on locus {printable} ({member_list})."
            )
        if len(domains) > 1:
            issues.append(
                f"audit:vector_family_consistency_check: vector node '{node}' "
                f"components disagree on physics_domain {sorted(domains)} "
                f"({member_list})."
            )
        if axis_tokens and not any(
            axis_tokens <= triple for triple in _CANONICAL_AXIS_TRIPLES
        ):
            issues.append(
                f"audit:vector_family_consistency_check: vector node '{node}' "
                f"uses non-canonical axis triple {sorted(axis_tokens)} — expected "
                f"x, y, z or radial, toroidal, vertical ({member_list})."
            )

        # (6) documentation completeness: a sibling merged onto an existing
        # stub via 'attach' rather than freshly composed can carry
        # documentation="" indefinitely, since attach never regenerates
        # docs. Once any sibling in the node is documented, an undocumented
        # sibling is a defect, not a pending-generation state.
        documented = {n for n in members if members[n]["documented"]}
        undocumented = set(members) - documented
        if documented and undocumented:
            issues.append(
                f"audit:vector_family_consistency_check: vector node '{node}' "
                f"has documented siblings {sorted(documented)} but empty "
                f"documentation on {sorted(undocumented)} — an attach-only "
                f"merge likely skipped docs generation ({member_list})."
            )

    return issues


def dd_path_uniqueness_check(names: list[dict[str, Any]]) -> list[str]:
    """Corpus-level audit: one DD path carries exactly one standard name.

    A DD path attached to two accepted names is an attach error — the review
    of the 90 live duplicates classified 86/90 as generic-vs-specific double
    attaches or locus mismatches. Groups the provided names by each of their
    ``source_paths`` (``dd_paths`` fallback) and flags every path claimed by
    more than one name. Corpus-level signature, mirroring
    :func:`vector_family_consistency_check`.
    """
    from collections import defaultdict

    claims: dict[str, set[str]] = defaultdict(set)
    for entry in names:
        name = (entry.get("id") or entry.get("name") or "").strip()
        if not name:
            continue
        paths = entry.get("source_paths") or entry.get("dd_paths") or []
        if not isinstance(paths, (list, tuple)):
            continue
        for raw in paths:
            if isinstance(raw, str):
                claims[_strip_source_scheme(raw)].add(name)

    issues: list[str] = []
    for path in sorted(claims):
        holders = claims[path]
        if len(holders) > 1:
            issues.append(
                f"audit:dd_path_uniqueness_check: DD path '{path}' is attached "
                f"to {len(holders)} names ({', '.join(sorted(holders))}) — one "
                f"path carries exactly one standard name."
            )
    return issues


def run_audits(
    candidate: dict[str, Any],
    existing_sns_in_domain: list[dict[str, Any]] | None = None,
    source_path: str | None = None,
    source_cocos_type: str | None = None,
) -> list[str]:
    """Run all audits on a candidate and return tagged issue strings.

    Each returned string has the format ``"audit:<check_name>: <detail>"``.

    Args:
        candidate: Standard name candidate dict (must include ``id``,
            ``description``, ``documentation``, ``unit`` at minimum).
        existing_sns_in_domain: Precomputed list of existing SNs in the
            same domain for synonym checking. Each dict needs ``name``,
            ``description_embedding``, ``unit``.
        source_path: The original source DD path for provenance verb check.
        source_cocos_type: COCOS transformation type from the source path.

    Returns:
        List of tagged issue strings.
    """
    all_issues: list[str] = []

    all_issues.extend(latex_def_check(candidate))
    all_issues.extend(description_notation_check(candidate))
    all_issues.extend(placeholder_check(candidate))
    all_issues.extend(unit_validity_check(candidate))
    all_issues.extend(generic_noun_check(candidate))
    all_issues.extend(tautology_check(candidate))
    all_issues.extend(spectral_suffix_check(candidate))
    all_issues.extend(abbreviation_check(candidate))
    all_issues.extend(american_spelling_check(candidate))
    all_issues.extend(name_description_consistency_check(candidate))
    all_issues.extend(description_verb_drift_check(candidate))
    all_issues.extend(structural_dim_tag_check(candidate))
    all_issues.extend(provenance_verb_check(candidate, source_path))
    all_issues.extend(synonym_check(candidate, existing_sns_in_domain or []))
    all_issues.extend(unit_dimension_check(candidate))
    all_issues.extend(name_unit_consistency_check(candidate, source_path))
    all_issues.extend(multi_subject_check(candidate))
    all_issues.extend(cocos_specificity_check(candidate, source_cocos_type))
    all_issues.extend(representation_artifact_check(candidate, source_path))
    all_issues.extend(causal_due_to_check(candidate))
    all_issues.extend(implicit_field_check(candidate))
    all_issues.extend(density_unit_consistency_check(candidate))
    all_issues.extend(position_coordinate_check(candidate))
    all_issues.extend(vector_field_component_check(candidate))
    all_issues.extend(segment_order_check(candidate))
    all_issues.extend(aggregator_order_check(candidate))
    all_issues.extend(diamagnetic_component_check(candidate))
    all_issues.extend(amplitude_of_prefix_check(candidate))
    all_issues.extend(mode_number_suffix_check(candidate))
    all_issues.extend(cumulative_prefix_check(candidate))
    all_issues.extend(pulse_schedule_reference_check(candidate, source_path))
    all_issues.extend(ratio_binary_operator_check(candidate))
    all_issues.extend(instrument_owned_observable_check(candidate))
    all_issues.extend(profile_suffix_check(candidate))
    all_issues.extend(repeated_token_check(candidate))
    all_issues.extend(adjacent_duplicate_token_check(candidate))
    all_issues.extend(length_soft_cap_check(candidate))
    all_issues.extend(instrument_stokes_bind_check(candidate))
    all_issues.extend(position_redundancy_check(candidate))
    all_issues.extend(process_qualifier_check(candidate))
    all_issues.extend(preposition_physical_base_check(candidate))
    all_issues.extend(canonical_locus_check(candidate))
    all_issues.extend(decomposition_audit_check(candidate))
    all_issues.extend(ggd_implementation_leakage_check(candidate))

    return all_issues


def has_critical_audit_failure(issues: list[str]) -> bool:
    """Return True if any issue is from a critical audit check."""
    for issue in issues:
        for check in CRITICAL_CHECKS:
            if f"audit:{check}:" in issue:
                return True
    return False


def derived_parent_structural_check(name: str, children: list[str]) -> list[str]:
    """Structural admission for a derived family parent (a partial name).

    A derived family parent is a deliberately partial name peeled from its
    children: ``internal_state_energy_flux`` generalises over the species
    subject that each concrete child (``deuterium_internal_state_energy_flux``,
    …) carries; ``magnetic_field`` generalises over the projection axis of its
    ``radial_magnetic_field`` / ``toroidal_magnetic_field`` children. Such a
    parent is NOT required to parse as a standalone full name — the peel
    legitimately drops the very segment (subject, projection, …) a complete
    name would have to carry, so the full-name grammar round-trip that gates
    standalone names is the wrong instrument for it.

    What a derived parent MUST satisfy is the structural contract that
    justifies materialising it:

    - it groups at least one child via ``HAS_PARENT`` — an orphan parent
      generalises nothing and is residue;
    - it is a genuine generalisation of a child — every token of the parent
      appears in at least one child, so the parent is a peel of a real family
      member rather than an unrelated string. Token *set* containment (not
      ordered subsequence) is used because qualifier binding can reorder tokens
      between a child and its peeled parent.

    Returns critical ``audit:derived_parent_structure_check`` issues when either
    clause is broken — the missed-acceptance-gate signal is preserved for a
    genuinely malformed parent — or ``[]`` when the parent is a sound peel.
    """
    parent = (name or "").strip()
    if not parent:
        return ["audit:derived_parent_structure_check: empty parent name"]

    child_ids = [c for c in (children or []) if c]
    if not child_ids:
        return [
            f"audit:derived_parent_structure_check: derived parent {parent!r} "
            "has no HAS_PARENT children — it generalises nothing"
        ]

    parent_tokens = set(parent.split("_"))
    if not any(parent_tokens <= set(child.split("_")) for child in child_ids):
        sample = ", ".join(sorted(child_ids)[:3])
        return [
            f"audit:derived_parent_structure_check: derived parent {parent!r} is "
            f"not a token-generalisation of any child ({sample}) — the peel is "
            "inconsistent with the family it heads"
        ]

    return []


# =============================================================================
# Semantic similarity gate (post-embed)
# =============================================================================


def semantic_similarity_check(
    name: str,
    description: str | None,
    *,
    critical_threshold: float | None = None,
    warning_threshold: float | None = None,
) -> tuple[float | None, list[str]]:
    """Compute cosine similarity between name-as-text and description.

    This check catches semantically ambiguous names where the name alone
    does not convey what is being measured (e.g. ``co_passing_density`` —
    density of what?).

    Both embeddings are computed fresh from the raw text fields using the
    project embedding server.  The stored ``sn.embedding`` is NOT used
    because its format (``"name — description"``) conflates name and
    description semantics.

    Args:
        name: The standard name string (e.g. ``"co_passing_density"``).
        description: The short description text.
        critical_threshold: Override for quarantine threshold.
        warning_threshold: Override for advisory threshold.

    Returns:
        ``(similarity, issues)`` — similarity is None on embed failure.
        Issues are tagged strings for ``validation_issues``.
    """
    from imas_codex.standard_names.defaults import (
        SEMANTIC_SIM_CRITICAL,
        SEMANTIC_SIM_WARNING,
    )

    crit = (
        critical_threshold if critical_threshold is not None else SEMANTIC_SIM_CRITICAL
    )
    warn = warning_threshold if warning_threshold is not None else SEMANTIC_SIM_WARNING

    if not description or not description.strip():
        return None, []

    name_text = name.replace("_", " ")
    desc_text = description[:500]

    try:
        from imas_codex.embeddings.description import embed_descriptions_batch

        items = [
            {"id": "name", "_text": name_text},
            {"id": "desc", "_text": desc_text},
        ]
        embed_descriptions_batch(items, text_field="_text")
        name_emb = items[0].get("embedding")
        desc_emb = items[1].get("embedding")
        if name_emb is None or desc_emb is None:
            logger.debug("semantic_similarity_check: embed returned None for %s", name)
            return None, []
    except Exception:
        logger.debug(
            "semantic_similarity_check: embed failed for %s", name, exc_info=True
        )
        return None, []

    name_vec = np.asarray(name_emb, dtype=np.float32)
    desc_vec = np.asarray(desc_emb, dtype=np.float32)
    norm_n = np.linalg.norm(name_vec)
    norm_d = np.linalg.norm(desc_vec)
    if norm_n < 1e-8 or norm_d < 1e-8:
        return None, []

    sim = float(np.dot(name_vec, desc_vec) / (norm_n * norm_d))

    issues: list[str] = []
    if sim < crit:
        issues.append(
            f"audit:semantic_similarity_check: "
            f"sim={sim:.3f} below critical threshold {crit:.2f} — "
            f"name is semantically ambiguous"
        )
    elif sim < warn:
        issues.append(
            f"audit:semantic_similarity_check_warning: "
            f"sim={sim:.3f} below warning threshold {warn:.2f} — "
            f"name may be semantically ambiguous"
        )

    return sim, issues


# ---------------------------------------------------------------------------
# Graph-side gate audits (live catalog, read-only)
# ---------------------------------------------------------------------------

#: Flux-surface reduction operator prefixes gated by the ISN grammar
#: (constant_on_flux_surface bases reject them — the reduction of a flux
#: function is a no-op). Used to pre-filter the live catalog cheaply before
#: running the authoritative ISN parse.
_FLUX_SURFACE_REDUCTION_TOKENS: tuple[str, ...] = (
    "flux_surface_averaged",
    "maximum_over_flux_surface",
    "minimum_over_flux_surface",
)


def find_flux_surface_reduction_violations(*, gc=None) -> list[dict[str, Any]]:
    """Return live names rejected by the flux-surface-reduction grammar gate.

    A live (non-superseded, non-exhausted) name carrying a flux-surface
    reduction operator on a base flagged ``constant_on_flux_surface`` in the
    ISN vocabulary (safety factor, magnetic shear, flux labels, pressure) is
    a no-op composition the grammar can no longer mint; any survivor in the
    graph is legacy debt to supersede. Read-only diagnostic.
    """
    from imas_standard_names.grammar.model import (  # noqa: PLC0415
        parse_standard_name,
    )

    from imas_codex.graph.client import GraphClient  # noqa: PLC0415

    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = (
            gc.query(
                """
                MATCH (n:StandardName)
                WHERE NOT coalesce(n.name_stage, '') IN ['superseded', 'exhausted']
                  AND any(tok IN $tokens WHERE n.id CONTAINS tok)
                RETURN n.id AS id, n.name_stage AS name_stage
                ORDER BY n.id
                """,
                tokens=list(_FLUX_SURFACE_REDUCTION_TOKENS),
            )
            or []
        )
    finally:
        if owns:
            gc.close()

    violations: list[dict[str, Any]] = []
    for row in rows:
        try:
            parse_standard_name(row["id"])
        except ValueError as exc:
            if "constant on a flux surface" in str(exc):
                violations.append(
                    {
                        "id": row["id"],
                        "name_stage": row.get("name_stage"),
                        "reason": str(exc),
                    }
                )
        except Exception:  # noqa: BLE001 - other parse failures are not this gate
            continue
    return violations


def find_removed_dd_sources(*, gc=None) -> list[dict[str, Any]]:
    """Return live names still fed by a DD path absent from the current DD.

    The DD build stamps IMASNodes absent from the current DD version as
    ``lifecycle_status='removed'`` and extraction excludes them, so any live
    name whose source resolves through a removed node is legacy debt from a
    pre-gate extract — re-anchor it to the renamed path or retire it.
    Read-only diagnostic.
    """
    from imas_codex.graph.client import GraphClient  # noqa: PLC0415

    owns = gc is None
    gc = gc or GraphClient()
    try:
        return list(
            gc.query(
                """
                MATCH (s:StandardNameSource {source_type:'dd'})-[:PRODUCED_NAME]->(n:StandardName)
                WHERE NOT coalesce(n.name_stage, '') IN ['superseded', 'exhausted']
                MATCH (l:IMASNode {id: s.source_id})
                WHERE l.lifecycle_status = 'removed'
                RETURN n.id AS id, s.id AS source_id, l.renamed_to AS renamed_to
                ORDER BY n.id
                """
            )
            or []
        )
    finally:
        if owns:
            gc.close()
