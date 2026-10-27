"""Common candidate, provenance, and telemetry helpers."""

from __future__ import annotations

import logging
import re as _re
from functools import cache as _cache, lru_cache
from typing import Any

__all__ = [
    "logger",
    "backfill_semantic_similarity_gate_resolution_method",
    "normalize_spelling",
    "normalize_prose_spelling",
    "_GREEK_WORD_TO_SYMBOL",
    "_GREEK_WORD_RE",
    "_GREEK_CMD_TO_SYMBOL",
    "_GREEK_CMD_RE",
    "_GREEK_STRANDED_BACKSLASH_RE",
    "_LATEX_CMD_ARG_RE",
    "_LATEX_CMD_BARE_RE",
    "_SUPERSCRIPT_BRACE_RE",
    "_SUBSCRIPT_BRACE_RE",
    "_MATH_DELIM_RE",
    "normalize_description_notation",
    "normalize_description_text",
    "_DOMAIN_UK_US",
    "_GRAMMAR_FIELDS",
    "_dedup_adjacent_tokens",
    "_SHAPE_PARAMETER_BASE_POLICY",
    "_isn_segment_tokens",
    "_isn_physical_bases",
    "shape_parameter_base_drift",
    "_shape_parameter_bases",
    "_rate_natured_bases",
    "_state_resolution_tokens",
    "_name_denotes_state",
    "_name_denotes_rate",
    "_ir_denotes_time_change",
    "_ir_claims_time_change_in_scope",
    "_name_claims_time_change",
    "_time_change_claim_is_binary_scoped",
    "_shape_parameter_surface",
    "_inject_shape_parameter_surface",
    "_SNAKE_CASE_RE",
    "is_well_formed_candidate",
    "_claimed_source_ids",
    "_strip_response_source_id",
    "_review_dd_source_bindings",
    "_review_dd_source_ids",
    "_sanitize_dd_gap_evidence",
    "_persist_dd_gap_evidence",
    "_claimed_source_outcome",
    "_sanitize_compose_result_sources",
    "_filter_persist_candidates_to_claimed_sources",
    "_withhold_binding_conflicted_sources",
    "_candidate_source_node_id",
    "_persist_description_checked_candidates",
    "_mark_source_prevalidation_failed",
    "provenance_canonical_names",
    "_NON_NAMEABLE_BARE_TOKENS",
    "is_non_nameable_coordinate",
    "_coerce_llm_metric",
    "_extract_llm_telemetry",
    "_load_known_physical_bases",
    "_auto_detect_physical_base_gaps",
    "_compute_token_reuse_hits",
    "_grammar_round_trip_failures",
    "_build_grammar_retry_reason",
    "_build_token_reuse_retry_reason",
    "_active_editorial_gap_guidance",
    "_stamp_dedup_decision",
]

logger = logging.getLogger(__name__)


def backfill_semantic_similarity_gate_resolution_method(
    _worker_dependency, /, gc: Any | None = None
) -> int:
    """Type legacy deterministic gate reviews without granting them authority.

    The model and reviewer-model pair is the durable identity of these rows.
    Scores are deliberately not part of the predicate because genuine reviewer
    verdicts can share the same normalized score.
    """

    def _backfill(client: Any) -> int:
        rows = list(
            client.query(
                "\n                MATCH (review:StandardNameReview)\n                WHERE review.model = $model_identity\n                  AND review.reviewer_model = $model_identity\n                  AND review.resolution_method IS NULL\n                SET review.resolution_method = $resolution_method\n                RETURN count(review) AS updated\n                ",
                model_identity=_worker_dependency("SEMANTIC_SIM_GATE_MODEL"),
                resolution_method=_worker_dependency(
                    "SEMANTIC_SIM_GATE_RESOLUTION_METHOD"
                ),
            )
            or []
        )
        return int(rows[0]["updated"]) if rows else 0

    if gc is not None:
        return _backfill(gc)
    from imas_codex.graph.client import GraphClient

    with GraphClient() as client:
        return _backfill(client)


def normalize_spelling(_worker_dependency, /, name: str) -> str:
    """Deterministic British→American spelling normalization.

    Splits the underscore-delimited name into tokens and converts each
    via ``breame.spelling.get_american_spelling``.  A small domain-specific
    supplement catches physics terms missing from breame's dictionary.
    """
    from breame.spelling import get_american_spelling

    tokens = name.split("_")
    return "_".join(
        _worker_dependency("_DOMAIN_UK_US").get(t, get_american_spelling(t))
        for t in tokens
    )


def normalize_prose_spelling(_worker_dependency, /, text: str) -> str:
    """British→American spelling normalization for prose text.

    Unlike ``normalize_spelling`` which operates on underscore-delimited
    name tokens, this function handles natural English prose — splitting
    on word boundaries and preserving punctuation, whitespace, and
    markdown formatting.
    """
    if not text:
        return text
    from breame.spelling import get_american_spelling

    def _replace(match: _re.Match) -> str:
        word = match.group(0)
        us = _worker_dependency("_DOMAIN_UK_US").get(
            word.lower(), get_american_spelling(word.lower())
        )
        if us == word.lower():
            return word
        if word[0].isupper():
            return us[0].upper() + us[1:]
        return us

    return _re.sub("[A-Za-z]+", _replace, text)


_GREEK_WORD_TO_SYMBOL: dict[str, str] = {"phi": "φ", "theta": "θ", "rho": "ρ"}
_GREEK_WORD_RE = _re.compile("\\b(phi|theta|rho)\\b")
_GREEK_CMD_TO_SYMBOL: dict[str, str] = {
    "phi": "φ",
    "theta": "θ",
    "rho": "ρ",
    "pi": "π",
    "Phi": "Φ",
    "Theta": "Θ",
    "Rho": "Ρ",
    "Pi": "Π",
}
_GREEK_CMD_RE = _re.compile("\\\\(phi|theta|rho|pi|Phi|Theta|Rho|Pi)(?![a-zA-Z])")
_GREEK_STRANDED_BACKSLASH_RE = _re.compile("\\\\(?=[φθρπΦΘΡΠ])")
_LATEX_CMD_ARG_RE = _re.compile("\\\\[a-zA-Z]+\\s*\\{([^{}]*)\\}")
_LATEX_CMD_BARE_RE = _re.compile("\\\\[a-zA-Z]+")
_SUPERSCRIPT_BRACE_RE = _re.compile("\\^\\{([^{}]*)\\}")
_SUBSCRIPT_BRACE_RE = _re.compile("_\\{([^{}]*)\\}")
_MATH_DELIM_RE = _re.compile("\\$\\$?")


def normalize_description_notation(_worker_dependency, /, text: str) -> str:
    """Normalize description prose to plain Unicode text.

    Descriptions are plain text (the ISN convention); LaTeX and math markup
    belong in the ``documentation`` field. Any LaTeX that leaks into a
    description is stripped to a readable plain form:

    - ``$…$`` / ``$$…$$`` delimiters removed (inner content kept);
    - ``\\phi \\theta \\rho \\pi`` → ``φ θ ρ π`` (plus capitals), and a
      stranded backslash in front of an already-Unicode symbol (``\\φ``) → φ;
    - spelled-out ``phi/theta/rho`` prose words → ``φ/θ/ρ`` (word-bounded, so
      DD tokens like ``phi_tor`` / ``b_field_phi`` are untouched);
    - ``\\mathbf{k}`` → ``k`` and other backslash commands dropped;
    - braced sub/superscripts (``^{1+}`` / ``_{i}``) flattened.

    Idempotent — a description already in plain form is returned unchanged.
    Applies to the description field only; ``documentation`` keeps LaTeX.
    """
    if not text:
        return text
    text = _worker_dependency("_GREEK_CMD_RE").sub(
        lambda m: _worker_dependency("_GREEK_CMD_TO_SYMBOL")[m.group(1)], text
    )
    text = _worker_dependency("_GREEK_STRANDED_BACKSLASH_RE").sub("", text)
    text = _worker_dependency("_GREEK_WORD_RE").sub(
        lambda m: _worker_dependency("_GREEK_WORD_TO_SYMBOL")[m.group(1)], text
    )
    text = _worker_dependency("_LATEX_CMD_ARG_RE").sub("\\1", text)
    text = _worker_dependency("_LATEX_CMD_BARE_RE").sub("", text)
    text = _worker_dependency("_SUPERSCRIPT_BRACE_RE").sub("\\1", text)
    text = _worker_dependency("_SUBSCRIPT_BRACE_RE").sub("_\\1", text)
    text = text.replace("{", "").replace("}", "")
    text = _worker_dependency("_MATH_DELIM_RE").sub("", text)
    text = text.replace("\\", "")
    text = _re.sub("[ \\t]{2,}", " ", text)
    return text


def normalize_description_text(_worker_dependency, /, text: str) -> str:
    """Full description-prose normalization: spelling, then Greek symbols."""
    return _worker_dependency("normalize_description_notation")(
        _worker_dependency("normalize_prose_spelling")(text)
    )


_DOMAIN_UK_US: dict[str, str] = {
    "linearised": "linearized",
    "linearise": "linearize",
    "discretised": "discretized",
    "discretise": "discretize",
    "symmetrised": "symmetrized",
    "symmetrise": "symmetrize",
    "parameterised": "parameterized",
    "parameterise": "parameterize",
    "centreline": "centerline",
    "centre": "center",
    "grey": "gray",
    "colour": "color",
    "favour": "favor",
    "favourite": "favorite",
    "honour": "honor",
    "labelled": "labeled",
    "labelling": "labeling",
    "travelled": "traveled",
    "travelling": "traveling",
    "catalogue": "catalog",
    "programme": "program",
    "analogue": "analog",
    "dialogue": "dialog",
}
_GRAMMAR_FIELDS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
    "geometric_base",
    "object",
)


def _dedup_adjacent_tokens(
    _worker_dependency, /, name: str, log: logging.Logger | None = None
) -> str:
    """Remove adjacent duplicate tokens introduced by grammar round-trip.

    The ISN grammar (≤ 0.7.0rc27) has a known bug where ``parse →
    compose`` doubles certain tokens — e.g.
    ``magnetic_field_probe`` → ``magnetic_magnetic_field_probe``.

    This function collapses any ``tok_tok`` pair into a single ``tok``
    while preserving legitimate compounds via
    :data:`~imas_codex.standard_names.audits._COMPOUND_SUBJECT_PAIRS`.
    """
    from imas_codex.standard_names.audits import _COMPOUND_SUBJECT_PAIRS

    tokens = name.split("_")
    result: list[str] = [tokens[0]]
    for tok in tokens[1:]:
        if tok == result[-1] and f"{tok}_{tok}" not in _COMPOUND_SUBJECT_PAIRS:
            if log:
                log.debug("Dedup adjacent token %r in %r", f"{tok}_{tok}", name)
            continue
        result.append(tok)
    return "_".join(result)


_SHAPE_PARAMETER_BASE_POLICY: tuple[str, ...] = (
    "triangularity",
    "elongation",
    "squareness",
)


@_cache
def _isn_segment_tokens(_worker_dependency, /, segment: str) -> frozenset[str]:
    """The token universe ISN registers for one grammar *segment*.

    Empty when ISN is absent or the segment is unknown. Callers must read empty
    as "the vocabulary cannot be consulted", never as "no token qualifies" — a
    rule keyed on an empty set would judge every name by a vocabulary it could
    not read.
    """
    from imas_codex.standard_names.segments import grammar_tokens_by_segment

    return frozenset(grammar_tokens_by_segment().get(segment) or ())


def _isn_physical_bases(_worker_dependency, /) -> frozenset[str]:
    """The ISN ``physical_base`` token universe, or empty when ISN is absent."""
    return _worker_dependency("_isn_segment_tokens")("physical_base")


def shape_parameter_base_drift(_worker_dependency, /) -> frozenset[str]:
    """Policy tokens that are no longer ISN ``physical_base`` tokens.

    Non-empty means ISN's vocabulary moved out from under the surface rule and
    the policy needs updating (a renamed or retired shape descriptor). Empty
    when ISN is unavailable — an absent vocabulary is not evidence of drift.
    """
    isn = _worker_dependency("_isn_physical_bases")()
    if not isn:
        return frozenset()
    return frozenset(_worker_dependency("_SHAPE_PARAMETER_BASE_POLICY")) - isn


@lru_cache(maxsize=1)
def _shape_parameter_bases(_worker_dependency, /) -> frozenset[str]:
    """Shape-descriptor bases that must carry a surface locus.

    The codex policy intersected with the live ISN ``physical_base``
    vocabulary: a token ISN no longer registers is dropped (and reported by
    :func:`shape_parameter_base_drift`) rather than applied blind. When ISN is
    unavailable the policy is returned as-is so the rule keeps working offline.
    """
    isn = _worker_dependency("_isn_physical_bases")()
    if not isn:
        return frozenset(_worker_dependency("_SHAPE_PARAMETER_BASE_POLICY"))
    return frozenset(_worker_dependency("_SHAPE_PARAMETER_BASE_POLICY")) & isn


@lru_cache(maxsize=1)
def _rate_natured_bases(_worker_dependency, /) -> frozenset[str]:
    """ISN ``physical_base`` tokens that count a quantity per unit time.

    A codex POLICY keyed on the live ISN vocabulary: a base that IS ``rate`` or
    ``frequency``, or that ends in ``_rate`` / ``_frequency``, already denotes a
    per-time quantity, so a name built on it reads as a rate without carrying a
    ``time_derivative_of_`` prefix. Derived at runtime, so a base ISN adds or
    renames is picked up with no codex edit; empty when ISN is unavailable,
    which only relaxes the rate reading back to the prefix form.
    """
    return frozenset(
        t
        for t in _worker_dependency("_isn_physical_bases")()
        if t in {"rate", "frequency"} or t.endswith(("_rate", "_frequency"))
    )


@lru_cache(maxsize=1)
def _state_resolution_tokens(_worker_dependency, /) -> frozenset[str]:
    """Grammar tokens that mark a name as resolved to ONE state of a species.

    The grammar carries state resolution two ways, and both are read from the
    live vocabulary rather than restated here: a dedicated ``state`` segment
    whose tokens name the KIND of state (a charge state, an internal excitation
    state), and the ``subject`` tokens that fold the state into the species
    (``ion_state``, ``neutral_state``, ``ion_charge_state``). Codex owns only
    the POLICY on the subject side — a subject that names a state ends in
    ``_state`` — so a new kind of state added upstream is recognised with no
    codex edit. Empty when the vocabulary cannot be consulted, which switches
    the rule off rather than judging every name species-level.
    """
    subjects = frozenset(
        t
        for t in _worker_dependency("_isn_segment_tokens")("subject")
        if t.endswith("_state")
    )
    return _worker_dependency("_isn_segment_tokens")("state") | subjects


def _name_denotes_state(_worker_dependency, /, sn_name: str) -> bool:
    """True when *sn_name* is resolved to one state of a species.

    Matched as whole grammar tokens, so a word that merely ends in ``state``
    (``steady_state``, a regime rather than a species state) never counts.
    """
    padded = f"_{sn_name}_"
    return any(
        f"_{t}_" in padded for t in _worker_dependency("_state_resolution_tokens")()
    )


def _name_denotes_rate(_worker_dependency, /, sn_name: str) -> bool:
    """True when *sn_name* is built on a rate-natured base.

    Matched on token boundaries, so a word that merely contains a base as a
    substring (``substrate`` ⊃ ``rate``) never counts as one.
    """
    padded = f"_{sn_name}_"
    return any(f"_{t}_" in padded for t in _worker_dependency("_rate_natured_bases")())


def _ir_denotes_time_change(_worker_dependency, /, ir: Any) -> bool:
    """Recursively find temporal-change semantics through the public ISN API."""
    from imas_standard_names import get_operator_semantics

    def denotes_time_change(token: Any) -> bool:
        return isinstance(token, str) and "temporal_change" in get_operator_semantics(
            token
        )

    for operator in getattr(ir, "operators", ()) or ():
        if denotes_time_change(getattr(operator, "op", None)):
            return True
        if any(
            _worker_dependency("_ir_denotes_time_change")(argument)
            for argument in getattr(operator, "args", ()) or ()
        ):
            return True
    return any(
        denotes_time_change(getattr(qualifier, "token", None))
        for qualifier in getattr(ir, "qualifiers", ()) or ()
    )


def _ir_claims_time_change_in_scope(_worker_dependency, /, ir: Any) -> bool:
    """True when a temporal operator governs *ir*'s own tense, scope-aware.

    A binary operator (``difference_of``, ``ratio_of``, ...) combines two
    independently-scoped operand expressions into one result whose tense is
    set by the binary operator itself, not by either operand: a difference of
    a power and a stored-energy time derivative is dimensionally a power (the
    derivative already carries the rate; the outer subtraction does not apply
    a second one). So a temporal operator buried inside ONE operand of a
    binary construction must not make the whole name read as a rate. A
    temporal operator anywhere in a non-binary (unary) chain still does,
    since a unary wrapper (``magnitude_of``, ``gradient_of``, ...) does not
    introduce an independent operand scope of its own.
    """
    from imas_standard_names import get_operator_semantics

    def denotes_time_change(token: Any) -> bool:
        return isinstance(token, str) and "temporal_change" in get_operator_semantics(
            token
        )

    for operator in getattr(ir, "operators", ()) or ():
        if denotes_time_change(getattr(operator, "op", None)):
            return True
        if getattr(operator, "kind", None) == "binary":
            continue
        if any(
            _worker_dependency("_ir_claims_time_change_in_scope")(argument)
            for argument in getattr(operator, "args", ()) or ()
        ):
            return True
    return any(
        denotes_time_change(getattr(qualifier, "token", None))
        for qualifier in getattr(ir, "qualifiers", ()) or ()
    )


def _name_claims_time_change(_worker_dependency, /, sn_name: str) -> bool:
    """Interpret temporal semantics structurally, with a legacy-safe fallback.

    A retired operator spelling is rewritten to its registered token first, so
    a name using the retired prefix still reads as claiming the temporal
    semantics the grammar continues to publish for it under that token.
    """
    from imas_codex.standard_names.audits import resolve_retired_operator_spellings

    sn_name = resolve_retired_operator_spellings(sn_name)
    try:
        from imas_codex.standard_names.grammar_adapter import parse_canonical_name

        return _worker_dependency("_ir_denotes_time_change")(
            parse_canonical_name(sn_name).ir
        )
    except (TypeError, ValueError):
        from imas_standard_names import get_operator_semantics

        parts = sn_name.split("_")
        return any(
            "temporal_change" in get_operator_semantics("_".join(parts[start:end]))
            for start in range(len(parts))
            for end in range(start + 1, len(parts) + 1)
        )


def _time_change_claim_is_binary_scoped(_worker_dependency, /, sn_name: str) -> bool:
    """True when *sn_name*'s ONLY temporal claim sits inside one operand of a
    binary construction (``difference_of``, ``ratio_of``, ...), not in the
    name's own outer operator scope.

    Distinguishes ``time_derivative_of_stored_energy`` (outer scope is
    temporal — still a rate) from
    ``difference_of_heating_power_and_time_derivative_of_stored_energy``
    (outer scope is the binary ``difference`` — the temporal operator only
    governs its own operand). Returns ``False`` — never scoped, keep the
    conservative reading — when strict parsing fails, matching the legacy
    fallback in :func:`_name_claims_time_change`.
    """
    from imas_codex.standard_names.audits import resolve_retired_operator_spellings

    resolved = resolve_retired_operator_spellings(sn_name)
    try:
        from imas_codex.standard_names.grammar_adapter import parse_canonical_name

        ir = parse_canonical_name(resolved).ir
    except (TypeError, ValueError):
        return False
    return _worker_dependency("_ir_denotes_time_change")(ir) and (
        not _worker_dependency("_ir_claims_time_change_in_scope")(ir)
    )


def _shape_parameter_surface(_worker_dependency, /, dd_path: str | None) -> str:
    """Surface a shape parameter from *dd_path* is defined *of*.

    - ``profiles_1d`` / ``flux_surface`` paths → ``flux_surface`` (radial profile)
    - all other paths (``boundary``, ``boundary_separatrix``, pulse-schedule
      control targets, …) → ``plasma_boundary`` (the LCFS)

    The default is the plasma boundary: "the triangularity" without further
    qualification conventionally means the boundary/separatrix value.
    """
    p = (dd_path or "").lower()
    if "/profiles_1d/" in p or "/flux_surface/" in p:
        return "flux_surface"
    return "plasma_boundary"


def _inject_shape_parameter_surface(
    _worker_dependency,
    /,
    candidate: Any,
    dd_path: str | None,
    log: logging.Logger | None = None,
) -> bool:
    """Force a surface locus onto a bare shape-parameter candidate.

    Mutates ``candidate.segments`` in place so ``compose_name()`` produces the
    surface-explicit leaf. Deterministic and idempotent: fires only when the
    base is a shape parameter (quantity kind) and the composer left the locus
    empty; a composer-supplied locus is always preserved. Returns ``True`` when
    an injection was applied.
    """
    seg = getattr(candidate, "segments", None)
    if seg is None:
        return False
    if (
        seg.base_kind != "quantity"
        or seg.base_token not in _worker_dependency("_shape_parameter_bases")()
    ):
        return False
    if seg.locus_token:
        return False
    surface = _worker_dependency("_shape_parameter_surface")(dd_path)
    seg.locus_token = surface
    seg.locus_relation = "of"
    seg.locus_type = "geometry"
    if log:
        log.debug(
            "shape-param surface injection: base=%s src=%s -> _of_%s",
            seg.base_token,
            dd_path,
            surface,
        )
    return True


_SNAKE_CASE_RE = _re.compile("[a-z][a-z0-9_]*")


def is_well_formed_candidate(
    _worker_dependency, /, raw_name: str
) -> tuple[bool, str | None]:
    """Check whether *raw_name* is a plausible standard-name candidate.

    Returns ``(True, None)`` if the name passes all checks, or
    ``(False, reason)`` if it should be rejected **before** MERGE.

    Rejection criteria:
    - Empty or whitespace-only
    - Contains newline, tab, ``{``, ``}``, ``\\``, or triple-dot
    - Not snake_case-shaped after stripping
    """
    if not raw_name or not raw_name.strip():
        return (False, "empty_or_whitespace")
    name = raw_name.strip()
    if any(ch in name for ch in ("\n", "\t", "{", "}", "\\")):
        return (False, "illegal_chars")
    if "..." in name:
        return (False, "triple_dot")
    if not _worker_dependency("_SNAKE_CASE_RE").fullmatch(name):
        return (False, "not_snake_case")
    return (True, None)


def _claimed_source_ids(_worker_dependency, /, batch: list[dict[str, Any]]) -> set[str]:
    """Return the authoritative bare source ids carried by a claimed batch."""
    return {
        source_id
        for item in batch
        if (source_id := (item.get("path") or item.get("source_id")))
    }


def _strip_response_source_id(_worker_dependency, /, source_id: Any) -> Any:
    """Remove only surrounding whitespace from a model-proposed source id."""
    return source_id.strip() if isinstance(source_id, str) else source_id


def _review_dd_source_bindings(
    _worker_dependency, /, item: dict[str, Any]
) -> dict[str, str | None]:
    """Return exact DD paths and versions from authoritative source edges."""
    bindings: dict[str, str | None] = {}
    for binding in item.get("source_bindings") or []:
        if not isinstance(binding, dict) or binding.get("source_type") != "dd":
            continue
        path = binding.get("dd_path") or binding.get("source_id")
        if isinstance(path, str) and (
            path := _worker_dependency("strip_dd_prefix")(path)
        ):
            bindings[path] = binding.get("dd_version")
    return bindings


def _review_dd_source_ids(_worker_dependency, /, item: dict[str, Any]) -> set[str]:
    """Return exact DD paths authorized by source-binding relationships."""
    return set(_worker_dependency("_review_dd_source_bindings")(item))


def _sanitize_dd_gap_evidence(
    _worker_dependency,
    /,
    evidence: list[Any],
    allowed_source_ids: set[str],
    *,
    phase: str,
) -> list[Any]:
    """Fence DD evidence to exact paths owned by the current claimed batch."""
    kept: list[Any] = []
    rejected: set[str] = set()
    for report in evidence:
        if isinstance(report, dict):
            path = _worker_dependency("_strip_response_source_id")(report.get("path"))
            reference_path = _worker_dependency("_strip_response_source_id")(
                report.get("reference_path")
            )
            report["path"] = path
            if "reference_path" in report:
                report["reference_path"] = reference_path
        else:
            path = _worker_dependency("_strip_response_source_id")(
                getattr(report, "path", None)
            )
            reference_path = _worker_dependency("_strip_response_source_id")(
                getattr(report, "reference_path", None)
            )
            report.path = path
            if hasattr(report, "reference_path"):
                report.reference_path = reference_path
        report_paths = {value for value in (path, reference_path) if value}
        if path not in allowed_source_ids or not report_paths.issubset(
            allowed_source_ids
        ):
            rejected.update(str(value) for value in report_paths)
            continue
        kept.append(report)
    if rejected:
        logger.warning(
            "Pool %s: ignored %d DD-gap path proposal(s) outside the claimed batch: %s",
            phase,
            len(rejected),
            ", ".join(sorted(rejected)[:8]),
        )
    return kept


def _persist_dd_gap_evidence(
    _worker_dependency,
    /,
    evidence: list[Any],
    allowed_source_ids: set[str],
    *,
    phase: str,
    reporter: str,
    observed_dd_version: str | None = None,
    source_versions: dict[str, str | None] | None = None,
) -> int:
    """Best-effort evidence persistence, isolated from pipeline outcomes."""
    fenced = _worker_dependency("_sanitize_dd_gap_evidence")(
        evidence, allowed_source_ids, phase=phase
    )
    if not fenced:
        return 0
    reports: list[dict[str, Any]] = []
    for evidence_item in fenced:
        if hasattr(evidence_item, "model_dump"):
            report = evidence_item.model_dump(exclude_none=True)
        else:
            report = dict(evidence_item)
        report["reporter"] = reporter
        binding_version = (source_versions or {}).get(str(report.get("path") or ""))
        if binding_version:
            report["observed_dd_version"] = binding_version
        elif observed_dd_version and (not report.get("observed_dd_version")):
            report["observed_dd_version"] = observed_dd_version
        reports.append(report)
    try:
        from imas_codex.standard_names.dd_gaps import write_dd_gaps

        result = write_dd_gaps(reports)
        return int(result.get("reported", 0))
    except Exception:
        logger.warning(
            "Pool %s: DD-gap evidence persistence failed without changing the pipeline outcome",
            phase,
            exc_info=True,
        )
        return 0


def _claimed_source_outcome(
    _worker_dependency,
    /,
    batch: list[dict[str, Any]],
    source_id: str,
    *,
    source_type: str,
    status: str,
    last_error: str | None = None,
    skip_reason: str | None = None,
    skip_reason_detail: str | None = None,
) -> dict[str, Any] | None:
    """Build an exact fenced outcome for one source in a claimed batch."""
    item = next(
        (
            claimed
            for claimed in batch
            if (claimed.get("path") or claimed.get("source_id")) == source_id
        ),
        None,
    )
    if not item:
        return None
    token = item.get("claim_token")
    seq = item.get("claim_seq")
    if not token or seq is None:
        return None
    return {
        "sns_id": f"{source_type}:{source_id}",
        "source_id": source_id,
        "claim_token": token,
        "claim_seq": seq,
        "status": status,
        "last_error": last_error,
        "skip_reason": skip_reason,
        "skip_reason_detail": skip_reason_detail,
    }


def _sanitize_compose_result_sources(
    _worker_dependency, /, result: Any, allowed_source_ids: set[str], *, phase: str
) -> None:
    """Fence every compose-result source projection to the claimed batch.

    The prompt contains related and cross-IDS paths for context, so an LLM can
    legitimately mention an unclaimed sibling. Context is not authorization:
    only the sources in the graph-claimed batch may be attached, skipped,
    marked with a vocabulary gap, or persisted through a candidate.
    """
    rejected: set[str] = set()
    kept_candidates = []
    for candidate in result.candidates:
        candidate.source_id = _worker_dependency("_strip_response_source_id")(
            candidate.source_id
        )
        candidate.dd_paths = [
            _worker_dependency("_strip_response_source_id")(path)
            for path in candidate.dd_paths
        ]
        if candidate.source_id not in allowed_source_ids:
            rejected.add(candidate.source_id)
            continue
        out_of_batch_paths = set(candidate.dd_paths) - allowed_source_ids
        rejected.update(out_of_batch_paths)
        candidate.dd_paths = [
            path for path in candidate.dd_paths if path in allowed_source_ids
        ]
        kept_candidates.append(candidate)
    result.candidates = kept_candidates
    kept_attachments = []
    for attachment in result.attachments:
        attachment.source_id = _worker_dependency("_strip_response_source_id")(
            attachment.source_id
        )
        if attachment.source_id in allowed_source_ids:
            kept_attachments.append(attachment)
        else:
            rejected.add(attachment.source_id)
    result.attachments = kept_attachments
    result.skipped = [
        _worker_dependency("_strip_response_source_id")(source_id)
        for source_id in result.skipped
    ]
    rejected.update(set(result.skipped) - allowed_source_ids)
    result.skipped = [sid for sid in result.skipped if sid in allowed_source_ids]
    kept_gaps = []
    for gap in result.vocab_gaps:
        gap.source_id = _worker_dependency("_strip_response_source_id")(gap.source_id)
        if gap.source_id in allowed_source_ids:
            kept_gaps.append(gap)
        else:
            rejected.add(gap.source_id)
    result.vocab_gaps = kept_gaps
    selected: dict[str, str] = {}
    overlaps: set[str] = set()

    def _reserve(source_id: str, disposition: str) -> bool:
        owner = selected.setdefault(source_id, disposition)
        if owner != disposition:
            overlaps.add(source_id)
            return False
        return True

    result.vocab_gaps = [
        gap for gap in result.vocab_gaps if _reserve(gap.source_id, "vocab_gap")
    ]
    result.attachments = [
        attachment
        for attachment in result.attachments
        if _reserve(attachment.source_id, "attachment")
    ]
    result.skipped = [
        source_id for source_id in result.skipped if _reserve(source_id, "skip")
    ]
    result.candidates = [
        candidate
        for candidate in result.candidates
        if _reserve(candidate.source_id, "generated")
    ]
    if overlaps:
        logger.warning(
            "Pool %s: resolved %d multi-disposition source(s) by fixed priority: %s",
            phase,
            len(overlaps),
            ", ".join(sorted(overlaps)[:8]),
        )
    result.dd_gaps = _worker_dependency("_sanitize_dd_gap_evidence")(
        list(getattr(result, "dd_gaps", None) or []), allowed_source_ids, phase=phase
    )
    if rejected:
        logger.warning(
            "Pool %s: ignored %d out-of-batch source proposal(s): %s",
            phase,
            len(rejected),
            ", ".join(sorted(rejected)[:8]),
        )


def _filter_persist_candidates_to_claimed_sources(
    _worker_dependency,
    /,
    candidates: list[dict[str, Any]],
    allowed_source_ids: set[str],
    *,
    phase: str,
) -> list[dict[str, Any]]:
    """Keep only candidates whose own source was separately claimed."""
    kept = [
        candidate
        for candidate in candidates
        if candidate.get("source_id") in allowed_source_ids
    ]
    rejected = len(candidates) - len(kept)
    if rejected:
        logger.warning(
            "Pool %s: deferred %d candidate(s) without a source claim", phase, rejected
        )
    return kept


def _withhold_binding_conflicted_sources(
    _worker_dependency,
    /,
    conflict: object,
    candidates: list[dict[str, Any]],
    *,
    source_type: str,
    phase: str,
    compose_model: str,
    dd_version: str | None,
    cocos_version: int | None,
    run_id: str | None,
) -> list[str]:
    """Record a compose refusal for a source already bound to a live identity.

    A source held by a live identity outside the migration's predecessor is an
    adjudicable state: recomposing cannot settle it, so releasing the claim to
    retry would spend an attempt on a decision only an operator can make. The
    same holds for a source bound to the live predecessor and a successor that
    has itself been retired — one live claim beside a settled one. The withheld
    source is left claimed, its holding identity and stage recorded (and the
    retired identity where one applies), and no winner id is returned for it.
    Every other candidate in the batch persists normally.
    """
    conflicted_source_ids = {
        item.source_id for item in getattr(conflict, "conflicts", ())
    }
    for item in getattr(conflict, "conflicts", ()):
        retired_identity = getattr(item, "retired_identity", None)
        if retired_identity is None:
            logger.warning(
                "Pool %s: source %s bound to live identity %r at stage %r — withheld for adjudication, no attempt spent",
                phase,
                item.source_id,
                item.holding_identity,
                item.holding_stage,
            )
        else:
            logger.warning(
                "Pool %s: source %s bound to live identity %r at stage %r alongside retired identity %r at stage %r — withheld for adjudication, no attempt spent",
                phase,
                item.source_id,
                item.holding_identity,
                item.holding_stage,
                retired_identity,
                getattr(item, "retired_stage", None),
            )
    remaining = [
        candidate
        for candidate in candidates
        if _worker_dependency("_candidate_source_node_id")(candidate, source_type)
        not in conflicted_source_ids
    ]
    if not remaining:
        return []
    from imas_codex.standard_names.graph_store.graph_catalog import (
        persist_generated_name_batch,
    )

    return persist_generated_name_batch(
        remaining,
        compose_model=compose_model,
        dd_version=dd_version,
        cocos_version=cocos_version,
        run_id=run_id,
        return_winner_ids=True,
    )


def _candidate_source_node_id(
    _worker_dependency, /, candidate: dict[str, Any], source_type: str
) -> str:
    source_id = candidate.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        return ""
    if source_type == "dd" and source_id.startswith("dd:"):
        return source_id
    return _worker_dependency("encode_source_path")(source_type, source_id)


def _persist_description_checked_candidates(
    _worker_dependency,
    /,
    candidates: list[dict[str, Any]],
    *,
    source_type: str,
    phase: str,
    compose_model: str,
    dd_version: str | None,
    cocos_version: int | None,
    run_id: str | None,
) -> int | list[str]:
    """Persist only candidates with prose, releasing rejected claims to retry.

    The structured response contract requires a non-empty description, but
    callers and test doubles can still supply unvalidated candidate-shaped
    objects. This final boundary prevents those objects from creating a drafted name
    before releasing each exact source claim through the canonical failure
    path. A source with any descriptionless proposal is entirely deferred so
    duplicate proposals cannot race the release with a richer sibling.
    """
    from imas_codex.standard_names.graph_store.graph_catalog import (
        persist_generated_name_batch,
    )
    from imas_codex.standard_names.graph_store.graph_claim_release import (
        release_generate_name_failed_claims,
    )

    rejected_source_ids = {
        candidate.get("source_id")
        for candidate in candidates
        if not isinstance(candidate.get("description"), str)
        or not candidate["description"].strip()
    }
    rejected_source_ids.discard(None)
    rejected_count = sum(
        1
        for candidate in candidates
        if not isinstance(candidate.get("description"), str)
        or not candidate["description"].strip()
    )
    releases_by_token: dict[str, set[str]] = {}
    for candidate in candidates:
        source_id = candidate.get("source_id")
        if source_id not in rejected_source_ids:
            continue
        claim_token = candidate.get("source_claim_token")
        if not isinstance(source_id, str) or not source_id or (not claim_token):
            continue
        source_node_id = _worker_dependency("_candidate_source_node_id")(
            candidate, source_type
        )
        releases_by_token.setdefault(str(claim_token), set()).add(source_node_id)
    released = 0
    release_expected = sum(len(ids) for ids in releases_by_token.values())
    for claim_token, source_ids in sorted(releases_by_token.items()):
        released += release_generate_name_failed_claims(
            source_ids=sorted(source_ids), claim_token=claim_token
        )
    if rejected_count:
        logger.warning(
            "Pool %s: rejected %d candidate(s) with empty descriptions; released %d/%d exact source claim(s) for retry",
            phase,
            rejected_count,
            released,
            release_expected,
        )
    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("source_id") not in rejected_source_ids
        and isinstance(candidate.get("description"), str)
        and candidate["description"].strip()
    ]
    if not eligible:
        return []
    from imas_codex.standard_names.provenance_lifecycle import (
        SourceBindingConflictError,
        SourceRetiredSuccessorConflictError,
    )

    try:
        return persist_generated_name_batch(
            eligible,
            compose_model=compose_model,
            dd_version=dd_version,
            cocos_version=cocos_version,
            run_id=run_id,
            return_winner_ids=True,
        )
    except (
        SourceBindingConflictError,
        SourceRetiredSuccessorConflictError,
    ) as conflict:
        return _worker_dependency("_withhold_binding_conflicted_sources")(
            conflict,
            eligible,
            source_type=source_type,
            phase=phase,
            compose_model=compose_model,
            dd_version=dd_version,
            cocos_version=cocos_version,
            run_id=run_id,
        )


def _mark_source_prevalidation_failed(
    _worker_dependency,
    /,
    source_id: str,
    *,
    source_type: str,
    candidate_name: str,
    reason: str,
    claim_token: str | None = None,
    claim_seq: int | None = None,
) -> bool:
    """Persist a terminal pre-validation failure with actionable diagnostics."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        persist_claimed_source_outcomes,
    )

    error = f"candidate pre-validation failed ({reason}): {candidate_name}"
    if not claim_token or claim_seq is None:
        return False
    winners = persist_claimed_source_outcomes(
        [
            {
                "sns_id": f"{source_type}:{source_id}",
                "claim_token": claim_token,
                "claim_seq": claim_seq,
                "status": "failed",
                "last_error": error,
            }
        ]
    )
    return bool(winners)


def provenance_canonical_names(
    _worker_dependency, /, candidates: list, log: logging.Logger | None = None
) -> dict[str, str]:
    """Map each base quantity to ONE canonical name for its estimator facets.

    Per the locked ``name-multiplicity-rule`` (collapse provenance), the
    measured / reconstructed / reference estimates of one physical quantity must
    share ONE standard name — but the per-facet composer is non-deterministic
    (e.g. ``pressure/measured`` -> ``plasma_pressure`` while
    ``pressure/reconstructed`` -> ``total_plasma_pressure``). Estimator facets of
    a quantity strip to the same ``base_path`` (and share a compose batch via
    ``batch_key``), so we group every value-provenance candidate by base and
    pick one canonical name; the caller overrides each facet's ``name_id`` (the
    MERGE identity) with it, collapsing the group onto a single StandardName.

    Canonical pick: most-frequently-composed name, ties broken by SHORTEST then
    lexicographic. Shortest implements the "physics, not instrument" /
    no-redundant-qualifier preference deterministically — it drops
    ``_of_flux_loop`` (vs ``poloidal_magnetic_flux``) and ``total_plasma_pressure``
    (vs ``plasma_pressure``). Returns ``{base_path: canonical_name}`` only for
    bases whose facets disagree (a no-op map otherwise).
    """
    from collections import Counter

    groups: dict[str, list[str]] = {}
    for c in candidates:
        term, base = _worker_dependency("detect_value_provenance")(
            getattr(c, "source_id", "") or ""
        )
        if not term:
            continue
        try:
            nm = _worker_dependency("normalize_spelling")(c.compose_name())
        except Exception:
            continue
        groups.setdefault(base, []).append(nm)
    canonical: dict[str, str] = {}
    for base, member_names in groups.items():
        distinct = set(member_names)
        if len(distinct) <= 1:
            continue
        counts = Counter(member_names)
        winner = min(distinct, key=lambda n: (-counts[n], len(n), n))
        canonical[base] = winner
        if log:
            log.info(
                "provenance canon: base=%s -> %s (collapsed %d variants: %s)",
                base,
                winner,
                len(distinct),
                sorted(distinct),
            )
    return canonical


_NON_NAMEABLE_BARE_TOKENS: frozenset[str] = frozenset(
    {
        "time",
        "time_stamp",
        "timestamp",
        "delay",
        "latency",
        "dead_time",
        "index",
        "count",
        "counter",
        "version",
    }
)


def is_non_nameable_coordinate(
    _worker_dependency, /, raw_name: str
) -> tuple[bool, str | None]:
    """Detect bare coordinate / infrastructure tokens that get no standard name.

    Returns ``(True, reason)`` when *raw_name* is a bare non-nameable token
    (a time coordinate / timestamp, signal-chain timing, an array counter, or
    pure metadata) that should be routed to ``skipped`` rather than composed.
    Returns ``(False, None)`` otherwise.

    This is intentionally conservative: it only fires on *bare* tokens (no
    qualifier/locus context). A qualified name like ``confinement_time`` or
    ``major_radius_of_magnetic_axis`` is a real quantity and passes through.
    """
    if not raw_name:
        return (False, None)
    name = raw_name.strip()
    if name in _worker_dependency("_NON_NAMEABLE_BARE_TOKENS"):
        return (True, f"non_nameable_coordinate:{name}")
    return (False, None)


def _coerce_llm_metric(_worker_dependency, /, value: Any) -> int:
    """Best-effort int coercion for LLM telemetry fields."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _extract_llm_telemetry(
    _worker_dependency,
    /,
    llm_out: Any,
    *,
    parsed_obj: Any = None,
    tokens_fallback: Any = None,
) -> tuple[int, int, int, int]:
    """Normalize split token/cache telemetry across LLM return shapes."""

    def _from_attrs(obj: Any) -> tuple[int, int, int, int]:
        return (
            _worker_dependency("_coerce_llm_metric")(getattr(obj, "input_tokens", 0)),
            _worker_dependency("_coerce_llm_metric")(getattr(obj, "output_tokens", 0)),
            _worker_dependency("_coerce_llm_metric")(
                getattr(obj, "cache_read_tokens", 0)
            ),
            _worker_dependency("_coerce_llm_metric")(
                getattr(obj, "cache_creation_tokens", 0)
            ),
        )

    metrics = _from_attrs(llm_out)
    if any(metrics):
        return metrics
    if parsed_obj is not None:
        metrics = _from_attrs(parsed_obj)
        if any(metrics):
            return metrics
    if isinstance(tokens_fallback, dict):
        return (
            _worker_dependency("_coerce_llm_metric")(
                tokens_fallback.get("input_tokens", 0)
            ),
            _worker_dependency("_coerce_llm_metric")(
                tokens_fallback.get("output_tokens", 0)
            ),
            _worker_dependency("_coerce_llm_metric")(
                tokens_fallback.get("cache_read_tokens", 0)
            ),
            _worker_dependency("_coerce_llm_metric")(
                tokens_fallback.get(
                    "cache_creation_tokens",
                    tokens_fallback.get("cache_write_tokens", 0),
                )
            ),
        )
    if isinstance(tokens_fallback, tuple | list):
        values = list(tokens_fallback)
        return (
            _worker_dependency("_coerce_llm_metric")(
                values[0] if len(values) > 0 else 0
            ),
            _worker_dependency("_coerce_llm_metric")(
                values[1] if len(values) > 1 else 0
            ),
            _worker_dependency("_coerce_llm_metric")(
                values[2] if len(values) > 2 else 0
            ),
            _worker_dependency("_coerce_llm_metric")(
                values[3] if len(values) > 3 else 0
            ),
        )
    return (0, 0, 0, 0)


@_cache
def _load_known_physical_bases(_worker_dependency, /) -> frozenset[str]:
    """Return the set of registered physical_base tokens from ISN.

    Reads the ``grammar/vocabularies/physical_bases.yml`` vocabulary file
    from the installed ``imas_standard_names`` package.  Falls back to an
    empty set if the package or file is unavailable.
    """
    try:
        from importlib.resources import files

        import yaml

        text = (
            files("imas_standard_names")
            .joinpath("grammar/vocabularies/physical_bases.yml")
            .read_text()
        )
        data = yaml.safe_load(text)
        bases = data.get("bases", {})
        if isinstance(bases, dict):
            return frozenset(bases.keys())
        return frozenset()
    except Exception:
        return frozenset()


def _auto_detect_physical_base_gaps(
    _worker_dependency, /, candidates: list, known_bases: frozenset[str] | None = None
) -> list[dict]:
    """Parse each candidate name and surface novel ``physical_base`` tokens.

    Returns a list of VocabGap-compatible dicts for bases not in the ISN
    registry.  These are auto-tracked so the LLM does not need to emit
    explicit ``vocab_gap`` exits for ``physical_base``.

    Args:
        candidates: Parsed ``StandardNameCandidate`` objects or dicts.
        known_bases: Pre-loaded set of registered physical_base tokens.
            Defaults to ``_load_known_physical_bases()`` if not provided.
            The flat registry under-reports the grammar's lexical-compound
            bases (``internal_inductance``, ``major_radius`` …); on the default
            path a base absent from the flat set is re-checked against the ISN
            parser and emitted as a gap only when the parser also rejects it.
            An explicit ``known_bases`` is treated as authoritative and the
            parser escape hatch is NOT applied.
    """
    from imas_codex.standard_names.segments import is_known_physical_base

    consult_parser = known_bases is None
    if known_bases is None:
        known_bases = _worker_dependency("_load_known_physical_bases")()
    gaps: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for c in candidates:
        try:
            if hasattr(c, "base_token"):
                base = c.base_token
                source_id = c.source_id
                base_kind = getattr(c, "base_kind", "quantity")
            elif isinstance(c, dict):
                base = c.get("id", "")
                source_id = c.get("source_id", "")
                try:
                    from imas_standard_names.grammar import parse_standard_name

                    parsed = parse_standard_name(base)
                    base = parsed.physical_base
                    base_kind = "quantity"
                except Exception:
                    continue
            else:
                continue
            if base_kind == "quantity" and base and (base not in known_bases):
                if consult_parser and is_known_physical_base(base):
                    continue
                key = (source_id, base)
                if key not in seen:
                    seen.add(key)
                    gaps.append(
                        {
                            "source_id": source_id,
                            "segment": "physical_base",
                            "token": base,
                            "reason": f"Novel physical_base proposed by compose: {base!r}",
                        }
                    )
        except Exception:
            continue
    return gaps


def _compute_token_reuse_hits(
    _worker_dependency, /, vocab_gaps: list
) -> dict[tuple[str, str], Any]:
    """Flag candidate-new tokens that duplicate a registered same-segment token.

    For each ``vocab_gap`` the compose LLM emitted this attempt, embeds the
    proposed token and compares it against its segment's registered vocabulary
    (free local embeddings, batched by segment so each segment's vocab is
    embedded once).  Returns a map ``(segment, token) -> TokenNeighbour`` for
    every proposal scoring at/above ``get_sn_dedup_threshold()``.

    Best-effort: never raises (the helper itself is defensive); the local
    compose model + local embeddings add no OpenRouter spend.  Returns ``{}``
    when there are no gaps or no near-synonym registered token.
    """
    if not vocab_gaps:
        return {}
    from imas_codex.settings import get_sn_dedup_threshold
    from imas_codex.standard_names.vocab_semantic_dedup import nearest_registered_tokens

    threshold = get_sn_dedup_threshold()
    if threshold > 1.0:
        return {}
    items = sorted({(vg.token, vg.segment) for vg in vocab_gaps})
    hits = nearest_registered_tokens(items, threshold=threshold)
    return {(hit.segment, hit.proposed_token): hit for hit in hits.values()}


def _grammar_round_trip_failures(
    _worker_dependency, /, candidates: list[Any]
) -> tuple[list[str], list[str]]:
    """Names that fail the grammar round trip, with per-failure repair advice.

    Returns ``(names, advice)``.  ``names`` identifies what failed (the composed
    name, or the source id when even composing it raises).  ``advice`` holds one
    actionable line per diagnosable failure.

    The diagnosis comes from the public parser's structured residue. For an
    unresolved base, ``describe_gap`` turns that residue into the slot to use
    instead — that ``square`` is an operator rather than a qualifier, say.
    Reporting only the failed name gives the composer nothing to change, so it
    re-proposes the same token on every retry.

    Vocabulary suggestions are deliberately NOT surfaced: the seat's prompt
    already carries the full vocabulary, and repeating it would crowd out the
    part that is new information — which token, and which slot.
    """
    names: list[str] = []
    advice: list[str] = []
    try:
        from imas_standard_names import ParseError

        from imas_codex.standard_names.grammar_adapter import parse_canonical_name
        from imas_codex.standard_names.segments import describe_gap
    except ImportError:
        return (names, advice)
    for candidate in candidates:
        candidate_name: str | None = None
        parse_error: ParseError | None = None
        try:
            candidate_name = candidate.compose_name()
            parse_canonical_name(candidate_name)
            continue
        except ParseError as exc:
            parse_error = exc
        except Exception:
            pass
        if parse_error is not None and parse_error.residue:
            try:
                diagnosis = describe_gap("physical_base", parse_error.residue).guidance
            except Exception:
                logger.debug(
                    "could not describe grammar failure for %s",
                    getattr(candidate, "source_id", "?"),
                    exc_info=True,
                )
            else:
                if diagnosis:
                    advice.append(diagnosis)
        try:
            names.append(candidate_name or candidate.compose_name())
        except Exception:
            names.append(candidate.source_id)
    return (names, advice)


def _build_grammar_retry_reason(
    _worker_dependency, /, failures: list[str], advice: list[str]
) -> str:
    """Render the grammar half of a retry directive.

    Leads with the repair advice when there is any, because that is what lets the
    composer converge; the bare "produce a different name" fallback only applies
    when nothing about the failure could be diagnosed.
    """
    if advice:
        return (
            "Previous attempt failed the grammar round trip. Fix each:\n"
            + "\n".join(f"- {line}" for line in dict.fromkeys(advice))
        )
    return f"Previous attempt failed: grammar round-trip failed for {', '.join(failures[:3])}. Consider expanded neighbour context and produce a different name."


def _build_token_reuse_retry_reason(
    _worker_dependency, /, hits: dict[tuple[str, str], Any]
) -> str:
    """Render a retry directive naming each flagged candidate-new token.

    The agent re-composes after seeing this: it either reuses the registered
    token (dedup achieved) or re-emits the same gap (confirming the quantity is
    genuinely distinct on that segment's axis, recorded as distinct-confirmed).
    """
    parts = []
    for hit in hits.values():
        parts.append(
            f"Segment `{hit.segment}`: your proposed new token `{hit.proposed_token}` is {hit.similarity:.2f} similar to registered `{hit.nearest_token}`. Reuse the registered token unless this quantity is genuinely DISTINCT on the {hit.segment} axis; if distinct, keep it and it will be recorded as a new-vocab request."
        )
    return "Vocabulary reuse check — re-compose to reuse-or-confirm:\n" + "\n".join(
        parts
    )


def _active_editorial_gap_guidance(
    _worker_dependency, /, vocab_gaps: list[Any]
) -> tuple[str, bool]:
    """Read reviewed decisions for only this result and render exact guidance."""
    if not vocab_gaps:
        return ("", False)
    gap_dicts = [
        {"segment": getattr(gap, "segment", None), "token": getattr(gap, "token", None)}
        for gap in vocab_gaps
    ]
    try:
        from imas_codex.standard_names.graph_store.graph_catalog_reset import (
            fetch_vocab_gap_adjudications,
        )
        from imas_codex.standard_names.vocab_adjudication import (
            editorial_retry_guidance,
        )

        rows = fetch_vocab_gap_adjudications(gap_dicts)
        return editorial_retry_guidance(rows)
    except Exception:
        logger.debug("Editorial vocabulary guidance lookup failed", exc_info=True)
        return ("", False)


def _stamp_dedup_decision(
    _worker_dependency, /, gap_dicts: list[dict], hits: dict[tuple[str, str], Any]
) -> None:
    """Stamp the compose-time token-reuse adjudication onto gap dicts in place.

    A gap whose ``(segment, token)`` was flagged and SURVIVED the chained retry
    (the agent re-emitted it after being shown the near-token) is
    ``distinct_confirmed`` and carries the nearest token + score.  Gaps that
    were never flagged are ``unchecked``.  (Gaps the agent dropped after the
    prompt — "reused" — are no longer in ``result.vocab_gaps`` and so are never
    written.)
    """
    from imas_codex.graph.models import VocabGapDedupDecision

    settled = {
        VocabGapDedupDecision.reuse_confirmed.value,
        VocabGapDedupDecision.distinct_confirmed.value,
        VocabGapDedupDecision.checked_no_reuse.value,
    }
    for g in gap_dicts:
        if g.get("dedup_decision") in settled:
            continue
        hit = hits.get((g.get("segment"), g.get("token")))
        if hit is not None:
            g["nearest_token"] = hit.nearest_token
            g["nearest_similarity"] = hit.similarity
            g["dedup_decision"] = VocabGapDedupDecision.distinct_confirmed.value
        else:
            g.setdefault("dedup_decision", VocabGapDedupDecision.unchecked.value)
