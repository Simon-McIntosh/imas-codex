"""Assign a closed-vocabulary quantity, unit and role to a plant signal by rule.

The three vocabularies are the schema enums ``SignalQuantity``,
``SignalVocabularyUnit`` and ``SignalVocabularyRole``. A signal's value on one
axis comes only from a source that states it, and four sources are consulted in
this precedence: the linked signal-list row, the controls metadata the signal
carries, the unit the archive reports (unit axis only), and the field pattern of
the signal's name. A source that says nothing about an axis gives no value; a
source whose text names two members of the axis is ambiguous and gives none.

Each axis has exactly one outcome. ``decided`` stores the member and credits the
first source in precedence that gave it; ``conflict`` stores nothing when two
sources give different members; ``undecided`` stores nothing when no source
gives a member. Nothing here infers a meaning: every term, unit spelling and
pattern is an explicit rule, and a value no rule reaches stays undecided.

The field-pattern rule is the only rule that reads the population rather than
one signal: other signals that share the signal's field code contribute their
own decided value, and the pattern states a member only when at least
``min_pattern_support`` of them agree unanimously. A signal never supplies its
own pattern, so a value is never supported by itself, and a consensus drawn from
a single device family does not transfer to an accession of another family.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from iter_codex.graph.models import (
    SignalQuantity,
    SignalVocabularyRole,
    SignalVocabularySource,
    SignalVocabularyUnit,
)
from iter_codex.units import normalize_unit_symbol

AXES = ("quantity", "unit", "role")

# Precedence: the first source in this order that gives the stored member is
# credited. ``archive_unit`` gives the unit axis only.
SOURCE_ORDER = (
    SignalVocabularySource.signal_list_row.value,
    SignalVocabularySource.controls_metadata.value,
    SignalVocabularySource.archive_unit.value,
    SignalVocabularySource.field_pattern.value,
)
UNIT_ONLY_SOURCES = frozenset({SignalVocabularySource.archive_unit.value})

# Every canonical unit symbol a member may be written as. The shared unit
# authority decides equivalence; these spellings only seed the lookup, so a
# source spelling reaches a member exactly when both parse to one unit.
UNIT_SPELLINGS: dict[str, tuple[str, ...]] = {
    "kelvin": ("K",),
    "degree_celsius": ("degC", "°C"),
    "volt": ("V",),
    "ohm": ("ohm", "Ω"),
    "millibar": ("mbar",),
    "bar": ("bar",),
    "megapascal": ("MPa",),
    "ampere": ("A",),
    "kiloampere": ("kA",),
    "millimetre": ("mm",),
    "percent": ("%",),
    "gram_per_second": ("g.s^-1", "g/s"),
    "kilogram_per_minute": ("kg.min^-1", "kg/min"),
    "millisecond": ("ms",),
    "microsecond": ("us",),
    "second": ("s",),
    "kelvin_per_hour": ("K.h^-1", "K/h"),
    "microstrain": ("m^-1.um", "μm/m"),
    "kilovolt_ampere": ("kVA",),
    "hertz": ("Hz",),
    "kilowatt_hour": ("kWh",),
    "newton": ("N",),
}

# Words and phrases that state a quantity. A term matches whole tokens only, so
# an identifier that merely contains it does not count. A term is never added to
# cover a device name unless the wording is that quantity's own designation.
QUANTITY_TERMS: dict[str, tuple[str, ...]] = {
    "current": ("current", "currents"),
    "voltage": ("voltage", "voltages", "volt"),
    "temperature": (
        "temp",
        "temps",
        "temperature",
        "temperatures",
        "thrm",
        "pt100",
        "pt-100",
    ),
    "pressure": ("pressure", "pressures"),
    "flow": ("flow", "flows", "massflow", "mass flow"),
    "valve_position": ("valve position", "valve pos"),
    "position": ("position", "positions"),
    "resistance": ("resistance", "resistive"),
    "strain": ("strain",),
    "frequency": ("frequency", "frequencies"),
    "energy": ("energy",),
    "force": ("force",),
    "speed": ("speed",),
    "duration": ("duration",),
    "ratio": ("ratio", "percentage"),
}

# Words and phrases that state a role. "power", "output" and "input" are
# deliberately absent: they name an interface or an equipment supply as often as
# they name a quantity, so a rule over them would decide from wording that does
# not state the role.
ROLE_TERMS: dict[str, tuple[str, ...]] = {
    "setpoint": ("setpoint", "set point", "set_point"),
    "state": (
        "status",
        "state",
        "running",
        "stopped",
        "open",
        "closed",
        "normal",
        "ready",
        "health",
    ),
    "alarm": ("alarm", "fault", "trip", "interlock", "warning", "error"),
    "command": ("command", "commande", "write control", "cmd"),
    "measurement": ("measurement", "measured", "measure", "monitoring", "reading"),
}


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Match a term at token boundaries, treating space, underscore and hyphen alike,
    so ``set point``, ``set_point`` and ``set-point`` are one rule."""
    parts = [re.escape(part) for part in re.split(r"[\s_-]+", phrase) if part]
    return re.compile(
        r"(?<![A-Za-z0-9])" + r"[\s_-]+".join(parts) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


@dataclass(frozen=True)
class VocabularyRules:
    """The declared rule set: closed members, their spellings and their terms."""

    quantity_terms: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(QUANTITY_TERMS)
    )
    role_terms: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(ROLE_TERMS)
    )
    unit_spellings: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(UNIT_SPELLINGS)
    )
    min_pattern_support: int = 3

    def __post_init__(self) -> None:
        _reject_unknown(self.quantity_terms, SignalQuantity, "quantity")
        _reject_unknown(self.role_terms, SignalVocabularyRole, "role")
        _reject_unknown(self.unit_spellings, SignalVocabularyUnit, "unit")
        if self.min_pattern_support < 1:
            raise ValueError("a field pattern needs positive support")

    def quantity_matchers(self) -> list[tuple[str, re.Pattern[str]]]:
        return _matchers(self.quantity_terms)

    def role_matchers(self) -> list[tuple[str, re.Pattern[str]]]:
        return _matchers(self.role_terms)

    def unit_lookup(self) -> dict[str, str]:
        return _unit_lookup(self.unit_spellings)


def _reject_unknown(mapping: Mapping[str, Any], enum: Any, axis: str) -> None:
    members = {member.value for member in enum}
    unknown = set(mapping) - members
    if unknown:
        raise ValueError(
            f"{axis} rules name members outside the vocabulary: {sorted(unknown)}"
        )


def _matchers(
    mapping: Mapping[str, tuple[str, ...]],
) -> list[tuple[str, re.Pattern[str]]]:
    return [
        (member, _phrase_pattern(term))
        for member, terms in sorted(mapping.items())
        for term in terms
    ]


def _unit_lookup(spellings: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for member, candidates in spellings.items():
        for spelling in candidates:
            normalized = normalize_unit_symbol(spelling)
            if normalized is None:
                raise ValueError(f"unit spelling {spelling!r} does not parse")
            previous = lookup.setdefault(normalized, member)
            if previous != member:
                raise ValueError(
                    f"unit spelling {spelling!r} resolves to two members: {previous}, {member}"
                )
    return lookup


@dataclass(frozen=True)
class SignalSources:
    """What one signal's sources state, before any precedence is applied."""

    signal_id: str
    field_code: str | None = None
    device_token: str | None = None
    row_texts: tuple[str, ...] = ()
    controls_text: str | None = None
    controls_unit: str | None = None
    archive_unit: str | None = None


def _text_values(
    texts: Iterable[str], matchers: Sequence[tuple[str, re.Pattern[str]]]
) -> set[str]:
    found: set[str] = set()
    for text in texts:
        if not text:
            continue
        for member, pattern in matchers:
            if pattern.search(text):
                found.add(member)
    return found


def values_from_text(
    texts: Iterable[str], matchers: Sequence[tuple[str, re.Pattern[str]]]
) -> str | None:
    """Return the one member the text states, or None when it states none or two."""
    found = _text_values(texts, matchers)
    if len(found) != 1:
        return None
    return found.pop()


def unit_member(raw: str | None, lookup: Mapping[str, str]) -> str | None:
    """Resolve one source's unit spelling to a vocabulary member."""
    if not raw:
        return None
    normalized = normalize_unit_symbol(raw)
    if normalized is None:
        return None
    return lookup.get(normalized)


def device_token(accessor: str | None) -> str | None:
    """The accession's device token: the letters before its number.

    ``MAG-…-CIS:MT1100-TT-XI`` yields ``MT`` and ``…:MP2941-MST-XI`` yields
    ``MP``. The token names the instrument family the accession belongs to, so
    a pattern credited by one family does not transfer to another.
    """
    if not accessor:
        return None
    tail = accessor.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    match = re.match(r"[A-Za-z]+", tail)
    return match.group(0).upper() if match else None


def _pattern_reaches(
    signal: SignalSources, supporters: Sequence[SignalSources]
) -> bool:
    """Whether a code's consensus is evidence about *this* signal's family.

    Supporters drawn from several device families state the code's meaning, so
    the consensus transfers to any family. Supporters drawn from one family
    state that family's practice: it transfers only when the signal's own token
    shares a prefix with it, since ``MP`` and ``MPT`` are one instrument line
    while ``MP`` and ``MT`` are not. A signal with no token takes no single-family
    consensus, because nothing shows it belongs to that family.
    """
    tokens = {other.device_token for other in supporters if other.device_token}
    if len(tokens) != 1:
        return True
    supporter = next(iter(tokens))
    own = signal.device_token
    if not own:
        return False
    return own.startswith(supporter) or supporter.startswith(own)


def _own_candidates(
    sources: SignalSources, rules: VocabularyRules, lookup: Mapping[str, str]
) -> dict[str, list[tuple[str, str]]]:
    """Candidates this signal's own sources give, in precedence order."""
    quantity_matchers = rules.quantity_matchers()
    role_matchers = rules.role_matchers()
    row_quantity = values_from_text(sources.row_texts, quantity_matchers)
    controls_quantity = (
        values_from_text([sources.controls_text], quantity_matchers)
        if sources.controls_text
        else None
    )
    controls_unit = unit_member(sources.controls_unit, lookup)
    archive_unit = unit_member(sources.archive_unit, lookup)
    row_role = values_from_text(sources.row_texts, role_matchers)
    controls_role = (
        values_from_text([sources.controls_text], role_matchers)
        if sources.controls_text
        else None
    )
    candidates: dict[str, list[tuple[str, str]]] = {
        "quantity": [],
        "unit": [],
        "role": [],
    }
    if row_quantity:
        candidates["quantity"].append((SOURCE_ORDER[0], row_quantity))
    if controls_quantity:
        candidates["quantity"].append((SOURCE_ORDER[1], controls_quantity))
    if controls_unit:
        candidates["unit"].append((SOURCE_ORDER[1], controls_unit))
    if archive_unit:
        candidates["unit"].append((SOURCE_ORDER[2], archive_unit))
    if row_role:
        candidates["role"].append((SOURCE_ORDER[0], row_role))
    if controls_role:
        candidates["role"].append((SOURCE_ORDER[1], controls_role))
    return candidates


def _pattern_values(
    signals: Sequence[SignalSources],
    rules: VocabularyRules,
    lookup: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Unanimous field-pattern member per signal, excluding the signal itself.

    A pattern is the field code alone. A member is stated only when at least
    ``min_pattern_support`` other signals with that code give it from their own
    sources, every one of them gives the same member, and the consensus is
    evidence about this signal's device family (see ``_pattern_reaches``).
    """
    own: dict[str, dict[str, list[tuple[str, str]]]] = {
        signal.signal_id: _own_candidates(signal, rules, lookup) for signal in signals
    }
    by_code: dict[str, list[SignalSources]] = defaultdict(list)
    for signal in signals:
        if signal.field_code:
            by_code[signal.field_code].append(signal)
    result: dict[str, dict[str, str]] = {}
    for members in by_code.values():
        if len(members) < rules.min_pattern_support + 1:
            continue
        for axis in AXES:
            counts: Counter[str] = Counter()
            per_signal: dict[str, str] = {}
            for signal in members:
                values = {value for _, value in own[signal.signal_id][axis]}
                if len(values) == 1:
                    value = values.pop()
                    per_signal[signal.signal_id] = value
                    counts[value] += 1
            supporters = [other for other in members if other.signal_id in per_signal]
            for signal in members:
                if not _pattern_reaches(
                    signal,
                    [other for other in supporters if other is not signal],
                ):
                    continue
                own_value = per_signal.get(signal.signal_id)
                remaining = counts - Counter({own_value: 1}) if own_value else counts
                if sum(remaining.values()) < rules.min_pattern_support:
                    continue
                if len(remaining) != 1:
                    continue
                result.setdefault(signal.signal_id, {})[axis] = next(iter(remaining))
    return result


def assign_vocabulary(
    signals: Sequence[SignalSources], rules: VocabularyRules | None = None
) -> list[dict[str, Any]]:
    """Assign every signal's three axes; one outcome per axis and no inference.

    The returned record carries the stored member (None unless decided), the
    credited source (None unless decided), the outcome, and every candidate that
    was weighed, so a reader can check the precedence without re-running rules.
    """
    rules = rules or VocabularyRules()
    lookup = rules.unit_lookup()
    saw: set[str] = set()
    for signal in signals:
        if signal.signal_id in saw:
            raise ValueError(f"signal identity repeated: {signal.signal_id}")
        saw.add(signal.signal_id)
    patterns = _pattern_values(signals, rules, lookup)
    result = []
    for signal in signals:
        own = _own_candidates(signal, rules, lookup)
        pattern = patterns.get(signal.signal_id, {})
        record: dict[str, Any] = {"id": signal.signal_id}
        for axis in AXES:
            candidates = list(own[axis])
            if axis in pattern:
                candidates.append((SOURCE_ORDER[3], pattern[axis]))
            members = {value for _, value in candidates}
            candidates.sort(key=lambda item: SOURCE_ORDER.index(item[0]))
            if len(members) > 1:
                outcome, member, source = "conflict", None, None
            elif members:
                member = members.pop()
                outcome = "decided"
                source = next(src for src, value in candidates if value == member)
            else:
                member, source, outcome = None, None, "undecided"
            record[f"vocabulary_{axis}"] = member
            record[f"vocabulary_{axis}_source"] = source
            record[f"vocabulary_{axis}_state"] = outcome
            record[f"{axis}_candidates"] = candidates
        result.append(record)
    return result


def summarise(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Count each axis by outcome and by credited source, summing to the cohort."""
    summary: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        states = Counter(record[f"vocabulary_{axis}_state"] for record in records)
        credited = Counter(
            record[f"vocabulary_{axis}_source"]
            for record in records
            if record[f"vocabulary_{axis}_state"] == "decided"
        )
        if sum(states.values()) != len(records):
            raise AssertionError("an axis must have exactly one outcome per signal")
        summary[axis] = {
            "decided": states["decided"],
            "conflict": states["conflict"],
            "undecided": states["undecided"],
            "decided_by_source": {
                source: credited.get(source, 0) for source in SOURCE_ORDER
            },
        }
    return summary


def vocabulary_properties(record: Mapping[str, Any]) -> dict[str, Any]:
    """The nine stored properties for one ledger record, with None left unset."""
    properties = {}
    for axis in AXES:
        for suffix in ("", "_source", "_state"):
            value = record[f"vocabulary_{axis}{suffix}"]
            if value is not None:
                properties[f"vocabulary_{axis}{suffix}"] = value
    return properties
