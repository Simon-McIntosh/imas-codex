"""Give each archive field code a role and a meaning from a source that states it.

An archive variable identifier ends in a short field code naming the kind of
channel a component exposes. Three sources state what a code means, in
precedence: a signal-list row or its legend, the controls metadata a signal with
that code carries, and the functional specification. This module turns what
those sources say into one table entry per code and component kind.

Nothing is inferred from the letters of a code. A source states a role when its
text names one member of the closed role vocabulary; a source naming two members
states neither, and one naming none gives no role. A code whose sources state no
role has no entry and is counted, and so does a code whose sources disagree: the
table records the disagreement rather than choosing. A meaning is a one-line
reading of a source's own wording, declared in private configuration beside the
code, so an entry exists only where both a role and a meaning are established.

A code's role may depend on the kind of component carrying it: one code may be a
command on a valve and a state on a cubicle. An entry is keyed by code and by
the component kind its source names, and a kind no entry covers resolves to no
role. An entry whose source names no kind applies to every kind, and a meaning
declared without a kind serves every entry for that code.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from iter_codex.discovery.signals.vocabulary import (
    ROLE_TERMS,
    _phrase_pattern,
    values_from_text,
)
from iter_codex.graph.models import SignalVocabularyRole, SignalVocabularySource

# The order a role is credited in. Only these sources establish a code's
# meaning; the archive unit states a unit, not a role.
TABLE_SOURCES = (
    SignalVocabularySource.signal_list_row.value,
    SignalVocabularySource.controls_metadata.value,
    SignalVocabularySource.functional_specification.value,
)

# Wording that states a role in a signal-list purpose or controls description,
# beyond the vocabulary rule set's own terms. Each names what a signal is or
# does, never what a code's letters suggest: an instrument noun states a
# measurement, a request or reset states a command, and a threshold states a
# setpoint.
SOURCE_ROLE_TERMS: dict[str, tuple[str, ...]] = {
    "measurement": (
        "transducer",
        "transmitter",
        "sensor",
        "gauge",
        "thermocouple",
        "pt100",
        "pt-100",
        "sensing",
        "analogue input",
        "analog input",
        "feedback",
    ),
    "command": (
        "request",
        "reset",
        "close control",
        "open control",
        "close cmd",
        "open cmd",
    ),
    "setpoint": ("threshold", "set value"),
    "state": ("switch", "operating", "operating time", "active"),
}


def declared_role_terms() -> dict[str, tuple[str, ...]]:
    return {
        member: tuple(terms) + SOURCE_ROLE_TERMS.get(member, ())
        for member, terms in ROLE_TERMS.items()
    }


@dataclass(frozen=True)
class FieldCodeRules:
    """The declared rules the table is read with.

    The role terms are the vocabulary assignment's own terms extended with the
    instrument and command wording a signal list uses. The assignment keeps its
    own rule set, so a reading here never changes a value it already stored.
    """

    role_terms: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self.role_terms is None:
            object.__setattr__(self, "role_terms", declared_role_terms())
        unknown = set(self.role_terms or {}) - {
            member.value for member in SignalVocabularyRole
        }
        if unknown:
            raise ValueError(
                f"role rules name members outside the vocabulary: {sorted(unknown)}"
            )

    def role_matchers(self) -> list[tuple[str, Any]]:
        return [
            (member, _phrase_pattern(term))
            for member, terms in sorted((self.role_terms or {}).items())
            for term in terms
        ]


def role_stated(text: str, rules: FieldCodeRules | None = None) -> str | None:
    """The one role the text states, or None when it states none or two."""
    rules = rules or FieldCodeRules()
    return values_from_text([text], rules.role_matchers())


@dataclass(frozen=True)
class FieldCodeStatement:
    """One source's statement about what one code means for one component kind."""

    code: str
    component_kind: str | None
    role: str | None
    source: str
    evidence: str


@dataclass(frozen=True)
class FieldCodeEntry:
    """A code's established role and meaning for one component kind."""

    code: str
    component_kind: str | None
    role: str
    meaning: str
    source: str
    support: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "component_kind": self.component_kind,
            "role": self.role,
            "meaning": self.meaning,
            "source": self.source,
            "support": self.support,
        }


@dataclass(frozen=True)
class FieldCodeConflict:
    """A code and kind whose sources disagree, so it has no entry."""

    code: str
    component_kind: str | None
    roles: tuple[str, ...]
    sources: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "component_kind": self.component_kind,
            "roles": list(self.roles),
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class FieldCodeTable:
    """Entries, conflicts and the codes no source establishes, for one facility."""

    entries: tuple[FieldCodeEntry, ...] = ()
    conflicts: tuple[FieldCodeConflict, ...] = ()
    unestablished: tuple[dict[str, Any], ...] = ()

    def entry(
        self, code: str | None, component_kind: str | None = None
    ) -> FieldCodeEntry | None:
        """The entry for this code and kind, falling back to the any-kind entry."""
        if not code:
            return None
        for entry in self.entries:
            if entry.code == code and entry.component_kind == component_kind:
                return entry
        for entry in self.entries:
            if entry.code == code and entry.component_kind is None:
                return entry
        return None

    def resolve(
        self, code: str | None, component_kind: str | None = None
    ) -> str | None:
        """The role for this code and kind, or None when none is established."""
        entry = self.entry(code, component_kind)
        return entry.role if entry is not None else None

    def codes(self) -> set[str]:
        return {entry.code for entry in self.entries}

    def summary(
        self, signals_by_code: Mapping[str, int] | None = None
    ) -> dict[str, Any]:
        """Count the table by source and role, and the catalogue it covers."""
        by_source = Counter(entry.source for entry in self.entries)
        by_role = Counter(entry.role for entry in self.entries)
        specific = sum(1 for entry in self.entries if entry.component_kind is not None)
        summary: dict[str, Any] = {
            "entries": len(self.entries),
            "entry_codes": len(self.codes()),
            "entries_by_source": {
                source: by_source.get(source, 0) for source in TABLE_SOURCES
            },
            "entries_by_role": {
                member.value: by_role.get(member.value, 0)
                for member in SignalVocabularyRole
            },
            "entries_for_one_kind": specific,
            "entries_for_any_kind": len(self.entries) - specific,
            "conflicts": len(self.conflicts),
            "conflict_codes": sorted({item.code for item in self.conflicts}),
            "unestablished": len(self.unestablished),
        }
        if signals_by_code is not None:
            codes = self.codes()
            catalogue = set(signals_by_code)
            summary["catalogue_signals"] = sum(signals_by_code.values())
            summary["codes_with_entry"] = len(codes & catalogue)
            summary["codes_without_entry"] = len(catalogue - codes)
            summary["entry_codes_not_in_catalogue"] = sorted(codes - catalogue)
            summary["signals_with_entry"] = sum(
                count for code, count in signals_by_code.items() if code in codes
            )
        return summary


def statements_from_texts(
    code: str | None,
    component_kind: str | None,
    source: str,
    texts: Iterable[str],
    rules: FieldCodeRules | None = None,
) -> list[FieldCodeStatement]:
    """One statement per text that states a role, attributed to its source."""
    if not code:
        return []
    rules = rules or FieldCodeRules()
    statements = []
    for text in texts:
        if not text:
            continue
        role = role_stated(text, rules)
        if role is not None:
            statements.append(
                FieldCodeStatement(
                    code=code.strip(),
                    component_kind=(component_kind or None),
                    role=role,
                    source=source,
                    evidence=text,
                )
            )
    return statements


def build_field_code_table(
    statements: Sequence[FieldCodeStatement],
    meanings: Mapping[tuple[str, str | None], str],
    codes_in_catalogue: Iterable[str] = (),
) -> FieldCodeTable:
    """Resolve each code and kind to one entry, or record why it has none.

    A source that states two roles for one code and kind contradicts itself, and
    two sources that state different roles for one code and kind contradict each
    other. Both leave that code and kind with no entry and a recorded conflict.
    An agreement is credited to the first source in ``TABLE_SOURCES`` that states
    the role, and an entry also needs a declared meaning.
    """
    by_key: dict[tuple[str, str | None], list[FieldCodeStatement]] = defaultdict(list)
    for statement in statements:
        if statement.role is None:
            continue
        by_key[(statement.code, statement.component_kind)].append(statement)

    entries: list[FieldCodeEntry] = []
    conflicts: list[FieldCodeConflict] = []
    unestablished: list[dict[str, Any]] = []

    for (code, kind), group in sorted(
        by_key.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        roles_by_source: dict[str, set[str]] = defaultdict(set)
        for statement in group:
            roles_by_source[statement.source].add(statement.role)
        roles = {statement.role for statement in group}
        self_contradictory = any(
            len(source_roles) > 1 for source_roles in roles_by_source.values()
        )
        if self_contradictory or len(roles) > 1:
            conflicts.append(
                FieldCodeConflict(
                    code=code,
                    component_kind=kind,
                    roles=tuple(sorted(roles)),
                    sources=tuple(
                        source for source in TABLE_SOURCES if source in roles_by_source
                    ),
                )
            )
            continue
        role = next(iter(roles))
        credited = next(
            source
            for source in TABLE_SOURCES
            if role in roles_by_source.get(source, set())
        )
        meaning = meanings.get((code, kind)) or meanings.get((code, None))
        if not meaning:
            unestablished.append(
                {
                    "code": code,
                    "component_kind": kind,
                    "role": role,
                    "reason": "role stated but no meaning declared",
                }
            )
            continue
        entries.append(
            FieldCodeEntry(
                code=code,
                component_kind=kind,
                role=role,
                meaning=meaning,
                source=credited,
                support=len(group),
            )
        )

    established = {entry.code for entry in entries}
    conflicted = {item.code for item in conflicts}
    for code in sorted(set(codes_in_catalogue) - established - conflicted):
        unestablished.append(
            {
                "code": code,
                "component_kind": None,
                "role": None,
                "reason": "no source states a role",
            }
        )

    return FieldCodeTable(
        entries=tuple(entries),
        conflicts=tuple(conflicts),
        unestablished=tuple(unestablished),
    )
