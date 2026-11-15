"""Conservative candidate interpretation of structured archive selectors.

The grammar is inferred from catalogued selector structure. Its correspondence
to the naming convention remains unconfirmed. Tokens are retained without
expanding abbreviations or claiming a verified engineering identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from imas_codex.graph.models import PlantComponent, SignalNameParseOutcome

PARSE_PROVENANCE = (
    "observed archive selector structure: hyphenated component prefix, colon, "
    "letter-plus-digit control function, hyphen, variable token; "
    "IDM 2UT8SH correspondence unconfirmed"
)
RULE_STATUS = "unconfirmed"
_COMPONENT = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)+")
_FUNCTION = re.compile(r"[A-Z]+[0-9]+")
_VARIABLE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*")


@dataclass(frozen=True)
class VariableNameParse:
    """Tokens from one untouched selector, with explicit syntactic coverage."""

    selector: str
    outcome: SignalNameParseOutcome
    plant_system_code: str | None = None
    plant_component_code: str | None = None
    control_function: str | None = None
    variable_token: str | None = None
    reason: str | None = None

    def components(self, facility: str) -> list[PlantComponent]:
        """Build only hierarchy levels actually present in the parsed prefix."""
        if self.plant_component_code is None:
            return []
        result = []
        code = ""
        parent = None
        for token in self.plant_component_code.split("-"):
            code = f"{code}-{token}" if code else token
            component = PlantComponent(
                id=f"{facility}:plant_component:{code}",
                facility_id=facility,
                code=code,
                name=token,
                description="Candidate component from an unconfirmed selector grammar",
                parent_component=parent,
                parse_provenance=PARSE_PROVENANCE,
            )
            result.append(component)
            parent = component.id
        return result

    def properties(self, facility: str) -> dict:
        """Return schema properties; no caller-supplied component is accepted."""
        return {
            "name_parse_outcome": self.outcome.value,
            "name_parse_provenance": PARSE_PROVENANCE,
            "name_parse_rule_status": RULE_STATUS,
            "name_parse_reason": self.reason,
            "plant_system_code": self.plant_system_code,
            "plant_component_code": self.plant_component_code,
            "control_function": self.control_function,
            "variable_token": self.variable_token,
            "candidate_plant_component": (
                f"{facility}:plant_component:{self.plant_component_code}"
                if self.plant_component_code is not None
                else None
            ),
        }


def parse_variable_name(selector: str) -> VariableNameParse:
    """Parse exact text without trimming, case folding or repairing a selector.

    A single colon and a complete multi-token prefix establish the candidate
    component boundary. An incomplete right side retains that prefix as partly
    parsed. Ambiguous boundaries or malformed prefixes derive no component.
    """
    if selector.count(":") != 1:
        return VariableNameParse(
            selector,
            SignalNameParseOutcome.not_conforming,
            reason="requires one unambiguous colon boundary",
        )
    prefix, tail = selector.split(":")
    if not _COMPONENT.fullmatch(prefix):
        return VariableNameParse(
            selector,
            SignalNameParseOutcome.not_conforming,
            reason="component prefix does not match the observed grammar",
        )
    function, delimiter, variable = tail.partition("-")
    function = function if _FUNCTION.fullmatch(function) else None
    variable = variable if delimiter and _VARIABLE.fullmatch(variable) else None
    complete = function is not None and variable is not None
    return VariableNameParse(
        selector=selector,
        outcome=(
            SignalNameParseOutcome.parsed
            if complete
            else SignalNameParseOutcome.partly_parsed
        ),
        plant_system_code=prefix.split("-", 1)[0],
        plant_component_code=prefix,
        control_function=function,
        variable_token=variable,
        reason=None if complete else "control function or variable token is incomplete",
    )
