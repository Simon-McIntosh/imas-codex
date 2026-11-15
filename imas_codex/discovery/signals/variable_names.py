"""Parse control paths and component-variable codes without interpreting codes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from imas_codex.graph.models import ControlNode, PlantComponent, SignalNameParseOutcome

PARSE_PROVENANCE = (
    "ITER variable naming convention: FunctionID:VariableID; control hierarchy "
    "and component category, sequence, field and discriminator codes only"
)
RULE_STATUS = "convention"
_CONTROL_PATH = re.compile(r"[A-Z0-9]{1,4}(?:-[A-Z0-9]{1,4}){0,4}")
_VARIABLE_ID = re.compile(r"[A-Z0-9-]{1,24}")
_COMPONENT_VARIABLE = re.compile(
    r"(?P<category>[A-Z]{1,3})(?P<sequence>[0-9]{1,4})-"
    r"(?P<field>[A-Z]{1,4})"
    r"(?P<discriminator>[0-9][A-Z0-9]{0,3}|-[A-Z][A-Z0-9]{0,3})?"
)


@dataclass(frozen=True)
class VariableNameParse:
    """One untouched selector and its syntactic control or component identity."""

    selector: str
    outcome: SignalNameParseOutcome
    control_path: str | None = None
    variable_id: str | None = None
    component_category: str | None = None
    component_sequence: str | None = None
    field_code: str | None = None
    field_discriminator: str | None = None
    reason: str | None = None

    @property
    def plant_component_code(self) -> str | None:
        if self.component_category is None:
            return None
        return f"{self.control_path}:{self.component_category}{self.component_sequence}"

    def control_nodes(self, facility: str) -> list[ControlNode]:
        """Preserve the one to five ordered control-breakdown hierarchy levels."""
        result = []
        path = ""
        parent = None
        for level, token in enumerate((self.control_path or "").split("-"), start=1):
            if not token:
                break
            path = f"{path}-{token}" if path else token
            node = ControlNode(
                id=f"{facility}:control_node:{path}",
                facility_id=facility,
                code=path,
                name=token,
                level=level,
                parent_control_node=parent,
                parse_provenance=PARSE_PROVENANCE,
            )
            result.append(node)
            parent = node.id
        return result

    def components(self, facility: str) -> list[PlantComponent]:
        """A physical component is one leaf under its complete control path."""
        if self.plant_component_code is None:
            return []
        return [
            PlantComponent(
                id=f"{facility}:plant_component:{self.plant_component_code}",
                facility_id=facility,
                code=self.plant_component_code,
                name=f"{self.component_category}{self.component_sequence}",
                control_node=f"{facility}:control_node:{self.control_path}",
                component_category=self.component_category,
                component_sequence=self.component_sequence,
                parse_provenance=PARSE_PROVENANCE,
            )
        ]

    def properties(self, facility: str) -> dict:
        """All identities come from the selector; no code meanings are inferred."""
        return {
            "name_parse_outcome": self.outcome.value,
            "name_parse_provenance": PARSE_PROVENANCE,
            "name_parse_rule_status": RULE_STATUS,
            "name_parse_reason": self.reason,
            "control_path": self.control_path,
            "control_node": f"{facility}:control_node:{self.control_path}"
            if self.control_path
            else None,
            "variable_id": self.variable_id,
            "plant_component_code": self.plant_component_code,
            "component_category": self.component_category,
            "component_sequence": self.component_sequence,
            "field_code": self.field_code,
            "field_discriminator": self.field_discriminator,
            "candidate_plant_component": f"{facility}:plant_component:{self.plant_component_code}"
            if self.plant_component_code
            else None,
        }


def parse_variable_name(selector: str) -> VariableNameParse:
    """Classify valid function-level variables separately from component variables.

    Control path fields are at most four alphanumeric characters. VariableID is
    at most 24 uppercase alphanumeric or hyphen characters. A component form has
    a category of up to three letters, up to four sequence digits, and a field
    of up to four letters. A discriminator is at most four alphanumeric
    characters; a leading letter requires a separating hyphen.
    """
    if selector.count(":") != 1:
        return VariableNameParse(
            selector,
            SignalNameParseOutcome.not_conforming,
            reason="requires one unambiguous colon boundary",
        )
    control_path, variable_id = selector.split(":")
    if not _CONTROL_PATH.fullmatch(control_path):
        return VariableNameParse(
            selector,
            SignalNameParseOutcome.not_conforming,
            reason="control path requires one to five fields of one to four uppercase alphanumeric characters",
        )
    if not _VARIABLE_ID.fullmatch(variable_id):
        return VariableNameParse(
            selector,
            SignalNameParseOutcome.not_conforming,
            reason="variable identifier requires one to 24 uppercase alphanumeric or hyphen characters",
        )
    component = _COMPONENT_VARIABLE.fullmatch(variable_id)
    if component is None:
        return VariableNameParse(
            selector,
            SignalNameParseOutcome.function_variable,
            control_path=control_path,
            variable_id=variable_id,
        )
    return VariableNameParse(
        selector,
        SignalNameParseOutcome.component_variable,
        control_path=control_path,
        variable_id=variable_id,
        component_category=component["category"],
        component_sequence=component["sequence"],
        field_code=component["field"],
        field_discriminator=component["discriminator"].removeprefix("-")
        if component["discriminator"]
        else None,
    )
