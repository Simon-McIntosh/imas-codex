"""Build private, source-grounded plant context without reading archive samples."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.icp.records import FunctionCategory, PlantNode
from imas_codex.discovery.signals.variable_names import parse_variable_name
from imas_codex.discovery.wiki.excel import _format_row_with_headers
from imas_codex.graph.models import (
    PlantContextDocument,
    PlantContextMatch,
    PlantFunctionCategory,
    PlantSignalRow,
    PlantStructure,
    SignalContextLink,
)

_SELECTOR = re.compile(
    r"(?<![A-Za-z0-9_:-])[A-Z0-9]{1,24}(?:-[A-Z0-9]{1,24}){0,4}:"
    r"[A-Z0-9-]{1,24}(?![A-Za-z0-9_:-])"
)


def fingerprint(*parts: str) -> str:
    """Hash a length-unambiguous sequence of native identities."""
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def in_system(control_path: str | None, system: str) -> bool:
    return bool(control_path) and (
        control_path == system or control_path.startswith(system + "-")
    )


def component_key(control: str | None, category: str | None, sequence: Any):
    """Compare component numbers numerically while retaining the full control path."""
    if not control or not category or sequence is None:
        return None
    text = str(sequence).strip()
    if not re.fullmatch(r"[0-9]+(?:\.0+)?", text):
        return None
    return control, category, int(text.split(".")[0])


@dataclass(frozen=True)
class SpreadsheetLayout:
    """Explicit source coordinates; columns and header rows are one-based."""

    sheet: str
    header_row: int | None = 1
    first_data_row: int = 2
    control_column: int | None = None
    category_column: int | None = None
    sequence_column: int | None = None
    field_column: int | None = None
    plant_column: int | None = None


def extract_signal_rows(
    content: bytes,
    document: PlantContextDocument,
    layouts: list[SpreadsheetLayout],
    *,
    plant_bindings: dict[str, str] | None = None,
    category_bindings: dict[str, str] | None = None,
) -> list[PlantSignalRow]:
    """Use the shared spreadsheet formatter, retaining every configured source row.

    Unlike an embedding preview, a coordinate-based import must not truncate at
    a text-row limit or skip a numeric sheet. Source layout is explicit so a
    title or revision sheet cannot silently become a signal table.
    """
    from openpyxl import load_workbook

    book = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    result = []
    try:
        if len({layout.sheet for layout in layouts}) != len(layouts):
            raise ValueError("A worksheet may be configured only once")
        if set(book.sheetnames) != {layout.sheet for layout in layouts}:
            raise ValueError("Every worksheet needs an explicit import layout")
        for layout in layouts:
            if layout.first_data_row < 1 or (
                layout.header_row is not None
                and not 1 <= layout.header_row < layout.first_data_row
            ):
                raise ValueError("Headers must precede positive data-row coordinates")
            ws = book[layout.sheet]
            if any(
                column is not None and not 1 <= column <= ws.max_column
                for column in (
                    layout.control_column,
                    layout.category_column,
                    layout.sequence_column,
                    layout.field_column,
                    layout.plant_column,
                )
            ):
                raise ValueError("Source columns must be inside the worksheet")
            headers = (
                [
                    str(v or "").strip()
                    for v in next(
                        ws.iter_rows(
                            min_row=layout.header_row,
                            max_row=layout.header_row,
                            values_only=True,
                        )
                    )
                ]
                if layout.header_row is not None
                else None
            )
            for number, cells in enumerate(
                ws.iter_rows(min_row=layout.first_data_row, values_only=True),
                start=layout.first_data_row,
            ):
                description = _format_row_with_headers(cells, headers)
                if not description:
                    continue

                def value(column, cells=cells):
                    raw = cells[column - 1] if column else None
                    return str(raw).strip() if raw is not None else None

                category = value(layout.category_column)
                selectors = sorted(
                    {
                        match.group()
                        for cell in cells
                        if isinstance(cell, str)
                        for match in _SELECTOR.finditer(cell)
                    }
                )
                result.append(
                    PlantSignalRow(
                        id=f"{document.id}:row:{fingerprint(layout.sheet, str(number))}",
                        facility_id=document.facility_id,
                        context_scope=document.context_scope,
                        source_document=document.id,
                        sheet=layout.sheet,
                        row_number=number,
                        description=description,
                        cells_json=json.dumps(
                            {
                                str(i + 1): v
                                for i, v in enumerate(cells)
                                if v is not None
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                        control_path=value(layout.control_column),
                        component_category=category,
                        component_sequence=value(layout.sequence_column),
                        field_code=value(layout.field_column),
                        plant_reference=value(layout.plant_column),
                        source_selectors=selectors,
                        plant_structure=(plant_bindings or {}).get(
                            value(layout.plant_column)
                        ),
                        function_category=(category_bindings or {}).get(category),
                    )
                )
    finally:
        book.close()
    return result


def structure_nodes(
    plants: list[PlantNode], facility: str, scope: str
) -> list[PlantStructure]:
    """Preserve official parent references, requiring their records to be present."""
    identities = {p.uid: f"{facility}:plant_structure:{p.uid}" for p in plants}
    result = []
    for p in plants:
        if p.parent_uid and p.parent_uid not in identities:
            raise ValueError("A referenced parent plant record was not read")
        result.append(
            PlantStructure(
                id=identities[p.uid],
                facility_id=facility,
                context_scope=scope,
                source_uid=p.uid,
                code=p.code,
                description=p.title,
                source_origin=p.source.origin,
                source_endpoint=p.source.endpoint,
                parent_structure=identities.get(p.parent_uid),
            )
        )
    return result


def category_nodes(
    categories: list[FunctionCategory], facility: str, scope: str
) -> list[PlantFunctionCategory]:
    return [
        PlantFunctionCategory(
            id=f"{facility}:function_category:{c.uid}",
            facility_id=facility,
            context_scope=scope,
            source_uid=c.uid,
            code=c.code,
            description=c.title,
            component_type=c.component_type,
            source_origin=c.source.origin,
            source_endpoint=c.source.endpoint,
        )
        for c in categories
    ]


def match_signal_row(
    selector: str,
    row: PlantSignalRow,
    system: str,
    *,
    source_systems: tuple[str, ...] = (),
    number_join: bool = False,
) -> PlantContextMatch:
    """Never join across systems or through a bare component number.

    A literal selector is exact. Joining separate source columns is derived,
    even when their field code agrees, because the source never spells the
    archive identity as one value. A derived association establishes component
    context, not the meaning of every field exposed by that component.
    """
    parsed = parse_variable_name(selector)
    if selector.count(":") != 1 or not in_system(selector.split(":")[0], system):
        return PlantContextMatch.absent
    if row.control_path and not any(
        in_system(row.control_path, source) for source in (system, *source_systems)
    ):
        return PlantContextMatch.absent
    if selector in (row.source_selectors or []):
        return PlantContextMatch.exact
    key = component_key(
        parsed.control_path, parsed.component_category, parsed.component_sequence
    )
    if key is not None and key == component_key(
        row.control_path, row.component_category, row.component_sequence
    ):
        return PlantContextMatch.derived
    row_key = component_key(
        row.control_path, row.component_category, row.component_sequence
    )
    if (
        number_join
        and key is not None
        and row_key is not None
        and key[1:] == row_key[1:]
    ):
        return PlantContextMatch.derived
    return PlantContextMatch.absent


@dataclass
class PlantContextBundle:
    facility: str
    scope: str
    system: str
    structures: list[PlantStructure]
    categories: list[PlantFunctionCategory]
    documents: list[PlantContextDocument]
    rows: list[PlantSignalRow]
    links: list[SignalContextLink] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)

    def node_groups(self):
        return {
            "PlantStructure": self.structures,
            "PlantFunctionCategory": self.categories,
            "PlantContextDocument": self.documents,
            "PlantSignalRow": self.rows,
            "SignalContextLink": self.links,
        }

    def link_signals(self, signals: list[dict]) -> dict:
        """Index rows by complete component key and retain all supporting rows."""
        if self.links or self.signals:
            raise ValueError("Build associations once per bundle")
        by_selector, by_component, by_number = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        source_systems = set()
        for document in self.documents:
            if document.system_code != self.system:
                raise ValueError("Document belongs to a different plant system")
            aliases = set(document.source_system_codes or []) - {self.system}
            if aliases and not document.system_binding_note:
                raise ValueError("A source system alias needs supporting evidence")
            source_systems.update(aliases)
        source_identities, archive_identities = defaultdict(set), defaultdict(set)
        for row in self.rows:
            if row.facility_id != self.facility or row.context_scope != self.scope:
                raise ValueError(
                    "Source rows must belong to the imported facility and scope"
                )
            for selector in row.source_selectors or []:
                by_selector[selector].append(row)
            key = component_key(
                row.control_path, row.component_category, row.component_sequence
            )
            if key is not None:
                by_component[key].append(row)
                if any(
                    in_system(row.control_path, source)
                    for source in (self.system, *source_systems)
                ):
                    by_number[key[1:]].append(row)
                    source_identities[key[1:]].add(
                        row.plant_reference or row.control_path
                    )
        for signal in signals:
            parsed = parse_variable_name(signal["selector"])
            key = component_key(
                parsed.control_path,
                parsed.component_category,
                parsed.component_sequence,
            )
            if key is not None:
                archive_identities[key[1:]].add(parsed.plant_component_code)
        counts = Counter({state.value: 0 for state in PlantContextMatch})
        seen = set()
        for signal in signals:
            if signal["id"] in seen:
                raise ValueError("Catalogue snapshot repeated a signal identity")
            seen.add(signal["id"])
            parsed = parse_variable_name(signal["selector"])
            if signal["selector"].count(":") != 1 or not in_system(
                signal["selector"].split(":")[0], self.system
            ):
                raise ValueError("Catalogue snapshot contains a different system")
            candidates = {
                row.id: row
                for row in (
                    by_selector[signal["selector"]]
                    + by_component[
                        component_key(
                            parsed.control_path,
                            parsed.component_category,
                            parsed.component_sequence,
                        )
                    ]
                    + by_number[
                        component_key(
                            parsed.control_path,
                            parsed.component_category,
                            parsed.component_sequence,
                        )[1:]
                        if parsed.component_category
                        else None
                    ]
                )
            }
            links = []
            for row in sorted(candidates.values(), key=lambda r: r.id):
                key = component_key(
                    row.control_path, row.component_category, row.component_sequence
                )
                number_join = bool(
                    key is not None
                    and len(source_identities[key[1:]]) == 1
                    and len(archive_identities[key[1:]]) == 1
                )
                kind = match_signal_row(
                    signal["selector"],
                    row,
                    self.system,
                    source_systems=tuple(source_systems),
                    number_join=number_join,
                )
                if (
                    kind == PlantContextMatch.derived
                    and key is not None
                    and len(source_identities[key[1:]]) > 1
                ):
                    kind = PlantContextMatch.absent
                if kind == PlantContextMatch.absent:
                    continue
                link = SignalContextLink(
                    id=f"{self.facility}:signal_context:{fingerprint(signal['id'], row.id)}",
                    facility_id=self.facility,
                    context_scope=self.scope,
                    signal=signal["id"],
                    source_row=row.id,
                    component=signal.get("component_id"),
                    match_kind=kind,
                    description=(
                        "Unchanged archive selector explicitly present in the source row"
                        if kind == PlantContextMatch.exact
                        else "Source-scoped category and numeric component join; unique source and archive component when control paths differ; field meaning remains unasserted"
                    ),
                )
                links.append(link)
            state = (
                PlantContextMatch.exact
                if any(link.match_kind == PlantContextMatch.exact for link in links)
                else PlantContextMatch.derived
                if links
                else PlantContextMatch.absent
            )
            self.links.extend(links)
            self.signals.append(
                {**signal, "state": state.value, "links": [link.id for link in links]}
            )
            counts[state.value] += 1
        return dict(counts)
