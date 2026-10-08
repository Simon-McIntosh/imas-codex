"""Extract literal EDAS data identifiers from source code calls."""

import re
from typing import NamedTuple


class EDASReference(NamedTuple):
    """One literal data identifier read through an EDAS API."""

    ref_type: str
    category: str | None
    data_name: str
    raw_string: str


_CALL = re.compile(
    r"\b(?P<name>(?:_?eddbread(?:Time|One|Para|Header)(?:_f)?|"
    r"get_time_(?:slice_data|nearest|seriese_data)|getseldata|"
    r"(?:uddb|pmdb|lcdb|mbdb)(?:read|_)[A-Za-z_0-9]*))\s*\(",
    re.IGNORECASE,
)
_LITERAL = re.compile(r"(?s)^\s*(?:[uUbBrR]|[uU][rR])?(['\"])(.*?)\1\s*$")
_KEYWORD = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*)$", re.DOTALL)


def _arguments(text: str, start: int) -> list[str]:
    """Split a call's outer arguments, leaving nested expressions intact."""
    args: list[str] = []
    depth = 0
    quote = None
    escaped = False
    begin = start
    for position in range(start, len(text)):
        char = text[position]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                args.append(text[begin:position].strip())
                return args
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[begin:position].strip())
            begin = position + 1
    return []


def _literal(value: str | None) -> str | None:
    if value is None:
        return None
    match = _LITERAL.fullmatch(value)
    return match.group(2) if match else None


def extract_edas_references(text: str) -> list[EDASReference]:
    """Recognize EDAS reads with literal names; ignore declarations and variables."""
    found: list[EDASReference] = []
    seen: set[tuple[str, str | None, str]] = set()
    for call in _CALL.finditer(text):
        line_start = text.rfind("\n", 0, call.start()) + 1
        prefix = text[line_start : call.start()].strip()
        if prefix.startswith(("#", "//", "!")) or re.search(
            r"\b(?:def|function|subroutine|extern\s+int)\s*$", prefix, re.I
        ):
            continue
        args = _arguments(text, call.end())
        if not args:
            continue
        positional = []
        named = {}
        for arg in args:
            keyword = _KEYWORD.match(arg)
            if keyword:
                named[keyword.group(1).lower()] = keyword.group(2)
            else:
                positional.append(arg)
        name = call.group("name").lower().lstrip("_")
        if name.startswith("eddbread"):
            ref_type = "edas_eddb"
            category = _literal(named.get("cat") or named.get("category"))
            data_name = _literal(
                named.get("dname") or named.get("data_name") or named.get("pid")
            )
            if category is None and len(positional) > 1:
                category = _literal(positional[1])
            if data_name is None and len(positional) > 2:
                data_name = _literal(positional[2])
        elif name.startswith("get_time_"):
            ref_type = "edas_eddb"
            category = _literal(named.get("inp_category") or named.get("cat"))
            data_name = _literal(named.get("inp_dname") or named.get("dname"))
            if len(positional) >= 4:
                if category is None:
                    category = _literal(positional[-1])
                if data_name is None:
                    data_name = _literal(positional[-2])
        elif name == "getseldata":
            ref_type = "edas_eddb"
            category = _literal(named.get("category") or named.get("cat"))
            data_name = _literal(named.get("dname") or named.get("data_name"))
            if category is None and len(positional) > 1:
                category = _literal(positional[1])
            if data_name is None and len(positional) > 2:
                data_name = _literal(positional[2])
        else:
            ref_type = "edas_" + name[:4]
            category = _literal(named.get("category") or named.get("cat"))
            data_name = _literal(
                named.get("dname") or named.get("data_name") or named.get("pid")
            )
            if data_name is None:
                if len(positional) > 2 and _literal(positional[2]):
                    data_name = _literal(positional[2])
                    if category is None:
                        category = _literal(positional[1])
                elif len(positional) > 1:
                    data_name = _literal(positional[1])
        if not data_name:
            continue
        key = (ref_type, category, data_name)
        if key in seen:
            continue
        seen.add(key)
        raw_string = f"{category}/{data_name}" if category else data_name
        found.append(EDASReference(ref_type, category, data_name, raw_string))
    return found
