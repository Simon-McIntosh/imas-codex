"""The persistent REPL keeps each response inside the MCP transport budget."""

from __future__ import annotations

import re

from iter_codex.llm import server as server_module
from iter_codex.llm.server import AgentsServer


def _registered_repl():
    server = AgentsServer(read_only=False, dd_only=False)
    return next(
        component.fn
        for key, component in server.mcp._local_provider._components.items()
        if key.startswith("tool:repl")
    )


def test_repl_small_output_is_returned_byte_identical(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "_get_repl", lambda: {})
    repl = _registered_repl()

    expected = "small result\n"

    assert repl("print('small result')") == expected


def test_repl_huge_output_is_truncated_under_cap_with_counts(monkeypatch) -> None:
    monkeypatch.setattr(server_module, "_get_repl", lambda: {})
    repl = _registered_repl()
    printed_chars = server_module._REPL_OUTPUT_MAX_CHARS + 12_345

    result = repl(f"print('x' * {printed_chars})")

    notice_match = re.search(
        r"\n\n\[repl output truncated: produced (\d+) characters; "
        r"dropped (\d+) characters\]$",
        result,
    )
    assert len(result) <= server_module._REPL_OUTPUT_MAX_CHARS
    assert notice_match is not None

    produced_chars = int(notice_match.group(1))
    dropped_chars = int(notice_match.group(2))
    retained_chars = len(result) - len(notice_match.group(0))
    assert produced_chars == printed_chars + 1
    assert dropped_chars == produced_chars - retained_chars
