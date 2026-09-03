"""The agent-facing edit surfaces must report the cascade as deferred.

``EditPlan.cascade_deferred`` is a plan the acceptance hook applies later:
``_edit_standard_name`` writes none of those descendant renames, and a root
that is withheld or exhausted leaves every row unperformed. These tests pin
the wording of both agent-facing surfaces — the tool result an agent reads
(``summary``, ``cascade_status``) and the description the MCP tool is
registered with — so neither can drift back to presenting the deferred rows
as steps that already happened.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

CASCADE = [
    {
        "from": "deuterium_deuterium_emissivity_due_to_fusion",
        "to": "deuterium_deuterium_source_rate_due_to_fusion",
    },
    {
        "from": "deuterium_tritium_emissivity_due_to_fusion",
        "to": "deuterium_tritium_source_rate_due_to_fusion",
    },
]


def _plan(**overrides: Any) -> Any:
    from imas_codex.standard_names.edit import EditPlan

    fields: dict[str, Any] = {
        "target": "emissivity_due_to_fusion",
        "mode": "rename",
        "axis": "name",
        "scope": "subtree",
        "entry": "review_name",
        "successor": "source_rate_due_to_fusion",
        "cascade_deferred": list(CASCADE),
        "applied": True,
        "run_id": "sn-edit-20260903T000000Z",
    }
    fields.update(overrides)
    return EditPlan(**fields)


def _call(plan: Any, **kwargs: Any) -> dict:
    from imas_codex.llm import sn_tools

    with patch("imas_codex.standard_names.edit.apply_edit", return_value=plan):
        return sn_tools._edit_standard_name(
            "emissivity_due_to_fusion",
            reason="the per-volume per-steradian base is source_rate, not emissivity",
            rename="source_rate_due_to_fusion",
            scope="subtree",
            **kwargs,
        )


def _flat(text: str) -> str:
    """Collapse the docstring's own wrapping so phrase checks survive it.

    Both surfaces wrap at their own indentation, so a phrase that reads as
    one sentence to the agent is split across lines in the source.
    """
    return " ".join(text.split())


def _mcp_tool_description() -> str:
    from imas_codex.llm.server import AgentsServer

    server = AgentsServer(dd_only=False, include_standard_names=True, read_only=False)
    components = server.mcp._local_provider._components
    keys = [k for k in components if k.startswith("tool:edit_standard_name@")]
    assert keys, "edit_standard_name is not registered on a writable server"
    return components[keys[0]].description or ""


class TestAppliedResultReportsTheDeferral:
    """An applied rename has written the root and none of its descendants."""

    def test_summary_states_the_descendants_are_unchanged_and_waiting(self) -> None:
        result = _call(_plan())

        assert "2 descendant(s) unchanged and deferred" in result["summary"]
        assert (
            "renamed only once source_rate_due_to_fusion reaches accepted"
            in result["summary"]
        )
        assert "withheld or exhausted" in result["summary"]

    def test_cascade_status_names_the_deferral_beside_the_rows(self) -> None:
        result = _call(_plan())

        assert "not yet applied" in result["cascade_status"]
        assert "reaches accepted" in result["cascade_status"]
        # The rows themselves are still returned — the agent needs to see
        # the consequence of the edit it is proposing.
        assert result["cascade_deferred"] == CASCADE

    def test_summary_never_presents_the_rows_as_completed_steps(self) -> None:
        result = _call(_plan())

        assert "cascade step(s)" not in result["summary"]
        assert "renamed 2 descendant" not in result["summary"]
        assert "descendant(s) staged" not in result["summary"]


class TestDryRunResultReportsTheDeferral:
    """A dry run has no successor yet, so it names the root it waits on."""

    def test_summary_names_the_deferral_without_a_successor_id(self) -> None:
        result = _call(_plan(successor=None, applied=False, run_id=None), dry_run=True)

        assert result["summary"].startswith("[dry-run]")
        assert "2 descendant(s) unchanged and deferred" in result["summary"]
        assert "the renamed root reaches accepted" in result["summary"]

    def test_dry_run_summary_drops_the_bare_step_count(self) -> None:
        result = _call(_plan(successor=None, applied=False, run_id=None), dry_run=True)

        assert "cascade step(s)" not in result["summary"]


class TestNoCascadeStaysQuiet:
    """A self-scope edit implies no descendants, so it claims no deferral."""

    def test_empty_cascade_omits_the_status_key(self) -> None:
        result = _call(_plan(scope="only_self", cascade_deferred=[]))

        assert "cascade_status" not in result
        assert "deferred" not in result["summary"]


class TestSurfaceDocumentationStatesTheDeferral:
    """Both agent-facing descriptions must carry the same true statement."""

    def test_tool_wrapper_docstring_states_nothing_is_written(self) -> None:
        from imas_codex.llm.sn_tools import _edit_standard_name

        doc = _flat(_edit_standard_name.__doc__ or "")
        assert "this call writes none of" in doc
        assert "reaches ``accepted``" in doc
        assert "must never be reported as renames that happened" in doc

    def test_mcp_tool_description_states_nothing_is_written(self) -> None:
        description = _flat(_mcp_tool_description())

        assert "DEFERRED" in description
        assert "writes none of those renames" in description
        assert "never report them as renames that happened" in description

    def test_mcp_tool_description_no_longer_implies_applied_work(self) -> None:
        description = _flat(_mcp_tool_description())

        assert "the full deferred cascade" in description
        assert "for family/subtree scope, the full cascade)" not in description

    def test_both_surfaces_share_the_deferral_vocabulary(self) -> None:
        """One wording for one fact, so the surfaces read alike."""
        from imas_codex.llm.sn_tools import _edit_standard_name

        wrapper = _flat(_edit_standard_name.__doc__ or "")
        described = _flat(_mcp_tool_description())
        for phrase in (
            "deferred",
            "acceptance hook re-walks the live",
            "withheld or exhausted",
        ):
            assert phrase in wrapper, f"{phrase!r} missing from the tool wrapper"
            assert phrase in described, f"{phrase!r} missing from the MCP description"
