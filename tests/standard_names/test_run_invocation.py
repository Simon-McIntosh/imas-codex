"""Behaviour of the ``SNRun`` invocation capture.

The captured command line is written to the graph, so the redaction contract
is load-bearing: a credential passed as an argument must not survive into the
stored string, and an ordinary argument must survive intact or the record
stops answering the question it exists for.
"""

from __future__ import annotations

import json

import pytest

from imas_codex.standard_names.run_invocation import (
    REDACTED,
    capture_run_invocation,
    redact_argv,
)


class TestRedaction:
    @pytest.mark.parametrize(
        "argv",
        [
            ["sn", "run", "--api-key", "sk-live-abcdefghijklmnopqrst"],
            ["sn", "run", "--openrouter-token", "abcdefghijklmnopqrst"],
            ["sn", "run", "--neo4j-password", "hunter2"],
            ["sn", "run", "--secret", "value"],
        ],
    )
    def test_credential_option_value_is_replaced(self, argv: list[str]) -> None:
        redacted = redact_argv(argv)
        assert redacted[-1] == REDACTED
        assert redacted[:-1] == argv[:-1]

    def test_inline_credential_value_is_replaced(self) -> None:
        redacted = redact_argv(["sn", "run", "--api-key=sk-live-secret-value"])
        assert redacted == ["sn", "run", f"--api-key={REDACTED}"]

    def test_bare_key_shaped_value_is_replaced(self) -> None:
        redacted = redact_argv(["sn", "run", "sk-abcdefghijklmnopqrstuvwx"])
        assert redacted == ["sn", "run", REDACTED]

    def test_ordinary_arguments_survive(self) -> None:
        argv = [
            "imas-codex",
            "sn",
            "run",
            "--only",
            "review_name",
            "--domain",
            "edge_physics",
            "-c",
            "80",
        ]
        assert redact_argv(argv) == argv

    def test_redaction_does_not_swallow_the_following_argument(self) -> None:
        """Only the credential itself is lost, not the flags after it."""
        redacted = redact_argv(
            ["sn", "run", "--api-key", "sk-abcdefghijklmnopqrstuv", "--only", "review"]
        )
        assert redacted == ["sn", "run", "--api-key", REDACTED, "--only", "review"]


class TestCapture:
    def test_invocation_is_a_reusable_command_line(self) -> None:
        captured = capture_run_invocation(
            flags={"cost_limit": 80.0},
            scope={},
            argv=["imas-codex", "sn", "run", "--only", "review_name"],
        )
        assert captured["invocation"] == "imas-codex sn run --only review_name"

    def test_arguments_needing_quotes_stay_recoverable(self) -> None:
        captured = capture_run_invocation(
            flags={},
            scope={},
            argv=["sn", "run", "--description", "edge physics drain"],
        )
        assert captured["invocation"] == "sn run --description 'edge physics drain'"

    def test_flags_and_scope_round_trip_as_json(self) -> None:
        captured = capture_run_invocation(
            flags={"cost_limit": 80.0, "flush": True, "compose_model": None},
            scope={"domains": ["edge_physics"], "only_pool": "review_name"},
            argv=["sn", "run"],
        )
        assert json.loads(captured["invocation_flags"]) == {
            "cost_limit": 80.0,
            "flush": True,
        }
        assert json.loads(captured["invocation_scope"]) == {
            "domains": ["edge_physics"],
            "only_pool": "review_name",
        }

    def test_unset_knobs_are_omitted_rather_than_stored_as_null(self) -> None:
        captured = capture_run_invocation(
            flags={"min_score": None, "cost_limit": 5.0},
            scope={"domains": [], "drain_scope_id": None},
            argv=["sn", "run"],
        )
        assert json.loads(captured["invocation_flags"]) == {"cost_limit": 5.0}
        assert json.loads(captured["invocation_scope"]) == {}

    def test_tuple_scope_values_serialize(self) -> None:
        """Callers pass click multi-options through as tuples."""
        captured = capture_run_invocation(
            flags={},
            scope={"domains": ("edge_physics", "core_plasma_physics")},
            argv=["sn", "run"],
        )
        assert json.loads(captured["invocation_scope"])["domains"] == [
            "edge_physics",
            "core_plasma_physics",
        ]

    def test_credentials_are_redacted_before_storage(self) -> None:
        captured = capture_run_invocation(
            flags={},
            scope={},
            argv=["sn", "run", "--api-key", "sk-live-do-not-store-this-value"],
        )
        assert "do-not-store-this-value" not in captured["invocation"]
        assert REDACTED in captured["invocation"]
