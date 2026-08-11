"""Write-boundary guards for semantic source attachments."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.attachment_audit import guard_source_pairings
from imas_codex.standard_names.workers import _is_attachment_consistent


@pytest.mark.parametrize(
    "primitive",
    ["thick_line", "outline", "rectangle", "oblique", "arcs_of_circle"],
)
def test_solid_hardware_geometry_cannot_source_optical_locus(primitive: str) -> None:
    path = f"pf_active/coil/element/geometry/{primitive}/r"

    accepted, reason = _is_attachment_consistent(
        path,
        "radial_coordinate_of_line_of_sight",
        dd_unit="m",
        sn_unit="m",
    )

    assert not accepted
    assert "geometry representation mismatch" in reason


def test_optical_path_cannot_source_solid_geometry_locus() -> None:
    accepted, reason = _is_attachment_consistent(
        "bremsstrahlung_visible/channel/line_of_sight/first_point/r",
        "radial_coordinate_of_rectangle",
        dd_unit="m",
        sn_unit="m",
    )

    assert not accepted
    assert "geometry representation mismatch" in reason


def test_valid_sightline_coordinate_remains_accepted() -> None:
    accepted, reason = _is_attachment_consistent(
        "bremsstrahlung_visible/channel/line_of_sight/first_point/r",
        "radial_coordinate_of_line_of_sight",
        dd_unit="m",
        sn_unit="m",
    )

    assert accepted, reason


def test_valid_hardware_geometry_coordinate_remains_accepted() -> None:
    accepted, reason = _is_attachment_consistent(
        "pf_active/coil/element/geometry/rectangle/r",
        "radial_coordinate_of_rectangle",
        dd_unit="m",
        sn_unit="m",
    )

    assert accepted, reason


def test_pairing_guard_preserves_existing_and_rejects_only_fresh_conflict() -> None:
    class FakeGraph:
        def query(self, _cypher: str, **_params):
            return [
                {
                    "source_id": "dd:existing",
                    "source_type": "dd",
                    "dd_path": "pf_active/coil/element/geometry/rectangle/r",
                    "dd_unit": "m",
                    "sn_unit": "m",
                    "already_bound": True,
                    "existing_dd_paths": [],
                    "name_stage": "drafted",
                },
                {
                    "source_id": "dd:fresh",
                    "source_type": "dd",
                    "dd_path": "pf_active/coil/element/geometry/rectangle/r",
                    "dd_unit": "m",
                    "sn_unit": "m",
                    "already_bound": False,
                    "existing_dd_paths": [],
                    "name_stage": "drafted",
                },
            ]

    result = guard_source_pairings(
        FakeGraph(),
        "radial_coordinate_of_line_of_sight",
        ["dd:fresh", "dd:existing"],
    )

    assert result.accepted_source_ids == ("dd:existing",)
    assert [item.source_node_id for item in result.rejected] == ["dd:fresh"]


def test_bulk_writer_does_not_materialize_rejected_dd_pairing() -> None:
    from imas_codex.standard_names.graph_ops import write_standard_names

    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def query(self, cypher: str, **params):
            self.calls.append((cypher, params))
            if "existing_dd_paths" in cypher:
                return [
                    {
                        "id": "radial_coordinate_of_line_of_sight",
                        "source_id": ("pf_active/coil/element/geometry/rectangle/r"),
                        "dd_unit": "m",
                        "sn_unit": "m",
                        "existing_dd_paths": [],
                    }
                ]
            return []

    gc = FakeGraph()
    written = write_standard_names(
        [
            {
                "id": "radial_coordinate_of_line_of_sight",
                "source_types": ["dd"],
                "source_id": "pf_active/coil/element/geometry/rectangle/r",
                "unit": "m",
            }
        ],
        gc=gc,
    )

    projection_calls = [
        params
        for cypher, params in gc.calls
        if "MERGE (src)-[:HAS_STANDARD_NAME]->(sn)" in cypher
    ]
    assert written == 1
    assert projection_calls == [{"batch": []}]


# ---------------------------------------------------------------------------
# Write-time prevention on the claimed-source persisters
# ---------------------------------------------------------------------------
#
# The compose-time gate sees only the candidate's same-batch siblings and never
# the units, so a source binding onto an ESTABLISHED name could contradict the
# sources that name already carries and still reach the graph — caught later by
# the retrospective reconcile rather than refused at the boundary. Both claimed
# source persisters lock their bindings through _lock_claimed_name_bindings,
# which is where the guard now runs.


_CONFLICTING_ATTACHMENT = {
    "sns_id": "dd:pf_active/coil/geometry/outline/r",
    "source_id": "pf_active/coil/geometry/outline/r",
    "standard_name": "radial_outline_of_poloidal_field_coil",
    "claim_token": "attachment-winner",
    "claim_seq": 5,
}


class _RecordingTransaction:
    """A transaction that answers by query content and records every call."""

    def __init__(self, *, guard_rows: list[dict], outcome: str) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        self.committed = False
        self._guard_rows = guard_rows
        self._outcome = outcome

    def run(self, cypher: str, **params):
        self.calls.append((cypher, params))
        if "names" in params:
            # Target-existence probe: compose may mint a name, and only an
            # established one carries sources a fresh pairing can contradict.
            return [{"id": name} for name in params["names"]]
        if "AS already_bound" in cypher:
            return self._guard_rows
        if "AS outcome" in cypher:
            return [
                {
                    "id": item["sns_id"],
                    "outcome": self._outcome,
                    "binding_kind": "stable_reuse",
                    "candidate_id": item["sn_id"],
                    "target_stage": "accepted",
                    "attempt_count": 1,
                }
                for item in params.get("batch", [])
            ]
        return [{"id": item["sns_id"]} for item in params.get("batch", [])]

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True

    def cypher_with(self, fragment: str) -> str:
        return next(cypher for cypher, _ in self.calls if fragment in cypher)

    def params_with(self, fragment: str) -> dict:
        return next(params for cypher, params in self.calls if fragment in cypher)


def _graph_returning(tx):
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    return gc


def _guard_row(dd_path: str, *, existing: list[str], unit: str = "m") -> dict:
    return {
        "source_id": _CONFLICTING_ATTACHMENT["sns_id"],
        "source_type": "dd",
        "dd_path": dd_path,
        "dd_unit": unit,
        "sn_unit": unit,
        "already_bound": False,
        "existing_dd_paths": existing,
        "name_stage": "accepted",
    }


def test_attachment_persist_refuses_a_pairing_the_name_contradicts() -> None:
    """A distinct vector field of one device cannot join a name's sources."""
    from unittest.mock import patch

    from imas_codex.standard_names.graph_ops import persist_claimed_attachments

    # The name already carries the coil's arc-of-circle description; the coil's
    # outline polygon is a different geometry of the same device.
    tx = _RecordingTransaction(
        guard_rows=[
            _guard_row(
                "pf_active/coil/geometry/outline/r",
                existing=["pf_active/coil/geometry/arcs_of_circle/r"],
            )
        ],
        outcome="attachment_collision",
    )
    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=_graph_returning(tx),
    ):
        winners = persist_claimed_attachments([dict(_CONFLICTING_ATTACHMENT)])

    assert winners == []
    # The refusal reaches the write as per-pairing state, not as a failed batch.
    locked = tx.params_with("AS outcome")["batch"]
    assert len(locked) == 1
    assert "geometry" in (locked[0]["attachment_error"] or "").lower()
    # No provenance or projection edge was written for the refused pairing.
    assert not [
        cypher
        for cypher, _ in tx.calls
        if "MERGE (src)-[:HAS_STANDARD_NAME]->(sn)" in cypher
    ]
    # The source is released with the guard's reason, not a lifecycle message.
    release = tx.cypher_with("sns.last_error = CASE")
    assert "b.attachment_error IS NOT NULL" in release
    assert "standard-name attachment rejected" in release
    assert tx.committed


def test_attachment_persist_lands_a_consistent_pairing() -> None:
    from unittest.mock import patch

    from imas_codex.standard_names.graph_ops import persist_claimed_attachments

    # Same device, same geometry primitive: a second coordinate of the outline.
    tx = _RecordingTransaction(
        guard_rows=[
            _guard_row(
                "pf_active/coil/geometry/outline/r",
                existing=["pf_active/coil/geometry/outline/z"],
            )
        ],
        outcome="winner",
    )
    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=_graph_returning(tx),
    ):
        winners = persist_claimed_attachments([dict(_CONFLICTING_ATTACHMENT)])

    assert winners == [_CONFLICTING_ATTACHMENT["sns_id"]]
    locked = tx.params_with("AS outcome")["batch"]
    assert locked[0].get("attachment_error") is None
    assert tx.cypher_with("MERGE (src)-[:HAS_STANDARD_NAME]->(sn)")


def test_guard_runs_before_the_binding_query_reserves_the_edge() -> None:
    """Ordering is the whole mechanism.

    The binding query stamps a provisional PRODUCED_NAME reservation, and the
    guard reads exactly that edge to decide a source is already bound. Run the
    other way round, it would admit every pairing it exists to judge.
    """
    from unittest.mock import patch

    from imas_codex.standard_names.graph_ops import persist_claimed_attachments

    tx = _RecordingTransaction(
        guard_rows=[
            _guard_row(
                "pf_active/coil/geometry/outline/r",
                existing=["pf_active/coil/geometry/arcs_of_circle/r"],
            )
        ],
        outcome="attachment_collision",
    )
    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=_graph_returning(tx),
    ):
        persist_claimed_attachments([dict(_CONFLICTING_ATTACHMENT)])

    order = [
        "guard" if "AS already_bound" in cypher else "bind"
        for cypher, _ in tx.calls
        if "AS already_bound" in cypher or "AS outcome" in cypher
    ]
    assert order[:2] == ["guard", "bind"]


def test_compose_persist_guards_a_binding_onto_an_established_name() -> None:
    """The stable-reuse branch is the phantom-attach case, and it is gated."""
    from unittest.mock import patch

    from imas_codex.standard_names.graph_ops import persist_generated_name_winners

    tx = _RecordingTransaction(
        guard_rows=[
            {
                "source_id": "dd:b_field_non_axisymmetric/control_surface/outline/z",
                "source_type": "dd",
                "dd_path": "b_field_non_axisymmetric/control_surface/outline/z",
                "dd_unit": "m",
                "sn_unit": "m",
                "already_bound": False,
                "existing_dd_paths": [
                    "b_field_non_axisymmetric/control_surface/normal_vector/z"
                ],
                "name_stage": "accepted",
            }
        ],
        outcome="attachment_collision",
    )
    candidate = {
        "id": "vertical_coordinate_of_control_surface",
        "source_id": "b_field_non_axisymmetric/control_surface/outline/z",
        "source_types": ["dd"],
        "kind": "scalar",
        "description": "Vertical coordinate of the control surface.",
        "unit": "m",
        "source_claim_token": "compose-winner",
        "source_claim_seq": 3,
    }
    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=_graph_returning(tx),
    ):
        finalized = persist_generated_name_winners(
            [candidate], compose_model="test/model"
        )

    assert finalized == []
    locked = tx.params_with("AS outcome")["batch"]
    assert "distinct" in (locked[0]["attachment_error"] or "").lower()
