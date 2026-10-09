"""A source passes its enrichment to members linked after its representative."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imas_codex.discovery.signals import parallel


class SourceGraph:
    def __init__(self, source_status: str, representative_status: str):
        self.source = {"status": source_status, "members_described": True}
        self.signals = {
            "rep": {
                "id": "rep",
                "status": representative_status,
                "description": "An enriched source",
                "name": "Source name",
                "physics_domain": "plasma_control",
                "keywords": ["source"],
            }
        }
        self.member_updates = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def query(self, statement, **params):
        if "rep.id AS representative_id" in statement:
            assert "sg:SignalSource {facility_id: $facility}" in statement
            assert "member.status = $discovered" in statement
            assert self.signals["rep"]["status"] in params["enriched_statuses"]
            if any(
                s["status"] == "discovered"
                for k, s in self.signals.items()
                if k != "rep"
            ):
                return [
                    {
                        "representative_id": "rep",
                        "enrichment": self.signals["rep"].copy(),
                    }
                ]
            return []
        if "RETURN count(s) AS cnt" in statement:
            return [
                {
                    "cnt": sum(
                        s["status"] == "discovered"
                        for k, s in self.signals.items()
                        if k != "rep"
                    )
                }
            ]
        if "SET sg.status = 'enriched'" in statement:
            self.source.update(
                status="enriched",
                description=params["description"],
                members_described=False,
            )
            return []
        if "RETURN count(s) AS updated" in statement:
            updated = 0
            for key, signal in self.signals.items():
                if key != "rep" and signal["status"] == "discovered":
                    signal.update(
                        status=params["enriched"],
                        description=params["description"],
                        name=params["name"],
                        physics_domain=params["physics_domain"],
                        keywords=params["keywords"],
                        enrichment_source="signal_source_propagation",
                    )
                    updated += 1
            self.member_updates += updated
            return [{"updated": updated}]
        raise AssertionError(f"Unexpected graph query: {statement}")


@pytest.mark.parametrize(
    ("source_status", "representative_status"),
    [("discovered", "enriched"), ("enriched", "checked")],
)
def test_next_pass_enriches_member_linked_after_representative(
    source_status, representative_status
):
    graph = SourceGraph(source_status, representative_status)
    with (
        patch.object(parallel, "GraphClient", return_value=graph),
        patch.object(parallel, "detect_signal_sources", return_value=(0, 0)),
    ):
        assert parallel.prepare_signal_sources("jt-60sa") == (0, 0, 0)
        graph.signals["late"] = {"id": "late", "status": "discovered"}
        assert parallel.prepare_signal_sources("jt-60sa") == (0, 0, 1)
        assert graph.signals["late"]["description"] == "An enriched source"
        assert graph.signals["late"]["name"] == "Source name"
        assert graph.signals["late"]["physics_domain"] == "plasma_control"
        assert graph.signals["late"]["enrichment_source"] == "signal_source_propagation"
        assert graph.source["status"] == "enriched"
        assert graph.source["members_described"] is False
        snapshot = (graph.source.copy(), graph.signals["late"].copy())
        assert parallel.prepare_signal_sources("jt-60sa") == (0, 0, 0)
        assert (graph.source, graph.signals["late"]) == snapshot
        assert graph.member_updates == 1
