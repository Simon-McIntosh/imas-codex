from imas_codex.standard_names.graph_store.graph_audits import (
    fetch_manifest_source_release_rows,
)


def test_manifest_source_projection_transcribes_recorded_causes() -> None:
    class _ProjectionClient:
        def __init__(self) -> None:
            self.calls = 0

        def query(self, _cypher: str, **_params):
            self.calls += 1
            if self.calls == 1:
                return [
                    {
                        "source_path": "failed/source",
                        "source_status": "failed",
                        "skip_reason": "older_skip",
                        "skip_reason_detail": "older detail",
                        "last_error": "compose claim-attempt cap reached",
                        "produced_sn_id": None,
                        "direct_ids": [],
                    },
                    {
                        "source_path": "skipped/source",
                        "source_status": "not_physical_quantity",
                        "skip_reason": "dd_node_category_ineligible",
                        "skip_reason_detail": "fit artifact",
                        "last_error": None,
                        "produced_sn_id": None,
                        "direct_ids": [],
                    },
                    {
                        "source_path": "silent/source",
                        "source_status": "extracted",
                        "skip_reason": None,
                        "skip_reason_detail": None,
                        "last_error": None,
                        "produced_sn_id": None,
                        "direct_ids": [],
                    },
                ]
            raise AssertionError("the fixture has no successor query")

    rows = fetch_manifest_source_release_rows(
        ["failed/source", "skipped/source", "silent/source"],
        gc=_ProjectionClient(),
    )

    reasons = {row["source_path"]: row["non_nameable_reason"] for row in rows}
    assert reasons == {
        "failed/source": "compose claim-attempt cap reached",
        "skipped/source": "dd_node_category_ineligible: fit artifact",
        "silent/source": "cause not recorded",
    }
