"""A source member can recover when its representative lacks context."""

import hashlib

from imas_codex.discovery.signals import parallel


class _Graph:
    def __init__(self, representative_status):
        self.representative_status = representative_status
        self.member = {
            "id": "jt-60sa:member",
            "status": "discovered",
            "accessor": "read member",
            "data_source_path": "general/member",
            "source_description": "Member's own catalogue text",
            "facility_id": "jt-60sa",
        }
        self.claim_token = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def query(self, statement, **params):
        if "RETURN count(s) > 0 AS has_work" in statement:
            return [{"has_work": self._claimable(statement)}]
        if "SET s.status = $skipped" in statement:
            return []
        if "SET s.claimed_at = datetime(), s.claim_token = $token" in statement:
            if self._claimable(statement):
                self.claim_token = params["token"]
            return []
        if "claim_token: $token" in statement:
            if self.claim_token == params["token"]:
                return [dict(self.member)]
            return []
        if "SET s.description = item.description" in statement:
            item = params["items"][0]
            self.member["description"] = item["description"]
            self.member["embedding"] = item["embedding"]
            self.member["embedded_text_hash"] = item["embedded_text_hash"]
            return [{"updated": 1}]
        if "SET s.status = $enriched" in statement:
            self.member["status"] = params["enriched"]
            self.claim_token = None
            return []
        raise AssertionError(statement)

    def _claimable(self, statement):
        return (
            self.member["status"] == "discovered"
            and self.representative_status == "underspecified"
            and "rep.status = 'underspecified'" in statement
        )


def _embed(rows, *, text_field):
    for row in rows:
        row["embedding"] = [float(len(row[text_field]))]


def test_underspecified_source_member_leaves_discovered_with_its_own_text(monkeypatch):
    graph = _Graph("underspecified")
    monkeypatch.setattr(parallel, "GraphClient", lambda: graph)
    monkeypatch.setattr(parallel, "embed_descriptions_batch", _embed)

    assert parallel.has_pending_enrich_work("jt-60sa") is True
    claimed = parallel.claim_signals_for_enrichment("jt-60sa", batch_size=1)
    assert [row["id"] for row in claimed] == [graph.member["id"]]
    assert claimed[0]["source_description"] == "Member's own catalogue text"

    description = f"Describes {claimed[0]['source_description']}"
    assert (
        parallel.mark_signals_enriched(
            [
                {
                    "id": claimed[0]["id"],
                    "description": description,
                    "physics_domain": "general",
                    "name": "Member signal",
                    "diagnostic": "",
                    "analysis_code": "",
                    "keywords": [],
                    "sign_convention": "",
                    "context_quality": "medium",
                }
            ]
        )
        == 1
    )
    assert graph.member["status"] == "enriched"
    assert graph.member["description"] == description
    assert graph.member["embedding"] == [float(len(description))]
    assert (
        graph.member["embedded_text_hash"]
        == hashlib.sha256(description.encode()).hexdigest()
    )
    assert parallel.has_pending_enrich_work("jt-60sa") is False


def test_member_waits_for_an_enriched_representative(monkeypatch):
    graph = _Graph("enriched")
    monkeypatch.setattr(parallel, "GraphClient", lambda: graph)
    assert parallel.has_pending_enrich_work("jt-60sa") is False
    assert parallel.claim_signals_for_enrichment("jt-60sa", batch_size=1) == []
