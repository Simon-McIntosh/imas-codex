"""EDDB catalogue classes select their readers and survive re-enumeration."""

import json
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from imas_codex.discovery.signals import parallel
from imas_codex.discovery.signals.scanners.edas import EDASScanner
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import FacilitySignal
from imas_codex.remote.scripts.check_edas import main as check_main


@pytest.mark.parametrize(
    ("letter", "data_class", "reader"),
    [
        ("T", "time_series", "eddbreadTime"),
        ("O", "one_point", "eddbreadOne"),
        ("P", "parameter", "eddbreadPara"),
        ("J", "parameter", "eddbreadPara"),
        ("M", "image", "eddbreadImage"),
        ("G", "binary", "eddbreadBinary"),
        ("N", "comment", "eddbreadComment"),
    ],
)
async def test_catalogue_class_selects_reader(letter, data_class, reader):
    payload = {
        "signals": [
            {
                "category": "CAM",
                "data_name": "channel",
                "data_class": letter,
                "units": "count",
            }
        ],
        "ncats": 1,
        "categories": ["CAM"],
    }
    remote = AsyncMock(return_value=json.dumps(payload))
    with patch("imas_codex.remote.executor.async_run_python_script", remote):
        result = await EDASScanner().scan(
            "jt-60sa",
            "jt-60sa",
            {"reference_shot": 101173, "api_path": "/api", "lib_path": "/lib"},
        )
    assert "error" not in result.stats
    signal = result.signals[0]
    assert signal.data_class == data_class
    assert signal.data_class_letter == letter
    assert signal.accessor.startswith(f"{reader}(")


@pytest.mark.parametrize("letter", ["A", "C", "K", "V"])
async def test_unclassified_signal_keeps_catalogue_letter(letter):
    payload = {
        "signals": [{"category": "CAM", "data_name": "channel", "data_class": letter}],
        "ncats": 1,
        "categories": ["CAM"],
    }
    remote = AsyncMock(return_value=json.dumps(payload))
    with patch("imas_codex.remote.executor.async_run_python_script", remote):
        result = await EDASScanner().scan(
            "jt-60sa",
            "jt-60sa",
            {"reference_shot": 101173, "api_path": "/api", "lib_path": "/lib"},
        )
    signal = result.signals[0]
    assert signal.data_class is None
    assert signal.data_class_letter == letter


@pytest.mark.parametrize("letter", ["J", "M", "G", "N", "A"])
async def test_check_payload_keeps_catalogue_letter(letter):
    interpreted = {"J": "parameter", "M": "image", "G": "binary", "N": "comment"}
    signal = FacilitySignal(
        id=f"jt-60sa:general/cam_{letter.lower()}",
        facility_id="jt-60sa",
        name=f"CAM/{letter}",
        accessor="",
        data_source_name="edas",
        data_source_path=f"CAM/{letter}",
        data_class=interpreted.get(letter),
        data_class_letter=letter,
    )
    remote = AsyncMock(return_value=json.dumps({"results": []}))
    with patch("imas_codex.remote.executor.async_run_python_script", remote):
        await EDASScanner().check(
            "jt-60sa",
            "jt-60sa",
            [signal],
            {"reference_shot": 101173, "api_path": "/api", "lib_path": "/lib"},
        )
    assert remote.call_args.args[1]["signals"][0]["data_class"] == letter


@pytest.mark.parametrize(
    ("letter", "reader", "response"),
    [
        ("P", "eddbreadPara", {"count": 1, "data": ["ready"]}),
        ("J", "eddbreadPara", {"count": 1, "data": ["ready"]}),
        ("M", "eddbreadImage", {"datasize": 4, "data": b"image"}),
        ("G", "eddbreadBinary", {"datasize": 4, "data": b"data"}),
        ("N", "eddbreadComment", {"data": "comment"}),
    ],
)
def test_remote_check_uses_class_reader(letter, reader, response, capsys):
    calls = []

    class Wrapper:
        def __init__(self, _library):
            pass

        def eddbOpen(self):
            return True

        def eddbClose(self):
            return True

        def __getattr__(self, name):
            def read(*args):
                calls.append((name, args))
                return True, response

            return read

    payload = {
        "ref_shot": "E101173",
        "api_path": "/api",
        "lib_path": "/lib",
        "signals": [
            {
                "id": "camera",
                "category": "CAM",
                "data_name": "channel",
                "data_class": letter,
            }
        ],
    }
    with (
        patch.object(sys, "stdin", StringIO(json.dumps(payload))),
        patch.dict(
            sys.modules, {"eddb_pwrapper": SimpleNamespace(eddbWrapper=Wrapper)}
        ),
    ):
        check_main()
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert [name for name, _ in calls] == [reader]
    assert result["success"] is True


def test_remote_check_refuses_unresolved_class(capsys):
    calls = []

    class Wrapper:
        def __init__(self, _library):
            pass

        def eddbOpen(self):
            return True

        def eddbClose(self):
            return True

        def __getattr__(self, name):
            def read(*_args):
                calls.append(name)
                return True, {"ntime": 1}

            return read

    payload = {
        "ref_shot": "E101173",
        "api_path": "/api",
        "lib_path": "/lib",
        "signals": [
            {
                "id": "unknown",
                "category": "CAM",
                "data_name": "channel",
                "data_class": "A",
            }
        ],
    }
    with (
        patch.object(sys, "stdin", StringIO(json.dumps(payload))),
        patch.dict(
            sys.modules, {"eddb_pwrapper": SimpleNamespace(eddbWrapper=Wrapper)}
        ),
    ):
        check_main()
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert calls == []
    assert result["success"] is False
    assert "class A" in result["error"]


@pytest.mark.graph
def test_rescan_refreshes_scanner_fields_without_replacing_enrichment():
    facility = f"edas-class-test:{uuid4()}"
    signal_id = f"{facility}:signal"
    scanned = {
        "id": signal_id,
        "facility_id": facility,
        "status": "discovered",
        "discovery_source": "edas",
        "accessor": "eddbreadImage('E101173', 'CAM', 'channel', 0)",
        "data_class": "image",
        "data_class_letter": "M",
        "unit": "pixel",
        "source_description": "Catalogue camera entry",
        "description": "Scanner wording",
        "name": "CAM/channel",
        "physics_domain": "general",
    }

    class TransactionClient:
        def __init__(self, transaction):
            self.transaction = transaction

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, statement, **params):
            return [dict(row) for row in self.transaction.run(statement, **params)]

    with GraphClient() as graph, graph.session() as session:
        transaction = session.begin_transaction()
        try:
            transaction.run("MERGE (:Facility {id: $id})", id=facility).consume()
            transaction.run(
                """CREATE (:FacilitySignal {
                    id: $id, facility_id: $facility, status: 'enriched',
                    description: 'Enriched description', name: 'Camera view',
                    physics_domain: 'diagnostics', keywords: ['camera'],
                    accessor: 'eddbreadTime()', data_class: 'time_series',
                    data_class_letter: 'T', unit: 'count', claim_token: 'held',
                    claimed_at: datetime(), source_description: 'Old catalogue'
                })""",
                id=signal_id,
                facility=facility,
            ).consume()
            with patch.object(
                parallel, "GraphClient", return_value=TransactionClient(transaction)
            ):
                assert parallel.ingest_discovered_signals([scanned]) == 1
            row = transaction.run(
                """MATCH (s:FacilitySignal {id: $id}) RETURN s AS signal""",
                id=signal_id,
            ).single()["signal"]
            assert row["accessor"] == scanned["accessor"]
            assert row["data_class"] == "image"
            assert row["data_class_letter"] == "M"
            assert row["unit"] == "pixel"
            assert row["source_description"] == "Catalogue camera entry"
            assert row["description"] == "Enriched description"
            assert row["name"] == "Camera view"
            assert row["physics_domain"] == "diagnostics"
            assert row["keywords"] == ["camera"]
            assert row["status"] == "enriched"
            assert row["claim_token"] == "held"
            assert row["claimed_at"] is not None
        finally:
            transaction.rollback()
