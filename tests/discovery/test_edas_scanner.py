"""Tests for the EDAS (JT-60SA) signal scanner.

Covers the typed-slot contract on FacilitySignal for the values the EDDB
catalogue returns: the scanner routes the data class, shot range and PID
into `data_class`, `shot_range` and `pid`, and `check()` reads the data
class back from the slot rather than parsing it out of `keywords`.
"""

from __future__ import annotations

import json
import re
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import yaml

from imas_codex.discovery.signals.parallel import DataDiscoveryState, seed_worker
from imas_codex.discovery.signals.scanners.base import ScanResult
from imas_codex.discovery.signals.scanners.edas import EDASScanner
from imas_codex.graph.models import DataAccess, FacilitySignal, SignalDataClass
from imas_codex.remote.scripts.check_edas import main as check_main
from imas_codex.remote.scripts.enumerate_edas import (
    attempt_database,
    enumerate_lcdb,
    enumerate_mbdb,
    enumerate_uddb,
    main as enumerate_main,
)
from imas_codex.remote.scripts.read_equilibrium_file import resolve_eqdb_path

# Two catalogue rows covering the two classes the scanner branches on:
# a time series with a shot range and no PID, and a one-point row with a PID.
ENUMERATE_FIXTURE = {
    "signals": [
        {
            "category": "MAG",
            "data_name": "CoilCur",
            "units": "A",
            "description": "coil current",
            "alias": "CC",
            "data_class": "T",
            "shot_range": "E080000-",
            "udp_id": "",
        },
        {
            "category": "MDAC",
            "data_name": "Status",
            "units": "",
            "description": "status",
            "alias": "",
            "data_class": "O",
            "shot_range": "E080000-E101173",
            "udp_id": "1234ABCDE",
        },
    ],
    "ncats": 2,
    "categories": ["MAG", "MDAC"],
}

CONFIG = {
    "reference_shot": 101173,
    "api_path": "/opt/edas/api",
    "lib_path": "/opt/edas/libeddb.so",
}


def _equilibrium_config():
    root = Path(__file__).parents[2]
    facility = yaml.safe_load(
        (root / "imas_codex/config/facilities/jt-60sa.yaml").read_text()
    )
    return root, facility["data_systems"]["edas"]


async def test_equilibrium_access_methods_and_inventory_field_signals():
    root, config = _equilibrium_config()
    inventory = json.loads(
        (
            root
            / "docs/evidence/fragments/jt60sa-discovery-completion/flux-map-inventory.json"
        ).read_text()
    )
    by_path = {record["path"]: record for record in inventory["files"]}
    for example in config["equilibrium_examples"]:
        recorded = by_path[example["path"]]
        assert recorded["format"] == example["format"]
        assert recorded["code"] == example.get("inventory_code", example["code"])
        assert recorded["shot"] == example["shot"]
        assert recorded["time_s"] == example["time"]
        assert recorded["grid"] == example["grid"]

    remote = AsyncMock(return_value=json.dumps(ENUMERATE_FIXTURE))
    with patch("imas_codex.remote.executor.async_run_python_script", remote):
        result = await EDASScanner().scan("jt-60sa", "nakasvr26", config)

    assert result.data_access.id == "jt-60sa:edas:eddb"
    assert {access.id for access in result.data_accesses} == {
        "jt-60sa:eqdb:local_record",
        "jt-60sa:equilibrium:g_eqdsk_file",
    }
    assert "dbsvr17p" in result.data_accesses[0].description
    equilibrium = [s for s in result.signals if s.physics_domain == "equilibrium"]
    assert len(equilibrium) == 22
    assert {s.data_source_path.split("/")[0] for s in equilibrium} == {
        "SELENE",
        "TOPICS",
        "SA",
        "LIUQE",
        "CHEASE",
    }
    assert {s.accessor for s in equilibrium if s.data_source_name == "EQDB"} == {
        "RG",
        "ZG",
        "NSR",
        "NSZ",
        "PSI",
    }
    assert {s.accessor for s in equilibrium if s.data_source_name == "G-EQDSK"} == {
        "r",
        "z",
        "psirz",
        "COCOS",
    }
    assert {s.cocos for s in equilibrium if s.data_source_name == "G-EQDSK"} == {
        2,
        7,
        17,
    }
    for signal in equilibrium:
        example = next(
            item
            for item in config["equilibrium_examples"]
            if item["code"] == signal.data_source_path.split("/")[0]
        )
        assert example["path"] in signal.description
        assert signal.example_shot == example["shot"]


async def test_eqdb_local_template_uses_requested_shot_and_time():
    _, config = _equilibrium_config()
    remote = AsyncMock(return_value=json.dumps(ENUMERATE_FIXTURE))
    with patch("imas_codex.remote.executor.async_run_python_script", remote):
        result = await EDASScanner().scan("jt-60sa", "nakasvr26", config)
    access = result.data_accesses[0]
    example = config["equilibrium_examples"][0]
    calls = []

    def fake_run(script, payload, **kwargs):
        calls.append((script, payload, kwargs))
        return json.dumps({"data": [1.0], "grid": example["grid"]})

    code = (
        access.connection_template
        + "\n"
        + access.data_template.format(
            root=example["root"],
            shot=example["shot"],
            time=example["time"],
            field="PSI",
            ssh_host="nakasvr26",
        )
    )
    with patch("imas_codex.remote.executor.run_python_script", side_effect=fake_run):
        exec(code, {})
    assert calls[0][0] == "read_equilibrium_file.py"
    assert calls[0][1] == {
        "format": "eqdb_record",
        "root": example["root"],
        "shot": example["shot"],
        "time": example["time"],
        "field": "PSI",
    }

    geqdsk = result.data_accesses[1]
    example = next(
        item for item in config["equilibrium_examples"] if item["code"] == "LIUQE"
    )
    code = (
        geqdsk.connection_template
        + "\n"
        + geqdsk.data_template.format(
            root=example["root"],
            shot=example["shot"],
            time=example["time"],
            filename_template=example["filename_template"],
            field="psirz",
            ssh_host="nakasvr26",
        )
    )
    with patch("imas_codex.remote.executor.run_python_script", side_effect=fake_run):
        exec(code, {})
    assert calls[1][1] == {
        "format": "g_eqdsk",
        "root": example["root"],
        "shot": example["shot"],
        "time": example["time"],
        "filename_template": example["filename_template"],
        "field": "psirz",
    }


def test_eqdb_local_path_uses_shot_time_and_single_file_override(tmp_path, monkeypatch):
    monkeypatch.delenv("EQDB_FILE", raising=False)
    first = resolve_eqdb_path(str(tmp_path), 101163, 5.0)
    second = resolve_eqdb_path(str(tmp_path), 101164, 5.0)
    assert first == tmp_path / "10/1011/101163/005000"
    assert second == tmp_path / "10/1011/101164/005000"
    assert first != second
    shorter = tmp_path / "10/1011/101163/05000"
    shorter.parent.mkdir(parents=True)
    shorter.touch()
    assert resolve_eqdb_path(str(tmp_path), 101163, 5.0) == shorter
    monkeypatch.setenv("EQDB_FILE", str(tmp_path / "single_record"))
    assert resolve_eqdb_path(str(tmp_path), 101163, 5.0) == tmp_path / "single_record"


async def test_equilibrium_check_reads_each_code_example_once():
    _, config = _equilibrium_config()
    scanner = EDASScanner()
    with patch(
        "imas_codex.remote.executor.async_run_python_script",
        AsyncMock(return_value=json.dumps(ENUMERATE_FIXTURE)),
    ):
        scanned = await scanner.scan("jt-60sa", "nakasvr26", config)
    equilibrium = [
        signal
        for signal in scanned.signals
        if signal.data_source_name in {"EQDB", "G-EQDSK"}
    ]
    reads = []

    async def fake_run(script, payload, **_kwargs):
        assert script == "read_equilibrium_file.py"
        example = next(
            item
            for item in config["equilibrium_examples"]
            if item["root"] == payload["root"] and item["shot"] == payload["shot"]
        )
        reads.append(example["code"])
        return json.dumps(
            {
                "format": example["format"],
                "grid": example["grid"],
                "path": example["path"],
                "cocos": example.get("cocos"),
            }
        )

    with patch("imas_codex.remote.executor.async_run_python_script", fake_run):
        results = await scanner.check("jt-60sa", "nakasvr26", equilibrium, config)
    assert len(results) == len(equilibrium)
    assert all(item["valid"] for item in results)
    assert sorted(reads) == sorted(
        example["code"] for example in config["equilibrium_examples"]
    )


async def test_scanner_access_methods_share_the_existing_persistence_writer():
    """All access nodes must exist before signal ingestion creates their edges."""
    methods = [
        DataAccess(
            id=f"jt-60sa:edas:access_{name}",
            facility_id="jt-60sa",
            method_type="edas",
            library="test",
            access_type="local",
            data_source=name,
            connection_template="",
            data_template="",
        )
        for name in ("eddb", "eqdb", "g_eqdsk")
    ]
    signal = FacilitySignal(
        id="jt-60sa:equilibrium/test_psi",
        facility_id="jt-60sa",
        name="psi",
        accessor="psi",
        data_access=methods[-1].id,
    )

    class Scanner:
        scanner_type = "edas"

        async def scan(self, **_kwargs):
            return ScanResult(
                signals=[signal],
                data_access=methods[0],
                data_accesses=[methods[1], methods[2], methods[0]],
            )

    state = DataDiscoveryState(
        facility="jt-60sa",
        ssh_host="nakasvr26",
        scanner_types=["edas"],
        facility_config={"data_systems": {"edas": CONFIG}},
        initial_version_counts={"total": 0},
        initial_signal_counts={"total": 0},
        cost_limit=10.0,
    )
    events = []
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.__exit__.return_value = None
    graph.query.side_effect = lambda query, **kw: (
        events.append(kw["id"]) if "MERGE (da:DataAccess" in query else []
    )

    with (
        patch(
            "imas_codex.discovery.signals.scanners.base.get_scanner",
            return_value=Scanner(),
        ),
        patch("imas_codex.discovery.signals.parallel.GraphClient", return_value=graph),
        patch(
            "imas_codex.discovery.signals.parallel.ingest_discovered_signals",
            side_effect=lambda rows: events.append("ingest") or len(rows),
        ),
    ):
        await seed_worker(state)

    assert events == [method.id for method in methods] + ["ingest"]


class TestConfiguredDatabases:
    def test_raw_check_uses_global_catalogue_and_rejects_unknown_pid(
        self, monkeypatch, capsys
    ):
        class ProcessedCatalogue:
            def __init__(self, path):
                pass

            def eddbOpen(self):
                return True

            def eddbClose(self):
                return True

        class RawCatalogue:
            def __init__(self, path):
                pass

            def uddbOpen(self):
                return True

            def uddbreadTable(self, pid):
                if pid == "2111UA001":
                    return True, {"data": [pid], "count": 1, "irc": 0}
                return False, {"data": [], "count": 0, "irc": 1111}

            def uddbClose(self):
                return True

        monkeypatch.setitem(
            sys.modules,
            "eddb_pwrapper",
            SimpleNamespace(eddbWrapper=ProcessedCatalogue),
        )
        monkeypatch.setitem(
            sys.modules, "uddb_pwrapper", SimpleNamespace(uddbWrapper=RawCatalogue)
        )
        monkeypatch.setattr(
            sys,
            "stdin",
            StringIO(
                json.dumps(
                    {
                        "ref_shot": "E101173",
                        "api_path": "/analysis/src/eddb",
                        "lib_path": "/analysis/lib/libeddb.so",
                        "signals": [
                            {"id": "known", "database": "UDDB", "pid": "2111UA001"},
                            {"id": "unknown", "database": "UDDB", "pid": "NO_SUCH_PID"},
                        ],
                    }
                )
            ),
        )
        check_main()
        results = json.loads(capsys.readouterr().out)["results"]
        assert [result["success"] for result in results] == [True, False]
        assert results[0]["dtype"] == "raw_catalogue"
        assert "absent from catalogue" in results[1]["error"]

    def test_mbdb_opened_case_exposes_native_field_list(self, monkeypatch):
        class CaseCatalogue:
            def __init__(self, path):
                self.mbdb = object()

            def mbdbSetDirectory(self, name):
                assert name == "mbdb"
                return True, {"irtn": 0}

            def mbdbROpen(self, **kwargs):
                assert kwargs["caseno"] == 12345
                assert kwargs["category"] == "data.t001"
                return True, {"irtn": 0}

            def mbdbRClose(self):
                return True, {"irtn": 0}

        monkeypatch.setitem(
            sys.modules, "mbdbWrapper", SimpleNamespace(mbdbWrapper=CaseCatalogue)
        )
        monkeypatch.setattr(
            "imas_codex.remote.scripts.enumerate_edas.Path.glob",
            lambda self, pattern: [
                Path("/analysis_DB/MBDB/owner/01/0123/012345/mbdb/data.t001.ldb")
            ],
        )
        monkeypatch.setattr(
            "imas_codex.remote.scripts.enumerate_edas._mbdb_data_names",
            lambda lib: ([("PSI", "T")], 0),
        )
        rows, attempt = enumerate_mbdb({})
        assert attempt["return_code"] == 0
        assert attempt["count"] == 1
        assert rows[0]["category"] == "MBDB/owner/12345/data.t001"
        assert rows[0]["data_name"] == "PSI"

    async def test_mbdb_signal_keeps_case_and_field(self):
        fixture = {
            "signals": [
                {
                    "database": "MBDB",
                    "category": "MBDB/owner/12345/data.t001",
                    "file_category": "data.t001",
                    "data_name": "PSI",
                    "data_kind": "T",
                    "case": 12345,
                    "root": "/analysis_DB/MBDB/owner",
                }
            ],
            "categories": [],
            "ncats": 0,
        }
        remote = AsyncMock(return_value=json.dumps(fixture))
        with patch("imas_codex.remote.executor.async_run_python_script", remote):
            result = await EDASScanner().scan("jt-60sa", "nakasvr26", CONFIG)
        signal = result.signals[0]
        assert signal.id == "jt-60sa:general/mbdb_owner_12345_data.t001_psi"
        assert signal.data_source_name == "MBDB"
        assert signal.data_source_path == "owner/12345/data.t001/PSI"
        assert signal.data_class == SignalDataClass.time_series
        check_remote = AsyncMock(
            return_value=json.dumps({"results": [{"id": signal.id, "success": True}]})
        )
        with patch("imas_codex.remote.executor.async_run_python_script", check_remote):
            checked = await EDASScanner().check(
                "jt-60sa", "nakasvr26", [signal], CONFIG
            )
        sent = check_remote.call_args.args[1]["signals"][0]
        assert (sent["database"], sent["case"], sent["category"]) == (
            "MBDB",
            12345,
            "data.t001",
        )
        assert checked[0]["valid"] is True

    def test_lcdb_wrapper_catalogue_is_enumerated(self, monkeypatch):
        class AnalysisCatalogue:
            def lcdb_shot(self, **kwargs):
                assert kwargs["root"] == "/analysis_DB/EDASDB/owner"
                return True, [63632]

            def lcdb_dname(self, shot, category, **kwargs):
                assert (shot, category) == (63632, "ane.s001")
                return True, ["DNAME", "NFIT"]

        monkeypatch.setitem(
            sys.modules,
            "lcdbWrapper",
            SimpleNamespace(LcdbWrapper=AnalysisCatalogue, lib=object()),
        )
        monkeypatch.setattr(
            "imas_codex.remote.scripts.enumerate_edas.os.scandir",
            lambda path: [SimpleNamespace(path=f"{path}/owner", is_dir=lambda: True)],
        )
        monkeypatch.setattr(
            "imas_codex.remote.scripts.enumerate_edas._lcdb_categories",
            lambda lib, root, shot: (["ane.s001"], 0),
        )
        rows, attempt = enumerate_lcdb({"lcdb_root": "/analysis_DB/EDASDB"})
        assert attempt["return_code"] == 0
        assert attempt["count"] == 2
        assert {row["data_name"] for row in rows} == {"DNAME", "NFIT"}
        assert all(row["shot"] == 63632 for row in rows)

    async def test_lcdb_signal_keeps_owner_category_and_shot(self):
        fixture = {
            "signals": [
                {
                    "database": "LCDB",
                    "category": "LCDB/owner/ane.s001",
                    "file_category": "ane.s001",
                    "data_name": "NFIT",
                    "root": "/analysis_DB/EDASDB/owner",
                    "shot": 63632,
                }
            ],
            "categories": [],
            "ncats": 0,
        }
        remote = AsyncMock(return_value=json.dumps(fixture))
        with patch("imas_codex.remote.executor.async_run_python_script", remote):
            result = await EDASScanner().scan("jt-60sa", "nakasvr26", CONFIG)
        signal = result.signals[0]
        assert signal.id == "jt-60sa:general/lcdb_owner_ane.s001_nfit"
        assert signal.data_source_name == "LCDB"
        assert signal.data_source_path == "owner/ane.s001/NFIT"
        assert signal.example_shot == 63632
        assert "lcdb_value(63632, 'ane.s001', ['NFIT']" in signal.accessor

    def test_eddb_only_failure_is_reported_as_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "imas_codex.remote.scripts.enumerate_edas.enumerate_eddb",
            lambda *args: (
                [],
                [],
                [{"database": "EDDB", "call": "eddbOpen()", "return_code": 1}],
            ),
        )
        monkeypatch.setattr(
            sys,
            "stdin",
            StringIO(json.dumps({"ref_shot": "E101173", **CONFIG})),
        )
        enumerate_main()
        assert "error" in json.loads(capsys.readouterr().out)

    def test_other_database_attempts_keep_wrapper_return_codes(self, monkeypatch):
        class PlantCatalogue:
            def __init__(self, path):
                assert path.endswith("libpmdb.so")

            def plantdread(self, **kwargs):
                assert kwargs["cat"] == "*"
                return False, {"irc": 301}

            def planthread(self, **kwargs):
                assert kwargs["dname"] == "*"
                return False, {"irc": 301}

        monkeypatch.setitem(
            sys.modules, "pmdb_wrapper", SimpleNamespace(pmdbWrapper=PlantCatalogue)
        )
        plant_attempt = attempt_database("PMDB", "E101173", {})
        assert plant_attempt["return_code"] == 301
        assert plant_attempt["header_return_code"] == 301
        with patch(
            "imas_codex.remote.scripts.enumerate_edas.os.scandir",
            side_effect=FileNotFoundError(2, "No such file or directory"),
        ):
            assert attempt_database("EQDB", "E101173", {})["return_code"] == 2

    def test_uddb_table_rows_are_emitted(self, monkeypatch):
        class RawCatalogue:
            def __init__(self, path):
                assert path == "/analysis/lib/libuddb.so"

            def uddbOpen(self):
                return True

            def uddbreadTable(self):
                return True, {
                    "data": ["2111UA001"],
                    "aliaslist": ["raw channel"],
                    "shotlist": ["E080000-"],
                    "irc": 0,
                }

            def uddbClose(self):
                return True

        monkeypatch.setitem(
            sys.modules, "uddb_pwrapper", SimpleNamespace(uddbWrapper=RawCatalogue)
        )
        rows, attempt = enumerate_uddb({})
        assert attempt["return_code"] == 0
        assert attempt["count"] == 1
        assert rows == [
            {
                "database": "UDDB",
                "category": "UDDB",
                "data_name": "2111UA001",
                "alias": "raw channel",
                "shot_range": "E080000-",
            }
        ]

    async def test_uddb_catalogue_produces_distinct_raw_signal(self):
        config = {**CONFIG, "databases": ["EDDB", "UDDB"]}
        received = {}

        async def remote_run(script, payload, **kwargs):
            received.update(payload)
            return json.dumps(
                {
                    **ENUMERATE_FIXTURE,
                    "signals": [
                        *ENUMERATE_FIXTURE["signals"],
                        {
                            "database": "UDDB",
                            "category": "UDDB",
                            "data_name": "2111UA001",
                            "alias": "raw channel",
                            "shot_range": "E080000-",
                        },
                    ],
                }
            )

        with patch("imas_codex.remote.executor.async_run_python_script", remote_run):
            result = await EDASScanner().scan("jt-60sa", "nakasvr26", config)

        assert received["databases"] == ["EDDB", "UDDB"]
        assert {s.id for s in result.signals} == {
            "jt-60sa:general/mag_coilcur",
            "jt-60sa:general/mdac_status",
            "jt-60sa:general/uddb_2111ua001",
        }
        raw = next(s for s in result.signals if s.data_source_name == "UDDB")
        assert raw.accessor == "uddbreadConvert('E101173', '2111UA001', t1, t2)"

    async def test_raw_signal_check_uses_uddb_pid(self):
        signal = FacilitySignal(
            id="jt-60sa:general/uddb_2111ua001",
            facility_id="jt-60sa",
            name="UDDB/2111UA001",
            accessor="uddbreadConvert('E101173', '2111UA001', t1, t2)",
            data_source_name="UDDB",
            data_source_path="UDDB/2111UA001",
        )
        remote = AsyncMock(
            return_value=json.dumps({"results": [{"id": signal.id, "success": True}]})
        )
        with patch("imas_codex.remote.executor.async_run_python_script", remote):
            results = await EDASScanner().check(
                "jt-60sa", "nakasvr26", [signal], CONFIG
            )
        assert remote.call_args.args[1]["signals"] == [
            {"id": signal.id, "database": "UDDB", "pid": "2111UA001"}
        ]
        assert results[0]["valid"] is True


class TestEdasScanTypedSlots:
    """scan() writes the catalogue values into the typed slots."""

    async def test_scan_populates_typed_slots(self):
        scanner = EDASScanner()
        remote = AsyncMock(return_value=json.dumps(ENUMERATE_FIXTURE))

        with patch("imas_codex.remote.executor.async_run_python_script", remote):
            result = await scanner.scan(
                facility="jt-60sa",
                ssh_host="nakasvr26",
                config=CONFIG,
            )

        assert "error" not in result.stats, result.stats
        by_id = {s.id: s for s in result.signals}
        time_series = by_id["jt-60sa:general/mag_coilcur"]
        one_point = by_id["jt-60sa:general/mdac_status"]

        assert time_series.data_class == SignalDataClass.time_series
        assert time_series.shot_range == "E080000-"
        assert time_series.pid is None

        assert one_point.data_class == SignalDataClass.one_point
        assert one_point.shot_range == "E080000-E101173"
        assert one_point.pid == "1234ABCDE"

    async def test_scan_keywords_carry_no_typed_values(self):
        scanner = EDASScanner()
        remote = AsyncMock(return_value=json.dumps(ENUMERATE_FIXTURE))

        with patch("imas_codex.remote.executor.async_run_python_script", remote):
            result = await scanner.scan(
                facility="jt-60sa",
                ssh_host="nakasvr26",
                config=CONFIG,
            )

        for signal in result.signals:
            keywords = signal.keywords or []
            assert not any(
                k.startswith(("class:", "shots:", "udpid:")) for k in keywords
            ), f"{signal.id} still parks typed values in keywords: {keywords}"

    async def test_scan_unknown_data_class_leaves_slot_unset(self):
        scanner = EDASScanner()
        fixture = {
            "signals": [
                {
                    "category": "MAG",
                    "data_name": "Weird",
                    "data_class": "Z",
                    "shot_range": "",
                    "udp_id": "",
                }
            ],
            "ncats": 1,
            "categories": ["MAG"],
        }
        remote = AsyncMock(return_value=json.dumps(fixture))

        with patch("imas_codex.remote.executor.async_run_python_script", remote):
            result = await scanner.scan(
                facility="jt-60sa", ssh_host="nakasvr26", config=CONFIG
            )

        assert result.signals[0].data_class is None


class TestEdasCheckReadsSlot:
    """check() derives the data class from the slot, not from keywords."""

    async def test_check_reads_data_class_from_slot(self):
        scanner = EDASScanner()
        signal = FacilitySignal(
            id="jt-60sa:general/mdac_status",
            facility_id="jt-60sa",
            accessor="eddbreadOne('E101173', 'MDAC', 'Status', None, 0, 0)",
            name="MDAC/Status",
            data_class=SignalDataClass.one_point,
        )
        assert signal.keywords is None

        captured: dict = {}

        async def fake_run(script, payload, **kwargs):
            captured["script"] = script
            captured["payload"] = payload
            return json.dumps(
                {"results": [{"id": signal.id, "success": True, "dtype": "f8"}]}
            )

        with patch("imas_codex.remote.executor.async_run_python_script", fake_run):
            results = await scanner.check(
                facility="jt-60sa",
                ssh_host="nakasvr26",
                signals=[signal],
                config=CONFIG,
            )

        assert captured["script"] == "check_edas.py"
        sent = captured["payload"]["signals"][0]
        assert sent["data_class"] == "O"
        assert results[0]["valid"] is True


# Rows from the PID-keyed second pass of the enumeration: each carries the
# nine-character PID No. as its udp_id, and cites the source catalogue data
# name so the accessor can pass both. The second row carries no PID.
PID_KEYED_FIXTURE = {
    "signals": [
        {
            "category": "PSRC",
            "data_name": "AbnormFact",
            "source_dname": "AbnormFact",
            "units": "-",
            "description": "Abnormality factor",
            "alias": "",
            "data_class": "O",
            "shot_range": "",
            "udp_id": "1626 A006",
            "pid_keyed": True,
        },
        {
            "category": "MDAC",
            "data_name": "DaqStrTime",
            "source_dname": "DaqStrTime",
            "units": "ms",
            "description": "Data acquisition start time",
            "alias": "",
            "data_class": "O",
            "shot_range": "",
            "udp_id": "",
            "pid_keyed": True,
        },
    ],
    "ncats": 2,
    "categories": ["PSRC", "MDAC"],
}


class TestEdasPidKeyedGroup:
    """The PID pass persists the one-point data keyed by its PID No."""

    async def test_pid_keyed_rows_map_pid_and_one_point(self):
        scanner = EDASScanner()
        remote = AsyncMock(return_value=json.dumps(PID_KEYED_FIXTURE))

        with patch("imas_codex.remote.executor.async_run_python_script", remote):
            result = await scanner.scan(
                facility="jt-60sa", ssh_host="nakasvr26", config=CONFIG
            )

        by_id = {s.id: s for s in result.signals}
        keyed = by_id["jt-60sa:general/psrc_1626_a006"]
        assert keyed.pid == "1626 A006"
        assert keyed.data_class == SignalDataClass.one_point
        # Four digits then five characters, nine in all.
        assert re.match(r"^\d{4}.{5}$", keyed.pid)
        # The accessor reads the PID-keyed datum by both the source data name
        # and the PID.
        assert "'AbnormFact'" in keyed.accessor
        assert "'1626 A006'" in keyed.accessor

    async def test_pid_keyed_row_with_empty_pid_keeps_pid_unset(self):
        scanner = EDASScanner()
        remote = AsyncMock(return_value=json.dumps(PID_KEYED_FIXTURE))

        with patch("imas_codex.remote.executor.async_run_python_script", remote):
            result = await scanner.scan(
                facility="jt-60sa", ssh_host="nakasvr26", config=CONFIG
            )

        empty = next(
            s for s in result.signals if s.data_source_path == "MDAC/DaqStrTime"
        )
        assert empty.pid is None
        assert empty.data_class == SignalDataClass.one_point


class TestEdasCheckReadsCategoryFromSourcePath:
    """check() keys the catalogue off data_source_path, not a rewritten name."""

    async def test_enriched_name_still_sends_source_path_category(self):
        scanner = EDASScanner()
        # Enrichment replaced name with a human-readable label; the catalogue
        # path survives on data_source_path.
        signal = FacilitySignal(
            id="jt-60sa:general/edas_plasma_current",
            facility_id="jt-60sa",
            accessor="eddbreadTime('E101173', 'PSRC', 'calIp', '0', '0.01')",
            name="Plasma Current (Ip)",
            data_source_path="PSRC/calIp",
            data_class=SignalDataClass.time_series,
        )

        captured: dict = {}

        async def fake_run(script, payload, **kwargs):
            captured["payload"] = payload
            return json.dumps(
                {"results": [{"id": signal.id, "success": True, "dtype": "f8"}]}
            )

        with patch("imas_codex.remote.executor.async_run_python_script", fake_run):
            await scanner.check(
                facility="jt-60sa",
                ssh_host="nakasvr26",
                signals=[signal],
                config=CONFIG,
            )

        sent = captured["payload"]["signals"][0]
        assert sent["category"] == "PSRC"
        assert sent["data_name"] == "calIp"

    async def test_name_fallback_when_source_path_absent(self):
        scanner = EDASScanner()
        signal = FacilitySignal(
            id="jt-60sa:general/edas_status",
            facility_id="jt-60sa",
            accessor="eddbreadOne('E101173', 'MDAC', 'Status', None, 0, 0)",
            name="MDAC/Status",
            data_class=SignalDataClass.one_point,
        )

        captured: dict = {}

        async def fake_run(script, payload, **kwargs):
            captured["payload"] = payload
            return json.dumps(
                {"results": [{"id": signal.id, "success": True, "dtype": "f8"}]}
            )

        with patch("imas_codex.remote.executor.async_run_python_script", fake_run):
            await scanner.check(
                facility="jt-60sa",
                ssh_host="nakasvr26",
                signals=[signal],
                config=CONFIG,
            )

        sent = captured["payload"]["signals"][0]
        assert sent["category"] == "MDAC"
        assert sent["data_name"] == "Status"
