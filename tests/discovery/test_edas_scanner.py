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
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from imas_codex.discovery.signals.scanners.edas import EDASScanner
from imas_codex.graph.models import FacilitySignal, SignalDataClass
from imas_codex.remote.scripts.enumerate_edas import (
    attempt_database,
    enumerate_lcdb,
    enumerate_uddb,
    main as enumerate_main,
)

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


class TestConfiguredDatabases:
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
                assert kwargs["cat"] == ""
                return False, {"irc": 301}

        class LargeAnalysisCatalogue:
            def __init__(self, path):
                assert path.endswith("libmbdb.so")

            def mbdbROpen(self, **kwargs):
                assert kwargs["category"] == "MBEQ"
                return False, {"irtn": 1012}

        monkeypatch.setitem(
            sys.modules, "pmdb_wrapper", SimpleNamespace(pmdbWrapper=PlantCatalogue)
        )
        monkeypatch.setitem(
            sys.modules,
            "mbdbWrapper",
            SimpleNamespace(mbdbWrapper=LargeAnalysisCatalogue),
        )
        assert attempt_database("PMDB", "E101173", {})["return_code"] == 301
        assert attempt_database("MBDB", "E101173", {})["return_code"] == 1012
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
