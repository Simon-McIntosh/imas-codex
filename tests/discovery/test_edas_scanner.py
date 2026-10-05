"""Tests for the EDAS (JT-60SA) signal scanner.

Covers the typed-slot contract on FacilitySignal for the values the EDDB
catalogue returns: the scanner routes the data class, shot range and PID
into `data_class`, `shot_range` and `pid`, and `check()` reads the data
class back from the slot rather than parsing it out of `keywords`.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from imas_codex.discovery.signals.scanners.edas import EDASScanner
from imas_codex.graph.models import FacilitySignal, SignalDataClass

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
