"""Resumable bounded archive cohorts with separate metadata and data provenance."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from imas_codex.discovery.signals.scanners.base import ScanResult, register_scanner
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import (
    ArchiveEndpoint,
    ArchiveSamplingMode,
    DataAccess,
    FacilitySignal,
    FacilitySignalStatus,
)
from imas_codex.remote.codac_uda import (
    ArchiveRequestError,
    archive_config,
    request_archive,
)

ARCHIVE_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $archives AS props
MERGE (a:ArchiveEndpoint {id: props.id})
SET a += props
MERGE (a)-[:AT_FACILITY]->(f)
RETURN count(a) AS written
"""

SIGNAL_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $signals AS props
MATCH (a:ArchiveEndpoint {id: props.metadata_archive})
MATCH (da:DataAccess {id: props.data_access})
MERGE (s:FacilitySignal {id: props.id})
ON CREATE SET s.status = props.status, s.discovered_at = datetime(),
              s.description = props.description
SET s.facility_id = props.facility_id, s.accessor = props.accessor,
    s.name = props.name, s.discovery_source = props.discovery_source,
    s.source_description = props.source_description, s.unit = props.unit,
    s.data_access = props.data_access, s.metadata_archive = props.metadata_archive,
    s.data_source_name = props.data_source_name,
    s.data_source_path = props.data_source_path,
    s.sampling_mode = props.sampling_mode, s.sampling_evidence = props.sampling_evidence,
    s.archive_cohorts = CASE WHEN $cohort IN coalesce(s.archive_cohorts, [])
                            THEN s.archive_cohorts
                            ELSE coalesce(s.archive_cohorts, []) + [$cohort] END
MERGE (s)-[:AT_FACILITY]->(f)
MERGE (s)-[:DATA_ACCESS]->(da)
WITH s, a
OPTIONAL MATCH (s)-[old:METADATA_ARCHIVE]->(previous)
WHERE previous <> a
DELETE old
MERGE (s)-[:METADATA_ARCHIVE]->(a)
RETURN count(DISTINCT s) AS written
"""

CONFIRM_QUERY = """
MATCH (s:FacilitySignal {id: $signal}), (a:ArchiveEndpoint {id: $archive})
WHERE a.excluded = false AND s.facility_id = a.facility_id
OPTIONAL MATCH (s)-[old:DATA_ARCHIVE]->(previous)
WHERE previous <> a
DELETE old
SET s.data_archive = a.id
MERGE (s)-[:DATA_ARCHIVE]->(a)
RETURN count(DISTINCT s) AS written
"""


def _save(path: Path, state: dict) -> None:
    """Replace a private checkpoint atomically; incomplete is the default state."""
    state["enumerated"] = len(state.get("selectors", state.get("entries", {})))
    state["metadata_completed"] = len(state.get("entries", {}))
    state["written"] = len(state.get("written_ids", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(state, stream)
        temporary = Path(stream.name)
    temporary.replace(path)


def _persist_archives(
    facility: str, archives: list[ArchiveEndpoint], access: DataAccess
) -> None:
    from imas_codex.discovery.signals.parallel import persist_data_access

    with GraphClient() as graph:
        rows = graph.query(
            ARCHIVE_QUERY,
            facility=facility,
            archives=[item.model_dump(mode="json") for item in archives],
        )
        if not rows or rows[0]["written"] != len(archives):
            raise RuntimeError(
                "archive write was not acknowledged; facility must exist"
            )
    persist_data_access(access)


def _persist_signals(facility: str, cohort: str, signals: list[FacilitySignal]) -> int:
    with GraphClient() as graph:
        rows = graph.query(
            SIGNAL_QUERY,
            facility=facility,
            cohort=cohort,
            signals=[signal.model_dump(mode="json") for signal in signals],
        )
    return rows[0]["written"] if rows else 0


def _confirm(signal: str, archive: str) -> None:
    with GraphClient() as graph:
        rows = graph.query(CONFIRM_QUERY, signal=signal, archive=archive)
    if not rows or rows[0]["written"] != 1:
        raise RuntimeError("data-serving archive confirmation was not acknowledged")


class CodacUDAScanner:
    """Enumerate and persist one configured cohort with recoverable progress.

    Checkpoints and selectors remain private cache data. A checkpoint becomes
    complete only after every pattern and every graph batch is acknowledged.
    Existing graph enrichment, claims and data-serving confirmation survive scans.
    """

    scanner_type = "codac_uda"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        settings = archive_config(
            facility, config if config.get("variable_patterns") else None
        )
        effective = settings.model_dump(mode="json")
        patterns = list(dict.fromkeys(settings.variable_patterns or []))
        if not patterns:
            raise ValueError(
                "declare a bounded variable cohort in private configuration"
            )
        role = settings.catalogue_role
        if role is None or role not in settings.readable_roles:
            raise ValueError(
                "declare a readable catalogue_role in private configuration"
            )
        # Resolve every candidate before any reader or graph write can begin.
        from imas_codex.config.codac_uda import resolve_archive_endpoint

        endpoints = [
            resolve_archive_endpoint(settings, item) for item in settings.readable_roles
        ]
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    facility,
                    role,
                    patterns,
                    [(item.role, item.host, item.port) for item in endpoints],
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        path = Path(
            config.get("_checkpoint_path")
            or Path.home()
            / ".cache/imas-codex-codac-uda/catalogues"
            / f"{facility}-{fingerprint}.json"
        )
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get("cohort") != fingerprint or state.get("complete"):
            state = {
                "cohort": fingerprint,
                "complete": False,
                "entries": {},
                "selectors": [],
                "metadata_batches": 0,
                "finished_patterns": [],
                "written_ids": [],
                "skipped": {},
                "warnings": [],
            }
        state.setdefault("selectors", list(state["entries"]))
        state.setdefault("metadata_batches", 0)
        state["complete"] = False
        state.pop("error", None)
        _save(path, state)
        archive_id = f"{facility}:codac_uda:{role}"
        access = DataAccess(
            id=f"{facility}:codac_uda",
            facility_id=facility,
            name="CODAC UDA bounded reads",
            method_type="python",
            library="uda_client_reader",
            access_type="local",
            data_source="codac_uda",
            connection_template="from imas_codex.remote.codac_uda import request_archive",
            data_template="data = request_archive('{facility}', '{archive_role}', dict(operation='read', name='{accessor}', start={start_ns}, end={end_ns}, mode='raw'))",
        )
        archives = [
            ArchiveEndpoint(
                id=f"{facility}:codac_uda:{item.role}",
                facility_id=facility,
                data_system="codac_uda",
                role=item.role,
                excluded=bool(item.excluded),
            )
            for item in settings.archives
        ]
        result = ScanResult(
            data_access=access,
            metadata={"checkpoint_path": str(path), "cohort": fingerprint},
        )
        written = set(state["written_ids"])

        async def persist_pending() -> None:
            pending = [
                self._signal(facility, archive_id, access.id, entry, fingerprint)
                for entry in state["entries"].values()
                if f"{facility}:general/codac_uda/{entry['name']}" not in written
            ]
            for offset in range(0, len(pending), 100):
                batch = pending[offset : offset + 100]
                count = await asyncio.to_thread(
                    _persist_signals, facility, fingerprint, batch
                )
                if count != len(batch):
                    raise RuntimeError(
                        f"incomplete graph batch: acknowledged {count} of {len(batch)}"
                    )
                written.update(signal.id for signal in batch)
                state["written_ids"] = sorted(written)
                _save(path, state)

        try:
            await asyncio.to_thread(_persist_archives, facility, archives, access)
            for pattern in patterns:
                if pattern in state["finished_patterns"]:
                    continue
                response = await asyncio.to_thread(
                    request_archive,
                    facility,
                    role,
                    {
                        "operation": "variables",
                        "pattern": pattern,
                        "limit": min(
                            settings.catalogue_limit,
                            config.get("_scan_limit", settings.catalogue_limit),
                        ),
                        "metadata_batch_size": settings.metadata_batch_size,
                        "metadata": False,
                    },
                    config=effective,
                )
                for entry in response["entries"]:
                    if entry["name"] in state["selectors"]:
                        state["skipped"]["overlapping_pattern"] = (
                            state["skipped"].get("overlapping_pattern", 0) + 1
                        )
                    else:
                        state["selectors"].append(entry["name"])
                state["warnings"].extend(response.get("warnings", []))
                state["finished_patterns"].append(pattern)
                _save(path, state)
            if written:
                state["skipped"]["already_written"] = len(written)
            if state["entries"]:
                state["skipped"]["metadata_already_fetched"] = len(state["entries"])
            await persist_pending()
            pending_metadata = [
                name for name in state["selectors"] if name not in state["entries"]
            ]
            for offset in range(0, len(pending_metadata), settings.metadata_batch_size):
                names = pending_metadata[offset : offset + settings.metadata_batch_size]
                response = await asyncio.to_thread(
                    request_archive,
                    facility,
                    role,
                    {
                        "operation": "metadata",
                        "names": names,
                        "metadata_batch_size": settings.metadata_batch_size,
                    },
                    config=effective,
                )
                received = response["entries"]
                if len(received) != len(names) or {
                    entry["name"] for entry in received
                } != set(names):
                    raise RuntimeError(
                        "metadata batch does not match its requested selectors"
                    )
                state["entries"].update({entry["name"]: entry for entry in received})
                state["warnings"].extend(response.get("warnings", []))
                state["metadata_batches"] += 1
                _save(path, state)
                await persist_pending()
            result.signals = [
                self._signal(facility, archive_id, access.id, entry, fingerprint)
                for entry in state["entries"].values()
            ]
            state["complete"] = (
                len(written) == len(result.signals) == len(state["selectors"])
            )
            _save(path, state)
            result.stats = {
                "complete": state["complete"],
                "persisted": True,
                "enumerated": len(result.signals),
                "written": len(written),
                "skipped": state["skipped"],
                "metadata_warnings": len(state["warnings"]),
                "metadata_completed": state["metadata_completed"],
                "metadata_batches": state["metadata_batches"],
            }
            return result
        except BaseException as exc:
            state["complete"] = False
            state["error"] = type(exc).__name__
            _save(path, state)
            raise

    @staticmethod
    def _signal(
        facility: str, archive: str, access: str, entry: dict, cohort: str
    ) -> FacilitySignal:
        name = entry["name"]
        mode = entry.get("sampling_mode", "unknown")
        if mode not in {item.value for item in ArchiveSamplingMode}:
            mode = "unknown"
        return FacilitySignal(
            id=f"{facility}:general/codac_uda/{name}",
            facility_id=facility,
            status=FacilitySignalStatus.discovered,
            name=name,
            accessor=name,
            data_access=access,
            data_source_name="codac_uda",
            data_source_path=name,
            metadata_archive=archive,
            archive_cohorts=[cohort],
            description=entry.get("description", {}).get("value"),
            source_description=entry.get("description", {}).get("value"),
            unit=entry.get("unit", {}).get("value"),
            discovery_source="codac_uda",
            sampling_mode=mode,
            sampling_evidence=entry.get("sampling_evidence")
            if mode != "unknown"
            else None,
        )

    async def check(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> list[dict]:
        settings = archive_config(
            facility, config if config.get("variable_patterns") else None
        )
        effective = settings.model_dump(mode="json")
        start, end = settings.validation_start_ns, settings.validation_end_ns
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 < end - start <= settings.max_window_ns
        ):
            raise ValueError(
                "declare a bounded validation window in private configuration"
            )
        results = []
        for signal in signals:
            attempts = []
            confirmed = None
            for role in settings.readable_roles:
                try:
                    reply = await asyncio.to_thread(
                        request_archive,
                        facility,
                        role,
                        {
                            "operation": "read",
                            "name": signal.accessor,
                            "start": start,
                            "end": end,
                            "mode": "envelope",
                            "bins": min(32, settings.max_samples),
                            "max_samples": settings.max_samples,
                        },
                        config=effective,
                    )
                    count = len(reply["time"])
                    if (
                        count
                        and len(reply["values"]) == count
                        and not reply.get("empty", False)
                    ):
                        archive = f"{facility}:codac_uda:{role}"
                        await asyncio.to_thread(_confirm, signal.id, archive)
                        signal.data_archive = archive
                        confirmed = archive
                        attempts.append({"role": role, "samples": count})
                        break
                    attempts.append({"role": role, "samples": 0, "reason": "empty"})
                except ArchiveRequestError as exc:
                    attempts.append(
                        {"role": role, "samples": 0, "reason": exc.error_type}
                    )
            results.append(
                {
                    "signal_id": signal.id,
                    "valid": confirmed is not None,
                    "data_archive": confirmed,
                    "attempts": attempts,
                }
            )
        return results


register_scanner(CodacUDAScanner())
