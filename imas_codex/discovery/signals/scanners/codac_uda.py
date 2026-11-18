"""Resumable bounded archive cohorts with separate metadata and data provenance."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from imas_codex.discovery.signals.scanners.base import ScanResult, register_scanner
from imas_codex.discovery.signals.variable_name_graph import persist_name_parses
from imas_codex.discovery.signals.variable_names import parse_variable_name
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import (
    ArchiveEndpoint,
    ArchiveSamplingMode,
    DataAccess,
    FacilitySignal,
    FacilitySignalStatus,
)
from imas_codex.remote.codac_uda import (
    ArchiveReader,
    archive_config,
    request_archive,
)

ARCHIVE_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $archives AS props
MERGE (a:ArchiveEndpoint {id: props.id})
FOREACH (_ IN CASE WHEN any(key IN keys(props) WHERE
    coalesce(a[key] <> props[key], a[key] IS NOT NULL OR props[key] IS NOT NULL))
    THEN [1] ELSE [] END | SET a += props)
MERGE (a)-[:AT_FACILITY]->(f)
RETURN count(a) AS written
"""

CONFIRM_QUERY = """
MATCH (s:FacilitySignal {id: $signal}), (a:ArchiveEndpoint {id: $archive})
MATCH (da:DataAccess)
WHERE a.excluded = false AND s.facility_id = a.facility_id
  AND da.id = a.facility_id + ':codac_uda:access:' + a.role
OPTIONAL MATCH (s)-[old:DATA_ARCHIVE]->(previous)
WHERE previous <> a
DELETE old
SET s.data_archive = a.id, s.data_access = da.id
MERGE (s)-[:DATA_ARCHIVE]->(a)
WITH s, da
OPTIONAL MATCH (s)-[old_access:DATA_ACCESS]->(previous_access)
WHERE previous_access <> da
DELETE old_access
MERGE (s)-[:DATA_ACCESS]->(da)
RETURN count(DISTINCT s) AS written
"""


BINDING_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $bindings AS props
MATCH (a:ArchiveEndpoint {id: props.archive}), (access:DataAccess {id: props.access})
WHERE a.excluded = false
MERGE (source:DataSource {id: props.source})
SET source.name = props.name, source.facility_id = $facility,
    source.archive_endpoint = a.id, source.source_format = 'codac_uda',
    source.shot_dependent = false
MERGE (source)-[:AT_FACILITY]->(f)
MERGE (source)-[:HOSTED_BY_ARCHIVE]->(a)
RETURN count(source) AS written
"""


def _access(facility: str, role: str) -> DataAccess:
    return DataAccess(
        id=f"{facility}:codac_uda:access:{role}",
        facility_id=facility,
        name=f"CODAC UDA {role} bounded reads",
        method_type="codac_uda",
        library="imas_codex.remote.codac_uda",
        access_type="local",
        data_source=f"{facility}:codac_uda:source:{role}",
        connection_template="from imas_codex.remote.codac_uda import ArchiveSignalLoader",
        data_template="data = ArchiveSignalLoader().load('{signal_id}', start={start_ns}, end={end_ns})",
    )


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


def persist_archive_endpoints(
    graph: GraphClient, facility: str, archives: list[ArchiveEndpoint]
) -> int:
    """Upsert archive endpoints and return acknowledged created-node counters.

    Signal and pulse catalogues share this writer without requiring a pulse
    import to create signal access methods or data source bindings.
    """
    if not archives:
        return 0

    def write(tx):
        result = tx.run(
            ARCHIVE_QUERY,
            facility=facility,
            archives=[item.model_dump(mode="json") for item in archives],
        )
        row = result.single()
        summary = result.consume()
        if row is None or row["written"] != len(archives):
            raise RuntimeError(
                "archive write was not acknowledged; facility must exist"
            )
        return summary.counters.nodes_created

    with graph.session() as session:
        return session.execute_write(write)


def _persist_archives(facility: str, archives: list[ArchiveEndpoint]) -> None:
    from imas_codex.discovery.signals.parallel import persist_data_access

    with GraphClient() as graph:
        persist_archive_endpoints(graph, facility, archives)
    bindings = []
    for archive in archives:
        if archive.excluded:
            continue
        method = _access(facility, archive.role)
        persist_data_access(method)
        bindings.append(
            {
                "archive": archive.id,
                "access": method.id,
                "source": method.data_source,
                "name": f"codac_uda:{archive.role}",
            }
        )
    with GraphClient() as graph:
        rows = graph.query(BINDING_QUERY, facility=facility, bindings=bindings)
    if not rows or rows[0]["written"] != len(bindings):
        raise RuntimeError("archive source bindings were not acknowledged")


def _persist_signals(facility: str, cohort: str, signals: list[FacilitySignal]) -> int:
    """Delegate signal writes to their owner, then persist derived name links."""
    from imas_codex.discovery.signals.parallel import ingest_discovered_signals

    count = ingest_discovered_signals(
        [signal.model_dump(mode="json") for signal in signals]
    )
    if count != len(signals):
        return count
    with GraphClient() as graph:
        persist_name_parses(
            graph,
            [
                {"id": signal.id, "facility": facility, "selector": signal.accessor}
                for signal in signals
            ],
        )
    return count


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
        if config.get("_catalogue_path"):
            raise ValueError(
                "use the private catalogue entry point outside the pilot scanner"
            )
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
                "failed": {},
            }
        state.setdefault("selectors", list(state["entries"]))
        state.setdefault("metadata_batches", 0)
        state.setdefault("failed", {})
        state["complete"] = False
        state.pop("error", None)
        _save(path, state)
        archive_id = f"{facility}:codac_uda:{role}"
        access = _access(facility, role)
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
            await asyncio.to_thread(_persist_archives, facility, archives)
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
                name
                for name in state["selectors"]
                if name not in state["entries"] and name not in state["failed"]
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
                from imas_codex.remote.archive_catalogue import (
                    metadata_entries,
                )

                received = metadata_entries(response)
                failed = [
                    entry
                    for entry in received
                    if entry.get("metadata_state") == "unavailable"
                ]
                answered = [entry["name"] for entry in received]
                if len(answered) != len(names) or set(answered) != set(names):
                    raise RuntimeError(
                        "metadata batch does not match its requested selectors"
                    )
                state["entries"].update({entry["name"]: entry for entry in received})
                state["failed"].update(
                    {entry["name"]: entry["metadata_error"] for entry in failed}
                )
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
                "failed": len(state["failed"]),
                "errors": state["failed"],
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
        metadata_state = entry.get("metadata_state", "available")
        if metadata_state == "unavailable":
            entry = {
                key: value
                for key, value in entry.items()
                if key
                in {
                    "name",
                    "metadata_state",
                    "metadata_error_code",
                    "metadata_error",
                    "metadata_errors",
                    "listed_archives",
                }
            }
        elif metadata_state != "available" or not entry.get("archive_metadata"):
            raise ValueError(
                "available archive metadata requires a successful document"
            )
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
            metadata_state=metadata_state,
            metadata_error_code=entry.get("metadata_error_code"),
            metadata_error=entry.get("metadata_error"),
            metadata_errors=json.dumps(entry["metadata_errors"])
            if entry.get("metadata_errors")
            else None,
            archive_labels=entry.get("archive_labels"),
            archive_format=entry.get("archive_format"),
            archive_lower_display_limit=entry.get("archive_lower_display_limit"),
            archive_upper_display_limit=entry.get("archive_upper_display_limit"),
            archive_lower_alarm_limit=entry.get("archive_lower_alarm_limit"),
            archive_upper_alarm_limit=entry.get("archive_upper_alarm_limit"),
            archive_lower_warning_limit=entry.get("archive_lower_warning_limit"),
            archive_upper_warning_limit=entry.get("archive_upper_warning_limit"),
            listed_archives=entry.get("listed_archives", [archive]),
            archive_metadata=json.dumps(entry["archive_metadata"], sort_keys=True)
            if "archive_metadata" in entry
            else None,
            data_type=entry.get("data_type"),
            structure_fields=json.dumps(entry["structure_fields"], sort_keys=True)
            if "structure_fields" in entry
            else None,
            first_sample_ns=entry.get("first_sample_ns"),
            last_sample_ns=entry.get("last_sample_ns"),
            archive_cohorts=[cohort],
            description=entry.get("description", {}).get("value"),
            source_description=entry.get("description", {}).get("value"),
            unit=entry.get("unit", {}).get("value"),
            discovery_source="codac_uda",
            sampling_mode=mode if metadata_state == "available" else None,
            sampling_evidence=entry.get("sampling_evidence")
            if mode != "unknown"
            else None,
            **parse_variable_name(name).properties(facility),
        )

    async def check(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
        *,
        read_parameters: dict[str, Any] | None = None,
    ) -> list[dict]:
        settings = archive_config(
            facility, config if config.get("variable_patterns") else None
        )
        effective = {
            "ssh_host": ssh_host,
            "data_systems": {"codac_uda": settings.model_dump(mode="json")},
        }
        parameters = {
            "start": settings.validation_start_ns,
            "end": settings.validation_end_ns,
            "mode": "envelope",
            "bins": min(32, settings.max_samples),
            "max_samples": settings.max_samples,
            **(read_parameters or {}),
        }
        results = []
        for signal in signals:
            attempts = []
            confirmed = None
            shared_signal = None
            for role in settings.readable_roles:
                reader = ArchiveReader(facility, role, config=effective)
                reply = await asyncio.to_thread(
                    reader.check, signal.accessor, **parameters
                )
                if reply["success"]:
                    shared_signal = reply["signal"]
                    archive = reader.archive
                    await asyncio.to_thread(_confirm, signal.id, archive)
                    signal.data_archive = archive
                    signal.data_access = _access(facility, role).id
                    confirmed = archive
                    attempts.append({"role": role, "samples": len(shared_signal.time)})
                    break
                attempts.append(
                    {
                        "role": role,
                        "samples": 0,
                        "reason": reply["error_classification"],
                        "error": reply["error"],
                    }
                )
            results.append(
                {
                    "signal_id": signal.id,
                    "valid": confirmed is not None,
                    "data_archive": confirmed,
                    "data_access": signal.data_access,
                    "signal": shared_signal,
                    "shape": list(shared_signal.values.shape)
                    if shared_signal is not None
                    else None,
                    "dtype": str(shared_signal.values.dtype)
                    if shared_signal is not None
                    else None,
                    "error_classification": None
                    if confirmed
                    else attempts[-1]["reason"],
                    "error": None if confirmed else attempts[-1]["error"],
                    "attempts": attempts,
                }
            )
        return results


register_scanner(CodacUDAScanner())
