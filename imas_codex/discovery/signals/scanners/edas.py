"""EDAS (Experiment Data Access System) scanner plugin for JT-60SA.

EDAS is JT-60SA's primary data access system. Data is organized by:
  (shot, category, data_name) — e.g., ('E012345', 'EDDB', 'tesTime')

Discovery strategy:
  1. SSH to JT-60SA, use eddbreadCatTable() to enumerate categories
  2. For each category, use eddbreadTable() to enumerate data names
     with their units, descriptions, aliases, and shot ranges
  3. Create FacilitySignal per (category, data_name) with EDAS accessor

Key insight: eddbreadTable returns a self-describing catalog — far richer
than TDI .fun parsing. It provides units, descriptions (Japanese/English),
aliases, data class, and shot validity ranges.

Remote execution uses the shared run_python_script() infrastructure
with scripts in imas_codex/remote/scripts/ (enumerate_edas.py, check_edas.py).

Config key: data_systems.edas
Facility: JT-60SA
"""

from __future__ import annotations

import json
import logging
from typing import Any

from imas_codex.discovery.signals.scanners.base import (
    ScanResult,
    register_scanner,
)
from imas_codex.graph.models import (
    DataAccess,
    FacilitySignal,
    FacilitySignalStatus,
    SignalDataClass,
)

logger = logging.getLogger(__name__)


# EDDB data-class letter -> SignalDataClass value. The catalogue assigns the
# letter; an unrecognised one leaves the slot unset rather than guessing.
DATA_CLASS_BY_EDDB_LETTER = {
    "T": "time_series",
    "O": "one_point",
    "P": "parameter",
}
EDDB_LETTER_BY_DATA_CLASS = {v: k for k, v in DATA_CLASS_BY_EDDB_LETTER.items()}


class EDASScanner:
    """Discover signals from JT-60SA EDAS system.

    EDAS discovery strategy:
    1. SSH to JT-60SA, use eddbreadCatTable + eddbreadTable API
    2. The database is self-describing: returns data names, units,
       descriptions (Japanese/English), aliases, and shot ranges
    3. Create FacilitySignal per (category, data_name) with EDAS accessor
    4. Japanese descriptions provide LLM enrichment context

    Config (data_systems.edas):
        api_path: str - Path to EDAS API source files
        lib_path: str - Path to libeddb.so shared library
        header_path: str - Path to C headers with signal definitions
        reference_shot: int - Shot for validation (E-prefix format)
    """

    scanner_type: str = "edas"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from EDAS via SSH eddbreadTable enumeration.

        Uses remote/scripts/enumerate_edas.py via async_run_python_script()
        for proper SSH execution with JSON encode/decode.
        """
        from imas_codex.remote.executor import async_run_python_script

        ref_shot = reference_shot or config.get("reference_shot")

        if not ref_shot:
            logger.warning(
                "EDAS scanner: no reference_shot configured for %s", facility
            )
            return ScanResult(stats={"error": "no reference_shot configured"})

        # Format shot as E-prefix string if numeric
        shot_str = str(ref_shot)
        if not shot_str.startswith("E"):
            shot_str = f"E{ref_shot:06d}"

        logger.info(
            "EDAS scanner: enumerating signals for shot %s on %s",
            shot_str,
            ssh_host,
        )

        # Read paths from config
        api_path = config.get("api_path")
        lib_path = config.get("lib_path")

        if not api_path or not lib_path:
            logger.error(
                "EDAS scanner: api_path and lib_path must be configured for %s",
                facility,
            )
            return ScanResult(
                stats={"error": "api_path and lib_path required in edas config"}
            )

        try:
            output = await async_run_python_script(
                "enumerate_edas.py",
                {
                    "ref_shot": shot_str,
                    "api_path": api_path,
                    "lib_path": lib_path,
                    "databases": config.get("databases", ["EDDB"]),
                    **{
                        key: config[key]
                        for key in (
                            "uddb_api_path",
                            "uddb_lib_path",
                            "pmdb_api_path",
                            "pmdb_lib_path",
                            "lcdb_api_path",
                            "lcdb_root",
                            "mbdb_api_path",
                            "mbdb_lib_path",
                            "mbdb_root",
                            "eqdb_root",
                        )
                        if key in config
                    },
                },
                ssh_host=ssh_host,
                timeout=180,
                python_command=config.get("python_command", "python3"),
                setup_commands=config.get("setup_commands"),
            )
            data = json.loads(output.strip().split("\n")[-1])
        except Exception as e:
            logger.error("EDAS enumeration failed on %s: %s", ssh_host, e)
            return ScanResult(stats={"error": str(e)[:300]})

        if "error" in data:
            logger.error("EDAS enumeration error: %s", data["error"])
            return ScanResult(stats={"error": data["error"]})

        raw_signals = data.get("signals", [])

        # Create DataAccess node
        data_access = DataAccess(
            id=f"{facility}:edas:eddb",
            facility_id=facility,
            name="EDAS EDDB Access",
            method_type="edas",
            library="eddb_pwrapper",
            access_type="local",
            data_source="eddb",
            connection_template=(
                "import sys\n"
                f"sys.path.insert(0, '{api_path}')\n"
                "from eddb_pwrapper import eddbWrapper\n"
                f"db = eddbWrapper('{lib_path}')\n"
                "db.eddbOpen()"
            ),
            # eddbreadTime takes its time bounds as strings; omitting both
            # returns the whole record (measured 740,900 points for a coil current).
            data_template=(
                "ok, rtn = db.eddbreadTime('{shot}', '{category}', '{data_name}', '0', '99')\n"
                "data = rtn['data'] if ok else None"
            ),
            cleanup_template="db.eddbClose()",
            setup_commands=config.get("setup_commands"),
        )

        # Convert to FacilitySignal nodes
        signals = []
        unknown_data_classes: set[str] = set()
        for raw in raw_signals:
            cat = raw["category"]
            dname = raw["data_name"]
            units = raw.get("units", "")
            description = raw.get("description", "")

            if raw.get("database") == "UDDB":
                signals.append(
                    FacilitySignal(
                        id=f"{facility}:general/uddb_{dname.lower()}",
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain="general",
                        name=f"UDDB/{dname}",
                        accessor=f"uddbreadConvert('{shot_str}', '{dname}', t1, t2)",
                        data_source_name="UDDB",
                        data_source_path=f"UDDB/{dname}",
                        description=description,
                        data_class=SignalDataClass.time_series,
                        shot_range=raw.get("shot_range") or None,
                        pid=dname,
                        aliases=[raw["alias"]] if raw.get("alias") else None,
                        discovery_source="edas",
                        example_shot=ref_shot,
                    )
                )
                continue

            if raw.get("database") == "LCDB":
                owner = raw["category"].split("/", 2)[1]
                source_category = raw["file_category"]
                shot = int(raw["shot"])
                root = raw["root"]
                signals.append(
                    FacilitySignal(
                        id=f"{facility}:general/lcdb_{owner}_{source_category}_{dname}".lower(),
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain="general",
                        name=f"LCDB/{owner}/{source_category}/{dname}",
                        accessor=f"lcdb_value({shot}, {source_category!r}, [{dname!r}], root={root!r})",
                        data_source_name="LCDB",
                        data_source_path=f"{owner}/{source_category}/{dname}",
                        description=f"LCDB {source_category} {dname} from {owner}",
                        discovery_source="edas",
                        example_shot=shot,
                    )
                )
                continue

            # A PID-keyed one-point row is its own signal group: EDDB addresses
            # it by the nine-character PID No. rather than by the catalogue
            # data name, so the PID is the signal's name within the scheme.
            pid_keyed = bool(raw.get("pid_keyed"))
            pid = (raw.get("udp_id") or "").strip()
            if pid_keyed:
                source_dname = raw.get("source_dname") or dname
                signal_id = (
                    f"{facility}:general/{cat.lower()}_{pid.lower().replace(' ', '_')}"
                )
                accessor = (
                    f"eddbreadOne('{shot_str}', '{cat}', '{source_dname}', "
                    f"'{pid}', 0, 0)"
                )
                # The PID pass enumerates the one-point and condition data, so
                # its data class is one_point whether or not the catalogue row
                # also carried an EDDB letter.
                data_class = SignalDataClass.one_point
            else:
                source_dname = dname
                signal_id = f"{facility}:general/{cat.lower()}_{dname.lower()}"
                eddb_class = raw.get("data_class", "")
                if eddb_class == "O":
                    accessor = (
                        f"eddbreadOne('{shot_str}', '{cat}', '{dname}', None, 0, 0)"
                    )
                else:
                    accessor = f"eddbreadTime('{shot_str}', '{cat}', '{dname}', t1, t2)"
                data_class = DATA_CLASS_BY_EDDB_LETTER.get(eddb_class)
                if eddb_class and data_class is None:
                    unknown_data_classes.add(eddb_class)

            signals.append(
                FacilitySignal(
                    id=signal_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain="general",  # Enriched by LLM
                    name=f"{cat}/{pid if pid_keyed else dname}",
                    accessor=accessor,
                    data_access=data_access.id,
                    data_source_name="edas",
                    data_source_path=f"{cat}/{source_dname}",
                    unit=units,
                    description=description,  # May be Japanese
                    data_class=data_class,
                    shot_range=raw.get("shot_range") or None,
                    pid=raw.get("udp_id") or None,
                    aliases=[raw["alias"]] if raw.get("alias") else None,
                    discovery_source="edas",
                    example_shot=ref_shot,
                )
            )

        if unknown_data_classes:
            logger.warning(
                "EDAS scanner: unrecognised data class(es) %s on %s; "
                "data_class left unset",
                sorted(unknown_data_classes),
                ssh_host,
            )

        logger.info(
            "EDAS scanner: discovered %d signals from %d categories (shot %s)",
            len(signals),
            data.get("ncats", 0),
            shot_str,
        )

        return ScanResult(
            signals=signals,
            data_access=data_access,
            metadata={
                "reference_shot": shot_str,
                "categories": data.get("categories", []),
                "ncats": data.get("ncats", 0),
                "database_attempts": data.get("attempts", []),
                "category_counts": data.get("category_counts", {}),
            },
            stats={
                "signals_discovered": len(signals),
                "categories_found": data.get("ncats", 0),
                "reference_shot": shot_str,
                "database_attempts": data.get("attempts", []),
            },
        )

    async def check(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> list[dict[str, Any]]:
        """Validate EDAS signals return data for reference shot.

        Uses remote/scripts/check_edas.py via async_run_python_script().
        """
        from imas_codex.remote.executor import async_run_python_script

        ref_shot = reference_shot or config.get("reference_shot")
        api_path = config.get("api_path")
        lib_path = config.get("lib_path")
        if not ref_shot:
            return [
                {"signal_id": s.id, "valid": False, "error": "no reference_shot"}
                for s in signals
            ]
        if not api_path or not lib_path:
            return [
                {
                    "signal_id": s.id,
                    "valid": False,
                    "error": "api_path/lib_path not configured",
                }
                for s in signals
            ]

        shot_str = str(ref_shot)
        if not shot_str.startswith("E"):
            shot_str = f"E{ref_shot:06d}"

        # The EDDB catalogue keys a signal by "<category>/<data name>". Read
        # that from data_source_path, which keeps the catalogue path for the
        # life of the row; fall back to name only when the path is absent,
        # since enrichment may replace name with a human-readable label.
        batch = []
        for s in signals:
            source = s.data_source_path or s.name or ""
            parts = source.split("/")
            if s.data_source_name == "LCDB" and len(parts) == 3:
                batch.append(
                    {
                        "id": s.id,
                        "database": "LCDB",
                        "owner": parts[0],
                        "category": parts[1],
                        "data_name": parts[2],
                        "shot": s.example_shot,
                    }
                )
                continue
            if s.data_source_name == "UDDB" and len(parts) == 2:
                batch.append({"id": s.id, "database": "UDDB", "pid": parts[1]})
                continue
            if len(parts) == 2:
                data_class = EDDB_LETTER_BY_DATA_CLASS.get(
                    getattr(s.data_class, "value", s.data_class), ""
                )
                batch.append(
                    {
                        "id": s.id,
                        "category": parts[0],
                        "data_name": parts[1],
                        "data_class": data_class,
                    }
                )
            else:
                batch.append({"id": s.id, "category": "", "data_name": ""})

        try:
            output = await async_run_python_script(
                "check_edas.py",
                {
                    "signals": batch,
                    "ref_shot": shot_str,
                    "api_path": api_path,
                    "lib_path": lib_path,
                    "lcdb_api_path": config.get("lcdb_api_path"),
                    "lcdb_root": config.get("lcdb_root"),
                },
                ssh_host=ssh_host,
                timeout=180,
                python_command=config.get("python_command", "python3"),
                setup_commands=config.get("setup_commands"),
            )
            response = json.loads(output.strip().split("\n")[-1])
            return [
                {
                    "signal_id": r["id"],
                    "valid": r.get("success", False),
                    "dtype": r.get("dtype"),
                    "error": r.get("error"),
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            logger.error("EDAS check failed: %s", e)
            return [
                {"signal_id": s.id, "valid": False, "error": str(e)[:200]}
                for s in signals
            ]


# Auto-register on import
register_scanner(EDASScanner())
