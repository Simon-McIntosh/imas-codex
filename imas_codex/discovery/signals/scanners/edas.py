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

EQDB_FIELDS = {
    "#GEO": ("RG", "ZG", "NSR", "NSZ"),
    "#EQU": ("PSI",),
}
GEQDSK_FIELDS = ("r", "z", "psirz", "COCOS")


def _equilibrium_access_methods(facility: str) -> list[DataAccess]:
    """Describe the two file formats independently of any exemplar path."""
    connection = "import json\nfrom imas_codex.remote.executor import run_python_script"
    common = (
        ", ssh_host='{ssh_host}', timeout=60, python_command='python', "
        "setup_commands=['module unload python/3.5.6', 'module load python/3.12']"
        "))\n"
        "data = result['data']"
    )
    eqdb = DataAccess(
        id=f"{facility}:eqdb:local_record",
        facility_id=facility,
        name="EQDB local record fields",
        description=(
            "Read the EQ31 or EQ11 record format at the EQDB client's local "
            "shot/time path. The compiled client's network mode against "
            "dbsvr17p:/JT60SA_EQ/EQDB is an alternate route when that server "
            "is reachable; it currently fails during defaults initialization."
        ),
        method_type="file",
        library="JT-60SA EQDB record format",
        access_type="local",
        data_source="EQDB",
        connection_template=connection,
        data_template=(
            "result = json.loads(run_python_script('read_equilibrium_file.py', "
            "dict(format='eqdb_record', root='{root}', shot={shot}, "
            "time={time}, field='{field}')" + common
        ),
        environment_variables="EQDSK_DIR; EQDB_FILE overrides the shot/time path",
        accesses_geometry="R,Z,psi grid",
    )
    geqdsk = DataAccess(
        id=f"{facility}:equilibrium:g_eqdsk_file",
        facility_id=facility,
        name="G-EQDSK file fields",
        description=(
            "Read a producer's G-EQDSK file by root, shot, time and its "
            "filename template; COCOS is returned only when the file declares it."
        ),
        method_type="file",
        library="G-EQDSK",
        access_type="local",
        data_source="G-EQDSK",
        connection_template=connection,
        data_template=(
            "result = json.loads(run_python_script('read_equilibrium_file.py', "
            "dict(format='g_eqdsk', root='{root}', shot={shot}, time={time}, "
            "filename_template='{filename_template}', field='{field}')" + common
        ),
        accesses_geometry="R,Z,psi grid",
    )
    return [eqdb, geqdsk]


def _equilibrium_signals(
    facility: str, examples: list[dict[str, Any]], access: list[DataAccess]
) -> list[FacilitySignal]:
    """Use inventory examples as evidence for stable code and format fields."""
    signals = []
    for example in examples:
        code = example["code"]
        is_record = example["format"] in {"selene_eq31", "eq11"}
        groups = EQDB_FIELDS.items() if is_record else (("G-EQDSK", GEQDSK_FIELDS),)
        for group, fields in groups:
            for field in fields:
                if field == "COCOS" and example.get("cocos") is None:
                    continue
                source = "EQDB" if is_record else "G-EQDSK"
                signals.append(
                    FacilitySignal(
                        id=f"{facility}:equilibrium/{source.lower().replace('-', '_')}_{code.lower()}_{field.lower()}",
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain="equilibrium",
                        name=f"{code} {source} {group}/{field}",
                        accessor=field,
                        data_access=access[0 if is_record else 1].id,
                        data_source_name=source,
                        data_source_path=f"{code}/{example['format']}/{group}/{field}",
                        description=(
                            f"{code} {example['format']} {group}/{field}; inventory "
                            f"example {example['path']} at shot {example['shot']} "
                            f"and {example['time']} s, grid {example['grid'][0]} "
                            f"by {example['grid'][1]}, SHA-256 {example['sha256']}."
                        ),
                        cocos=example.get("cocos"),
                        discovery_source="edas",
                        example_shot=example["shot"],
                    )
                )
    return signals


def _database_access_methods(
    facility: str, config: dict[str, Any], databases: set[str]
) -> dict[str, DataAccess]:
    """Describe the native routes used by the catalogued database signals."""
    routes = {route["name"]: route for route in config.get("database_routes", [])}
    methods = {}
    for name in databases & {"UDDB", "LCDB", "MBDB"}:
        route = routes.get(name, {})
        if name == "UDDB":
            api = config.get("uddb_api_path", "")
            library = config.get("uddb_lib_path", "")
            connection = (
                f"import sys\nsys.path.insert(0, {api!r})\n"
                "from uddb_pwrapper import uddbWrapper\n"
                f"db = uddbWrapper({library!r})\ndb.uddbOpen()"
            )
            read = (
                "ok, result = db.uddbreadConvert(shot='{shot}', pid='{pid}', "
                "t1='{t1}', t2='{t2}', datavol=10000, ch=1)"
            )
            cleanup = "db.uddbClose()"
        elif name == "LCDB":
            api = config.get("lcdb_api_path", "")
            connection = (
                f"import sys\nsys.path.insert(0, {api!r})\n"
                "from lcdbWrapper import LcdbWrapper\ndb = LcdbWrapper()"
            )
            read = (
                "ok, result = db.lcdb_value({shot}, '{category}', "
                "['{data_name}'], root='{root}')"
            )
            cleanup = None
            library = "lcdbWrapper"
        else:
            api = config.get("mbdb_api_path", "")
            library = config.get("mbdb_lib_path", "")
            connection = (
                f"import sys\nsys.path.insert(0, {api!r})\n"
                "from mbdbWrapper import mbdbWrapper\n"
                f"db = mbdbWrapper({library!r})\ndb.mbdbSetDirectory('mbdb')\n"
                "db.mbdbROpen(mbdbroot='{root}', caseno={case}, category='{category}')"
            )
            read = "ok, result = db.mbdbRTimes('{data_name}', '{t1}', '{t2}')"
            cleanup = "db.mbdbRClose()"
        methods[name] = DataAccess(
            id=f"{facility}:edas:{name.lower()}",
            facility_id=facility,
            name=f"EDAS {name} Access",
            description=(
                f"Catalogue: {route.get('catalogue_call', 'configured native wrapper')}; "
                f"metadata: {route.get('metadata_source', 'native wrapper')}."
            ),
            method_type="edas",
            library=library,
            access_type="local",
            data_source=name,
            connection_template=connection,
            data_template=read,
            cleanup_template=cleanup,
            setup_commands=config.get("setup_commands"),
        )
    return methods


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
                    "uddb_header_shots": [
                        item["shot"]
                        for item in config.get("equilibrium_examples", [])
                        if item.get("shot")
                    ],
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
        database_access = _database_access_methods(
            facility,
            config,
            {raw.get("database", "EDDB") for raw in raw_signals},
        )

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
                sample_shot = raw.get("metadata_shot") or shot_str
                signals.append(
                    FacilitySignal(
                        id=f"{facility}:general/uddb_{dname.lower()}",
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain="general",
                        name=f"UDDB/{dname}",
                        accessor=f"uddbreadConvert('{sample_shot}', '{dname}', t1, t2)",
                        data_access=database_access["UDDB"].id,
                        data_source_name="UDDB",
                        data_source_path=f"UDDB/{dname}",
                        unit=units,
                        description=description,
                        data_class=SignalDataClass.time_series,
                        shot_range=raw.get("shot_range") or None,
                        pid=dname,
                        aliases=[raw["alias"]] if raw.get("alias") else None,
                        discovery_source="edas",
                        example_shot=int(str(sample_shot).removeprefix("E")),
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
                        data_access=database_access["LCDB"].id,
                        data_source_name="LCDB",
                        data_source_path=f"{owner}/{source_category}/{dname}",
                        unit=units,
                        description=(
                            f"Dataset comment: {description}"
                            if description
                            else f"LCDB {source_category} {dname} from {owner}"
                        ),
                        discovery_source="edas",
                        example_shot=shot,
                    )
                )
                continue

            if raw.get("database") == "MBDB":
                _, owner, case, source_category = raw["category"].split("/", 3)
                kind = raw.get("data_kind")
                signals.append(
                    FacilitySignal(
                        id=f"{facility}:general/mbdb_{owner}_{case}_{source_category}_{dname}".lower(),
                        facility_id=facility,
                        status=FacilitySignalStatus.discovered,
                        physics_domain="general",
                        name=f"MBDB/{owner}/{case}/{source_category}/{dname}",
                        accessor=(
                            f"mbdbRPoint({dname!r})"
                            if kind == "P"
                            else f"mbdbRTimes({dname!r}, t1, t2)"
                        ),
                        data_access=database_access["MBDB"].id,
                        data_source_name="MBDB",
                        data_source_path=f"{owner}/{case}/{source_category}/{dname}",
                        description=f"MBDB {source_category} {dname} from {owner} case {case}",
                        data_class=(
                            SignalDataClass.one_point
                            if kind == "P"
                            else SignalDataClass.time_series
                            if kind == "T"
                            else None
                        ),
                        discovery_source="edas",
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

        examples = config.get("equilibrium_examples", [])
        equilibrium_access = _equilibrium_access_methods(facility) if examples else []
        signals.extend(_equilibrium_signals(facility, examples, equilibrium_access))
        access_methods = [data_access, *database_access.values(), *equilibrium_access]
        access_ids = {method.id for method in access_methods}
        missing = [
            signal.id for signal in signals if signal.data_access not in access_ids
        ]
        if missing:
            logger.error("EDAS scanner: %d signals lack an access method", len(missing))
            return ScanResult(
                stats={"error": "signals lack data access", "ids": missing}
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
            data_accesses=[*database_access.values(), *equilibrium_access],
            metadata={
                "reference_shot": shot_str,
                "categories": data.get("categories", []),
                "ncats": data.get("ncats", 0),
                "database_attempts": data.get("attempts", []),
                "category_counts": data.get("category_counts", {}),
                "equilibrium_examples": examples,
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

        equilibrium = [s for s in signals if s.data_source_name in {"EQDB", "G-EQDSK"}]
        edas_signals = [
            s for s in signals if s.data_source_name not in {"EQDB", "G-EQDSK"}
        ]
        equilibrium_results = []
        examples = {
            item["code"]: item for item in config.get("equilibrium_examples", [])
        }
        by_code: dict[str, list[FacilitySignal]] = {}
        for signal in equilibrium:
            code = (signal.data_source_path or "").split("/", 1)[0]
            by_code.setdefault(code, []).append(signal)
        for code, code_signals in by_code.items():
            example = examples.get(code)
            error = None
            record = None
            if example is None:
                error = f"no equilibrium inventory example for {code}"
            else:
                payload = {
                    "format": (
                        "eqdb_record"
                        if example["format"] in {"selene_eq31", "eq11"}
                        else "g_eqdsk"
                    ),
                    "root": example["root"],
                    "shot": example["shot"],
                    "time": example["time"],
                }
                if example.get("filename_template"):
                    payload["filename_template"] = example["filename_template"]
                try:
                    output = await async_run_python_script(
                        "read_equilibrium_file.py",
                        payload,
                        ssh_host=ssh_host,
                        timeout=60,
                        python_command=config.get("python_command", "python3"),
                        setup_commands=config.get("setup_commands"),
                    )
                    record = json.loads(output.strip().split("\n")[-1])
                    if record["format"] != example["format"]:
                        raise ValueError("equilibrium format differs from inventory")
                    if record["grid"] != example["grid"]:
                        raise ValueError("equilibrium grid differs from inventory")
                    if record["path"] != example["path"]:
                        raise ValueError("equilibrium path differs from inventory")
                    if record["sha256"] != example["sha256"]:
                        raise ValueError("equilibrium file hash differs from example")
                    if (
                        example.get("cocos") is not None
                        and record.get("cocos") != example["cocos"]
                    ):
                        raise ValueError("COCOS differs from inventory")
                except Exception as exc:
                    error = str(exc)[:200]
            for signal in code_signals:
                equilibrium_results.append(
                    {
                        "signal_id": signal.id,
                        "valid": error is None,
                        "dtype": "file_grid" if error is None else None,
                        "error": error,
                    }
                )

        if not edas_signals:
            return equilibrium_results

        # The EDDB catalogue keys a signal by "<category>/<data name>". Read
        # that from data_source_path, which keeps the catalogue path for the
        # life of the row; fall back to name only when the path is absent,
        # since enrichment may replace name with a human-readable label.
        batch = []
        for s in edas_signals:
            source = s.data_source_path or s.name or ""
            parts = source.split("/")
            if s.data_source_name == "MBDB" and len(parts) == 4:
                batch.append(
                    {
                        "id": s.id,
                        "database": "MBDB",
                        "owner": parts[0],
                        "case": int(parts[1]),
                        "category": parts[2],
                        "data_name": parts[3],
                        "data_class": getattr(s.data_class, "value", s.data_class),
                    }
                )
                continue
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
                batch.append(
                    {
                        "id": s.id,
                        "database": "UDDB",
                        "pid": parts[1],
                        "shot": f"E{s.example_shot:06d}"
                        if s.example_shot
                        else shot_str,
                    }
                )
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
                    "uddb_api_path": config.get("uddb_api_path"),
                    "uddb_lib_path": config.get("uddb_lib_path"),
                    "lcdb_api_path": config.get("lcdb_api_path"),
                    "lcdb_root": config.get("lcdb_root"),
                    "mbdb_api_path": config.get("mbdb_api_path"),
                    "mbdb_lib_path": config.get("mbdb_lib_path"),
                    "mbdb_root": config.get("mbdb_root"),
                },
                ssh_host=ssh_host,
                timeout=180,
                python_command=config.get("python_command", "python3"),
                setup_commands=config.get("setup_commands"),
            )
            response = json.loads(output.strip().split("\n")[-1])
            edas_results = [
                {
                    "signal_id": r["id"],
                    "valid": r.get("success", False),
                    "dtype": r.get("dtype"),
                    "error": r.get("error"),
                }
                for r in response.get("results", [])
            ]
            by_id = {
                item["signal_id"]: item for item in equilibrium_results + edas_results
            }
            return [by_id[s.id] for s in signals if s.id in by_id]
        except Exception as e:
            logger.error("EDAS check failed: %s", e)
            edas_results = [
                {"signal_id": s.id, "valid": False, "error": str(e)[:200]}
                for s in edas_signals
            ]
            by_id = {
                item["signal_id"]: item for item in equilibrium_results + edas_results
            }
            return [by_id[s.id] for s in signals]


# Auto-register on import
register_scanner(EDASScanner())
