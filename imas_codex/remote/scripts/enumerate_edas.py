#!/usr/bin/env python3
"""Enumerate JT-60SA EDAS categories and data names.

This script runs on the JT-60SA host where the eddb_pwrapper module
is available. It uses eddbreadCatTable()/eddbreadTable() to enumerate
all categories and their data names with metadata.

A second pass enumerates the one-point and condition data that EDDB keys
by PID No. (a nine-character key, four digits then five characters). An
empty PID on a valid one-point data name makes eddbreadOne return the
category's whole PID-keyed block; eddbreadPara does not enumerate it
(irc 1014) and an empty data name returns irc 1031. Those rows are
emitted with ``pid_keyed: true`` and their ``udp_id`` set to the PID.

Requirements:
- Python 3.8+ (stdlib only except eddb_pwrapper)
- eddb_pwrapper or edas_eddb_api (available at /analysis/lib)

Usage:
    echo '{"ref_shot": "E012345"}' | python3 enumerate_edas.py

Input (JSON on stdin):
    {
        "ref_shot": "E012345",
        "api_path": "/analysis/src/eddb",
        "lib_path": "/analysis/lib/libeddb.so"
    }

Output (JSON on stdout):
    {
        "signals": [
            {
                "category": "EDDB",
                "data_name": "tesTime",
                "alias": "",
                "units": "s",
                "description": "Time base"
            },
            ...
        ],
        "shot": "E012345",
        "categories": ["EDDB", ...],
        "ncats": 5
    }
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# One record of the PID-keyed block that eddbreadOne returns for an empty
# PID: PID="1651 A001" NAME="Data acquisition start time" UNIT="ms" DATA="-15000"
PID_RECORD_RE = re.compile(
    r'PID="([^"]*)"\s+NAME="([^"]*)"\s+UNIT="([^"]*)"\s+DATA="([^"]*)"'
)


def _uddb_header_value(header, key):
    """Read a quoted or bare header value without consuming the next key."""
    match = re.search(
        rf"\b{re.escape(key)}[ \t]*=(?:[ \t]*\"([^\"]*)\"|[ \t]*'([^']*)'|([^\s]*))",
        header,
    )
    return (
        next((value for value in match.groups() if value is not None), "")
        if match
        else ""
    )


def main():
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(0)

    ref_shot = config.get("ref_shot", "")
    api_path = config.get("api_path", "")
    lib_path = config.get("lib_path", "")
    databases = [name.upper() for name in config.get("databases", ["EDDB"])]
    if not ref_shot:
        print(json.dumps({"error": "No ref_shot specified"}))
        sys.exit(0)
    if not api_path or not lib_path:
        print(json.dumps({"error": "api_path and lib_path are required"}))
        sys.exit(0)

    signals = []
    attempts = []
    categories = []

    # The EDDB catalogue remains the source of the existing signal identities.
    if "EDDB" in databases:
        eddb_signals, categories, eddb_attempts = enumerate_eddb(
            ref_shot, api_path, lib_path
        )
        signals.extend(eddb_signals)
        attempts.extend(eddb_attempts)

    for database in databases:
        if database == "EDDB":
            continue
        if database == "UDDB":
            rows, attempt = enumerate_uddb(config)
            signals.extend(rows)
        elif database == "LCDB":
            rows, attempt = enumerate_lcdb(config)
            signals.extend(rows)
        elif database == "MBDB":
            rows, attempt = enumerate_mbdb(config)
            signals.extend(rows)
        else:
            attempt = attempt_database(database, ref_shot, config)
        attempts.append(attempt)

    if databases == ["EDDB"] and attempts[0]["return_code"] != 0:
        print(json.dumps({"error": "EDDB catalogue unavailable", "attempts": attempts}))
        return

    counts = Counter(row["category"] for row in signals)
    print(
        json.dumps(
            {
                "signals": signals,
                "shot": ref_shot,
                "categories": categories,
                "ncats": len(categories),
                "category_counts": dict(sorted(counts.items())),
                "attempts": attempts,
                "absent_categories": sorted(set(categories) - counts.keys()),
            }
        )
    )


def enumerate_eddb(ref_shot, api_path, lib_path):
    # Import eddb_pwrapper from configured api_path
    try:
        sys.path.insert(0, api_path)
        from eddb_pwrapper import eddbWrapper
    except ImportError:
        return (
            [],
            [],
            [{"database": "EDDB", "call": "import eddb_pwrapper", "return_code": 1}],
        )

    # eddbWrapper Python wrapper — needs library path
    db = eddbWrapper(lib_path)
    # eddbOpen returns rtn_bool — True on success
    ok = db.eddbOpen()
    if not ok:
        return [], [], [{"database": "EDDB", "call": "eddbOpen()", "return_code": 1}]

    # Read the registered category catalogue.
    # eddbreadCatTable returns (rtn_bool, rtn_data) where rtn_data is dict
    # with keys: count, catlist, desclist, rolist, ircgrp, irc
    cat_ok = False
    cat_data = {}
    try:
        cat_ok, cat_data = db.eddbreadCatTable()
        if cat_ok and cat_data:
            categories = cat_data.get("catlist", [])
        else:
            categories = []
    except Exception:
        # Fallback: use known categories from exploration
        categories = []

    signals = []
    pid_seen = set()
    for cat in categories:
        if not cat or not cat.strip():
            continue
        cat = cat.strip()
        try:
            # Read the data names valid for this category and shot.
            # Use shot=None for catalog listing (returns latest/all data names)
            # eddbreadTable returns (rtn_bool, rtn_data) where rtn_data is dict
            # with keys: count, data, dnamelist, aliaslist, udpidlist,
            #            classlist, shotlist, unitlist, desclist, ircgrp, irc
            # The table for a shot lists the data names valid at that shot;
            # without a shot the catalogue lists every name ever registered.
            tbl_ok, tbl_data = db.eddbreadTable(ref_shot, cat)
            if not tbl_ok or tbl_data is None:
                continue

            dnames = tbl_data.get("dnamelist", [])
            aliases = tbl_data.get("aliaslist", [])
            units = tbl_data.get("unitlist", [])
            descs = tbl_data.get("desclist", [])
            classes = tbl_data.get("classlist", [])
            shot_ranges = tbl_data.get("shotlist", [])
            udp_ids = tbl_data.get("udpidlist", [])

            def _at(seq, i):
                return seq[i].strip() if i < len(seq) and seq[i] else ""

            for i, dname in enumerate(dnames):
                if not dname or not dname.strip():
                    continue
                signals.append(
                    {
                        "category": cat,
                        # dnamelist carries the full "/CAT/name" path
                        "data_name": dname.strip().split("/")[-1],
                        "alias": _at(aliases, i),
                        "units": _at(units, i),
                        "description": _at(descs, i),
                        # T = time series, O = one-point, P = parameter
                        "data_class": _at(classes, i),
                        "shot_range": _at(shot_ranges, i),
                        "udp_id": _at(udp_ids, i),
                    }
                )

            # Second pass: the one-point and condition data are keyed by
            # PID No. An empty PID on a valid one-point data name makes
            # eddbreadOne return the category's whole PID-keyed block, whose
            # records map a PID to the one-point data name announced in the
            # table by the same alias. The table's own udpidlist column is
            # empty, so this pass is the only source of the PIDs.
            one_point = [
                (dnames[i].strip().split("/")[-1], _at(aliases, i))
                for i in range(len(dnames))
                if dnames[i] and dnames[i].strip() and _at(classes, i) == "O"
            ]
            if one_point:
                dname_by_pid = {
                    pid.strip(): dname for dname, pid in one_point if pid.strip()
                }
                try:
                    pid_ok, pid_data = db.eddbreadOne(
                        ref_shot, cat, one_point[0][0], "", 0, 0
                    )
                except Exception:
                    pid_ok, pid_data = False, None
                if pid_ok and isinstance(pid_data, dict):
                    for blob in pid_data.get("data") or []:
                        for line in (blob or "").split("\n"):
                            match = PID_RECORD_RE.search(line)
                            if not match:
                                continue
                            pid = match.group(1).strip()
                            if not pid or (cat, pid) in pid_seen:
                                continue
                            pid_seen.add((cat, pid))
                            signals.append(
                                {
                                    "category": cat,
                                    "data_name": dname_by_pid.get(pid, pid),
                                    "source_dname": dname_by_pid.get(pid, ""),
                                    "alias": "",
                                    "units": match.group(3).strip(),
                                    "description": match.group(2).strip(),
                                    "data_class": "O",
                                    "shot_range": "",
                                    "udp_id": pid,
                                    "pid_keyed": True,
                                }
                            )
        except Exception:
            pass

    db.eddbClose()
    return (
        signals,
        categories,
        [
            {
                "database": "EDDB",
                "call": f"eddbreadCatTable(); eddbreadTable('{ref_shot}', category)",
                "return_code": 0 if cat_ok else (cat_data or {}).get("irc", 1),
                "count": len(signals),
            }
        ],
    )


def enumerate_uddb(config):
    sys.path.insert(0, config.get("uddb_api_path", "/analysis/src/uddb"))
    try:
        from uddb_pwrapper import uddbWrapper

        db = uddbWrapper(config.get("uddb_lib_path", "/analysis/lib/libuddb.so"))
        if not db.uddbOpen():
            return [], {"database": "UDDB", "call": "uddbOpen()", "return_code": 1}
        try:
            ok, table = db.uddbreadTable()
            headers = {}
            if ok:
                shots = [config.get("ref_shot"), *config.get("uddb_header_shots", [])]
                shots = list(
                    dict.fromkeys(
                        shot if str(shot).startswith("E") else f"E{int(shot):06d}"
                        for shot in shots
                        if shot
                    )
                )
                for pid in table.get("data") or []:
                    name = unit = source_shot = name_shot = unit_shot = ""
                    for shot in shots:
                        header_ok, response = db.uddbreadHeader(shot=shot, pid=pid)
                        if not header_ok:
                            continue
                        source_shot = source_shot or shot
                        header = (response or {}).get("data") or ""
                        if not name:
                            name = _uddb_header_value(header, "NAME")
                            if name:
                                name_shot = shot
                        if not unit:
                            unit = _uddb_header_value(header, "UNIT")
                            if unit:
                                unit_shot = shot
                        if name and unit:
                            break
                    headers[pid] = (name, unit, source_shot, name_shot, unit_shot)
        finally:
            db.uddbClose()
    except Exception as exc:
        return [], {
            "database": "UDDB",
            "call": "uddbreadTable()",
            "return_code": 1,
            "error": str(exc)[:200],
        }

    attempt = {
        "database": "UDDB",
        "call": "uddbreadTable()",
        "return_code": 0 if ok else table.get("irc", 1),
        "count": len(table.get("data") or []),
    }
    if not ok:
        return [], attempt
    aliases = table.get("aliaslist") or []
    ranges = table.get("shotlist") or []
    rows = []
    for i, pid in enumerate(table.get("data") or []):
        if not pid or not pid.strip():
            continue
        pid = pid.strip()
        alias = (aliases[i] or "").strip() if i < len(aliases) else ""
        name, unit, source_shot, name_shot, unit_shot = headers.get(
            pid, ("", "", "", "", "")
        )
        source_shots = dict.fromkeys(
            shot for shot in (source_shot, name_shot, unit_shot) if shot
        )
        rows.append(
            {
                "database": "UDDB",
                "category": "UDDB",
                "data_name": pid,
                "alias": alias,
                "shot_range": (ranges[i] or "").strip() if i < len(ranges) else "",
                "units": unit,
                "description": name or (alias if alias != pid else ""),
                "metadata_source": (
                    "; ".join(f"uddbreadHeader({shot}, {pid})" for shot in source_shots)
                    if source_shots
                    else "uddbreadTable()"
                ),
                "metadata_shot": source_shot,
                "description_source_shot": name_shot,
                "unit_source_shot": unit_shot,
            }
        )
    return rows, attempt


def _lcdb_categories(lib, root, shot):
    """Read the category names from the same library used by LcdbWrapper."""
    import ctypes

    croot = ctypes.c_char_p(root.encode("utf-8"))
    cshot = ctypes.c_int(shot)
    count = ctypes.c_int()
    width = ctypes.c_int()
    code = lib.lcdbCategoryCount(croot, cshot, ctypes.byref(count), ctypes.byref(width))
    if code != 0:
        return [], code
    if count.value < 0 or count.value > 10000 or width.value < 0 or width.value > 1024:
        return [], 1
    names = ((ctypes.c_char * (width.value + 1)) * count.value)()
    code = lib.lcdbCategoryList(
        croot, cshot, ctypes.byref(count), ctypes.byref(width), ctypes.byref(names)
    )
    if code != 0:
        return [], code
    return [
        ctypes.cast(item, ctypes.c_char_p).value.decode("utf-8") for item in names
    ], 0


def _lcdb_file_metadata(path):
    """Read declared text metadata near the start of a dataset namelist."""
    try:
        with path.open(encoding="utf-8", errors="replace") as source:
            header = source.read(65536)
    except OSError:
        return {}
    fields = {}
    for match in re.finditer(
        r"^[ \t]*([A-Za-z][A-Za-z0-9_]*)[ \t]*=[ \t]*(?:'([^']*)'|\"([^\"]*)\"|([^,\r\n]*))",
        header,
        re.MULTILINE,
    ):
        key = match.group(1).upper()
        if key.endswith("UNIT") or key in {"COMMENT", "DESCRIPTION", "DESC"}:
            fields[key] = next(
                (part.strip() for part in match.groups()[1:] if part is not None), ""
            )
    return fields


def _lcdb_field_unit(name, metadata):
    """Apply an axis unit only to the data arrays on that axis."""
    field = name.upper()

    def declared(value, key):
        return ("" if value.strip().upper() == "DEBUG" else value), key

    for key in (f"{field}_UNIT", f"{field}UNIT"):
        if metadata.get(key):
            return declared(metadata[key], key)
    if re.match(r"^X(?:EXP|FIT)DATA", field) and metadata.get("XUNIT"):
        return declared(metadata["XUNIT"], "XUNIT")
    if re.match(r"^Y(?:EXP|FIT)DATA", field) and metadata.get("YUNIT"):
        return declared(metadata["YUNIT"], "YUNIT")
    return "", ""


def enumerate_lcdb(config):
    """Enumerate every readable owner, shot, category and data name."""
    api_path = config.get("lcdb_api_path", "/analysis/src/lcdbWrapper")
    root_base = config.get("lcdb_root", "/analysis_DB/EDASDB")
    call = f"lcdb_shot(root=owner); lcdbCategoryList(owner, shot); lcdb_dname(shot, category, root=owner) under {root_base}"
    try:
        sys.path.insert(0, api_path)
        from lcdbWrapper import LcdbWrapper, lib

        db = LcdbWrapper()
        owners = sorted(entry.path for entry in os.scandir(root_base) if entry.is_dir())
    except Exception as exc:
        return [], {
            "database": "LCDB",
            "call": call,
            "return_code": 1,
            "error": str(exc)[:200],
        }

    found = {}
    roots_with_shots = 0
    shots_seen = 0
    categories_seen = 0
    failures = 0
    for root in owners:
        try:
            ok, shots = db.lcdb_shot(root=root)
            if not ok or not shots:
                continue
            roots_with_shots += 1
            shots_seen += len(shots)
            owner = os.path.basename(root)
            for shot in sorted(shots):
                categories, code = _lcdb_categories(lib, root, shot)
                if code != 0:
                    failures += 1
                    continue
                categories_seen += len(categories)
                for category in categories:
                    ok, names = db.lcdb_dname(shot, category, root=root)
                    if not ok:
                        failures += 1
                        continue
                    source_file = (
                        Path(root)
                        / f"{shot // 10000:02d}"
                        / f"{shot // 100:04d}"
                        / f"{shot:06d}"
                        / "lcdb"
                        / f"{category}.ldb"
                    )
                    metadata = _lcdb_file_metadata(source_file)
                    source_exists = source_file.is_file()
                    for name in names:
                        key = (owner, category, name)
                        unit, unit_key = _lcdb_field_unit(name, metadata)
                        found[key] = {
                            "database": "LCDB",
                            "category": f"LCDB/{owner}/{category}",
                            "file_category": category,
                            "data_name": name,
                            "root": root,
                            "shot": shot,
                            "units": unit,
                            "description": (
                                metadata.get("DESCRIPTION")
                                or metadata.get("DESC")
                                or metadata.get("COMMENT")
                                or ""
                            ),
                            "metadata_source": str(source_file),
                            "metadata_file_present": source_exists,
                            "unit_source_key": unit_key,
                            "source_unit_value": metadata.get(unit_key, ""),
                        }
        except Exception:
            failures += 1
    rows = list(found.values())
    return rows, {
        "database": "LCDB",
        "call": call,
        "return_code": 0 if rows else 1,
        "count": len(rows),
        "roots_scanned": len(owners),
        "roots_with_shots": roots_with_shots,
        "shots": shots_seen,
        "categories": categories_seen,
        "failed_calls": failures,
    }


def _mbdb_data_names(lib):
    """Read the opened case's names through the MBDB library catalogue API."""
    import ctypes

    count = ctypes.c_int()
    width = ctypes.c_int()
    code = lib.mbdbDatanmCount(ctypes.byref(count), ctypes.byref(width))
    if code != 0:
        return [], code
    if count.value < 0 or count.value > 10000 or width.value < 0 or width.value > 1024:
        return [], 1
    names = ((ctypes.c_char * (width.value + 1)) * count.value)()
    kinds = (ctypes.c_char * count.value)()
    code = lib.mbdbDatanmList(
        ctypes.byref(count), width, ctypes.byref(names), ctypes.byref(kinds)
    )
    if code != 0:
        return [], code
    return [
        (ctypes.cast(name, ctypes.c_char_p).value.decode("utf-8"), kinds[i].decode())
        for i, name in enumerate(names)
    ], 0


def enumerate_mbdb(config):
    """Open every readable case file and enumerate its field names."""
    api_path = config.get("mbdb_api_path", "/analysis/src/mbdb")
    root_base = config.get("mbdb_root", "/analysis_DB/MBDB")
    route = "mbdbSetDirectory('mbdb'); mbdbROpen(owner, case, file category)"
    try:
        sys.path.insert(0, api_path)
        from mbdbWrapper import mbdbWrapper

        db = mbdbWrapper(config.get("mbdb_lib_path", "/analysis/lib/libmbdb.so"))
        candidates = sorted(Path(root_base).glob("*/??/????/??????/mbdb/*.ldb"))
        set_ok, set_result = db.mbdbSetDirectory("mbdb")
        if not set_ok:
            return [], {
                "database": "MBDB",
                "call": route,
                "return_code": set_result.get("irtn", 1),
            }
    except Exception as exc:
        return [], {
            "database": "MBDB",
            "call": route,
            "return_code": 1,
            "error": str(exc)[:200],
        }

    found = {}
    calls = []
    for path in candidates:
        root = str(path.parents[4])
        owner = path.parents[4].name
        case = int(path.parents[1].name)
        category = path.name.removesuffix(".ldb")
        ok, response = db.mbdbROpen(mbdbroot=root, caseno=case, category=category)
        code = response.get("irtn", 0 if ok else 1)
        calls.append(
            {"root": root, "case": case, "category": category, "return_code": code}
        )
        if not ok:
            continue
        try:
            names, code = _mbdb_data_names(db.mbdb)
            if code != 0:
                calls[-1]["name_return_code"] = code
                continue
            for name, kind in names:
                found[(owner, case, category, name)] = {
                    "database": "MBDB",
                    "category": f"MBDB/{owner}/{case}/{category}",
                    "file_category": category,
                    "data_name": name,
                    "data_kind": kind,
                    "root": root,
                    "case": case,
                }
        finally:
            db.mbdbRClose()
    rows = list(found.values())
    return rows, {
        "database": "MBDB",
        "call": route,
        "return_code": 0 if rows else (calls[0]["return_code"] if calls else 2),
        "count": len(rows),
        "candidate_files": len(candidates),
        "calls": calls,
        "field_list_call": "mbdbDatanmCount(); mbdbDatanmList() after a successful open",
    }


def attempt_database(database, ref_shot, config):
    """Record a bounded catalogue probe when no usable channel list is exposed."""
    call = "wrapper import"
    try:
        if database == "PMDB":
            sys.path.insert(0, config.get("pmdb_api_path", "/analysis/src/pmdb"))
            from pmdb_wrapper import pmdbWrapper

            db = pmdbWrapper(config.get("pmdb_lib_path", "/analysis/lib/libpmdb.so"))
            ok, response = db.plantdread(cat="*", dname="*", t1="", t2="")
            header_ok, header = db.planthread(
                cat="*", dname="*", t1="", descriptor="", count=1
            )
            call = "plantdread(cat='*', dname='*', t1='', t2=''); planthread(cat='*', dname='*', t1='', descriptor='', count=1)"
            code = (response or {}).get("irc", 0 if ok else 1)
            return {
                "database": database,
                "call": call,
                "return_code": code,
                "header_return_code": (header or {}).get("irc", 0 if header_ok else 1),
                "reason": "the native wrapper exposes keyed reads but no catalogue method",
            }
        elif database == "EQDB":
            path = config.get("eqdb_root", "/analysis_DB/EQDB")
            with os.scandir(path) as entries:
                count = sum(1 for _ in entries)
            return {
                "database": database,
                "call": f"scandir('{path}')",
                "return_code": 0,
                "count": count,
                "reason": "central EQDB directory probe; local format fields come from configured exemplars",
            }
        else:
            return {
                "database": database,
                "call": "unsupported database",
                "return_code": 2,
            }
    except OSError as exc:
        return {
            "database": database,
            "call": call if database != "EQDB" else f"scandir('{path}')",
            "return_code": exc.errno or 1,
            "error": str(exc)[:200],
        }
    except Exception as exc:
        return {
            "database": database,
            "call": call,
            "return_code": 1,
            "error": str(exc)[:200],
        }
    return {
        "database": database,
        "call": call,
        "return_code": code,
        "reason": "no enumerated channel catalogue",
    }


if __name__ == "__main__":
    main()
