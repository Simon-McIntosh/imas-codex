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

# One record of the PID-keyed block that eddbreadOne returns for an empty
# PID: PID="1651 A001" NAME="Data acquisition start time" UNIT="ms" DATA="-15000"
PID_RECORD_RE = re.compile(
    r'PID="([^"]*)"\s+NAME="([^"]*)"\s+UNIT="([^"]*)"\s+DATA="([^"]*)"'
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
    rows = [
        {
            "database": "UDDB",
            "category": "UDDB",
            "data_name": pid.strip(),
            "alias": (aliases[i] or "").strip() if i < len(aliases) else "",
            "shot_range": (ranges[i] or "").strip() if i < len(ranges) else "",
        }
        for i, pid in enumerate(table.get("data") or [])
        if pid and pid.strip()
    ]
    return rows, attempt


def attempt_database(database, ref_shot, config):
    """Record a bounded catalogue probe when no usable channel list is exposed."""
    call = "wrapper import"
    try:
        if database == "PMDB":
            sys.path.insert(0, config.get("pmdb_api_path", "/analysis/src/pmdb"))
            from pmdb_wrapper import pmdbWrapper

            db = pmdbWrapper(config.get("pmdb_lib_path", "/analysis/lib/libpmdb.so"))
            ok, response = db.plantdread(cat="", dname="", t1="", t2="")
            call = "plantdread(cat='', dname='', t1='', t2='')"
            code = (response or {}).get("irc", 0 if ok else 1)
        elif database == "LCDB":
            sys.path.insert(0, config.get("lcdb_api_path", "/analysis/src/lcdbWrapper"))
            from lcdbWrapper import LcdbWrapper

            root = config.get("lcdb_root", "/analysis_DB/EDASDB/public")
            ok, shots = LcdbWrapper().lcdb_shot(root=root)
            call = f"lcdb_shot(root='{root}')"
            code = 0 if ok else 1
            if ok:
                return {
                    "database": database,
                    "call": call,
                    "return_code": code,
                    "count": len(shots),
                    "reason": "shot list has no channel catalogue",
                }
        elif database == "MBDB":
            sys.path.insert(0, config.get("mbdb_api_path", "/analysis/src/mbdb"))
            from mbdbWrapper import mbdbWrapper

            root = config.get("mbdb_root", "/analysis_DB/MBDB/public")
            db = mbdbWrapper(config.get("mbdb_lib_path", "/analysis/lib/libmbdb.so"))
            ok, response = db.mbdbROpen(
                mbdbroot=root, caseno=int(ref_shot[1:]), category="MBEQ"
            )
            if ok:
                db.mbdbRClose()
            call = f"mbdbROpen(mbdbroot='{root}', caseno={int(ref_shot[1:])}, category='MBEQ')"
            code = (response or {}).get("irtn", 0 if ok else 1)
        elif database == "EQDB":
            path = config.get("eqdb_root", "/analysis_DB/EQDB")
            with os.scandir(path) as entries:
                count = sum(1 for _ in entries)
            return {
                "database": database,
                "call": f"scandir('{path}')",
                "return_code": 0,
                "count": count,
                "reason": "no field catalogue or confirmed client read route",
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
