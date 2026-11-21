#!/usr/bin/env python3
"""Validate JT-60SA EDAS signals return data for a reference shot.

This script runs on the JT-60SA host where eddb_pwrapper is available.
It reads each signal with the call selected by its EDDB catalogue class.

Requirements:
- Python 3.8+ (stdlib only except eddb_pwrapper)
- eddb_pwrapper (available at /analysis/src/eddb)
- libeddb.so (available at /analysis/lib/libeddb.so)

Usage:
    echo '{"signals": [...], "ref_shot": "E012345"}' | python3 check_edas.py

Input (JSON on stdin):
    {
        "signals": [
            {"id": "jt-60sa:general/eddb_testime", "category": "EDDB", "data_name": "tesTime"},
            ...
        ],
        "ref_shot": "E012345",
        "api_path": "/analysis/src/eddb",
        "lib_path": "/analysis/lib/libeddb.so"
    }

Output (JSON on stdout):
    {
        "results": [
            {"id": "jt-60sa:general/eddb_testime", "success": true, "dtype": "ndarray"},
            {"id": "jt-60sa:general/eddb_bad", "success": false, "error": "eddbreadOne returned None"},
            ...
        ]
    }
"""

import json
import re
import sys


def main():
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(0)

    signals = config.get("signals", [])
    ref_shot = config.get("ref_shot", "")
    api_path = config.get("api_path", "")
    lib_path = config.get("lib_path", "")

    if not ref_shot:
        print(
            json.dumps(
                {
                    "results": [
                        {"id": s["id"], "success": False, "error": "no ref_shot"}
                        for s in signals
                    ]
                }
            )
        )
        sys.exit(0)

    if not api_path or not lib_path:
        print(
            json.dumps(
                {
                    "results": [
                        {
                            "id": s["id"],
                            "success": False,
                            "error": "api_path and lib_path are required",
                        }
                        for s in signals
                    ]
                }
            )
        )
        sys.exit(0)

    try:
        sys.path.insert(0, api_path)
        from eddb_pwrapper import eddbWrapper
    except ImportError:
        print(
            json.dumps(
                {
                    "results": [
                        {
                            "id": s["id"],
                            "success": False,
                            "error": f"eddb_pwrapper not available at {api_path}",
                        }
                        for s in signals
                    ]
                }
            )
        )
        sys.exit(0)

    db = eddbWrapper(lib_path)
    ok = db.eddbOpen()
    if not ok:
        print(
            json.dumps(
                {
                    "results": [
                        {
                            "id": s["id"],
                            "success": False,
                            "error": "eddbOpen() failed",
                        }
                        for s in signals
                    ]
                }
            )
        )
        sys.exit(0)

    results = []
    for sig in signals:
        try:
            if sig.get("database") == "MBDB":
                sys.path.insert(0, config.get("mbdb_api_path") or "/analysis/src/mbdb")
                from mbdbWrapper import mbdbWrapper

                db = mbdbWrapper(
                    config.get("mbdb_lib_path") or "/analysis/lib/libmbdb.so"
                )
                base = config.get("mbdb_root") or "/analysis_DB/MBDB"
                db.mbdbSetDirectory("mbdb")
                opened, response = db.mbdbROpen(
                    mbdbroot=f"{base}/{sig['owner']}",
                    caseno=int(sig["case"]),
                    category=sig["category"],
                )
                if not opened:
                    results.append(
                        {
                            "id": sig["id"],
                            "success": False,
                            "error": f"MBDB open failed (irtn={response.get('irtn')})",
                        }
                    )
                    continue
                try:
                    if sig.get("data_class") == "one_point":
                        ok, _value = db.mbdbRPoint(sig["data_name"])
                    else:
                        ok, _value = db.mbdbRTimes(sig["data_name"], "0", "0.01")
                finally:
                    db.mbdbRClose()
                results.append(
                    {
                        "id": sig["id"],
                        "success": bool(ok),
                        "dtype": "analysis" if ok else None,
                        "error": None if ok else "MBDB value unavailable",
                    }
                )
                continue
            if sig.get("database") == "LCDB":
                sys.path.insert(
                    0, config.get("lcdb_api_path") or "/analysis/src/lcdbWrapper"
                )
                from lcdbWrapper import LcdbWrapper

                root = config.get("lcdb_root") or "/analysis_DB/EDASDB"
                ok, values = LcdbWrapper().lcdb_value(
                    int(sig["shot"]),
                    sig["category"],
                    [sig["data_name"]],
                    root=f"{root}/{sig['owner']}",
                )
                has_value = (
                    ok
                    and isinstance(values, dict)
                    and values.get(sig["data_name"]) is not None
                )
                results.append(
                    {
                        "id": sig["id"],
                        "success": bool(has_value),
                        "dtype": "analysis_value" if has_value else None,
                        "error": None if has_value else "LCDB value unavailable",
                    }
                )
                continue
            if sig.get("database") == "UDDB":
                sys.path.insert(0, config.get("uddb_api_path") or "/analysis/src/uddb")
                from uddb_pwrapper import uddbWrapper

                raw = uddbWrapper(
                    config.get("uddb_lib_path") or "/analysis/lib/libuddb.so"
                )
                opened = raw.uddbOpen()
                if not opened:
                    results.append(
                        {"id": sig["id"], "success": False, "error": "UDDB open failed"}
                    )
                    continue
                try:
                    shot = sig.get("shot") or ref_shot
                    header_ok, header = raw.uddbreadHeader(shot=shot, pid=sig["pid"])
                    if not header_ok:
                        results.append(
                            {
                                "id": sig["id"],
                                "success": False,
                                "error": f"UDDB header unavailable (irc={(header or {}).get('irc')})",
                            }
                        )
                        continue
                    header_text = (header or {}).get("data") or ""
                    match = re.search(
                        r"\bSTIME=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
                        header_text,
                    )
                    start = float(match.group(1)) if match else 0.0
                    t1, t2 = f"{start:.6f}", f"{start + 0.0001:.6f}"
                    ok, response = raw.uddbreadConvert(
                        shot=shot, pid=sig["pid"], t1=t1, t2=t2, datavol=1, ch=1
                    )
                    needed = (response or {}).get("datavol", 0)
                    if (
                        not ok
                        and (response or {}).get("irc") == 1062
                        and 1 < needed <= 10000
                    ):
                        ok, response = raw.uddbreadConvert(
                            shot=shot,
                            pid=sig["pid"],
                            t1=t1,
                            t2=t2,
                            datavol=needed,
                            ch=1,
                        )
                    values = (response or {}).get("data")
                    has_value = bool(ok and values is not None and len(values) > 0)
                finally:
                    raw.uddbClose()
                results.append(
                    {
                        "id": sig["id"],
                        "success": has_value,
                        "dtype": "raw_value" if has_value else None,
                        "error": None
                        if has_value
                        else f"UDDB value unavailable (irc={(response or {}).get('irc')})",
                    }
                )
                continue
            # A catalogue hit says a name is registered, not that this shot
            # carries it. Refuse letters without a known read call instead of
            # treating their payloads as time series.
            data_class = sig.get("data_class")
            if data_class not in {"T", "O", "P", "J", "M", "G", "N"}:
                results.append(
                    {
                        "id": sig["id"],
                        "success": False,
                        "error": f"unrecognised EDDB data class {data_class or '<unset>'}",
                    }
                )
                continue
            if data_class == "T":
                ok, rtn = db.eddbreadTime(
                    ref_shot, sig["category"], sig["data_name"], "0", "0.01"
                )
                has_value = bool(ok and (rtn or {}).get("ntime", 0))
                dtype = "time_series"
            elif data_class == "O":
                ok, rtn = db.eddbreadOne(
                    ref_shot, sig["category"], sig["data_name"], None, 0, 0
                )
                has_value = bool(ok and (rtn or {}).get("count", 0))
                dtype = "one_point"
            elif data_class in {"P", "J"}:
                ok, rtn = db.eddbreadPara(
                    ref_shot, sig["category"], sig["data_name"], None, 0, 0
                )
                has_value = bool(ok and (rtn or {}).get("count", 0))
                dtype = "parameter"
            elif data_class == "M":
                ok, rtn = db.eddbreadImage(
                    ref_shot, sig["category"], sig["data_name"], 0
                )
                has_value = bool(ok and (rtn or {}).get("datasize", 0))
                dtype = "image"
            elif data_class == "G":
                ok, rtn = db.eddbreadBinary(
                    ref_shot, sig["category"], sig["data_name"], 0
                )
                has_value = bool(ok and (rtn or {}).get("datasize", 0))
                dtype = "binary"
            else:
                ok, rtn = db.eddbreadComment(
                    ref_shot, sig["category"], sig["data_name"], 0
                )
                has_value = bool(ok and (rtn or {}).get("data"))
                dtype = "comment"
            if has_value:
                results.append({"id": sig["id"], "success": True, "dtype": dtype})
            else:
                irc = (rtn or {}).get("irc") if isinstance(rtn, dict) else None
                results.append(
                    {
                        "id": sig["id"],
                        "success": False,
                        "error": f"no data for {ref_shot} (irc={irc})",
                    }
                )
        except Exception as e:
            results.append(
                {
                    "id": sig["id"],
                    "success": False,
                    "error": str(e)[:200],
                }
            )

    db.eddbClose()
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
