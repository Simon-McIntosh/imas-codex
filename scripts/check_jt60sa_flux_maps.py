"""Inventory and verify readable JT-60SA equilibrium grids on nakasvr26.

Run this script on the analysis host. Collection walks only readable entries
under /home and /analysis_DB, and never follows directory symlinks.
"""

import argparse
import json
import math
import os
import pwd
import re
import struct
from collections import Counter

FLOAT = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[EeDd][-+]?\d+)?")
SHOT = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
GEQDSK_NAME = re.compile(r"(?:eqdsk|geqdsk|^g\d{6})", re.I)


def _finite_grid(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite or empty flux grid")
    if max(values) <= min(values):
        raise ValueError("constant flux grid")


def _shot_time(header, path):
    match = SHOT.search(header) or SHOT.search(os.path.basename(path))
    shot = int(match.group(1)) if match else None
    match = re.search(r"t\s*=\s*(\d+(?:\.\d+)?)", header, re.I)
    if match:
        return shot, float(match.group(1))
    match = re.search(r"(\d+)\s*ms", header, re.I)
    if match:
        return shot, int(match.group(1)) / 1000.0
    match = re.search(r"[tT](\d+(?:\.\d+)?)", os.path.basename(path))
    return shot, float(match.group(1)) if match else None


def _binary_record(data, offset):
    if offset + 8 > len(data):
        raise ValueError("truncated record marker")
    size = struct.unpack_from(">I", data, offset)[0]
    end = offset + size + 8
    if end > len(data) or struct.unpack_from(">I", data, end - 4)[0] != size:
        raise ValueError("Fortran record markers disagree")
    return data[offset + 4 : end - 4], end


def _selene(data, path, include_grid):
    header, offset = _binary_record(data, 0)
    body, offset = _binary_record(data, offset)
    while offset < len(data):
        _, offset = _binary_record(data, offset)
    if len(header) < 12 or len(body) < 24:
        raise ValueError("short SELENE header or body")
    shot = struct.unpack_from(">i", header, 0)[0]
    time = struct.unpack_from(">d", header, 4)[0]
    nr, nz = struct.unpack_from(">6i", body, 0)[-2:]
    if not 4 <= nr <= 2048 or not 4 <= nz <= 2048:
        raise ValueError("implausible SELENE grid dimensions")
    needed = 24 + 8 * (nr + nz + nr * nz)
    if len(body) < needed:
        raise ValueError("short SELENE flux grid")
    r = struct.unpack_from(f">{nr}d", body, 24)
    z = struct.unpack_from(f">{nz}d", body, 24 + 8 * nr)
    psi = struct.unpack_from(f">{nr * nz}d", body, 24 + 8 * (nr + nz))
    if not all(a < b for a, b in zip(r, r[1:], strict=False)) or not all(
        a < b for a, b in zip(z, z[1:], strict=False)
    ):
        raise ValueError("SELENE coordinate axis is not increasing")
    _finite_grid(psi)
    result = {
        "format": "selene_eq31",
        "code": "SELENE",
        "shot": shot,
        "time_s": time,
        "grid": [nr, nz],
        "psi_range": [min(psi), max(psi)],
    }
    if include_grid:
        result.update(r=list(r), z=list(z), psi=list(psi))
    return result


def _eq11(data, path, include_grid):
    header, offset = _binary_record(data, 0)
    body, offset = _binary_record(data, offset)
    while offset < len(data):
        _, offset = _binary_record(data, offset)
    nr, nz = struct.unpack_from(">2i", body)
    if not 4 <= nr <= 2048 or not 4 <= nz <= 2048:
        raise ValueError("implausible EQ11 grid dimensions")
    needed = 8 + 8 * (nr + nz + nr * nz)
    if len(body) < needed:
        raise ValueError("short EQ11 flux grid")
    r = struct.unpack_from(f">{nr}d", body, 8)
    z = struct.unpack_from(f">{nz}d", body, 8 + 8 * nr)
    psi = struct.unpack_from(f">{nr * nz}d", body, 8 + 8 * (nr + nz))
    if not all(a < b for a, b in zip(r, r[1:], strict=False)) or not all(
        a < b for a, b in zip(z, z[1:], strict=False)
    ):
        raise ValueError("EQ11 coordinate axis is not increasing")
    _finite_grid(psi)
    title = header[12:].decode("ascii", "replace").strip()
    parent = os.path.basename(os.path.dirname(path))
    shot = int(parent) if SHOT.fullmatch(parent) else None
    time = struct.unpack_from(">d", header, 4)[0]
    result = {
        "format": "eq11",
        "code": "TOPICS" if "topics" in title.lower() else "unidentified EQ11",
        "shot": shot,
        "time_s": time,
        "grid": [nr, nz],
        "psi_range": [min(psi), max(psi)],
        "header": title,
    }
    if include_grid:
        result.update(r=list(r), z=list(z), psi=list(psi))
    return result


def _geqdsk(data, path, include_grid):
    lines = data.decode("ascii").splitlines()
    fields = lines[0].split()
    nr, nz = int(fields[-2]), int(fields[-1])
    if not 4 <= nr <= 2048 or not 4 <= nz <= 2048:
        raise ValueError("implausible G EQDSK grid dimensions")
    numbers = [
        float(x.replace("D", "E").replace("d", "e"))
        for x in FLOAT.findall("\n".join(lines[1:]))
    ]
    offset = 20 + 4 * nr
    psi = numbers[offset : offset + nr * nz]
    if len(psi) != nr * nz:
        raise ValueError("short G EQDSK flux grid")
    _finite_grid(psi)
    header = lines[0]
    shot, time = _shot_time(header, path)
    if "LIUQE" in header.upper() or "MEQ" in header.upper():
        code = "LIUQE/MEQ"
        basis = "header"
    elif header.lstrip().startswith("SA"):
        code = "SA"
        basis = "header"
    elif "input_sa.txt" in header.lower() or "/work_sa/" in path.lower():
        code = "SA"
        basis = "header/path"
    else:
        code = "unidentified G EQDSK"
        basis = "unidentified"
    result = {
        "format": "g_eqdsk",
        "code": code,
        "shot": shot,
        "time_s": time,
        "grid": [nr, nz],
        "psi_range": [min(psi), max(psi)],
        "header": header,
        "code_basis": basis,
    }
    if include_grid:
        result.update(psi=psi)
    return result


def read_grid(path, include_grid=False):
    with open(path, "rb") as stream:
        data = stream.read()
    if len(data) < 64:
        raise ValueError("file too short for an equilibrium grid")
    if struct.unpack_from(">I", data)[0] == 252:
        title = data[16:20]
        if title == b"EQ31":
            result = _selene(data, path, include_grid)
        elif title == b"EQ11":
            result = _eq11(data, path, include_grid)
        else:
            raise ValueError("unrecognised Fortran equilibrium record")
    else:
        result = _geqdsk(data, path, include_grid)
    result["path"] = path
    result["bytes"] = len(data)
    result["owner"] = pwd.getpwuid(os.stat(path).st_uid).pw_name
    return result


def _candidates(roots, errors):
    for root in roots:
        for base, dirs, files in os.walk(
            root, followlinks=False, onerror=lambda err: errors.append(str(err))
        ):
            dirs[:] = [
                name for name in dirs if not os.path.islink(os.path.join(base, name))
            ]
            parts = base.split(os.sep)
            in_local = "LOCALDB" in parts and ("EQDSK" in parts or "2M" in parts)
            in_eqdb = any(part.upper().startswith("EQDB_") for part in parts)
            for name in files:
                path = os.path.join(base, name)
                if in_local or in_eqdb or GEQDSK_NAME.search(name):
                    yield path


def collect(selected_roots):
    if selected_roots:
        roots = selected_roots
    else:
        homes = [
            entry.path
            for entry in os.scandir("/home")
            if entry.is_dir(follow_symlinks=False)
            and os.access(entry.path, os.R_OK | os.X_OK)
        ]
        roots = sorted(homes) + ["/analysis_DB"]
    roots = [os.path.realpath(root) for root in roots]
    for root in roots:
        if not (root == "/analysis_DB" or os.path.dirname(root) == "/home"):
            raise ValueError(
                "collection root lies outside readable homes and /analysis_DB"
            )
        if not os.path.isdir(root) or not os.access(root, os.R_OK | os.X_OK):
            raise ValueError("collection root is not a readable directory")
    for root in roots:
        errors = []
        rejected = Counter()
        files = []
        for path in _candidates([root], errors):
            try:
                files.append(read_grid(path))
            except (OSError, ValueError, IndexError, struct.error, UnicodeError) as err:
                rejected[type(err).__name__ + ": " + str(err)] += 1
        result = {
            "host": os.uname().nodename,
            "root": root,
            "files": sorted(files, key=lambda item: item["path"]),
            "rejected": dict(rejected),
            "walk_error_count": len(errors),
            "walk_error_samples": errors[:20],
        }
        print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--collect-root", action="append")
    parser.add_argument("--grid")
    parser.add_argument("--check", nargs="+")
    args = parser.parse_args()
    if args.collect:
        collect(args.collect_root)
    elif args.grid:
        print(json.dumps(read_grid(args.grid, include_grid=True), sort_keys=True))
    elif args.check:
        seen = set()
        for path in args.check:
            item = read_grid(path)
            seen.add(item["format"])
            print(
                "PASS {} {} shot={} time={} grid={} psi_range={}".format(
                    item["format"],
                    path,
                    item["shot"],
                    item["time_s"],
                    item["grid"],
                    item["psi_range"],
                )
            )
        if not {"selene_eq31", "eq11", "g_eqdsk"}.issubset(seen):
            raise SystemExit("each equilibrium format requires a readable sample")
    else:
        parser.error("select --collect, --grid, or --check")


if __name__ == "__main__":
    main()
