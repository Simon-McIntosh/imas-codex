"""Read JT-60SA local EQDB records and G-EQDSK files by format.

This stdlib-only script is sent to the facility host by run_python_script.
The record decoders are shared with the equilibrium inventory checker.
"""

import json
import math
import os
import re
import struct
import sys
from pathlib import Path

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
    if len(numbers) < 20:
        raise ValueError("short G EQDSK geometry header")
    rdim, zdim, _, rleft, zmid = numbers[:5]
    if not all(math.isfinite(value) for value in (rdim, zdim, rleft, zmid)):
        raise ValueError("nonfinite G EQDSK geometry")
    if rdim <= 0 or zdim <= 0:
        raise ValueError("nonpositive G EQDSK grid extent")
    r = [rleft + i * rdim / (nr - 1) for i in range(nr)]
    z = [zmid - zdim / 2 + i * zdim / (nz - 1) for i in range(nz)]
    header = lines[0]
    shot, time = _shot_time(header, path)
    if "LIUQE" in header.upper() or "MEQ" in header.upper():
        code = "LIUQE/MEQ"
        basis = "header"
    elif "CHEASE" in header.upper():
        code = "CHEASE"
        basis = "header"
    elif header.lstrip().upper().startswith("EFIT"):
        code = "EFIT"
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
    cocos = re.search(r"COCOS[_ -]?(\d{1,2})", path + " " + header, re.I)
    if cocos:
        result["cocos"] = int(cocos.group(1))
    if include_grid:
        result.update(r=r, z=z, psi=psi)
    return result


def read_grid(path, include_grid=False):
    """Parse one equilibrium file, validating records and its finite flux grid."""
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
    result["path"] = str(path)
    result["bytes"] = len(data)
    return result


def resolve_eqdb_path(root, shot, time):
    """Apply the EQDB client's local file convention to a shot and time."""
    explicit_file = os.environ.get("EQDB_FILE")
    if explicit_file:
        return Path(explicit_file)
    base = Path(root or os.environ.get("EQDSK_DIR") or Path.home() / "LOCALDB/EQDSK")
    shot_name = f"{int(shot):06d}"
    time_name = f"{round(float(time) * 1000):06d}"
    return base / shot_name[:2] / shot_name[:4] / shot_name / time_name


def resolve_geqdsk_path(root, shot, time, filename_template):
    """Resolve a producer's file pattern within its root directory."""
    if not filename_template:
        raise ValueError("G-EQDSK access requires a filename template")
    filename = filename_template.format(shot=int(shot), time=float(time))
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("G-EQDSK filename must stay within root")
    return Path(root) / relative


def read_equilibrium(config):
    """Resolve, parse, and verify an equilibrium file for a template read."""
    kind = config["format"]
    shot = int(config["shot"])
    time = float(config["time"])
    if kind == "eqdb_record":
        path = resolve_eqdb_path(config.get("root"), shot, time)
    elif kind == "g_eqdsk":
        path = resolve_geqdsk_path(
            config["root"], shot, time, config.get("filename_template")
        )
    else:
        raise ValueError(f"unsupported equilibrium format: {kind}")
    result = read_grid(str(path), include_grid=bool(config.get("field")))
    if kind == "eqdb_record" and result["format"] not in {"selene_eq31", "eq11"}:
        raise ValueError("file is not an EQDB record")
    if kind == "g_eqdsk" and result["format"] != "g_eqdsk":
        raise ValueError("file is not G-EQDSK")
    if result["shot"] is not None and result["shot"] != shot:
        raise ValueError("equilibrium shot does not match request")
    if result["time_s"] is not None and abs(result["time_s"] - time) > 0.001:
        raise ValueError("equilibrium time does not match request")
    field = config.get("field")
    if field:
        field_name = {
            "RG": "r",
            "ZG": "z",
            "PSI": "psi",
            "psirz": "psi",
            "NSR": "nr",
            "NSZ": "nz",
            "COCOS": "cocos",
        }.get(field, field)
        if field_name in {"nr", "nz"}:
            result["data"] = result["grid"][0 if field_name == "nr" else 1]
        elif field_name == "cocos":
            if "cocos" not in result:
                raise ValueError("COCOS is not declared by this G-EQDSK file")
            result["data"] = result["cocos"]
        elif field_name in {"r", "z", "psi"}:
            result["data"] = result[field_name]
        else:
            raise ValueError(f"unsupported equilibrium field: {field}")
        for name in ("r", "z", "psi"):
            result.pop(name, None)
    return result


def main():
    print(json.dumps(read_equilibrium(json.load(sys.stdin))))


if __name__ == "__main__":
    main()
