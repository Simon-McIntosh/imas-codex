"""Inventory and verify readable JT-60SA equilibrium grids on nakasvr26.

Run this script on the analysis host. Collection walks only readable entries
under /home and /analysis_DB, and never follows directory symlinks.
"""

import argparse
import json
import os
import pwd
import re
import struct
from collections import Counter

from imas_codex.remote.scripts.read_equilibrium_file import (
    read_grid as read_equilibrium_grid,
)

GEQDSK_NAME = re.compile(r"(?:eqdsk|geqdsk|^g\d{6})", re.I)


def facility_basis(result):
    path = result["path"].lower()
    header = result.get("header", "").lower()
    if "/localdb/" in path:
        return "LOCALDB path"
    if any(part.startswith("eqdb_") for part in path.split(os.sep)):
        return "EQDB directory"
    if "/jt60sa/" in path or "jt-60sa" in header or "jt60sa" in header:
        return "machine named in path or header"
    if "/work_sa/" in path and "input_sa.txt" in header:
        return "SA simulation path and header"
    if "/equil_runs/" in path and re.search(r"e\d{6}", os.path.basename(path)):
        return "shot-tagged reconstruction path"
    return None


def read_grid(path, include_grid=False):
    result = read_equilibrium_grid(path, include_grid=include_grid)
    result["owner"] = pwd.getpwuid(os.stat(path).st_uid).pw_name
    result["facility_basis"] = facility_basis(result)
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
                "PASS {} {} code={} facility={} shot={} time={} grid={} psi_range={}".format(
                    item["format"],
                    path,
                    item["code"],
                    item["facility_basis"],
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
