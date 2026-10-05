#!/usr/bin/env python3
"""Report the remote interpreter version and required-module importability.

Executed through the facility-aware layer on the interpreter and shell the
facility declares, so the answer describes the environment a remote scan
actually uses rather than the login shell's default. Reads one JSON object on
stdin naming the modules the facility requires and writes one JSON object on
stdout. A module that fails to import is reported, never raised, and the
script exits 0 so an unmet environment is a finding rather than a transport
error.

The script itself must parse under the old interpreter it is meant to detect,
so it uses only stdlib and Python 3.5 syntax (no f-strings, no annotations).

Usage:
    echo '{"modules": [{"name": "numpy", "path": null}]}' | python3 probe_environment.py

Input (JSON on stdin):
    {"modules": [{"name": "numpy", "path": null},
                 {"name": "eddb_pwrapper", "path": "/analysis/src/eddb"}]}

Output (JSON on stdout):
    {"python_version": "3.12.9",
     "modules": [{"name": "numpy", "importable": true, "error": null},
                 {"name": "eddb_pwrapper", "importable": false,
                  "error": "ModuleNotFoundError: No module named 'eddb_pwrapper'"}]}

A module entry may carry a ``path`` that is added to ``sys.path`` before the
import, for a module installed beside a source tree rather than into the
interpreter's environment.
"""

import json
import sys


def _module_report(entry):
    """Import one module and report the caller-facing verdict.

    A ``path`` on the entry is prepended to ``sys.path`` before the import.
    The import result is returned as a plain mapping; a failure carries the
    exception class and message so the caller can name the missing module.
    """
    report = {"name": entry.get("name"), "importable": False, "error": None}
    name = entry.get("name")
    if not name:
        report["error"] = "no module name given"
        return report

    path = entry.get("path")
    if path and path not in sys.path:
        sys.path.insert(0, path)

    try:
        __import__(name)
        report["importable"] = True
    except Exception as exc:
        report["error"] = "%s: %s" % (exc.__class__.__name__, exc)
    return report


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError as exc:
        payload = {}
        sys.stderr.write("invalid probe input: %s\n" % exc)

    entries = payload.get("modules") or []
    result = {
        "python_version": "%d.%d.%d" % sys.version_info[:3],
        "modules": [_module_report(entry) for entry in entries],
    }
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
