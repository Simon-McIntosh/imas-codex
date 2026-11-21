"""Capture how an ``sn run`` was invoked, for post-hoc run diagnosis.

An ``SNRun`` row records what a run *did* — spend, counts, stop reason — but
not what it was *asked* to do.  A row that stopped on ``no_eligible_work`` or
``budget_saturated`` is therefore ambiguous: the scope may have been empty, or
the scope flags may have pointed the run away from work that was sitting
eligible the whole time.  Persisting the command line together with the
resolved knobs and the resolved scope removes that ambiguity without needing
the operator's shell history.

Command lines reach this module verbatim, so a credential passed as an
argument would otherwise be written to the graph.  :func:`redact_argv`
strips those before anything is stored; environment variables are never
read here, so secrets held the normal way cannot leak through this path.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from typing import Any

REDACTED = "<redacted>"

# Option names whose value is a credential.  Matched case-insensitively
# against the option word alone, so ``--api-key``, ``--openrouter-token``
# and ``--neo4j-password`` are all covered.
_SECRET_OPTION = re.compile(
    r"^--?[\w-]*(key|token|secret|password|passwd)[\w-]*$", re.I
)

# Bare values shaped like a provider credential.  Vendor keys are long,
# prefixed, and contain no path or whitespace characters, which keeps this
# clear of ordinary arguments such as domain names or file paths.
_SECRET_VALUE = re.compile(r"^(sk|pk|rk|ghp|gho|xoxb|hf)[-_][A-Za-z0-9_-]{16,}$")


def redact_argv(argv: list[str]) -> list[str]:
    """Return *argv* with credential-bearing arguments replaced.

    Handles both spellings a credential can take on a command line: the
    ``--api-key VALUE`` pair, where the following token is the secret, and
    ``--api-key=VALUE``, where it is the tail of the same token.
    """
    out: list[str] = []
    consume_next = False
    for token in argv:
        if consume_next:
            out.append(REDACTED)
            consume_next = False
            continue
        option, sep, _value = token.partition("=")
        if _SECRET_OPTION.match(option):
            if sep:
                out.append(f"{option}={REDACTED}")
            else:
                out.append(token)
                consume_next = True
            continue
        if _SECRET_VALUE.match(token):
            out.append(REDACTED)
            continue
        out.append(token)
    return out


def _json_scalar(value: Any) -> Any:
    """Coerce *value* to something ``json.dumps`` accepts, losslessly if it can."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple | set):
        return [_json_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_scalar(v) for k, v in value.items()}
    return str(value)


def _encode(values: dict[str, Any]) -> str:
    """Serialize a flag or scope mapping, dropping keys that carry nothing.

    Empty collections and ``None`` are omitted so the stored object shows the
    knobs a run actually set rather than the full option surface.
    """
    populated = {
        key: _json_scalar(value)
        for key, value in values.items()
        if value is not None and value != () and value != [] and value != {}
    }
    return json.dumps(populated, sort_keys=True, separators=(",", ":"))


def capture_run_invocation(
    *,
    flags: dict[str, Any],
    scope: dict[str, Any],
    argv: list[str] | None = None,
) -> dict[str, str]:
    """Return the three ``SNRun`` invocation properties for this process.

    *flags* carries the resolved run knobs and *scope* the resolved work
    restriction; both are stored as JSON so a later reader sees the values
    the orchestrator ran with rather than the defaults of whichever release
    they happen to be reading the row from.
    """
    source_argv = list(sys.argv if argv is None else argv)
    return {
        "invocation": shlex.join(redact_argv(source_argv)),
        "invocation_flags": _encode(flags),
        "invocation_scope": _encode(scope),
    }


__all__ = ["REDACTED", "capture_run_invocation", "redact_argv"]
