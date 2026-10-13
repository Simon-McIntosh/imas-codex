#!/usr/bin/env python
"""Positive control for the comparator-rerank batch acceptance measure.

The measure is a paired held-out batch run with
``IMAS_CODEX_SN_COMPOSE_RERANK_COMPARATORS`` off (arm A) and on (arm B). Both
arms must feed the same harness compose path, and the switch must be shown to
change the compose prompt that the harness actually renders — otherwise the
two arms are the same run and any acceptance difference is noise.

This driver renders the harness compose prompt exactly as
``imas_codex.standard_names.benchmark._run_model`` builds it (the
``sn/generate_name_dd`` template with that function's ``user_context``), under
both switch settings, and compares the rendered-message hashes. It makes no
paid call.

It also renders the same template once with a populated
``nearby_existing_names`` block, which is the block the switch reranks in
production (``imas_codex/standard_names/workers.py:4032-4033``). That render is
the instrument check: the template must render the block, so a differing hash
there shows the comparator block is injective in the rendered prompt (touching
it MUST change the prompt). If arm A and arm B hash identically while the
populated render differs, the harness's compose prompt carries no comparator
block and the switch cannot move it.

Usage:  uv run python ssog-rerank-batch-acceptance-run.py [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SWITCH_ENV = "IMAS_CODEX_SN_COMPOSE_RERANK_COMPARATORS"


def _item() -> dict:
    """One representative held-out DD item in the extraction item shape."""
    return {
        "id": "equilibrium/time_slice/profiles_1d/electrons/temperature",
        "source_id": "equilibrium/time_slice/profiles_1d/electrons/temperature",
        "path": "equilibrium/time_slice/profiles_1d/electrons/temperature",
        "description": (
            "Electron temperature on the equilibrium flux-surface grid, "
            "defined at each flux surface."
        ),
        "unit": "eV",
        "physics_domain": "equilibrium",
    }


def _nearby_block() -> list[dict]:
    """A populated nearby-name comparator block, as ``workers`` supplies it."""
    return [
        {
            "id": "ion_temperature",
            "description": "Ion temperature on the flux-surface grid.",
            "kind": "scalar",
            "unit": "eV",
        },
        {
            "id": "electron_density",
            "description": "Electron number density on the flux-surface grid.",
            "kind": "scalar",
            "unit": "m^-3",
        },
        {
            "id": "electron_temperature_at_pedestal_top",
            "description": "Electron temperature at the pedestal top.",
            "kind": "scalar",
            "unit": "eV",
        },
    ]


def _harness_messages(
    context: dict, items: list[dict], nearby: list | None
) -> list[dict]:
    """Build the compose messages exactly as ``benchmark._run_model`` does.

    Mirrors ``imas_codex/standard_names/benchmark.py`` lines 1613-1642: the
    harness sets ``ids_name``, ``existing_names`` and ``cluster_context`` and
    nothing else. ``nearby`` is passed only in the instrument check, which
    supplies the block the harness omits, to prove the template renders it.
    """
    from imas_codex.llm.prompt_loader import render_prompt

    user_context = {
        "items": items,
        "ids_name": "equilibrium",
        "existing_names": [],
        "cluster_context": "",
    }
    if nearby is not None:
        user_context["nearby_existing_names"] = nearby
    user_prompt = render_prompt("sn/generate_name_dd", {**context, **user_context})
    system_prompt = render_prompt(
        "sn/generate_name_system", {**context, "compose_scored_examples": []}
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _hash(messages: list[dict]) -> str:
    from imas_codex.standard_names.benchmark import _rendered_messages_hash

    return _rendered_messages_hash(messages)


def _render_under(switch: str | None, context: dict, nearby: list | None) -> dict:
    if switch is None:
        os.environ.pop(SWITCH_ENV, None)
    else:
        os.environ[SWITCH_ENV] = switch
    from imas_codex.standard_names.context import comparator_rerank_enabled

    msgs = _harness_messages(context, [_item()], nearby)
    return {
        "switch_env": switch,
        "rerank_enabled": comparator_rerank_enabled(),
        "hash": _hash(msgs),
        "user_prompt_chars": len(msgs[1]["content"]),
        "has_nearby_section": "Nearby Existing Standard Names" in msgs[1]["content"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from imas_codex.standard_names.context import build_compose_context

    context = build_compose_context()

    arm_off = _render_under(None, context, nearby=None)
    arm_on = _render_under("1", context, nearby=None)
    instrument = _render_under(None, context, nearby=_nearby_block())

    switch_changes_harness_prompt = arm_off["hash"] != arm_on["hash"]
    instrument_is_injective = instrument["hash"] != arm_off["hash"]
    instrument_renders_block = instrument["has_nearby_section"]

    result = {
        "switch_env": SWITCH_ENV,
        "arm_off": arm_off,
        "arm_on": arm_on,
        "instrument_check_populated_nearby": instrument,
        "switch_changes_harness_compose_prompt": switch_changes_harness_prompt,
        "comparator_block_is_injective_in_template": instrument_is_injective,
        "template_renders_nearby_block_when_supplied": instrument_renders_block,
        "positive_control_passed": switch_changes_harness_prompt,
    }

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
