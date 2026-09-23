"""Fresh-process import contracts for the dependency-light pool registry."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


# Each case starts a fresh interpreter and imports the CLI, so the cost is the
# process rather than the assertion: 6.98 s for the slowest case run alone on a
# debug partition, the highest floor on this surface.
#
# Under a full-surface run the figure does not replicate. Three samples of this
# same case gave 7.83 s, 7.53 s and 27.89 s, and across those runs a different
# test absorbed the stretch each time while whole-run wall time tracked it
# (317 s, 378 s, 494 s). So contention is a property of the run rather than of
# any test, and no per-test ceiling can anticipate where the squeeze lands — a
# contended login node put this case past thirteen minutes, which is why heavy
# runs belong on a partition rather than behind a bigger number here.
#
# The ceiling is therefore cheap insurance for the case with the highest floor,
# not a claim that this is the test at risk. The samples do not support that
# claim and it should not be repeated from this comment.
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "imports",
    [
        "import imas_codex.standard_names.turn as t; print(t.TURN_PHASES)",
        (
            "import imas_codex.cli.sn; "
            "import imas_codex.standard_names.turn as t; "
            "print(t.TURN_PHASES)"
        ),
    ],
)
def test_pool_phase_import_orders_succeed_in_fresh_process(imports: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    current_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(repo_root), current_path) if value
    )

    result = subprocess.run(
        [sys.executable, "-c", imports],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "review_name" in result.stdout
    assert "refine_name" in result.stdout


def test_operational_module_reexports_registry_objects() -> None:
    from imas_codex.standard_names import pool_registry, pools

    assert pools.POOL_NAMES is pool_registry.POOL_NAMES
    assert pools.POOL_WEIGHTS is pool_registry.POOL_WEIGHTS
    assert pools.POOL_NAMES == tuple(pools.POOL_WEIGHTS)
