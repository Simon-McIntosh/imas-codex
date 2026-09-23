"""Fresh-process import contracts for the dependency-light pool registry."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


# Each case starts a fresh interpreter and imports the CLI, so the cost is the
# process rather than the assertion. Measured on a debug partition: 6.98 s for
# the slowest case run alone, but 27.89 s for that same case when the whole
# standard-names surface runs beside it — which is the condition a gate uses,
# and 93% of the suite's 30 s default. A contended login node stretches it far
# further; one such run was still going after thirteen minutes. The ceiling is
# sized for the surface-run figure with room above it, not for the solo one.
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
