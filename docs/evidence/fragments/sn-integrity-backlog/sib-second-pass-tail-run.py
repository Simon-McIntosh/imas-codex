#!/usr/bin/env python
"""Record the model-build prerequisite before a source-cohort retry.

This recorder runs only the prerequisite. It never resets sources or starts
the paid pipeline. Give each invocation a fresh log path to preserve failures.
"""

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    root = args.checkout.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    command = ["uv", "run", "--no-sync", "build-models", "--force"]
    environment = dict(
        os.environ, UV_PROJECT_ENVIRONMENT=str(root / ".venv"), PYTHONPATH=str(root)
    )
    with args.log.open("x") as log:
        log.write(
            f"# revision: {revision}; tree: {root}; command: {shlex.join(command)}\n"
        )
        log.flush()
        result = subprocess.run(
            command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        log.write(f"EXIT={result.returncode}\n")
    print(f"Model-build receipt: {args.log}; EXIT={result.returncode}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
