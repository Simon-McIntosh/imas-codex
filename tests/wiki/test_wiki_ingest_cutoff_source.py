"""Check that wiki ingestion reads its configured cutoff at import time."""

import os
import subprocess
import sys


def test_wiki_ingest_cutoff_uses_environment_in_fresh_interpreter():
    environment = os.environ.copy()
    environment["IMAS_CODEX_WIKI_INGEST_THRESHOLD"] = "0.37"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from imas_codex.discovery.wiki.graph_ops import "
            "CONTENT_INGEST_THRESHOLD; print(CONTENT_INGEST_THRESHOLD)",
        ],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
        timeout=30,
    )
    assert result.stdout.strip() == "0.37"
