"""The benchmark banner must name the fixture the extractor actually reads.

Two committed fixtures back ``sn bench``: the curated reference dataset, and
the physics hard-case set selected by ``--physics``. The banner and the
extractor take the same branch, so a banner that reports one fixture's size
while the extractor reads the other names a population the run never used.

The test drives the banner through the CLI, keeps the config the banner built,
and then hands that same config to the real extractor, which records the paths
it is given. The two figures therefore describe one run rather than two.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.benchmark_reference import REFERENCE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]

# The fixture that backs each mode, named independently of the code under
# test: the fixture is the file, and the file's contents are the paths.
FIXTURES: dict[bool, tuple[Path, list[str]]] = {
    False: (
        REPO_ROOT / "imas_codex" / "standard_names" / "benchmark_reference.py",
        list(REFERENCE_NAMES),
    ),
    True: (
        REPO_ROOT / "research" / "physics_bench_paths.json",
        [
            entry["path"]
            for entry in json.loads(
                (REPO_ROOT / "research" / "physics_bench_paths.json").read_text()
            )
        ],
    ),
}


def _paths_the_extractor_reads(config: object) -> list[str]:
    """Run the real extractor on this config, recording the paths it is handed."""
    from imas_codex.standard_names.benchmark import _extract_candidates

    read: list[list[str]] = []

    def _recording_extractor(*args: object, **kwargs: object) -> list[MagicMock]:
        explicit = kwargs.get("explicit_paths") or (args[0] if args else [])
        read.append(list(explicit))
        batch = MagicMock()
        # One item per path handed over, so nothing is reported missing and
        # extraction runs to completion without reaching the Data Dictionary.
        batch.items = [{"path": path} for path in explicit]
        return [batch]

    with patch(
        "imas_codex.standard_names.sources.dd.extract_dd_candidates",
        new=_recording_extractor,
    ):
        _extract_candidates(config)

    assert read, "the extractor was never reached"
    return read[-1]


def _drive_banner(*, physics: bool) -> tuple[str, object]:
    """Drive the banner in one mode and return its output and built config."""
    report = MagicMock()
    report.provenance.codex_version = None
    report.provenance.codex_commit = None
    report.dataset_hash = None
    report.extraction_count = 0
    report.extraction_source_ids = []

    argv = ["bench", "--models", "unit/fixture-probe", "--max-candidates", "1000"]
    if physics:
        argv.append("--physics")

    with (
        patch(
            "imas_codex.standard_names.benchmark.run_benchmark",
            new=AsyncMock(return_value=report),
        ) as run_benchmark,
        patch("imas_codex.standard_names.benchmark.render_comparison_table"),
    ):
        invocation = CliRunner().invoke(sn, argv)

    assert invocation.exit_code == 0, invocation.output
    config = run_benchmark.call_args.args[0]
    return invocation.output, config


def test_banner_names_the_fixture_the_extractor_reads_in_each_mode() -> None:
    for physics, (fixture_path, fixture_paths) in FIXTURES.items():
        mode = "physics" if physics else "default"
        output, config = _drive_banner(physics=physics)
        read_by_extractor = _paths_the_extractor_reads(config)

        # The extractor read the whole fixture, so the figures below describe
        # one population rather than two.
        assert len(read_by_extractor) == len(fixture_paths), (
            f"{mode}: extractor read {len(read_by_extractor)} paths, "
            f"{fixture_path} holds {len(fixture_paths)}"
        )
        assert set(read_by_extractor) == set(fixture_paths), (
            f"{mode}: the extractor's paths are not {fixture_path}'s"
        )

        banner_line = next(
            (line for line in output.splitlines() if "Reference paths:" in line),
            None,
        )
        assert banner_line, output
        match = re.search(r"Reference paths:\s*(\d+)/(\d+)", banner_line)
        assert match, banner_line
        shown, total = int(match.group(1)), int(match.group(2))

        label = str(fixture_path.relative_to(REPO_ROOT))
        assert total == len(fixture_paths), (
            f"{mode}: banner total {total} is not {fixture_path} "
            f"({len(fixture_paths)} paths) that the extractor read"
        )
        assert shown == total, (
            f"{mode}: banner shows {shown}/{total} with the cap lifted to 1000"
        )
        assert label in banner_line, (
            f"{mode}: banner line {banner_line!r} does not name {label}"
        )
