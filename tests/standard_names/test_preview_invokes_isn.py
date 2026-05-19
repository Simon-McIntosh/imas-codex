"""Tests that ``run_preview`` shells out to ISN's catalog-site serve command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from imas_codex.standard_names.preview import PreviewHandle, run_preview


@pytest.fixture()
def staging_dir(tmp_path: Path) -> Path:
    """Create a minimal staging directory shaped like ``sn export`` output."""
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "catalog.yml").write_text(
        yaml.safe_dump({"catalog_name": "test"}), encoding="utf-8"
    )
    sn_dir = staging / "standard_names" / "test_domain"
    sn_dir.mkdir(parents=True)
    (sn_dir / "test_name.yml").write_text(
        yaml.safe_dump({"name": "test_name"}), encoding="utf-8"
    )
    return staging


class TestRunPreview:
    """``run_preview`` delegates to ISN's ``standard-names serve`` subcommand."""

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_invokes_isn_serve_command(
        self, mock_popen: MagicMock, staging_dir: Path
    ) -> None:
        mock_popen.return_value = MagicMock()

        run_preview(staging_dir)

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        # The subprocess must invoke ``standard-names serve`` from the
        # active venv against the staging dir's standard_names/ catalog.
        assert cmd[0].endswith("standard-names"), cmd
        assert cmd[1] == "serve"
        assert cmd[2] == str(staging_dir / "standard_names")
        assert "--port" in cmd
        assert "8000" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_custom_port(self, mock_popen: MagicMock, staging_dir: Path) -> None:
        mock_popen.return_value = MagicMock()

        handle = run_preview(staging_dir, port=9090)

        cmd = mock_popen.call_args[0][0]
        assert "--port" in cmd
        assert "9090" in cmd
        assert handle.url == "http://127.0.0.1:9090"

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_custom_host(self, mock_popen: MagicMock, staging_dir: Path) -> None:
        mock_popen.return_value = MagicMock()

        handle = run_preview(staging_dir, host="0.0.0.0")

        cmd = mock_popen.call_args[0][0]
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert handle.url == "http://0.0.0.0:8000"

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_default_port_url(self, mock_popen: MagicMock, staging_dir: Path) -> None:
        mock_popen.return_value = MagicMock()

        handle = run_preview(staging_dir)
        assert handle.url == "http://127.0.0.1:8000"

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_returns_handle(self, mock_popen: MagicMock, staging_dir: Path) -> None:
        mock_popen.return_value = MagicMock()

        handle = run_preview(staging_dir)
        assert isinstance(handle, PreviewHandle)
        assert handle.process is not None
        assert handle.staging_dir == str(staging_dir)

    def test_missing_staging_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run_preview(tmp_path / "nonexistent")

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        staging = tmp_path / "no_manifest"
        staging.mkdir()
        with pytest.raises(FileNotFoundError, match="catalog.yml"):
            run_preview(staging)

    def test_missing_standard_names_dir_raises(self, tmp_path: Path) -> None:
        staging = tmp_path / "no_sn_dir"
        staging.mkdir()
        (staging / "catalog.yml").write_text(
            yaml.safe_dump({"catalog_name": "test"}), encoding="utf-8"
        )
        with pytest.raises(FileNotFoundError, match="standard_names"):
            run_preview(staging)

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_stop_terminates_process(
        self, mock_popen: MagicMock, staging_dir: Path
    ) -> None:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        handle = run_preview(staging_dir)
        handle.stop()

        mock_process.terminate.assert_called_once()

    @patch(
        "imas_codex.standard_names.preview.subprocess.Popen",
        side_effect=FileNotFoundError("standard-names not found"),
    )
    def test_missing_cli_raises_importerror(
        self, mock_popen: MagicMock, staging_dir: Path
    ) -> None:
        with pytest.raises(ImportError, match="imas-standard-names"):
            run_preview(staging_dir)
