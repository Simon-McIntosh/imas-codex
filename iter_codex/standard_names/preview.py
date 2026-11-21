"""Preview a staging directory via the ISN catalog-site server.

ISN (≥ rc20) ships a Vite SPA and exposes ``standard-names serve`` to
run it locally with hot-reload. ``run_preview`` is a thin wrapper that
shells out to that subcommand against the staging directory's
``standard_names/`` catalog. The previous MkDocs-based path
(``MKDOCS_SERVE_TEMPLATE``, ``_generate_site_content``) was removed in
ISN's Vite migration and no longer exists.

Press Ctrl-C to stop the preview server.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PreviewHandle:
    """Handle to a running preview server process."""

    process: subprocess.Popen | None
    url: str | None
    staging_dir: str
    temp_dir: str | None = None

    def stop(self) -> None:
        """Stop the preview server and clean up temporary files."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


def run_preview(
    staging_dir: str | Path,
    *,
    port: int | None = None,
    host: str | None = None,
) -> PreviewHandle:
    """Launch a local preview of a staging directory.

    Delegates to ISN's ``standard-names serve`` command, which builds
    the Vite SPA dataset and runs the dev server with hot-reload.

    Parameters
    ----------
    staging_dir:
        Path to the staging directory produced by ``sn export``.
        Must contain ``catalog.yml`` and ``standard_names/``.
    port:
        Port number for the dev server (default: 8000).
    host:
        Host to bind to (default: ``127.0.0.1``). Pass ``0.0.0.0``
        to allow remote access (e.g., over an SSH tunnel).

    Returns
    -------
    PreviewHandle with the subprocess, URL, and temp directory.

    Raises
    ------
    FileNotFoundError
        If the staging directory or ``catalog.yml`` is missing.
    ImportError
        If the ``standard-names`` CLI is not available on PATH.
    """
    staging = Path(staging_dir)
    if not staging.is_dir():
        raise FileNotFoundError(f"Staging directory not found: {staging}")

    catalog_yml = staging / "catalog.yml"
    if not catalog_yml.is_file():
        raise FileNotFoundError(f"No catalog.yml in staging directory: {staging}")

    sn_dir = staging / "standard_names"
    if not sn_dir.is_dir():
        raise FileNotFoundError(f"No standard_names/ directory in staging: {staging}")

    effective_port = port or 8000
    effective_host = host or "127.0.0.1"
    url = f"http://{effective_host}:{effective_port}"

    # Resolve the ``standard-names`` console script from the active
    # interpreter's bin dir (``sys.executable`` is ``<venv>/bin/python``)
    # so we use the same install codex is running under, regardless of
    # PATH ordering. ISN's ``serve`` subcommand takes the catalog
    # directory as a positional argument plus ``--port`` / ``--host``
    # options matching our knobs.
    cli = Path(sys.executable).parent / "standard-names"
    cmd = [
        str(cli),
        "serve",
        str(sn_dir),
        "--port",
        str(effective_port),
        "--host",
        effective_host,
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(staging),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ImportError(
            "imas-standard-names is required for preview. "
            "Install with: uv add imas-standard-names"
        ) from exc

    logger.info("Preview server starting at %s (catalog: %s)", url, sn_dir)

    return PreviewHandle(
        process=process,
        url=url,
        staging_dir=str(staging),
        temp_dir=None,
    )
