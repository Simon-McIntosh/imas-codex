"""Fortran include files (.bf, .inc) reach the code scanner.

The code scanner enumerates remote files by extension: it lists only the
extensions that are keys of ``EXTENSION_TO_LANGUAGE``, read through
``_get_extensions_list``.  A Fortran include file whose extension is absent
from that map can never be listed, so it can never get a ``CodeFile`` row and
never reaches the code pipeline.  The JT-60SA EQDBMS field list lives in such
include files (``#AAA.bf`` and its siblings under
``/analysis/src/client_eqdbms.EQ32.Rev1/inc/F/``).

These tests pin the two Fortran include extensions into both extension lists
(the scanner map and the ``file_types.yaml`` Fortran entry), pin the language
they map to, and pin the shell-quoting of a path whose name begins with ``#``
along the listing and fetch command builders that carry it to the remote host.
"""

from __future__ import annotations

import io
import shlex
import tarfile
from unittest.mock import MagicMock, patch

import yaml

from imas_codex.config.discovery_config import DiscoveryConfig
from imas_codex.discovery.code.scanner import _get_extensions_list
from imas_codex.ingestion.readers.remote import (
    EXTENSION_TO_LANGUAGE,
    _fetch_batch_tar,
    _fetch_sequential,
    detect_language,
)
from imas_codex.remote.scripts import scan_files

# A filename that begins with "#", which opens a comment in a shell.
HASH_INCLUDE = "/analysis/src/client_eqdbms.EQ32.Rev1/inc/F/#AAA.bf"


def _config_patterns_dir():
    from imas_codex.config import discovery_config as dc

    return dc.__file__.rsplit("/", 1)[0] + "/patterns"


# ---------------------------------------------------------------------------
# The two extension lists name the Fortran include extensions
# ---------------------------------------------------------------------------


class TestFortranIncludeExtensions:
    def test_scanner_extension_list_has_bf_and_inc(self):
        """The scanner enumerates .bf and .inc, so include files are listed."""
        extensions = _get_extensions_list()
        assert "bf" in extensions, ".bf missing from the scan extension list"
        assert "inc" in extensions, ".inc missing from the scan extension list"

    def test_detect_language_maps_bf_and_inc_to_fortran(self):
        assert detect_language("client_eqdbms/inc/F/#AAA.bf") == "fortran"
        assert detect_language("client_eqdbms/inc/F/#GEO.bf") == "fortran"
        assert detect_language("some/header.inc") == "fortran"
        assert detect_language("some/HEADER.INC") == "fortran"

    def test_extension_map_has_bf_and_inc_as_fortran(self):
        assert EXTENSION_TO_LANGUAGE[".bf"] == "fortran"
        assert EXTENSION_TO_LANGUAGE[".inc"] == "fortran"

    def test_file_types_fortran_entry_has_bf_and_inc(self):
        """The pattern-config Fortran list names the same two extensions."""
        path = _config_patterns_dir() + "/file_types.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        fortran = data["code"]["fortran"]["extensions"]
        assert "bf" in fortran
        assert "inc" in fortran

    def test_both_lists_agree_on_the_include_extensions(self):
        """Both lists name .bf and .inc, so neither alone decides admission."""
        config = DiscoveryConfig.load()
        code_exts = {e.lower() for e in config.file_types.code_extensions}
        scan_exts = set(_get_extensions_list())
        assert {"bf", "inc"} <= code_exts
        assert {"bf", "inc"} <= scan_exts


# ---------------------------------------------------------------------------
# The "#"-named file reaches the remote command shell-quoted
# ---------------------------------------------------------------------------


class TestHashPathQuoting:
    def test_fetch_sequential_quotes_the_remote_path(self):
        with patch("imas_codex.ingestion.readers.remote.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout=b"c field list\n")
            list(_fetch_sequential("jt-60sa", [HASH_INCLUDE]))

        argv = run.call_args[0][0]
        assert argv[:3] == ["ssh", "jt-60sa", "cat"]
        # The whole name, shell-quoted, is what the remote shell receives.
        assert argv[3] == shlex.quote(HASH_INCLUDE)
        assert HASH_INCLUDE in shlex.split(" ".join(argv))[3]

    def test_fetch_batch_tar_quotes_the_remote_path(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz"):
            pass
        with patch("imas_codex.ingestion.readers.remote.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout=buf.getvalue())
            list(_fetch_batch_tar("jt-60sa", [HASH_INCLUDE]))

        argv = run.call_args[0][0]
        assert argv[:2] == ["ssh", "jt-60sa"]
        remote_cmd = argv[2]
        assert shlex.quote(HASH_INCLUDE) in remote_cmd
        # The whole, quoted name is one shell word passed to tar.
        assert HASH_INCLUDE in shlex.split(remote_cmd)

    def test_scan_files_find_quotes_the_scanned_path(self):
        """The find fallback interpolates the path into a shell command."""
        scanned = "/analysis/src/client_eqdbms/inc/F#eqdbms"
        with patch.object(scan_files.subprocess, "run") as run:
            run.return_value = MagicMock(stdout="", returncode=0)
            scan_files.scan_path_find(scanned, ["bf"], 3, 100, 0)

        cmd = run.call_args[0][0][2]
        assert shlex.quote(scanned) in cmd
        # The quoted path is exactly one word when the shell parses it.
        assert scanned in shlex.split(cmd)
