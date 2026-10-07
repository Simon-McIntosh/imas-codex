"""The remote directory enumerator must quote the path it scans.

``_enumerate_files_find`` in
``imas_codex/remote/scripts/discover_files.py`` builds a ``find`` command as a
``str`` and hands it to ``sh -c``. A scanned path carrying a space, a ``#`` or
another shell metacharacter is silently mangled unless the path is quoted.
"""

import shlex
from unittest.mock import MagicMock, patch

from imas_codex.remote.scripts import discover_files

# A path with both a space and a "#": the space splits the command into two
# shell words and the "#" opens a comment.
SUSPICIOUS_PATH = "/analysis/a dir#frag"


class TestFindCommandQuoting:
    def test_builder_quotes_a_path_with_space_and_hash(self):
        cmd = discover_files._build_find_command(SUSPICIOUS_PATH, ["f90"], 1, 1024)

        assert shlex.quote(SUSPICIOUS_PATH) in cmd
        # The quoted path is exactly one word when the shell parses it.
        assert SUSPICIOUS_PATH in shlex.split(cmd)

    def test_enumerate_find_hands_the_quoted_path_to_the_shell(self):
        with patch.object(discover_files.subprocess, "run") as run:
            run.return_value = MagicMock(stdout="", returncode=0)
            discover_files._enumerate_files_find(SUSPICIOUS_PATH, ["f90"], 1, 100, 1024)

        argv = run.call_args[0][0]
        assert argv[:2] == ["sh", "-c"]
        assert SUSPICIOUS_PATH in shlex.split(argv[2])

    def test_enumerate_find_discovers_files_in_a_path_with_space_and_hash(
        self, tmp_path
    ):
        target = tmp_path / "a dir#frag"
        target.mkdir()
        (target / "mgset0.f90").write_text("program x\nend program x\n")

        files, truncated = discover_files._enumerate_files_find(
            str(target), ["f90"], 1, 100, 1024 * 1024
        )

        assert truncated is False
        assert [f.rsplit("/", 1)[-1] for f in files] == ["mgset0.f90"]
