#!/usr/bin/env python3
"""Remote file discovery script with pattern pre-filtering.

Combines file enumeration (fd/find) with rg pattern matching in a single
SSH call. Only files at depth=1 (directly in the directory) are scanned
since the paths pipeline has already walked subdirectories.

For each discovered file, runs rg pattern matching to provide enrichment
evidence that feeds into LLM scoring. Files with zero pattern matches
are still returned but marked as unenriched — the triage LLM decides
whether to keep them based on path/naming signals.

Requirements:
- Python 3.8+ (stdlib only, no external dependencies)
- Optional: fd (fast-find) for faster file enumeration
- Optional: rg (ripgrep) for pattern matching

Usage:
    echo '{"paths": [...], "extensions": [...], "pattern_categories": {...}}' | python3 discover_files.py

Input (JSON on stdin):
    {
        "paths": ["/path/to/scan", ...],
        "extensions": ["py", "f90", "c", ...],
        "max_depth": 1,
        "max_files_per_path": 500,
        "max_file_size": 1048576,
        "pattern_categories": {
            "mdsplus": "MDSplus|mdsplus|Tree\\(",
            "imas_read": "get_ids|imas\\.open|DBEntry",
            ...
        }
    }

Output (JSON on stdout):
    [
        {
            "path": "/path/to/scan",
            "files": [
                {
                    "path": "/path/to/scan/code.py",
                    "patterns": {"mdsplus": 3, "imas_read": 1},
                    "total_matches": 4,
                    "line_count": 142
                },
                ...
            ],
            "truncated": false,
            "error": null
        },
        ...
    ]
"""

import json
import os
import re
import shlex
import subprocess
import sys

# Default file size limit: 1 MB
DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024

# Directories to skip
_SKIP_DIR_PATTERNS = {
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    ".tox",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".local",
    "site-packages",
    "dist-packages",
    ".venv",
    "venv",
}


def has_command(cmd):
    # type: (str) -> bool
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for d in path_dirs:
        if os.path.isfile(os.path.join(d, cmd)):
            return True
    return False


def sanitize_str(s):
    # type: (str) -> str
    return s.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")


def _should_skip_path(path):
    # type: (str) -> bool
    parts = path.split(os.sep)
    return any(p in _SKIP_DIR_PATTERNS for p in parts)


def _enumerate_files_fd(path, extensions, max_depth, max_files, max_file_size):
    # type: (str, List[str], int, int, int) -> tuple
    ext_args = []
    for ext in extensions:
        ext_args.extend(["-e", ext])

    size_args = []
    if max_file_size > 0:
        if max_file_size >= 1024 * 1024 and max_file_size % (1024 * 1024) == 0:
            size_str = f"-{max_file_size // (1024 * 1024)}m"
        elif max_file_size >= 1024 and max_file_size % 1024 == 0:
            size_str = f"-{max_file_size // 1024}k"
        else:
            size_str = f"-{max_file_size}b"
        size_args = ["--size", size_str]

    cmd = (
        ["fd", ".", "--type", "f", "--max-depth", str(max_depth)]
        + size_args
        + ext_args
        + [path]
    )
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30,
        )
        files = [
            sanitize_str(line.strip())
            for line in result.stdout.strip().splitlines()
            if line.strip() and not _should_skip_path(line.strip())
        ]
        truncated = len(files) > max_files
        if truncated:
            files = files[:max_files]
        return files, truncated
    except (subprocess.TimeoutExpired, Exception):
        return [], False


def _build_find_command(path, extensions, max_depth, max_file_size):
    # type: (str, List[str], int, int) -> str
    ext_predicates = " -o ".join(f'-name "*.{ext}"' for ext in extensions)
    size_filter = f"-size -{max_file_size}c" if max_file_size > 0 else ""
    # The scanned path reaches a shell in this fallback, so quote it: a
    # directory or file name may carry a space or other shell metacharacter. A
    # "#" opens a comment in a shell, and a space splits the path into two.
    return (
        f"find {shlex.quote(path)} -maxdepth {max_depth} -type f "
        f"{size_filter} \\( {ext_predicates} \\) 2>/dev/null"
    )


def _enumerate_files_find(path, extensions, max_depth, max_files, max_file_size):
    # type: (str, List[str], int, int, int) -> tuple
    cmd = _build_find_command(path, extensions, max_depth, max_file_size)
    try:
        result = subprocess.run(
            ["sh", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30,
        )
        files = [
            sanitize_str(line.strip())
            for line in result.stdout.strip().splitlines()
            if line.strip() and not _should_skip_path(line.strip())
        ]
        truncated = len(files) > max_files
        if truncated:
            files = files[:max_files]
        return files, truncated
    except (subprocess.TimeoutExpired, Exception):
        return [], False


def _count_lines(path):
    # type: (str) -> int
    try:
        result = subprocess.run(
            ["wc", "-l", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split()[0])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return 0


def _batch_pattern_counts(files, pattern_categories):
    # type: (List[str], Dict[str, str]) -> Dict[str, Dict[str, int]]
    if not files or not pattern_categories:
        return {}
    compiled = {
        name: re.compile(pattern) for name, pattern in pattern_categories.items()
    }
    combined = "|".join(
        "(?:" + pattern + ")" for pattern in pattern_categories.values()
    )
    try:
        result = subprocess.run(
            ["rg", "--json", "--no-messages", "-e", combined, "--"] + files,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode not in (0, 1):
        return {}
    matches = {}
    for record in result.stdout.splitlines():
        event = json.loads(record)
        if event.get("type") != "match":
            continue
        data = event["data"]
        path = data["path"].get("text", "")
        line = data["lines"].get("text", "")
        counts = matches.setdefault(path, {})
        for name, regex in compiled.items():
            if regex.search(line):
                counts[name] = counts.get(name, 0) + 1
    return matches


def _enrich_file(path, pattern_categories, has_rg, matches=None):
    # type: (str, Dict[str, str], bool) -> Dict[str, Any]
    info = {
        "path": sanitize_str(path),
        "patterns": {},
        "total_matches": 0,
        "line_count": _count_lines(path),
    }
    if has_rg and pattern_categories:
        info["patterns"] = matches or {}
        info["total_matches"] = sum(info["patterns"].values())
    return info


def discover_path(
    path,
    extensions,
    max_depth,
    max_files,
    max_file_size,
    pattern_categories,
    use_fd,
    has_rg,
):
    # type: (str, List[str], int, int, int, Dict[str, str], bool, bool) -> Dict[str, Any]
    if not os.path.isdir(path):
        return {
            "path": sanitize_str(path),
            "files": [],
            "truncated": False,
            "error": "not_a_directory",
        }

    if use_fd:
        files, truncated = _enumerate_files_fd(
            path, extensions, max_depth, max_files, max_file_size
        )
    else:
        files, truncated = _enumerate_files_find(
            path, extensions, max_depth, max_files, max_file_size
        )

    # Enrich each file with pattern matching
    enriched_files = []
    match_counts = _batch_pattern_counts(files, pattern_categories) if has_rg else {}
    for f in files:
        enriched_files.append(
            _enrich_file(f, pattern_categories, has_rg, match_counts.get(f))
        )

    return {
        "path": sanitize_str(path),
        "files": enriched_files,
        "truncated": truncated,
        "error": None,
    }


def main():
    input_data = json.loads(sys.stdin.read())
    paths = input_data.get("paths", [])
    extensions = input_data.get("extensions", ["py"])
    max_depth = input_data.get("max_depth", 1)
    max_files_per_path = input_data.get("max_files_per_path", 500)
    max_file_size = input_data.get("max_file_size", DEFAULT_MAX_FILE_SIZE)
    pattern_categories = input_data.get("pattern_categories", {})

    use_fd = has_command("fd")
    has_rg = has_command("rg")

    results = []
    for path in paths:
        result = discover_path(
            path,
            extensions,
            max_depth,
            max_files_per_path,
            max_file_size,
            pattern_categories,
            use_fd,
            has_rg,
        )
        results.append(result)

    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
