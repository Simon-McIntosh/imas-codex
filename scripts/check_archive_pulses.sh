#!/usr/bin/env bash
# Run the archive pulse checks with scratch space owned by this invocation.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export TMPDIR=${TMPDIR:-/tmp}
gate_dir=$(mktemp -d "$TMPDIR/archive-pulse-gate.XXXXXX")
trap 'status=$?; rm -rvf "$gate_dir"; printf "EXIT=%d\n" "$status"' EXIT
export TMPDIR=$gate_dir
cd "$repo"
revision=$(git rev-parse HEAD 2>/dev/null || printf unversioned)
printf '# revision: %s; tree: %s; command: bash %q\n' "$revision" "$repo" "$repo/scripts/check_archive_pulses.sh"
printf '# command: bash %q\n' "$repo/scripts/check_archive_pulses.sh"
command=(uv run --no-sync pytest -p no:cacheprovider -m ''
    "$repo/tests/discovery/test_archive_endpoint_writes.py"
    "$repo/tests/remote/test_pulse_catalogue.py"
    "$repo/tests/graph/test_archive_pulses.py"
    "$repo/tests/graph/test_archive_catalogue.py"
    "$repo/tests/graph/test_variable_name_graph.py"
    --basetemp="$gate_dir/pytest")
printf '# pytest_command: '; printf '%q ' "${command[@]}"; echo
"${command[@]}"
