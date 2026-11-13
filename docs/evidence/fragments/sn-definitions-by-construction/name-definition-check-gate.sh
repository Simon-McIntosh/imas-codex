#!/bin/bash
# Gate the name-versus-definition checks on a compute partition, and run the
# negative control that removes the seat's hold so the refusal test can be seen
# to fail. The seat module is restored from a backup whatever the exit path.
set -u
W=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$W" || exit 90
export TMPDIR=/tmp
export UV_NO_SYNC=1
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv
export PYTHONPATH="$W"

SEAT=imas_codex/standard_names/review/name_definition_seat.py
TESTS="tests/standard_names/test_name_definition_agreement.py tests/standard_names/test_review_checks.py"

echo "# revision: $(git rev-parse HEAD)"
echo "# command: uv run --no-sync pytest $TESTS -p no:cacheprovider"

uv run --no-sync pytest $TESTS -p no:cacheprovider
gate=$?

echo "== negative control: the seat raises no hold =="
cp "$SEAT" "$SEAT.bak.$$"
trap 'mv -f "$SEAT.bak.$$" "$SEAT"' EXIT
python3 - "$SEAT" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
marker = '    outcome = verdict.get("verdict")'
assert text.count(marker) == 1, "control marker not unique"
open(path, "w").write(text.replace(marker, '    return None\n' + marker, 1))
PY
uv run --no-sync pytest tests/standard_names/test_name_definition_agreement.py -p no:cacheprovider \
  -k "persisted_as_a_review_hold"
control=$?
echo "control_exit=$control"

if [ "$gate" -eq 0 ] && [ "$control" -ne 0 ]; then
  echo "EXIT=0"
  exit 0
fi
echo "EXIT=1"
exit 1