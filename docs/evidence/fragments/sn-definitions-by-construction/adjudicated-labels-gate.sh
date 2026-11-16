#!/bin/bash
# Gate for the adjudicated label set. Run from anywhere:
#   bash docs/evidence/fragments/sn-definitions-by-construction/adjudicated-labels-gate.sh
# Prints every command, records its status, and closes with EXIT=<n>, which is
# 0 only when every step behaved. The deliberately broken row must be refused.
set -u
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
SCRIPT=docs/evidence/fragments/sn-definitions-by-construction/adjudicated-labels.py
SET_JSON=docs/evidence/fragments/sn-name-definition-graph-reading/reader-verdicts-adjudicated.json
PAGE=docs/evidence/fragments/sn-definitions-by-construction/sdbc-adjudicated-labels.html
export UV_NO_SYNC=1
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv
export PYTHONPATH="$ROOT"
T=$(mktemp -d "${TMPDIR:-/tmp}/adjudicated-labels-gate-XXXXXX") || exit 2
trap 'rm -rf "$T"' EXIT
rc=0

run() {
  echo "+ $*"
  "$@"
  local status=$?
  echo "  status=$status"
  [ $status -ne 0 ] && rc=1
}

echo "# command: bash docs/evidence/fragments/sn-definitions-by-construction/adjudicated-labels-gate.sh"
echo "worktree $ROOT at $(git rev-parse HEAD)"
echo "== lint and format"
run uv run --no-sync ruff check --no-cache "$SCRIPT"
run uv run --no-sync ruff format --check "$SCRIPT"

echo "== check the committed set against the rule and its inputs"
run uv run --no-sync python "$SCRIPT" check

echo "== check must refuse a deliberately broken row"
uv run --no-sync python - "$SET_JSON" "$T/broken.json" <<'PY'
import json
import sys

document = json.loads(open(sys.argv[1]).read())
row = next(row for row in document["rows"] if row["dispute"] == "readers split")
row["adjudicated_verdict"] = "flagged" if row["adjudicated_verdict"] == "stands" else "stands"
open(sys.argv[2], "w").write(json.dumps(document, indent=1) + "\n")
print("broke", row["name"])
PY
uv run --no-sync python "$SCRIPT" check --document "$T/broken.json"
status=$?
echo "  status=$status (1 is the expected refusal)"
[ $status -ne 1 ] && rc=1

echo "== rebuild from a clean checkout of the committed head"
mkdir -p "$T/head"
git archive HEAD | tar -x -C "$T/head"
(
  cd "$T/head" || exit 2
  # Sequential commands in one subshell report only the last one's status, so a
  # failed build followed by a report from the committed JSON would close EXIT=0.
  PYTHONPATH="$T/head" uv run --no-sync python "$SCRIPT" build || exit $?
  PYTHONPATH="$T/head" uv run --no-sync python "$SCRIPT" report
)
status=$?
echo "  rebuild status=$status"
[ $status -ne 0 ] && rc=1

echo "== the rebuilt artifacts must match the committed ones byte for byte"
for rel in "$SET_JSON" "$PAGE"; do
  if cmp -s "$T/head/$rel" "$ROOT/$rel"; then
    echo "  identical $rel"
  else
    echo "  DIFFERS $rel"
    rc=1
  fi
done

echo "EXIT=$rc"
exit $rc