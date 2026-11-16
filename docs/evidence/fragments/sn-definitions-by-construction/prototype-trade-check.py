"""Check the prototype page's trade prose against the generator that writes it.

The page states how each arm stands against the adopted question. That sentence
is derived from the scored table by
``name-check-prototype.py:trade_sentence``, so this check imports the generator,
rebuilds the page's inputs from the committed scores and asserts the delivered
page carries the sentence the generator emits — for the settled labels and for
the adjudicated subset. The generator is imported rather than re-implemented:
its ``report`` phase needs the question run, which is gone, while the sentence
function is a pure function of the scores.

Usage:
    python prototype-trade-check.py [--page PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORES = HERE / "name-check-prototype.json"
PAGE = HERE / "sdbc-name-check-prototype.html"
GENERATOR = HERE / "name-check-prototype.py"


def generator():
    spec = importlib.util.spec_from_file_location("name_check_prototype", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules["name_check_prototype"] = module
    spec.loader.exec_module(module)
    return module


def scored(table: dict) -> dict:
    """The row shape the sentence reads: recall and specificity, not precision."""
    return {
        arm: {"recall": row.get("recall"), "specificity": row.get("precision")}
        for arm, row in table.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, default=PAGE)
    args = parser.parse_args()

    module = generator()
    payload = json.loads(SCORES.read_text())
    contenders = [
        arm
        for arm in module.ARM_ORDER
        if not arm.startswith("always-") and arm not in ("E-operators", "E-units")
    ]
    page = args.page.read_text()
    failures = []
    for label, key in (("settled", "arms"), ("adjudicated", "arms_adjudicated")):
        sentence = module.trade_sentence(scored(payload[key]), contenders)
        if not sentence:
            failures.append(f"{label}: the sentence is empty")
        elif sentence not in page:
            failures.append(f"{label}: the page does not carry the derived sentence")
        print(f"{label}: {sentence}")
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"checked 2 sentences, mismatched {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
