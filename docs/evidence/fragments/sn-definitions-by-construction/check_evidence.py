"""Check the committed evidence against figures read independently of it.

The expected values are in expected_values.json, which was written by a read of
the graph that took its names from the manifest and stated the sourceless count
in a different form from the collector. A failure here means the record and the
independent read disagree; the exit status is non-zero.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", default=str(HERE / "expected_values.json"))
    args = ap.parse_args()
    exp = json.loads(Path(args.expected).read_text())
    bulk = json.loads((HERE / "live-rewrite-bulk.json").read_text())
    man = json.loads((HERE / "live-rewrite-manifest.json").read_text())
    page = (HERE / "sdbc-live-rewrite-bulk.html").read_text()

    problems = []

    def check(label, got, want):
        if got != want:
            problems.append(f"{label}: found {got!r}, expected {want!r}")

    check("candidates", bulk["candidates"], exp["candidates"])
    check("outcomes", bulk["outcomes"], exp["stages"])
    check("manifest outcomes", man["final"]["counts"], exp["stages"])
    check("names by stage total", sum(man["final"]["counts"].values()), exp["candidates"])
    check("settled names", sum(bulk["outcomes"].values()), exp["settled"])
    check("manifest entries", len(man["entries"]), exp["candidates"])
    check(
        "manifest entries with an after state",
        sum(1 for e in man["entries"] if e.get("after")),
        exp["candidates"],
    )
    check("sourceless now", bulk["sourceless"]["now"], exp["sourceless"])
    check("sourceless names", bulk["sourceless"]["names"], exp["sourceless_names"])
    check("waiting names", len(bulk["waiting_equivalent_electron"]), 56)
    check("reviews", len(bulk["reviews"]), exp["stages"]["reviewed"])
    check("holds", len(bulk["holds"]), exp["stages"]["exhausted"])
    check("candidate rows", len(bulk["before_after"]), exp["candidates"])

    spend = bulk["spend"]
    class_sum = round(sum(spend["per_class_usd"].values()), 4)
    if abs(class_sum - spend["node_total_usd"]) > 0.01:
        problems.append(
            f"class rows sum to {class_sum}, node total {spend['node_total_usd']}"
        )
    if spend["node_total_usd"] > exp["ceiling_usd"]:
        problems.append(f"node spend {spend['node_total_usd']} above the ceiling")

    for marker in (
        "/imas-codex/figures/sn-definitions-by-construction/rewrite-outcomes-by-class.svg",
        "2,609 was a mis-measurement",
        "Waiting on the grammar word, 56 names",
    ):
        if marker not in page:
            problems.append(f"page is missing {marker!r}")
    if not page.rstrip().endswith("</body></html>"):
        problems.append("page does not close")

    for p in problems:
        print("FAIL", p)
    print("checks failed:", len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())