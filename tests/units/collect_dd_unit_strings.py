"""Collect every distinct unit string the Data Dictionary carries.

Writes one string per line to ``dd_unit_strings.txt`` beside this script. The
list feeds the normalisation regression guard: every string that resolved at the
base revision must still resolve to the same canonical symbol. Run against a
live graph; each query is a bounded DISTINCT read over unit strings, not a scan.
"""

from pathlib import Path

from imas_codex.graph.client import GraphClient

OUT = Path(__file__).with_name("dd_unit_strings.txt")

QUERIES = (
    "MATCH (n:IMASNode) WHERE n.units IS NOT NULL AND trim(n.units) <> '' "
    "RETURN DISTINCT n.units AS u",
    "MATCH (f:FacilitySignal) WHERE f.unit IS NOT NULL AND trim(f.unit) <> '' "
    "RETURN DISTINCT f.unit AS u",
    "MATCH (x:Unit) WHERE x.symbol IS NOT NULL AND trim(x.symbol) <> '' "
    "RETURN DISTINCT x.symbol AS u",
)


def main() -> None:
    gc = GraphClient()
    strings: set[str] = set()
    for q in QUERIES:
        for row in gc.query(q):
            u = row["u"]
            if u and u.strip():
                strings.add(u)
    ordered = sorted(strings)
    OUT.write_text("\n".join(ordered) + "\n")
    print(f"wrote {len(ordered)} strings to {OUT}")


if __name__ == "__main__":
    main()
