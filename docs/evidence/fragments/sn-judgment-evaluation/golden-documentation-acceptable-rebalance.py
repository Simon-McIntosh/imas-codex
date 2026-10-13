"""Rebalance the documentation-acceptability golden: swap acceptable reserve rows in for matched sheet negatives; print the plan, or write the golden with --write."""

import collections
import json
import sys

SHEET = "/home/ITER/mcintos/Code/imas-codex/docs/evidence/fragments/sn-judgment-evaluation/golden-documentation-acceptable-sheet.json"
OUT = "/home/ITER/mcintos/Code/imas-codex/docs/evidence/fragments/sn-judgment-evaluation/golden-documentation-acceptable.json"
LABELS = "/home/ITER/mcintos/Code/imas-codex/docs/evidence/fragments/sn-judgment-evaluation/golden-documentation-acceptable-labels.jsonl"
rows = {int(r["draw_rank"]): r for r in json.load(open(SHEET))}
lab = {}
for line in open(LABELS):
    d = json.loads(line)
    assert rows[d["draw_rank"]]["unit"] == d["name"], d
    lab[d["draw_rank"]] = d
sheet = [k for k in rows if k <= 400]
assert all(k in lab for k in sheet)
adds = sorted(k for k in lab if k > 400 and lab[k]["acceptable"])
removed, swaps = set(), []
for k in adds:
    r = rows[k]
    pool = [
        s
        for s in sheet
        if not lab[s]["acceptable"] and not rows[s]["lead_overlap"] and s not in removed
    ]
    for key, how in (
        ("pattern_family", "family"),
        ("physics_domain", "domain"),
        (None, "any"),
    ):
        cand = [s for s in pool if key is None or rows[s][key] == r[key]]
        if cand:
            out, matched = max(cand), how
            break
    removed.add(out)
    swaps.append((k, out, matched))
final = sorted([s for s in sheet if s not in removed] + adds)
pos = sum(lab[k]["acceptable"] for k in final)
print(
    f"swaps {len(swaps)}  final {len(final)}  acceptable {pos} ({pos / len(final):.3f})  overlap {sum(rows[k]['lead_overlap'] for k in final)}"
)
print("match", collections.Counter(h for _, _, h in swaps))
print(
    "splits",
    collections.Counter(rows[k]["split"] for k in final),
    "acceptable by split",
    collections.Counter(rows[k]["split"] for k in final if lab[k]["acceptable"]),
)
for k, o, h in swaps[:40]:
    print(f"  +{k} {rows[k]['unit'][:48]:48} -{o} {rows[o]['unit'][:40]:40} {h}")
if "--write" in sys.argv:
    swap_for = {k: o for k, o, _ in swaps}
    out_rows = []
    for k in final:
        r = dict(rows[k])
        r["label"] = 1 if lab[k]["acceptable"] else 0
        r["label_check"] = lab[k]["check"]
        r["label_note"] = lab[k]["note"]
        if k in swap_for:
            r["replaces_draw_rank"] = swap_for[k]
        out_rows.append(r)
    enc = ",\n".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in out_rows
    )
    open(OUT, "w").write("[\n" + enc + "\n]\n")
    print("wrote", OUT, len(out_rows))
