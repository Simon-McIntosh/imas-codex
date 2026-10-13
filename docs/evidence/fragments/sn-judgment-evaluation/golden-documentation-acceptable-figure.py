"""Bar chart of the first failing rubric check over the not-acceptable documentation-golden rows."""

import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("data-ink")
G = "/home/ITER/mcintos/Code/imas-codex/docs/evidence/fragments/sn-judgment-evaluation/golden-documentation-acceptable.json"
OUT = "/home/ITER/mcintos/Code/imas-codex/docs/figures/sn-judgment-evaluation/golden-b-first-failing-check.svg"
rows = json.load(open(G))
c = collections.Counter(r["label_check"] for r in rows if r["label"] == 0)
order = ["house_style", "links", "complete", "subject", "physics", "scope"]
labels = ["house style", "links", "complete", "subject", "physics", "scope"]
DET, SEM = "#4f7cac", "#b5734a"
colors = [DET, DET, DET, SEM, SEM, SEM]
fig, ax = plt.subplots(figsize=(14, 5.2), dpi=100)
y = range(len(order))[::-1]
vals = [c[k] for k in order]
ax.barh(list(y), vals, color=colors, height=0.62)
ax.set_yticks(list(y), labels)
ax.set_xlabel("not-acceptable rows, by first failing check")
for yi, v in zip(y, vals, strict=True):
    ax.text(v + 2, yi, str(v), va="center", fontsize=20)
ax.text(60, 3.0, "deterministic: 207 (81.5%)", color=DET, fontsize=20)
ax.text(60, 1.0, "semantic: 47 (18.5%)", color=SEM, fontsize=20)
ax.set_xlim(0, 175)
fig.tight_layout()
fig.savefig(OUT)
print(OUT)
