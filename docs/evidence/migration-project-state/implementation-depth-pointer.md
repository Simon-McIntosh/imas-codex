# §1 · Implementation depth — where the measurement lives, and its headline

The full classification is **`docs/evidence/imas-codex-replan/feature-depth.md`**,
committed at `bef439dbd` with a figure at
`docs/figures/sn-catalog-audit-instrument/review-context-keys.svg`. It is not under
this directory because three committed documents already reference that location, and
moving a file to satisfy a directory convention while breaking live references is the
worse trade. This pointer exists so §1 is not a dead end.

## The headline inverts the impression the plan history gives

`sn-catalog-audit-instrument` had **13 nodes land with 13 independent reviews**, all
merged. Anyone reading that history would put it near done. Measured:

| | |
|---|---|
| features classified | **25** |
| deep | **14** |
| shallow | **11** |
| impl set | **0.56** |

**Deep** means a real non-test call site reaches it. The rejected alternative —
weighting the partials — gives **0.72**, which is defensible and **not
reconstructible without knowing the weights**. The reproducible number won, which is
the right call: a figure a reader cannot rebuild is the thing a census exists to find.

## The 11 shallow features span three states

- **3 named and never shipped** — a manifest-scoped release-batch selector; scoring a
  description against the bindings its identity holds; refusing rather than scoring
  when a channel did not load.
- **2 instruments living only in `tests/`** — the declared-attribute and
  defaulted-attribute static checks. They run. No production path reaches them.
- **6 built but not connected.**

## The sharpest one, and why the criterion had to be a call site

`f-scai-audit-findings-reach-no-template`: Layer 1 audit findings are extracted
correctly, set on the reviewer's render context, and **read by no prompt template**.
Built every review, discarded every review.

Established by a render probe carrying its own control — two other markers render in
both templates, the audit marker in neither — and corroborated by nothing under
`imas_codex/llm/` naming the key.

**The existing test passes, and that is the point.** It asserts on the *extractor*,
so it is green on a value nothing consumes. Its done-when therefore requires
asserting on a **rendered prompt**. A test proving a thing is built is not a test
proving it is used, which is why depth had to be measured at the call site rather
than at the test.

## What a successor should take from this

Do not trust a plan's landing history as a proxy for its depth. Thirteen reviewed,
merged nodes produced a surface that is 44 % unreachable. The same question is open
for every other plan in the repository: the census covered this one because it is the
flagship, not because it is the worst.
