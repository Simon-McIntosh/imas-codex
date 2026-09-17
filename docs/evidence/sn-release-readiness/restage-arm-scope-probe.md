# The review restage arm fires under a drain scope

Verdict: the restage arm does move reviewed at-or-above-threshold rows under a
drain scope, so the scoping mechanism is the one the measurement implicates.

## Counts

Cohort: 30 rows at `reviewed`, score 0.85 or above, quorum shortfall non-null.
Sample: the ten highest-scoring of the thirty.

| arm | scope | moved | left |
|---|---|---|---|
| control | `scope_run_id` | 0 | 10 |
| treatment, one call | `drain_scope_id` | 2 | 8 |
| treatment, repeated | `drain_scope_id` | 9 | 1 |

The control is a live pool, so the control is aimed rather than blind: the
identical call returns non-empty and writes stages as soon as the scope shape is
a drain scope. All ten sample rows were read back at `reviewed` afterwards and
the lease was cleared, so the graph is as it was found. Nothing was spent.

## Where the arm is, and a refusal

The arm is at `graph_ops.py:16020`, armed only when `restage_review_axis` is set, and
`claim_review_name_batch` sets it to `"name"` only when `drain_scope_id` is passed.
The sanctioned operator could not be used at all: `prepare_manifest_drain_scope`
refused all 17 DD paths of the sample as ambiguous, so a drain scope could not be
opened over these rows through the CLI route. That is a finding in its own right.

The node's full record lives in the node's manifest, which was written
before this file: `/run/user/39486/claude-39486/-home-ITER-mcintos-Code-imas-codex/3c425733-c198-454e-9e36-681e003069da/manifests/restage-arm.md`

Reproduction: two scratch probes, named in the manifest. `probe.py` takes the
cohort, runs the control and reaches the scope refusal; `probe3.py` stamps the
lease, repeats the claim and restores the graph. Neither is committed. Spend is
zero: every call here is a claim and no model call was made.