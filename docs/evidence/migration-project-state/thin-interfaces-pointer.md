<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="migration-project-state">
<meta name="plan-status" content="active">
<meta name="plan-title" content="Migration Project State">
<meta name="plan-evidence-for" content="migration-project-state">

# Thin interfaces to shared machinery — where the survey lives

Plan `imas-codex:migration-project-state` §3. The survey itself is
[the shared-infrastructure coupling record](../imas-codex-replan/shared-infrastructure-coupling.md),
landed on `main` at merge `79419367c`, with its figure at
`docs/figures/sn-catalog-audit-instrument/shared-infrastructure-coupling.svg`.
It is left at that path rather than moved here because three committed documents
already reference it; a pointer costs nothing and a move would break them.

## What it found, so this section is not empty without opening it

**11 couplings across all five required surfaces**, each with a `file:line` site,
the failure it risks, and a one-sentence refactor. Two carry a verifiable
artifact today; the other nine are latent.

| Surface | Findings | The one that already shows damage |
|---|---|---|
| reckon crew and plan layer | 2 | `catalog_release.py` mints `minted_from` as an **absolute reckon-worktree path**, and **twelve committed manifests already carry it** — a provenance field naming a directory that is routinely deleted |
| local lane and router | 2 | the SN pipeline's lane choice and retry loop **bypass the router admission gate**, so the guard that protects the lane under load is not in the path |
| fleet and SLURM placement | 3 | `srun`/`sbatch`/`squeue`/`scancel` issued outside the ledger in `cli/compute.py` and `cli/services.py` |
| imas-python data access | 2 | `ids/assembler.py:439-440` opens `imas.DBEntry` with **no pinned DD version** |
| GPFS paths and state files | 2 | `graph/neo4j_ops.py` hand-rolls a lock with a 5 s alarm around `fcntl.lockf` on GPFS |

**No `h5py` access to IMAS data was found**, which is the one negative result
worth stating rather than omitting.

## Scope, stated so it is not over-read

The node surveys and does not repair: **none of the 11 refactors are landed.**
Couplings expressed only in YAML, shell templates or docs tooling were not
swept, so the count is a floor. Its gate log was written to `/tmp`, which is
node-local and will not survive the fleet move — the finding is in the committed
record, the log is not.
