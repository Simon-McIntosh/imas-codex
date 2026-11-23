# Brief: one coherent physics-domain list, reviewed against the IDS sources

The lead ruled on 2026-11-23 that the domain vocabulary is reviewed whole before
the grammar library changes. This node is that review. Its output is a proposal
for the lead. Nothing is written to the vocabulary, the graph or the grammar
library.

## Why now
Section 6 of `sn-audited-group-release` divides `transport` into five parts:
`transport_coefficients`, `profile_quantities`, `distribution_quantities`,
`summary_quantities` and `source_quantities`. The write half refuses every one
of them because none is a value of `PhysicsDomain`
(`docs/evidence/fragments/sn-audited-group-release/sagr-transport-parts-write.html`).
The three sources of domain truth already disagree. A few examples read while
briefing this node:
- the grammar library lists `core_sources` and `core_transport` under
  `core_plasma_physics`, while codex's `ids_domains.json` maps them to
  `transport`;
- the grammar library puts `edge_transport` and `transport_solver_numerics`
  under `transport`, while codex puts them in `edge_plasma_physics` and
  `computational_workflow`;
- 1042 live names carry `transport`, which is 47.4 % of what the earlier census
  measured as settled (`domain-groups.json`).

## Sources to read (all three, plus the live graph)
- The grammar library's vocabulary, with its per-domain IDS lists:
  `/home/ITER/mcintos/Code/imas-standard-names/imas_standard_names/grammar/vocabularies/physics_domains.yml`
  (32 domains). Read only.
- codex's own domain definitions: `iter_codex/definitions/physics/domains.yaml`
  (scope, includes, excludes, precedence),
  `iter_codex/definitions/physics/ids_domains.json` (87 IDSs, one domain each),
  `iter_codex/definitions/physics/path_domain_overrides.json` and
  `domain_gold_set.json`. The test that holds them together is
  `tests/graph/test_physics_domain_guidance.py`.
- How a DD path gets its domain: `iter_codex/graph/dd_domain_classifier.py` and
  `iter_codex/core/physics_domain.py`.
- The live graph: the domain on each live `StandardName`, the domain on each DD
  path its sources bind, and the IDS each source path belongs to.

## Reuse
- `iter_codex/standard_names/domain_audit.py` `audit_domains` already measures
  each name's domain against its sources' domains. The earlier run that used it
  is `docs/evidence/fragments/sn-audited-group-release/domain-groups.py`, with
  its data in `domain-groups.json`. Extend that reading; do not write new
  queries for what it already returns.
- `data-structure-groups.json` and `sagr-transport-parts-write.json` already
  hold the transport placement per name. Read the five parts from them.

## What the page must show
1. Every current domain, from the union of the grammar library and codex: its
   description, its IDS list in each source, codex's guidance entry, the live
   count of standard names carrying it, and the live count of DD paths carrying
   it.
2. Every IDS in the DD version the graph holds, with the domain each of the
   three sources gives it and the distribution of domains its live names
   actually carry. List every disagreement.
3. A proposed list. Each domain is kept, added, merged, split, renamed or
   retired, with a one-line reason. Each proposed domain has a description, a
   category, an IDS list, and scope, includes and excludes in the form
   `domains.yaml` uses.
   - Say for each proposed domain whether it names an area of physics or a
     grouping of data structures. The grammar library's vocabulary is published
     as a physics classification.
   - Judge the five transport parts inside the list. They may be kept as
     domains or placed some other way, but each needs a stated reason.
4. A mapping from current domains to proposed ones. Applied to the live names,
   it must give a count for every proposed domain, and the counts must sum to
   the live total, with no name left without a domain.
5. One figure if it carries what a table cannot, for example a matrix of IDS
   against domain showing where the sources disagree.

The final list is the lead's to rule. Propose it; do not settle it.

## Done when
- `docs/evidence/fragments/sn-audited-group-release/physics-domain-review.py
  --check` reads every count from the live graph and the three sources in one
  run. It asserts:
  - every IDS maps to a proposed domain;
  - every live name maps to a proposed domain;
  - the counts sum to the live total.
  It exits 0 when all three hold.
- A planted negative control makes the same `--check` exit non-zero: drop one
  IDS from the proposed mapping in a copy. The log of that run is kept.
- The `.json` data file and the `.html` page are committed beside the script.
  The page's numbers come from the data file.
- Nothing outside the declared write paths changes:
  - no graph write;
  - no edit to any vocabulary or definition file;
  - no edit to the grammar library;
  - no plan edit.

## Working conditions
- This node reads the live graph, so it runs under the implement role. Every
  graph read is bounded by a named label and property. Any whole-catalog pass
  runs as one job:
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a log under ~/.cache/imas-codex-live-rewrite/>`.
  A bare `srun --partition` hangs where you start.
- Copy, never symlink, the five gitignored generated files from the main
  checkout with `cp -n`, and never stage them:
  - `iter_codex/graph/models.py`
  - `iter_codex/graph/dd_models.py`
  - `iter_codex/config/models.py`
  - `iter_codex/graph/schema_context_data.py`
  - `agents/schema-reference.md`
- Set `UV_NO_SYNC=1`, and run Python as
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Manifest rules:
  - lists are comma-separated, and commits are bare hashes;
  - status is `complete`, `blocked` or `failed`, nothing else;
  - commit the proposal before you write `complete`.
- Never kill by pattern.
- Commit locally with a conventional subject and a body, and stage explicit
  paths only.
- A figure is saved through the repository's style module. Look at its render
  before committing.
