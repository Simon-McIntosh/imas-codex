# Brief: a review check that a name and its definition say the same thing (sn-definitions-by-construction §5)

## Reuse map (extend these; do not add a parallel gate)
- The named review checks a name already passes through: `imas_codex/llm/prompts/sn/review_checks.yaml`, exercised by `tests/standard_names/test_review_checks.py`. The new check is one more entry there and its refusal is theirs.
- Name-versus-unit consistency: `imas_codex/standard_names/audits.py` `name_unit_consistency_check` and `unit_dimension_check`. A unit whose dimension does not fit the quantity the name states is a deterministic finding from that owner, not a judgment.
- Abstract parents: the pipeline already knows which names are derived parents (see `imas_codex/core/AGENTS.md`, derived parents and `HAS_PARENT`); read that state, do not re-infer it from prose.

## The labels, and how they stay held out
- `docs/evidence/fragments/sn-definitions-by-construction/name-definition-labels.json`: 381 published names, each with the class under the lead's species ruling (use that column: a bare species name counts ions, so an ion-only definition agrees), and the pattern that produced each mismatch. 71 are mismatched.
- `docs/evidence/fragments/sn-name-definition-second-reading/labels.json`: a second reader's labels on 60 of them. Treat a name as mismatched for scoring only by the first reader's ruling column; report the check's verdict on the 15 names the readers disagree on separately.
- Split once, by a committed seed, into a development third and a held-out two thirds, stratified by class. Shape the check only on the development third. Report recall and false-flag rate on the held-out part, with counts. Do not look at held-out rows' labels until the check is frozen, and say so in the evidence.
- The labels were produced by a reader, not by this check or its model. State who produced them in the evidence.

## Patterns the first reading found (the classes the check must separate)
Equivalent-electron counting under a plain species name (13), an unstated weighting (9), a definition restricted to one device object (8), an unstated normalisation (4), abstract parents published as quantities (19).

## You may not settle
- A flagged name's disposition where the choice is between renaming and rewriting the definition and the physics is not obvious: list it with both options for the lead. The section asks for one disposition per flagged name; give one where it is clear and mark the rest for the lead.
- Any write to the live graph. Running the check over every published name reads the published catalog files at `/home/ITER/mcintos/Code/imas-standard-names-catalog/standard_names`.

## Spend
The check's judgments over 1,975 names run on the configured review or judgment seat; keep under USD 10, measure the first hundred and stop if the rate projects past it.

## Standing rules for this node
- Read your plan section from your worktree; it is the authority for the endpoint. This brief adds the reuse map and the questions you may not settle.
- Export `UV_NO_SYNC=1`, use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent, then run `build-models --force` in your worktree if the schema differs.
- Heavy gates (any test file opening the graph, any pass over the whole catalog) run as their own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-<node>/>`. A bare `srun --partition` from this node hangs. Never run them on the node you start on.
- Do not write the live graph. Use the disposable_neo4j fixtures in `tests/standard_names/conftest.py` for every control.
- Write your manifest with what you have before any long wait. Status is `complete`, `blocked` or `failed`.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
