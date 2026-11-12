# Brief: shared definition parts rendered from one stored wording (sn-definitions-by-construction §2)

## Reuse map (extend these; do not add a parallel mechanism)
- The one renderer: `imas_codex/standard_names/worker_tasks/worker_generate_docs.py` `_documentation_with_registry_form`, with `_fill_template_sentence` and `_instantiated_template_sentences` doing the substitution. Operator, population and locus all go through it; do not build one table or wrapper per kind of shared part.
- The comparison of a written definition with the stored wording already exists: `imas_codex/standard_names/docs_gates.py` `house_style_findings`, `_template_findings`, `registry_template_texts`, `locus_definition_sentence`, and the `HouseStyleFindingClass` values. A shared part altered after rendering is one more class there, refused where those findings are already acted on.
- Family wording agreement and its cost: `imas_codex/standard_names/harmonize.py` `doc_sig`, `drift`, `group_signature`, `measure_doc_call_cost`. The wording census extends these; the committed census scripts in the evidence directory are the starting readings, not a second owner.
- The stored wordings: `~/Code/imas-standard-names/imas_standard_names/grammar/vocabularies/conventions/{concepts,notation,templates}.yml`. That repository is not in your write scope. A wording this repository needs and the library lacks is recorded here where a reviewer can read it, and listed in your manifest as a proposal for the library.
- The writer's prompt: `imas_codex/llm/prompts/sn/generate_docs_user.md` and its system prompt. The model is asked for what is specific to the name; the shared sentences are not its to write.

## Rulings that bind this node
- Species population (lead, 2026-11-12): a bare species density counts ions, summed over all charge states, with neutral atoms excluded. A name that includes neutral atoms says so in the name. This is the one population statement every species name carries.
- Flux-surface average (lead, earlier ruling): it is defined by its surface form. Only a second, line-form display equation is forbidden. Do not tighten this to forbid restating the canonical equation with the quantity's own symbol.
- The catalog is COCOS 17 throughout, and documentation never names a COCOS.

## Starting evidence
`docs/evidence/fragments/sn-definitions-by-construction/published-operator-census.txt`, `published-family-census.txt` and `published-registry-conformance.txt`, each with its script: 8 forms of the flux-surface average over 32 names, 20 of 20 species families with more than one wording, 548 of 1,975 published names with a conformance finding.

## You may not settle
- Any wording the lead has not ruled where two published wordings contradict in content, not only in phrasing: list the pair in your manifest and render neither.
- Any write to the live graph: this node proves the writer on a disposable graph. The live rewrite is §6.
- Model seats in `pyproject.toml`.

## Spend
The writer's calls run on the configured documentation seat. Keep the node's metered spend under USD 5 and record it; the two-names-per-operator and five-member-family proofs need about fifteen written names.

## Standing rules for this node
- Read your plan section from your worktree; it is the authority for the endpoint. This brief adds the reuse map and the questions you may not settle.
- Export `UV_NO_SYNC=1`, use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent.
- Heavy gates (any test file opening the graph, any pass over the whole catalog) run as their own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-<node>/>`. A bare `srun --partition` from this node hangs. Never run them on the node you start on.
- Do not write the live graph. Use a disposable graph or fixtures for every control.
- Write your manifest with what you have before any long wait. Status is `complete`, `blocked` or `failed`.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
