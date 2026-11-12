# Brief: close the gaps the review found in the rendered shared wording (sn-definitions-by-construction §2)

The mechanism is on main: settled shared parts are given to the model before it writes, and a contradicting definition is refused before storage (`imas_codex/standard_names/docs_gates.py`, `worker_tasks/worker_generate_docs.py`, `worker_tasks/worker_refine_docs.py`). The review of the landed head found four gaps. This node repairs them and adds nothing else.

## The four gaps
1. **No terminal state.** In the refine worker a raised altered-wording refusal is caught by the generic handler, not by the refine-failure path, so the name is released back to review and a model that keeps contradicting the settled population is re-claimed and re-charged every pass. Give it the bounded-attempt terminal state the refine path already has for its own failures; do not add a second counter. The generate path must be bounded the same way.
2. **A reported finding that holds nothing.** The altered-wording class is listed in the conformance note but is absent from the set of classes that hold a name at review, so only the pre-storage refusal acts on it. Either it holds a name at review like its sibling classes or the report does not present it as a hold; say which and why in the evidence.
3. **Half the refine repair is untested.** The refine test plants documentation that contradicts the population whatever the description says, so removing the description repair leaves it green. Add a test that fails when the one-line description stops carrying the population.
4. **A clause that never runs.** One branch of the stored-sentence comparison in `docs_gates.py` is unreachable because an earlier return already handled it. Remove it or make it reachable, with a test either way.

## Measure
- For gap 1: a test that feeds the same contradicting answer on every attempt and reads that the name reaches a terminal state within the existing attempt bound, with the number of model calls it cost; shown failing at the base revision.
- Each new test is shown failing with its repair removed.
- The section's two-arm control is not re-run here.

## Standing rules for this node
- Read your plan section from your worktree; it is the authority for the endpoint. This brief adds the reuse map and the questions you may not settle.
- Export `UV_NO_SYNC=1`, use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent.
- Heavy gates (any test file opening the graph) run as their own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-<node>/>`. A bare `srun --partition` from this node hangs.
- Do not write the live graph. No metered model calls are needed.
- Write your manifest with what you have before any long wait. Status is `complete`, `blocked` or `failed`.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
