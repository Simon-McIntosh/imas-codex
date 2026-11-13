# Brief: the name-against-definition check as open narrowing phrases, each put to Jev as one closed question

Your worktree starts from the revision that holds the chat seat (`imas_codex/standard_names/review/name_definition_seat.py`), the slot probes (`name_definition_probes.py`) and their scoring harness (`docs/evidence/fragments/sn-definitions-by-construction/name-definition-probes-run.py`). Build on them; add no second harness.

## What three rounds measured
On 208 held-out names (36 mismatched, 172 agreeing): chat seat 27/36 found at 12/172 flagged; slot probes 15/36 at 13/172; union 28/36 at 23/172. The probes failed at candidate generation: a closed slot vocabulary cannot name what definitions add. The restriction, normalisation and weighting probes found 0 of 36. Eight mismatches were found by no arm: an "effective" form, the real part of a complex quantity, a named sub-population, one module's surface. They are listed in `name-definition-probes-results.json` under `missed_by_every_arm`; read them first.

## The design to build
1. **Open extraction, definition only.** One chat call per name reads the description and documentation with the name withheld ("the quantity") and returns every phrase that narrows, normalises, aggregates, weights, restricts, re-types or re-references the quantity, each quoted verbatim from the text, with no fixed vocabulary. Ask for more rather than fewer; extraction recall is what the earlier design lacked. Also run this extraction on Jev's own question kinds where one fits, as a second arm, and report both.
2. **One closed question per phrase, to Jev.** For each phrase: given the name's words, the recorded conventions and this phrase, does the name already state or imply it? Options: stated by the name; implied by the standard meaning of the name's own quantity or by a recorded convention; narrower, different or additional. Return the option probabilities. Conventions are data, as in the probes module (a bare species counts ions over all charge states, neutrals excluded; COCOS 17; a temperature is a mean kinetic energy; a flux is per unit area; a density is per unit volume); add one only from a lead ruling recorded on the plans, and list any other you would need as a question.
3. **Code decides.** A name is flagged when any phrase's "narrower, different or additional" probability exceeds a threshold. The finding carries the phrase, the name's words and the disposition class it implies. It joins the named review checks; it adds no gate or command.
4. **Reverse direction.** A qualifier the name states that the definition never supports is the same question with the roles swapped; include it if it finds development mismatches inside the ceiling, and report it apart.

## Tuning and scoring
- Development: all 313 agreed names in `docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json`. They have been scored three times and are development data now. Tune prompts, the convention list and thresholds here only, and report the development reading per arm.
- Scoring: the fresh names in `docs/evidence/fragments/sn-name-definition-fresh-labels/` (`sheet-part-*.json`; two readers' `labels-a-part-*.json` and `labels-b-part-*.json`), read from `/home/ITER/mcintos/Code/imas-codex` at main. They may not exist when you start. Do not open the label files until your thresholds are frozen and committed; state the freezing commit in the evidence. Only names where both readers agree score. Score once.
- If the labels have not landed when everything else is done, write the manifest as `blocked` with wait condition "fresh labels", having committed the frozen check and the development readings. You will be resumed.
- Bars on the fresh agreed names: at least 80 % of mismatched names found, at most 10 % of agreeing names flagged, with counts. Score the chat seat and the union on the same names beside it. You may not weaken a bar, relabel a name or settle whether a label is right; list every disagreement with its phrase.

## Done when
- The check and a test that plants a definition restricted to one device part under an unrestricted name, shown failing with the phrase question removed, and a test that a convention suppresses its flag, shown failing with the convention removed.
- Development readings and, once labels exist, fresh readings for the phrase check, the chat seat and their union, committed under `docs/evidence/fragments/sn-definitions-by-construction/` with cost per name split between chat calls and Jev judgments.
- The gate passes on `all_debug` with a final `EXIT=<n>` line in a committed log.

## Standing rules
- Export `UV_NO_SYNC=1`; use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent.
- Any pass over more than a few dozen names, and any test file that opens the graph, runs as its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-name-definition-phrases/>`. A bare `srun --partition` from the node you start on hangs.
- Never write the live graph. Spend ceiling for this node USD 12.
- Write the manifest before any long wait and evidence files as results arrive. Status is `complete`, `blocked` or `failed`.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
