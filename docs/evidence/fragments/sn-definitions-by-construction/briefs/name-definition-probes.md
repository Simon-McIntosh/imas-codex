# Brief: the name-against-definition check as narrow Jev probes compared in code (sn-definitions-by-construction §5)

Your plan section states the endpoint and both bars. This brief states the design the lead asked for, the reuse map, the reference labels and what you may not settle. Your worktree starts from the revision that holds the two earlier instruments; build on it.

## Why the earlier instruments fell short
- The Jev check `definition_qualifier_absent_from_name` in `imas_codex/llm/prompts/sn/review_checks.yaml` asks one seven-way question that makes the model read two texts, compare them and classify the difference in one judgment. It answered `agrees` for 31 of 47 held-out mismatches. Its misses are narrow: a device part, one layer, an energy channel, a spectral band, an equivalent-electron count, a weighting.
- The word rules (`imas_codex/standard_names/audits.py`) find only wording they were written for: 14 of 47 held out.
- The chat seat (`imas_codex/standard_names/review/name_definition_seat.py`) reads well but stops at 41 of 55; adding the word rules reaches 46 of 55 and breaks the false-flag ceiling.

## The design: extract, then compare
A judgment model is strongest on one narrow question about one text. So Jev never compares. Code does.

1. **Code reads the name.** From the grammar parse of the name (segments already carried in the review state: physical base, transformation, process, object, position, region, component, subject) build the set of qualifiers the name states, as a closed vocabulary of slots: aggregation (local, flux-surface average, volume average, line average, line integral, volume or surface integral, time-accumulated, extremum), normalisation, weighting, counting unit, restriction (device, device part or layer or channel, species, charge state, population such as trapped or co-passing, reaction branch, spectral band or energy channel), locus, reference (relative to what), value kind (scalar, vector, component, complex or angular), and one-quantity versus parent. A token-to-slot table is the word-rule part; keep it as data.
2. **Jev reads the definition alone.** One probe per slot, each a closed question about the definition text only, with the name withheld or reduced to "the quantity": for example "Over what is the quantity aggregated?" with the aggregation options and `none stated`; "Is the quantity divided by or expressed relative to a scale? Which?"; "What is counted: particles, equivalent electrons, atoms, something else, not a count?"; "Is the quantity restricted to one part, layer, channel, band or branch? Quote it." Each probe returns its option probabilities and, where the question type allows, the phrase it rests on. Use the question kinds the Jev layer already has; add none.
3. **Code compares slot by slot, in both directions.** A slot the definition asserts above its threshold and the name does not state is a candidate "definition says more". A slot the name states and the definition contradicts or omits is a candidate "name says more". Each candidate carries the slot, the definition phrase and the name tokens.
4. **Jev confirms each candidate.** One closed question per candidate: is the added phrase already the standard meaning of the name's own quantity, or a catalog convention, or a real extra qualifier? The conventions are data, not prose in the prompt: a bare species name counts ions summed over charge states with neutrals excluded; the catalog is COCOS 17; a temperature is a mean kinetic energy; a flux is per unit area; add others only from recorded lead rulings on the plans. This step is what removes false flags such as `plasma_temperature` and `vertical_electron_velocity`.
5. **A flag is a typed finding**: slot, direction, the definition phrase, the name tokens, and the disposition class it implies (rename, rewrite the definition, hold, stands with reason). It joins the named review checks; it adds no gate or command.

Thresholds per slot are set on the tuning split only. The chat seat stays in the tree as a comparison arm; whether it remains wired is decided by the readings, and you report them side by side.

## Prototype before you build
Before writing the full probe set, run three probes (aggregation, restriction, counting unit) over the 14 names the seat missed and the 15 it wrongly flagged, both listed in `docs/evidence/fragments/sn-definitions-by-construction/name-definition-seat-results.json`, and write the per-name probe answers to the evidence directory. If a probe cannot separate those names, change the question, not the threshold, and say what you changed. This table is a deliverable.

## Reference labels and the split (never produce your own)
- `docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json`: 55 both-flag, 258 both-pass, 68 disputed. Only the 313 agreed names score. Read it from `/home/ITER/mcintos/Code/imas-codex` if your worktree lacks it.
- Tuning split: the agreed names that were in the earlier development arm (see `split` in `name-definition-check-results.json`). Held-out: every other agreed name. Probe wording, the token table, the convention list and thresholds may be tuned on the tuning split and on the prototype names above; report the prototype names apart from the held-out score, since they are now seen.
- Bars, unchanged, on the held-out names: at least 80 % of both-flag names found, at most 10 % of both-pass names flagged, with counts. Report the same two figures over all 313 beside them, and per slot.

## You may not settle
- Whether a label is right. List every held-out disagreement with the slot and phrase so the lead can read the pattern.
- The bars. If they do not hold, report `blocked` with the per-slot table; do not weaken a bar or move a name between splits.
- A new convention. If a false flag needs a convention no lead ruling records, list it as a question.

## If the bars hold
Run the check over every published name read from the live graph read-only, list findings by slot with one disposition each, and record judgments paid and cost. Ceiling for this node USD 15 in all.

## Done when
- The prototype table is committed.
- The probe check is one of the named review checks, with a test that plants a normalised quantity under an unnormalised name and is shown failing with the comparison removed, and a test that a convention suppresses its flag, shown failing with the convention removed.
- Held-out and all-313 readings, per slot, for the probe check, the chat seat and their union, are committed under `docs/evidence/fragments/sn-definitions-by-construction/`, with cost per name.
- The gate passes on `all_debug` with a final `EXIT=<n>` line in a committed log.

## Standing rules for this node
- Export `UV_NO_SYNC=1`; use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent.
- Any pass over more than a few dozen names, and any test file that opens the graph, runs as its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-name-definition-probes/>`. A bare `srun --partition` from the node you start on hangs.
- Read the live graph read-only. Never write it.
- Write the manifest with what you have before any long wait, and evidence files as results arrive. Status is `complete`, `blocked` or `failed`, nothing else.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
