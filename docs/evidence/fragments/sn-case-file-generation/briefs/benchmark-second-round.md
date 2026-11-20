# Brief: second benchmark round, adjudicated blind by the coordinator

## Authority

Plan `sn-case-file-generation`, section 3 ("A benchmark chooses the structure and
the model"). Lead rulings of 2026-11-20, recorded on that section: the three
changes named by the failure read are accepted, the coordinating session
adjudicates the generated records, and every row-level record is written inside
the repository. The round's spend ceiling is USD 30, inside the lead's repair
ceiling of USD 400.

## What the first round got wrong

Read `docs/evidence/fragments/sn-case-file-generation/benchmark-failure-read.html`
and its `.json` before anything else. In short:

- The judge model passed 9 of 50 records where a blind read passed 29. It scored
  the physics group, which its own rule does not mention and the case file does
  not state.
- The prompt showed the incumbent but never asked the generator to keep a sound
  one, and the output had no way to say "keep". 37 of 50 incumbents already
  carried a right name and all were retyped.
- The withheld-enclosing-text control could not separate: that text is a median
  1.5 % of the case file and absent for 20 of 87 rows.
- The row-level records lived only in the run directory and were lost.

## Deliverable

Extend `docs/evidence/fragments/sn-case-file-generation/benchmark.py` (reuse its
split, seed, arms machinery, cost accounting and reference loading; do not write
a second harness) to run a second round and stop at a blind adjudication packet.
You do not score the held-out records. No judge model is called anywhere.

### Arms

Each arm writes one record per member from the same case files as round one.

1. `keep-or-repair/frontier`: the incumbent is shown, and the instruction says
   to keep it unless the case file contradicts or under-identifies it. The
   output schema gains a `decision` of `keep`, `repair` or `blocked`, and a
   `reason` naming the words of the case file that forced a repair. A `keep`
   returns the incumbent's name and definition unchanged.
2. `keep-or-repair/local`: the same on the local model.
3. `one-pass/frontier`: round one's best arm, unchanged, as the comparison.
4. `incumbent`: returns the incumbent unchanged (control).
5. `keep-or-repair/frontier` on a **perturbed case file** (control): for each
   member, the source description and enclosing-structure text are replaced by
   those of a different member drawn by the seeded generator from another
   physics group. This control must produce a different record from arm 1 on
   most members. Report the share that changed. If fewer than half change, say
   so plainly: the case file is then not what drives the answer.
6. The Kimi model: not run unless its endpoint answers. Report it as not run
   with the exact command that will run it; adding it later rescores nothing.

Use the frontier and local generation models round one used. State both model
identifiers in the evidence.

### Iteration on the development half

On the 89 development members only, run arms 1 and 2, then make at most two
prompt or case-file changes, each stated with the failures it answers and the
before and after figures. For this iteration only, you may use a mechanical
proxy: the share of records whose name is in the reference's right-name list and
whose unit agrees. Label it as a proxy everywhere it appears. It decides nothing
about the held-out half.

Address in this iteration the two generator failures the failure read counted:
a detail the case file states being dropped (8 of 50), and blocking on a name
the generator could derive (7 of 50).

### Held-out half: records and the blind packet

Run every arm once on the 87 held-out members with the final prompt. Then write,
all inside the repository under
`docs/evidence/fragments/sn-case-file-generation/`:

- `benchmark-second-round-records.jsonl`: one line per (member, arm) with the
  full generated record, the decision and reason, tokens, cost and seconds.
- `benchmark-second-round-packet.jsonl`: the adjudication packet. One line per
  **distinct** (member, name, definition, unit) record, so arms that returned the
  same record share one line. Each line carries a packet id, the member's case
  file source text (the data-dictionary description and enclosing-structure
  text), the record's name, definition, documentation and unit, and nothing that
  reveals the arm, the decision, the incumbent or the reference. Order is
  shuffled by the seeded generator.
- `benchmark-second-round-key.json`: the mapping from packet id to the (member,
  arm) pairs it stands for, and for each member the reference's right names and
  definition points. The adjudicator does not open this file until its verdicts
  are committed. State that rule at the top of the file.
- `benchmark-second-round.json` and `.html`: the round's evidence so far: arms,
  models, prompt versions and hashes, the development iteration with its proxy
  figures, the perturbed control's changed share, the count of distinct packet
  lines, per-arm keep, repair and blocked counts, spend beside the USD 30
  ceiling, and a clear statement that held-out scores are pending adjudication.

The adjudicator will return two verdicts per packet line, one for the name and
one for the definition, each right, wrong or undecidable, and will accept any
name that denotes the same quantity. Group is not scored. Add a `score`
subcommand to `benchmark.py` that reads a verdicts file of that shape plus the
key and writes the per-arm figures the plan's done-when asks for: share right
with a 95 % interval for name, for definition and for both; share blocked by
type; share where the arm changed an incumbent that was right; member-by-member
comparison between arms; cost and time per name. Prove `score` on a synthetic
verdicts file in a test or in `--check`, not on real verdicts.

## Measure

- `python docs/evidence/fragments/sn-case-file-generation/benchmark.py --check`
  passes on the committed files, and fails when one line of the records file is
  altered, when a packet line reveals an arm, and when the key and packet
  disagree on a packet id.
- The packet has no field naming an arm, a decision, the incumbent or a
  reference answer. Show the field list in the manifest.
- Spend is stated beside the USD 30 ceiling. Stop and write the manifest if the
  next call would exceed it.
- No stored standard-name text and no graph property changes. Reads of the graph
  are bounded and are not needed if the archived case files serve.

## Constraints

- Heavy work does not run on the fleet node. The generation calls are network
  bound and may run where you are; any suite or bulk compute goes to
  `all_debug` with `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch`.
- Use the shared environment: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv
  PYTHONPATH="$PWD" uv run --no-sync …`. Never `pip install`.
- Commit locally with a conventional subject and a body. Do not push, merge,
  rebase or stash. Stage explicit paths.
- Write the records file as you go, line by line, and commit it before building
  the packet, so a stopped run loses nothing already paid for.
- Write the manifest with what you have before any long wait. Recognised status
  words are `complete`, `blocked` and `failed`.
- No plan labels, version tags or ticket numbers in file names, symbols or
  comments.
