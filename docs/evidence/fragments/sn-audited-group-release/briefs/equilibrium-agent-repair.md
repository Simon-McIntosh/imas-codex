# Agent repair of the equilibrium names

## Why
Plan `sn-audited-group-release` section 4 (read it in your worktree; it is the authority for the endpoint). The blind adjudication judged 208 equilibrium names: 82 sound, 93 defective, 33 it could not decide; two more await a fresh adjudicator. The lead ruled on 2026-11-17 that every defective or undecidable member is repaired now by an agent's proposal through the signed edit path, and that the release bar is zero known defects.

## Inputs
- `docs/evidence/fragments/sn-audited-group-release/equilibrium-adjudication.json` and `.py` on main: each member's frozen source packet, its verdict, its classes, the adjudicator's reason with the source words it rests on, and the 14 shared causes with their members. `equilibrium-pending-packets.json`: the two members not yet judged; leave them alone.
- The live graph and the `sn edit` signed path. Read the `sn-edit` skill and `imas_codex/standard_names/AGENTS.md` before the first edit: redesign against hint mode, the mandatory reason, the dry run before the apply, and following an edit through review to acceptance.
- The catalog's rulings: a bare species density counts ions over all charge states and excludes neutrals; the catalog is COCOS 17 throughout and a definition never names a COCOS, so derive signs yourself; the grammar's controlled vocabulary is closed, so a qualifier the vocabulary lacks is reported, not invented.

## Deliverable
`equilibrium-repair.py`, `equilibrium-repair.json`, `equilibrium-repair.html` under that directory.

1. Work the 126 members in this order: the three largest shared causes first (scope qualifier lost, 18; sources that cannot share one definition, 13; boundary forced to the last closed flux surface, 11), then the remaining causes, then the members with no shared cause, then the 33 undecidable.
2. For each member, read its packet and the adjudicator's reason, decide the correction, and submit one proposal through `sn edit` with the source words it rests on: the name, the definition, the unit, the domain or the source bindings, as the defect requires. Dry run first, then apply, then follow it through name review to acceptance. A member bound to sources whose quantities differ is split or rebound through the pipeline's own write path, never by rewording one definition to cover both. If a rename collides with a live name, record the collision and the pair; do not force it.
3. An undecidable member gets a proposal only if the fuller source text you can read now decides it; otherwise list it with exactly what the source would need to say, as a question for the lead. Do not guess.
4. A member whose proposal the name review refuses twice is recorded with both attempts and their scores, restored to its accepted identity, and listed for the lead.
5. Pilot first: repair six members spanning the three largest causes, record cost per member and acceptance, and stop if the projection for all 126 exceeds the ceiling. Spend ceiling USD 30 for this node, stated beside recorded spend. Take the count of live names without a source before and after every pipeline command; it is 10, and you stop if it moves.
6. The record: per member, the identity before and after, the proposal's reason, the review score, the outcome, the cost; per defect class and per shared cause, the counts repaired, refused, left for the lead. Do not judge your own repairs sound: a fresh blind adjudicator does that after you.

## Rules
- You are the only live-graph writer of standard names while you run. Never edit stored text with ad-hoc Cypher; `sn edit` and the pipeline's own paths only. Run every `sn` command from current main code.
- Heavy gates run as `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug ...` with logs on GPFS.
- The gate is the script's `--check` against the live graph: every repaired member's stored identity equals the record, accepted, with a source. The negative control is that it fails when a record row is altered.
- Write the record and the manifest incrementally, after each batch, so a cut-off turn loses nothing. Commit locally with a subject and a body; do not push.
