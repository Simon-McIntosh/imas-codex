NEEDS-HELP: the live restore-point census shows both PR-edited targets at `docs_stage='drafted'`, but this node authorizes restoring only `pulse_duration`

tried: Read the live plan at head `8c4e7e83`, verified the atomic approval preflight landed at `ea575887`, and captured the graph/catalog controls before mutation. The graph census is `4683` StandardName nodes: `approved=0`, `contested=0`, `accepted=2336`, and `with_catalog_pr_number=0`. Both edited targets have `name_stage='accepted'`, no catalog provenance, and `docs_stage='drafted'`: `breakdown_initial_time` has docs score `0.7083333333333334`; `pulse_duration` has docs score `0.5083333333333333`. The approval preflight requires every edited target to have both name and docs stages accepted. No Cypher write, approval, resolution, undo, catalog commit, tag write, or upstream write was attempted.

options: (1) authorize a two-row lifecycle restoration, setting `docs_stage='accepted'` on both edited targets after checking their restore-point provenance; (2) authorize a second single-field restoration for `breakdown_initial_time` in addition to the already specified `pulse_duration` restoration; or (3) revise the approval preflight/flow so the earlier resolved edit is eligible without a direct lifecycle repair, which requires source and test paths outside this node's scope.

leaning: option 2, because the repaired undo now restores `docs_stage='accepted'` on both approved and contested reversions, and the two drafted values are residue from the pre-fix undo rather than a new adjudication. It is the narrowest mutation that makes the current live state match the repaired undo's intended postcondition.

cost-if-wrong: the wrong authorization could erase a legitimately pending documentation-review state and allow PR 3 fold-back to act on a target that has not actually returned to its catalog restore point. The cycle would then need another undo and a provenance-backed lifecycle correction before it could be trusted.

## Evidence

- Baseline graph log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T000011001254-n-west-stage6c-final-cycle/logs/00-baseline-graph.log` (exit `0`).
- Baseline controls log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T000011001254-n-west-stage6c-final-cycle/logs/01-baseline-controls.log` (exit `0`).
- Fork catalog main is `3edfcc138483bc23e734a7514e15d7c9dce9ee89`; its tree `f2954894015c95f31ff4b1782a4a1240b3e054ed` equals merge `6a5c44d3`'s tree.
- The fork has the cut-time tag `v0.3.0rc1+west-task-2e` and no `graph-merged:` receipt tag.
- Upstream default branch is `main` at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`.
- Review manifest working bytes equal HEAD: git blob `dd46e21250c3e4aad9259a2f58d87a2feff5fbab`; SHA-256 `d7f5b833cddcf17ae67318719a9b14d3ea1dd4a5e337b4a8c7a3b43eee9f122a`.
- Quantitative completion: `0/3` requested lifecycle commands run and `0/1` authorized lifecycle repairs applied; external state remains at the verified baseline.
