# Legacy DD source lifecycle release

Date: 2026-08-25
Scope: disposable-graph implementation evidence only

## Outcome

The generic signed-manifest registry now contains one closed lifecycle-release
program for exactly these three historical DD source identities:

- `dd:ntms/time_slice/mode`
- `dd:summary/pedestal_fits`
- `dd:waves/coherent_wave`

The program is reachable through `apply_signed_manifest`; its canonical signed
authority is emitted by `build_repair_authority`. The loader requires all three
rows, rejects any additional or missing source identity, requires the exact
release property set and guard set, and writes one `StandardNameChange` receipt
for every admitted row.

## Disposable-graph measure

The focused suite ran against a fresh loopback-only Neo4j instance on SLURM job
`1254498`. The fixture required `IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1` and refused
the configured project graph URI before creating any data. The instance was
destroyed after the run.

| Case | Authority rows | Admitted | Refused | Changed | Receipts | Replay |
|---|---:|---:|---:|---:|---:|---|
| Exact legacy cohort | 3 | 3 | 0 | 3 | 3 | `changed=0`, `persistent_writes=0` |
| One source still has an accepted live target | 3 | 2 | 1 | not applied | 0 | not applicable |
| One source is the final producer of a terminal target | 3 | 2 | 1 | not applied | 0 | not applicable |

The refusal reasons were returned verbatim:

- `source still has a live target`
- `target would lose its last producing source`

The successful case left all three sources at `status=extracted`, reset
`attempt_count=0`, and cleared `produced_sn_id`, `composed_at`, `claimed_at`, and
`claim_token`. The exact replay found the original three receipts, verified the
postconditions, performed no writes, and returned `changed=0`.

## Production boundary

This node did **not** connect to or mutate the production graph. The only graph
mutation occurred inside the disposable instance created for the focused test.
No production authority was built, signed, previewed, or applied by this node.

Test log:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T082012674341-n-ddresidueprogram/test-dd-residue-release-graph.log`
