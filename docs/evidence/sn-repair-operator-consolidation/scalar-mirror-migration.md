# Scalar mirror migration evidence

## Outcome

The scalar and backing-projection repair now enters through
`apply_signed_manifest`. The compatibility export
`repair_scalar_projection_mismatches` is a fixed `functools.partial` with the
closed adapter `semantic-mirror-repair`, mutation program
`restore-semantic-mirror`, and guards `sole-live-target-authority`,
`exact-upstream-backing`, and `out-of-allowlist-immutability`. The literal
public function definition is absent from `graph_ops.py`.

The backing-projection restore is deliberately its own closed mutation
program. The generic `add_relationship` program that revives a
`PRODUCED_NAME` relationship was not widened: scalar-mirror restoration still
requires exactly one live target and one exact upstream backing, then restores
only the scalar mirror and a missing `HAS_STANDARD_NAME` projection.

## Byte-unchanged behavioural gate

`tests/standard_names/test_signed_source_dispositions.py` was not edited. Run
together with `tests/graph/test_cypher_property_check.py` on disposable Neo4j
2026.01.4, it executed 17 tests: **17 passed, 0 failed, 0 skipped**.

The scalar-mirror cases retained these exact outcomes:

| Case | Admitted | Refused | Identity and result |
|---|---:|---:|---|
| scalar, projection, and combined mismatch cohort | 3 | 0 | `dd:mirrorscalar/value`, `dd:mirrorprojection/value`, and `dd:mirrorboth/value` admitted; 2 scalars changed, 2 projections added, and exactly 1 `StandardNameChange` cohort receipt written |
| ambiguous live-target closure | 0 | 1 | `dd:mirrorambiguous/value` refused verbatim: `source does not have exactly one live target`; graph snapshot remained byte-identical |
| backing-projection drift after preview | 0 | 1 conflict | `dd:mirrorprojectiondrift/value` refused at apply verbatim: `fresh semantic-mirror manifest does not match signed hash`; transaction wrote nothing and the graph snapshot remained byte-identical |

Applied replay returned `already_applied`, `changed=0`, and a byte-identical
snapshot. The unchanged suite also kept the out-of-allowlist identity
`dd:mirroroutside/value` byte-identical and left the unsourced-name census
unchanged.

An additional disposable-graph receipt probe exercised the cohort shape that
contains an already-clean row. It requested 2 sources, admitted 1 repair,
carried 1 already-clean source, refused 0, changed 1 scalar, added 1 projection,
and wrote exactly 1 receipt whose target projection contained both
`dd:mirrorreceiptmismatch/value` and `dd:mirrorreceiptclean/value`. Replay was
`already_applied`, `changed=0`, with a byte-identical snapshot. This proves an
already-clean row remains part of the signed cohort receipt rather than being
dropped merely because it needs no mutation.

## Relocation proof

The transaction core from the first `requested = sorted(set(source_ids))`
statement through `client.close()` is **291 lines / 14,358 bytes** before and
after relocation. Both byte streams have SHA-256
`c6d5c1331f2cb73abd740fb25a179cf94ad2094a92c2ad67ad3a5cdb773af8f4`.
The only new logic around that unchanged core is the closed adapter validation
and fixed-partial compatibility signature; no receipt string, refusal string,
compare-and-set, lock, replay, or mutation statement changed.

The focused test log is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T085515726418-n-scalarmirrormigrate/logs/focused-tests.log`.
The already-clean receipt probe is in
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T085515726418-n-scalarmirrormigrate/logs/already-clean-receipt.log`.
