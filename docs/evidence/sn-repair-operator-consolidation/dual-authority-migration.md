# Dual-authority retirement migration

Date: 2026-08-25

## Result

`retire_signed_dual_authority_targets` is now a fixed
`functools.partial` of `apply_signed_manifest`. The partial selects the closed
`dual-authority-retirement` adapter, the `release-and-supersede` compound
mutation, and the exact authority-join, lifecycle, structural-child, and
collateral guards. The former public function body is absent from
`graph_ops.py`.

The adapter reads both committed authority objects in their original shapes.
It preserves the source payload and per-row signatures, independently verifies
the retirement-authority digest, and joins only the exact signed intersection.
No authority artifact was edited, converted, or re-signed.

## Byte-unchanged equivalence gate

The existing disposable-Neo4j suite
`tests/standard_names/test_dual_authority_retirement.py` remained byte-unchanged
and passed all **5 of 5** cases with **0 skipped and 0 failed**. Together with
the Cypher property guard, the focused invocation completed at **8 passed, 0
failed**.

The clean signed cohort contained exactly **19 admitted source rows**, **20
jointly signed source-target bindings**, and **16 admitted retirement target
identities**, with **0 refusals**. Apply released all **20** `PRODUCED_NAME`
bindings and all **20** backing `HAS_STANDARD_NAME` projections, superseded all
**16** targets, and wrote exactly **16** `StandardNameChange` receipt rows using
the unchanged singular operation `retire_signed_dual_authority_target`.
Representative retired identities include
`beam_area_of_neutral_beam_injector` and `bremsstrahlung_count`.

Both refusal cases remained all-or-nothing and write-free:

- adding `unsigned_retirement_target` outside the joined authority produced
  exactly **1 refused source identity**, left **18 otherwise admissible source
  rows**, and applied **0** changes;
- adding `new_structurally_legitimate_child` beneath the first signed target
  produced exactly **1 refused target identity**, left all **19 source rows
  otherwise admissible**, and applied **0** changes with the verbatim
  `target has acquired a live HAS_PARENT child` reason.

The atomic-final-binding case retained all non-retired bindings. Exact replay
returned `already_applied`, `changed=0`, `persistent_writes=0`, and **16**
ledger rows, with the full graph snapshot byte-identical before and after.

## Verification

- Disposable Neo4j 2026.01.4: loopback Bolt port 54687, authentication
  disabled, no production endpoint contacted.
- `tests/graph/test_cypher_property_check.py`: **3 passed**.
- Legacy suite diff: zero bytes; the test file was not edited.
- Production spelling check: the literal
  `def retire_signed_dual_authority_targets` has **0** matches in
  `imas_codex/standard_names/graph_ops.py`.
- Runtime export check: `isinstance(export, functools.partial)` and
  `export.func is apply_signed_manifest` both hold.

Complete output is retained in the worker run directory under
`logs/pytest-focused.log`; the zero-byte suite-diff proof is
`logs/test-suite-byte-unchanged.log`.
