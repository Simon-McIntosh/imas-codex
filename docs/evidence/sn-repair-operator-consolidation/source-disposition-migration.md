# Source-disposition operator migration

Date: 2026-08-25

## Result

`apply_adjudicated_source_dispositions` is now a fixed `functools.partial` of
`apply_signed_manifest`. The partial fixes the named `catalog-disposition`
adapter, the closed `select-survivor-and-release-bindings` compound mutation,
and the signed-adjudication, last-producing-source, structural-legitimacy, and
out-of-allowlist-immutability guards. The former public function body is absent
from `graph_ops.py`; its legacy signature remains visible at runtime.

The adapter reads `imas-codex.catalog-edit-dual-binding-adjudication.v1`
unchanged and accepts the optional
`imas-codex.refused-target-orphan-adjudication.v2` join unchanged. No committed
authority was edited, converted, or re-signed. Their file digests remain
`5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e`
and `2c2d38f3241ec3057d24a5d05c27840f5e4ffe99520063059ab31c1e9d4bca36`,
respectively.

## Byte-unchanged equivalence gate

The existing `tests/standard_names/test_signed_source_dispositions.py` suite
remained byte-unchanged and executed **14 cases with 14 passed, 0 skipped, and
0 failed** on disposable Neo4j 2026.01.4. The three Cypher-property checks also
passed, making the combined invocation **17 passed, 0 skipped, 0 failed** in
17.47 seconds.

The two selection modes retained these exact counts and identities:

| Selection mode | Signed parent result | Executable result | Applied identities | Refused or excluded identities |
|---|---:|---:|---|---|
| Complete signed cohort | 3 admitted, 0 refused | 3 admitted, 0 refused; 3 bindings and 3 projections removed; 2 scalars changed | `dd:dispositionretain/value` → `dispositionretain_semantic`; `dd:dispositionretarget/value` → `dispositionretarget_semantic`; `dd:dispositionmissing/value` → `dispositionmissing_semantic` | None. `dd:dispositionoutside/value` remained outside the allowlist and byte-identical. |
| Admitted subset | 1 admitted, 1 refused | 1 admitted, 0 refused; 1 binding and 1 projection removed; 1 scalar changed | `dd:dispositionsubsetsafe/value` → `dispositionsubsetsafe_semantic` | `dd:dispositionsubsetprotected/value` was excluded with target `dispositionsubsetprotected_catalog` and the verbatim reason `removal would leave target with zero live producing sources`. |

Each applied mode wrote exactly **one cohort `StandardNameChange` receipt**.
Exact replay returned `already_applied` with `changed=0`; the complete cohort,
the admitted subset, its excluded row, and the outside-allowlist cohort all had
byte-identical graph snapshots before and after replay.

The byte-unchanged suite also retained every refusal boundary:

- `dd:dispositionprojection/value` remained refused with `live backing
  projection set changed from adjudication` and no writes;
- `dd:dispositionlastbinding/value` remained refused 0 admitted / 1 refused
  with target `dispositionlastbinding_catalog` and `removal would leave target
  with zero live producing sources`;
- `dd:dispositionnonstructural/value` remained refused 0 admitted / 1 refused
  with target `dispositionnonstructural_catalog` and `removed target is outside
  signed structural legitimacy authority`;
- claim drift, scalar drift, and a late incoming binding all retained the
  fail-closed `fresh source-disposition manifest does not match signed hash`
  conflict with byte-identical state after rollback;
- the signed structural join retained 1 admitted / 0 refused for
  `dd:dispositionstructural/value`, removing the last producer only while the
  exact signed direct child kept `dispositionstructural_catalog` structurally
  authoritative.

## Auditable relocation

The selected source boundary was
`imas_codex/standard_names/graph_ops.py:19147-19542` at the source commit. The
19,366-byte fragment started at the retry decorator immediately preceding the
public function and ended after its `finally` client-close branch.

After excluding only the adapter seam—the retry decorator, private adapter
name and fixed-control parameters, and graph-helper imports—the entire
18,878-byte transaction core is byte-equivalent. Its pre-move and post-move
SHA-256 are both
`7a55ecba3977e46a88f07463eaeb2cbb8ae2b71c94e9e0761a9fc8875cdb1279`.
The public runtime assertions report `partial=True` and
`partial.func is apply_signed_manifest`.

Complete output and the pre/post fragments are retained in the worker run
directory. The isolated graph used loopback Bolt port 55687 with authentication
disabled; its log records both `Started` and `Stopped`, and its Bolt and HTTP
ports were closed after validation.
