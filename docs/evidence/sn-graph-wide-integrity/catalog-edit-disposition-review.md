# Adversarial source-disposition authorization review

## Verdict

**HOLD.** The committed adjudication and the 216-row preview are internally
consistent, the signature verifier implements the committed ASCII-escaped
contract, and the scalar/claim compare-and-set is fail-closed. The canonical
checkout census nevertheless proves the proposed apply is unsafe: the 216 rows
produce 240 removal pairs over 151 distinct removed targets, and **89 of those
151 targets would be left with zero live producing sources**. The 89 are **86
accepted names and 3 reviewed names**.

The result is identical under both audited source-status scopes: all source
statuses and composed/attached sources only. The current graph baseline is
exactly 85 live names with no producing source; this apply would raise it to
**174**, newly orphaning 86 accepted names and regressing the plan's recovered
zero-provenance-orphan invariant. The locked orphan policy requires individual
adjudication, not silent detachment. `StandardName` identity is read through
the `id` property; there is no `name` property.

## Required checks

| Check | Evidence and independent review | Result |
|---|---|---|
| Committed authority identity | Artifact file SHA-256 `5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e`; declared payload SHA-256 `c227e70ec5cd940577ca778ce5ec63e4df3a63bf68c3e845eba92d0a4b9a0efb`. | PASS |
| ASCII-escaped signature contract | `_catalog_edit_adjudication_signature_hash` uses sorted compact JSON with default ASCII escaping and `allow_nan=False`. `_validate_signed_source_adjudication` verifies the outer payload and every row before graph access. The committed regression pins `c227e70e…` and all 216 row signatures. It does not alter `_authority_payload_hash`. | PASS |
| Closed disposition set and row identity | Exactly 216 unique `dd:` source rows; 111 `retain_scalar_target` + 102 `retarget_scalar_target` + 3 `select_missing_scalar` = 216. Candidate lists must be unique/sorted; survivor must be a candidate; removed targets must equal every non-survivor. Representative binding: `dd:summary/gas_injection_accumulated/ethylene/value` selects `accumulated_ethylene_count` over `ethylene_count` under DD-path identity. | PASS |
| Scalar compare-and-set | Apply matches the exact source element id and the adjudicated prior scalar, including null equality, requires both claim fields null, writes the survivor, and requires the returned source-id set to equal all expected sources. The mutation query rechecks the survivor scalar and survivor binding. | PASS |
| Claim compare-and-set | Preview refuses an existing `claimed_at` or `claim_token`; apply locks every signed source, rebuilds and rehashes the full source-local manifest, and both scalar and deletion queries require null claim state. Disposable regressions exercise both scalar and claim drift. | PASS |
| Binding/projection arithmetic | The preview reports 240 binding removals and 240 projection removals. The validator derives each row's removals as its exact candidate set minus its one survivor; the action builder pairs each removed binding with one exact backing projection, and the mutation requires every expected row to return. | PASS, conditional on unchanged exact manifest |
| Scalar-change arithmetic | 102 retargets + 3 missing-scalar selections = 105 changes; 111 retained scalars are unchanged. This equals the preview's 105. | PASS |
| Write-free preview | `would_apply` for 216 admitted, 0 refused; `StandardNameChange` stayed 7,451 and `LLMCost` stayed 27,467; the reported 9,282 out-of-allowlist source closures retained digest `b9864539…`. | PASS for the preview transaction |
| Last live binding of a removed name | Coordinator census from the canonical checkout: 240 removal pairs cover 151 distinct removed targets; 89/151 would retain zero live producing sources, split as 86 accepted + 3 reviewed. ANY-status and composed/attached-only scopes are identical. Baseline live unsourced names would move 85 → 174. The instrument does not collect, refuse, lock, or hash this global incoming-binding closure. | **FAIL — HOLD** |

The row and removal arithmetic proves that every selected source retains one
target. The independent target-level cardinality proves that 89 removed targets
do not retain another source. Both facts are true, and the second is decisive.

## Findings

### HIGH — Name-level orphan safety is outside the signed closure

`_signed_source_disposition_authority` signs the 216 selected sources, their
target nodes, selected source-to-target bindings, one exact DD backing, and its
projections. It never signs the complete incoming binding set of each removed
target. Consequently the apply can delete a target's last binding while still
matching the exact preview hash. After deletion it merely recomputes
`target.source_paths`; an empty list is accepted rather than refused.

The measured impact is not hypothetical: 89/151 removed targets lose their
last live binding. Even after those rows are refused, an unrelated
source-to-target edge remains outside the current manifest and can disappear
between preview and apply without necessarily changing any signed participant
property. The instrument therefore needs the complete incoming-binding set as
a signed, locked, rehashed compare-and-set closure.

### HIGH — The proposed apply would increase live unsourced names from 85 to 174

The canonical census measured 89 last-binding removals: 86 accepted names and
3 reviewed names. Counts are robust to whether all source statuses or only
composed/attached sources are treated as producing. Applied unchanged, the
operation raises the live-unsourced population from 85 to 174 and regresses the
plan's zero-provenance-orphan invariant. These 89 targets must be refused and
routed to individual adjudication under the locked orphan policy.

### MEDIUM — Existing regression coverage proves local closure, not target survival

The six committed tests cover all three disposition modes, exact application,
write-free replay, out-of-allowlist source immutability, scalar/claim drift,
projection refusal, and signature tampering. None seeds a removed target whose
only incoming binding is selected for deletion, and none requires a refusal or
lifecycle disposition for that case. The preview report's claim of complete
collateral hashing is therefore complete only at the source-closure level, not
at the removed-name incoming-edge level.

## Authorization conditions

Authorize the live apply only after all of the following are true:

1. The complete global incoming-binding closure for every removed target is
   included in the manifest's signed, locked, rehashed compare-and-set state.
2. The operator fail-closes every last-live-binding removal. The measured 89
   refused targets are routed to individual adjudication under the locked
   orphan policy; they are not silently orphaned.
3. The incoming-binding set/count used by the decision is included in the
   manifest's locked, rehashed compare-and-set closure; a separate stale census
   is insufficient authority.
4. A disposable regression proves the operator refuses a last-live-binding
   removal without its signed lifecycle disposition and remains write-free on
   refusal.
5. The 216-row preview is regenerated after the guard lands and reports the
   exact admitted/refused partition, including 89 last-binding refusals unless
   individual adjudication changes their authority, with zero writes.

## Evidence log

The coordinator produced the target-level census from the writable canonical
checkout. This delivery's earlier failed-attempt evidence is retained at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T063914061340-catalog-edit-disposition-review/readonly-authority-and-orphan-audit.log`.
Attempt one failed at `uv` cache initialization; attempt two, with caching
disabled, failed at read-only generated-model creation. Both exited before the
read-only graph census and before any graph mutation. That is an expected
review-sandbox limitation, not a gap in this finalized review. A corrective
node implementing the signed global closure, refusal, and routing is already in
flight.
