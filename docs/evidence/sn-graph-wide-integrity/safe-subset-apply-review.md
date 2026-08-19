# Guarded source-disposition safe-subset review

## Verdict

**HOLD.** The regenerated guard correctly identifies a semantically safe
138-row subset, but the current signed preview does not authorize applying that
subset. Its exact outcome is `refused`, and
`apply_adjudicated_source_dispositions` rolls back before acquiring the signed
global-closure locks whenever any refusal is present. Consequently manifest
SHA-256 `e475dfaad8e736c8d27c262b11eb317a69ee7024dc186f787250853dab4090a1`
authorizes **zero mutations**, not the 138 admitted rows.

The data partition and safety guard pass adversarial review. Authorization is
held only because no public, exact-hash executable authority exists for the
138-row mutation. Producing one requires a new write-free preview whose signed
manifest and apply path name exactly the admitted action set; slicing actions by
hand or treating the current refused hash as partial-apply authority is not
permitted.

## Quantitative checks

| Check | Independent result | Verdict |
|---|---|---|
| Complete row partition | 138 admitted + 78 unique refused source rows = 216 requested rows. | PASS |
| Refused target identity | The union of `target_ids` across the 78 refusal rows contains exactly 89 names. It is byte-for-byte equal to both independently measured last-binding sets: ANY source status and non-stale producer scope. The preview's numbered 89-name list has zero missing and zero extra names. | PASS |
| Signed global incoming closure | 151 removed-target closures sign 676 incoming `PRODUCED_NAME` relationships: 400 attached + 272 composed + 4 stale = 676. Each entry carries target, source, relationship element identity and properties. | PASS |
| Lock and re-hash contract | `_lock_signed_source_disposition_authority` locks all closure source nodes, target nodes, and incoming relationships. The refusal-free apply path rebuilds `_signed_source_disposition_authority` after locking and requires `_authority_payload_hash(locked_manifest)` to equal the preview digest before mutation. | PASS in code and disposable refusal-free apply; not reached by this refused production preview |
| Stale-source semantics | All four stale incoming relationships remain in the signed closure, so their appearance, disappearance, or status drift invalidates the manifest. Pre-apply orphan detection counts only source status other than `stale`; the post-delete invariant independently requires at least one remaining non-stale source. | PASS |
| Admitted-row producer floor | The 138 admitted rows schedule 138 binding removals over 59 distinct targets. Their target set intersects the independent 89-name last-binding set at exactly 0. Because that independent set was computed under the larger all-216 removal schedule, applying fewer removals is monotonic: every one of the 59 admitted targets retains at least one non-stale producer. | PASS |
| Executable safe-subset authority | The live preview reports `outcome=refused`; the public operator returns `changed=0`/`would_change=0` before `_lock_signed_source_disposition_authority`. There is no subset selector, no separately signed 138-row authority, and no exact preview hash that reaches the 138-row mutation. | **FAIL — HOLD** |

The admitted subset would remove 138 exact `PRODUCED_NAME` bindings and 138
matching `HAS_STANDARD_NAME` projections while changing 51 scalars. The 78
refused rows protect 89 targets, previously classified as 86 accepted and three
reviewed names. Representative refusals remain correctly aligned with source
meaning and name identity:

- `dd:bremsstrahlung_visible/channel/intensity` protects both
  `bremsstrahlung_count` and
  `time_derivative_of_bremsstrahlung_count_at_detector_pixel`;
- `dd:core_sources/source/profiles_1d/ion/momentum/radial` protects
  `momentum_source`, `plasma_momentum_source`, and
  `radial_plasma_momentum_source`;
- `dd:balance_of_plant/power_electric_plant_operation/system/power` protects
  `net_absorbed_power_of_plant_system`.

These rows remain routed to individual adjudication under the locked orphan
policy. This review does not alter or pre-empt the concurrent adjudication
record.

## Closure and concurrency review

The global closure is materially complete for last-binding authority. For each
of the 151 removed `StandardName.id` targets it includes every incoming source
node and `PRODUCED_NAME` relationship, including the four stale sources. The
lock set is the union of the original source-local participants and this global
incoming closure:

- every source node is write-locked through its element identity;
- every removed target node is write-locked;
- every incoming producing relationship is write-locked;
- the complete authority is queried again after locking; and
- any refusal or digest difference raises before scalar or relationship
  mutation.

The status property is inside each signed source property map. A source changing
between stale and non-stale therefore changes the manifest hash, and source-node
locking prevents that status from changing after the locked re-read. Creating a
new incoming relationship also changes the global closure; the added drift
regression proves a stale preview is refused. After deletion, the operator
queries every removed target and rolls back unless each retains a non-stale
incoming source.

This contract is sufficient for a refusal-free exact subset. It is not a means
to reinterpret an atomic refused preview as a partial apply.

## Findings

### HIGH — No exact executable authority exists for the 138-row apply

The current manifest contains 138 actions and 78 refusals, but the public apply
path checks `if refusals` and returns `outcome=refused` before locking or
mutation. Passing `apply=True` with `e475dfaa…` still changes zero rows. The
later scalar compare-and-set also expects the complete adjudicated source set,
so bypassing the refusal return and feeding the 138 actions is not a supported
or safe workaround.

Authorization requires one of the sanctioned designs to be implemented and
previewed: either a signed subset selector bound into the full 216-row authority
or a separately governed 138-row adjudication. In either case, the generated
manifest must identify the exact mutation set, reach the lock/re-hash path, and
return `would_apply` with zero refusals before its hash can be authorized.

### MEDIUM — The production 676-edge closure has not exercised its locked apply path

The code structure and disposable tests establish the lock/re-hash mechanism,
including a successful refusal-free three-row apply and refusal on incoming-edge
drift. The production preview with 676 relationships is necessarily rolled back
before locking because it contains refusals. A new exact 138-row preview may
have a different removed-target population and closure cardinality; its own
digest and counts must be reviewed rather than inherited from `e475dfaa…`.

No semantic safety defect was found in the selected 138 rows. The HOLD is an
authority/executability finding.

## Required authorization evidence

Before applying the safe subset, require all of the following from one fresh
write-free invocation:

1. an exact signed authority naming only the executable 138-row action set while
   retaining traceable linkage to the unchanged 216-row adjudication;
2. `outcome=would_apply`, `admitted=138`, `refused=0`, 138 exact binding and
   projection removals, and 51 scalar changes;
3. the newly computed global incoming closure count and digest for the exact
   subset, with every source, target, and relationship locked and re-hashed;
4. zero admitted targets at a post-apply non-stale producer count below one;
5. unchanged outside-allowlist closure, `StandardNameChange`, and `LLMCost`
   counters; and
6. apply only with the exact hash returned by that preview, followed by
   write-free replay and a final producer-floor census.

## Evidence inspected

- live plan version 178 and its independent 89-target HOLD census;
- guarded implementation commit `83e2cc7c`, merged at `52af3add`;
- regenerated preview and manifest hash `e475dfaa…`;
- independent baseline log containing 151 removed targets, identical 89-name
  loss sets, and status counts 400 attached / 272 composed / 4 stale;
- focused disposable suite: 8 passed, 0 failed, 0 skipped; and
- this review's independent set/arithmetic audit:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260819T070832660445-safe-subset-apply-review/safe-subset-static-audit.log`.
