NEEDS-HELP: the one-row signed supersede committed, but the closed supersede operator does not write the required `superseded_by` scalar, so the exact postcondition cannot be met inside this node's evidence-only fence.

tried: Read the live plan and production graph, then ran one process that re-read the cohort, required `area_of_flux_surface` to have exactly zero live children, derived a fresh one-row authority in that same invocation, previewed it, applied its signed hash, replayed it, and re-read both identities. The apply committed exactly one supersede plus one receipt. The first unmet assertion was `superseded_by`: the predecessor is now lifecycle-superseded but the scalar is null.

options: (1) expand scope to `imas_codex/standard_names/signed_manifest.py` and its signed-operator contract tests, make a supersede authority sign its canonical target and atomically write the `superseded_by` scalar plus successor mirror, then add a governed recovery transition for this already-committed row; (2) authorize a separate signed scalar-and-mirror repair under the existing ledger, explicitly amending the done-when from one apply to an original supersede plus one recovery receipt; (3) authorize an exact rollback of this manifest's one lifecycle mutation and one receipt, patch the operator, then regenerate and reapply from the original state. Option 3 is destructive and requires new authority.

leaning: option 1, with an explicit amended recovery measure. It preserves the immutable receipt and graph history, fixes the generic operator for every future supersede, and avoids deleting a committed change merely to reconstruct a cleaner transcript.

cost-if-wrong: choosing recovery when an exact rollback was required leaves a two-transition history and requires the evidence gate to be amended; choosing rollback when recovery was intended requires deleting a valid receipt and reversing lifecycle state, then regenerating the authority and rerunning all apply, replay, counter, lifecycle, producer, and collateral checks.

# Flux-surface umbrella supersede — partial apply and exact blocker

## Material outcome

The signed production transition was admitted and committed, and its replay is
write-free. The exact node remains **blocked**, not complete, because the live
post-state is:

| Measure | Required | Live result | Verdict |
|---|---|---|---|
| Umbrella live children immediately before signing | 0 | **0** | pass |
| Signed authority rows / admitted / refused | 1 / 1 / 0 | **1 / 1 / 0**, no refusals | pass |
| Apply outcome / changed | `applied` / 1 | **`applied` / 1** | pass |
| Apply mutations / receipt rows | 1 / 1 | **1 / 1** | pass |
| `StandardNameChange` delta | receipt count | **+1 = 1 receipt** | pass |
| Same-hash replay | `already_applied`, changed 0, writes 0 | **`already_applied`, changed 0, persistent writes 0** | pass |
| Umbrella lifecycle | `name_stage=superseded` | **`name_stage=superseded`, `status=superseded`** | pass |
| Umbrella successor scalar | canonical target | **null** | **fail** |
| Canonical target lifecycle | remains accepted | **accepted, drafted docs, valid, pipeline origin** | pass |
| Canonical target producers | unchanged from pre-read | **21 before, 21 after** | pass |

The required successor is
`poloidal_plane_cross_sectional_area_of_flux_surface`. It remains an accepted,
valid identity with all 21 producing sources. The umbrella remains childless.

## In-invocation ordering and signed authority

One Python process opened the production `GraphClient` and executed this order:

1. live pre-apply re-read;
2. assertion that `area_of_flux_surface` was accepted and had zero live
   `HAS_PARENT` children;
3. assertion that the canonical target was accepted, valid, and live;
4. fresh authority construction and signing;
5. signed preview;
6. signed apply using the preview hash;
7. persistent post-apply re-read;
8. same-hash replay;
9. persistent post-replay re-read and quantitative assertions.

The authority was therefore derived **after** the zero-child live read and
**inside the applying invocation**, not carried from the earlier refused cohort.

| Digest | SHA-256 |
|---|---|
| Authority file | `0039b5f248177b798baf67cdcf9f35e7b0f34fddb47043ad0d414076d8e1a14d` |
| Signed authority payload | `039c0d6097ecde14072878f92d5fc5a4b76b0d7c5e775eac5874c6a0b3c02973` |
| Preview/apply/replay manifest | `e8b3265a09208fa82ebd3527b8d0744d40ccf96bdf13256ece3e15d7d77e078e` |

The preview partition was `authority_rows=1`, `admitted=1`, `refused=0`,
`would_change=1`, with an empty refusal list. The apply returned
`outcome=applied`, `changed=1`, `mutations=1`, `receipt_rows=1`, and
`persistent_writes=2`: one lifecycle mutation and one immutable receipt. The
runner passed its assertions that the `StandardNameChange` delta was +1, equal
to the receipt count, and that `LLMCost` did not change before it reached the
successor-scalar assertion.

The receipt is:

- id:
  `sn-change:signed-manifest:e8b3265a09208fa82ebd3527b8d0744d40ccf96bdf13256ece3e15d7d77e078e:faf45258d8d7587981353eea`;
- operation: `supersede_legacy_spelling`;
- row:
  `area_of_flux_surface=>poloidal_plane_cross_sectional_area_of_flux_surface`;
- run: `r-20260822T210631596956-n-supersede2`.

An independent same-hash replay after the failed exact assertion returned
`outcome=already_applied`, `changed=0`, `receipt_rows=1`, and
`persistent_writes=0`.

## Persistent graph state and target preservation

The fresh diagnostic re-read after the committed apply and replay reports:

| Property | `area_of_flux_surface` | Canonical target |
|---|---|---|
| `name_stage` | `superseded` | `accepted` |
| vocabulary `status` | `superseded` | null |
| `docs_stage` | — | `drafted` |
| `validation_status` | `valid` | `valid` |
| `origin` | — | `pipeline` |
| `superseded_by` | **null** | null |
| live children | 0 | — |
| producing sources | — | **21** |

The applying invocation captured the target's complete 21-row producing-source
closure before mutation. The post-read contains the same 21 source ids and the
same `PRODUCED_NAME` relationship element ids and properties. The signed
out-of-allowlist immutability guard also re-hashed that target closure before
commit. Thus the target's lifecycle and producer authority survived unchanged;
the failure is confined to the absent predecessor successor scalar.

The receipt itself reinforces the implementation gap: its signed `row_id`
contains the canonical target, but `to_name` remains
`area_of_flux_surface`, so neither the predecessor property nor the receipt's
dedicated target field records the successor.

## Why this cannot be repaired inside the fence

The generic `RepairMutationKind.supersede` branch in
`imas_codex/standard_names/signed_manifest.py` writes
`superseded_from_stage`, `name_stage`, `status`, claim fields, and optional
source-path clearing. It does not consume a signed canonical-target argument,
does not set `superseded_by`, and does not create the schema's successor mirror.
The receipt writer likewise sets `to_name` from the predecessor identity.

This node may write only this evidence file. A raw Cypher property edit would
bypass the signed authority, and a second ad hoc manifest would violate the
exact one-supersede transition being measured. The safe repair therefore needs
a source-and-test scope expansion plus an explicit decision for recovering the
already-committed partial transition.

## Reproducible artifacts

All operational artifacts are retained under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T210631596956-n-supersede2/`:

- `umbrella_supersede_runner.py` — the single-process read, sign, preview,
  apply, replay, and assertion program;
- `umbrella-supersede-authority.json` — the builder-emitted signed authority;
- `umbrella-supersede-apply.log` — the production run and first failing exact
  assertion, with `EXIT_MARKER=1`;
- `umbrella-supersede-diagnostic.log` — read-only receipt and persistent-state
  diagnostic, with `EXIT_MARKER=0`;
- `umbrella-supersede-replay.log` — independent exact-hash replay returning no
  writes, with `EXIT_MARKER=0`;
- `umbrella-supersede-evidence-check.log` — the named exact postcondition check,
  failing only `superseded_by`, with `EXIT_MARKER=1`.
