# Production successor-relocation retirement

## Outcome

The production invocation derived and signed its authority from the live graph,
previewed that exact authority, applied the admitted subset, recovered its
receipts by the applying `run_id` plus manifest digest, and replayed the same
manifest without a write. The fresh cohort was smaller than the prior read-only
projection: **73 live unauthorized successor-relocation relationships**, not the
previously projected 78. The closed program admitted **72** and refused **1**;
the admitted 72 were moved back to their current derivation parents and the
refused relationship was left unchanged.

| Measure | Result |
|---|---:|
| Fresh signed authority rows | 73 |
| Preview admitted | 72 |
| Preview refused | 1 |
| Applied relationships | 72 |
| Receipt rows recovered by exact run and manifest | 72 |
| Replay changed | 0 |
| Replay persistent writes | 0 |
| Unauthorized non-self relationships after apply | 1 |
| Stale qualifier pairs before → after | 52 → 0 |

The refusal is a recorded outcome rather than an apply failure:

| Child | Incumbent tip | Current derivation parent | Verbatim refusal |
|---|---|---|---|
| `ratio_of_parallel_ion_velocity_to_magnetic_field_magnitude` | `parallel_bulk_ion_velocity` | `parallel_ion_velocity` | `signed successor-rewire closure does not match exact incumbent parent` |

Its signed relationship carried
`operator=ratio, operator_kind=binary, role=a, separator=to`. It remains the one
post-apply member of the freshly re-measured non-self class; no attempt was made
to bypass the closed program's exact-parent closure guard.

## Same-invocation authority and receipt identity

The applying process started from source commit
`d2edae90904734b08cee4a35a21b55f507c285be`. It re-ran current structural
derivation and production parent admission over **2,470 live names**, yielding
**2,118 raw** and **1,503 admitted** derived-parent rows. It then joined those
rows to the current successor lineages and the current live `HAS_PARENT`
relationships before constructing the authority. No per-row parent map from the
prior read-only projection was loaded or reused.

- Authority file SHA-256:
  `6fdc9ca2764cb12d171891cff955b0db8f0f5cf1b2883c4703aef2dc80f1686a`
- Authority payload SHA-256:
  `0c570dcb6e8c271eb3936ac2957b1967946f01168050b80a2b14da1fbce6dcc8`
- Authorized manifest SHA-256:
  `dc82dfb3f6e69d62d5f38bfd0284876b3f05b5287620f5f2380546ce8fea92f5`
- Applying run:
  `r-20260825T141822953482-n-retireapply`

The receipt recovery query required both the exact applying `run_id` and the
exact manifest digest above. It returned **72 rows**, matching the apply's 72
admitted rows and `receipt_rows=72`. No operation-name lookup was used as the
recovery key. Exact replay returned `outcome=already_applied`, `changed=0`,
`persistent_writes=0`, and the same 72 receipt rows.

## Collateral and zero sanity

The external collateral instrument covered the `HAS_PARENT` dimension this
node owned while excluding every signed cohort child. It contained **1,411
relationships** with SHA-256
`ea87d5dc03e000a056ed1b3fd86825afc34f60371427696c4083255ca3e60799`
before apply, after apply, and after replay. This graph-specific digest did not
include source-origin, documentation, review, or other axes concurrent nodes
could write. The signed executor also enforced its own out-of-allowlist closure
inside the transaction.

The post-apply unauthorized-class value of **1** is supported by these schema
and endpoint coverage counts, which were identical again after replay:

| Sanity measure | Covered / candidates |
|---|---:|
| `StandardName.id` | 4,658 / 4,658 |
| `StandardName.name_stage` | 4,658 / 4,658 |
| `HAS_PARENT` source `StandardName.id` | 1,485 / 1,485 |
| `HAS_PARENT` target `StandardName.id` | 1,485 / 1,485 |
| `HAS_PARENT.operator_kind` | 1,485 / 1,485 |

The independent stale-qualifier census saw **711 live qualifier edges**. Its
stale subset moved from the carried live baseline of **52** to **0**, and stayed
0 after replay. Thus the remaining refused binary-ratio relationship is not
hidden inside the qualifier result.

## Export-cycle release

Before apply, the global `HAS_PARENT`/`HAS_ERROR` ordering graph named exactly
these two cycle participants:

- `spectral_signal_to_noise_ratio_of_spectrometer_channel`
- `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel`

After apply and again after replay, the global cycle-participant set was empty.
That set is a superset of the export ordering-cycle withheld set, so both named
identities are explicitly **released from export withholding**.

## Retained artifacts

- Full structured result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953482-n-retireapply/production-result.json`
  (SHA-256
  `82a90ac0131b213c511be5190034a46283f8b1318c7ff0e0cba7108f6e0816c7`)
- Fresh signed authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953482-n-retireapply/production-authority.json`
- Successful applying invocation log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953482-n-retireapply/production-apply-second.log`
- Preflight failure log, before any preview or graph mutation:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953482-n-retireapply/production-apply.log`

The first invocation stopped at authority construction because the builder
required the closed selection id `artifact-rows`; it did not reach preview or
mutation. The corrected invocation passed Ruff, built the required authority,
and produced all results above in one process.
