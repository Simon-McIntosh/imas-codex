# Stale derived-source rebind removal

Date: 2026-08-21

## Outcome

The exact two producing edges recreated by structural reconciliation on stale
derived sources were removed from the production graph through the signed,
ledgered stale-source detach operator. Both source nodes remain stale, neither
retains a scalar or relationship target, and both structural parents retain
their standing through three live children each.

| Measure | Before | After | Result |
|---|---:|---:|---|
| Rows returned by `_STALE_LIVE_BINDINGS_QUERY` | 9 | 7 | exactly -2 |
| In-scope `PRODUCED_NAME` edges | 2 | 0 | exactly -2 |
| In-scope stale source nodes | 2 | 2 | status preserved |
| `StandardNameChange` | 7,704 | 7,706 | exactly +2 receipts |
| `LLMCost` | 27,591 | 27,591 | unchanged |
| Untouched source-closure rows | 9,614 | 9,614 | 0 changed row digests |

No ratchet ceiling constant was edited. The stale-live ratchet therefore still
reports its remaining measured residue honestly: 7 is above the frozen ceiling
of 3. Those seven are the previously classified three standing refusals plus
four transient `toroidal_momentum_flux` bindings; this node removed only the two
authorized genuine-regrowth rows.

## Exact identities and structural closure

| Stale source | Removed target | Source before | Source after | Live children before -> after | Per-identity receipt |
|---|---|---|---|---:|---|
| `derived:electron_diffusivity` | `electron_diffusivity` | status `stale`; scalar and edge selected `electron_diffusivity` | status `stale`; scalar null; no live edge | 3 -> 3: `effective_electron_diffusivity`, `parallel_electron_diffusivity`, `poloidal_electron_diffusivity` | `sn-change:stale-source-detach:8dc0b5c50d299639509c7da88421263760e4962bad6c9dd40177e160ae138dd5` |
| `derived:ion_diffusivity` | `ion_diffusivity` | status `stale`; scalar and edge selected `ion_diffusivity` | status `stale`; scalar null; no live edge | 3 -> 3: `effective_ion_diffusivity`, `parallel_ion_diffusivity`, `poloidal_ion_diffusivity` | `sn-change:stale-source-detach:f3463f60ab4896b5d51057a0148736135d9d54a35045135452033ca2f0fd6d39` |

Each parent had zero non-stale producing sources before and after. The operator
therefore admitted each removal only because the signed complete incoming
closure also proved three live structural children. Had either parent had zero
non-stale producers and zero live children, the last-producing-source guard
would have refused the whole transaction.

## Fresh authority and fail-closed apply

The applying invocation loaded the two already-adjudicated rows from
`stale-source-lifecycle.json`, required the exact identity set and count of two,
and wrote a fresh signed two-row authority. It also required the live stale
ratchet count to be exactly nine and the two expected source/target pairs to be
present. Any other applying count records a refusal before preview or mutation;
there is no partial-cohort path.

- Parent authority file SHA-256:
  `f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad`
- Parent authority rows SHA-256:
  `316d95c3e41efb29259bcef7e2ea17e8e003a4453279214afb75b732370f2198`
- Fresh two-row authority file SHA-256:
  `0f1c3cc01f99ef6d4b85f502c5aa60818253fefa8cd23639ed4cd21c48302025`
- Fresh canonical rows SHA-256:
  `ac08b85aba67c1a6b3ececf75f7768544a5d9910d03fedff3535ba50724c47d4`
- Closure-sensitive preview manifest SHA-256:
  `86cd341fc0b6d210febff59bb08cbb5631983a2ebf93857b3e92bcf7ccaa814b`

The preview was generated inside the applying driver and returned exactly two
would-change rows and two prospective receipt rows. The same invocation then
re-read `StandardNameChange=7,704` and `LLMCost=27,591`; both matched the
in-invocation preflight rather than any earlier measurement. The applying
transaction independently re-read and locked the complete source, target,
incoming-producer, and live-child closure. It compared canonical closure bytes
to the preview manifest before deleting either edge. The immediate replay
returned `already_applied`, `changed=0`, with the two receipts intact.

## Untouched-closure proof

Before and after the transaction, the applying driver canonicalized every
source closure outside the two-source allowlist separately: source properties,
all `PRODUCED_NAME` relationships, backing `FROM_DD_PATH` or `FROM_SIGNAL`
relationships, and every backing `HAS_STANDARD_NAME` projection. All 9,614
per-row SHA-256 digests were identical. The changed-source list is empty and
the digest of the ordered per-row digest set is identical before and after:

`7b50b995730f9a24208bf538e64adc310eecd1d4f0f5e69d26a3b0eb3ba19afd`

The public operator independently checked its own untouched-closure aggregate
over the same 9,614 rows:

`59efa0801704e671db5400233fdfdbebeaaaeba42dd3fa524e62a7d175fa19c0`

No backing projection existed for either derived source, so the exact mutation
removed two `PRODUCED_NAME` edges and zero `HAS_STANDARD_NAME` projections.

## Verification and durable artifacts

The production invocation exited successfully and its machine receipt has
SHA-256
`0b7c2ad357f5d93c64ba5618a5112a5046430efd826dd95c96bb8144e1c04f85`.
The focused live-graph test was then run with the explicit graph marker:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD \
  uv run --no-sync pytest -p no:cacheprovider -m graph \
  tests/graph/test_sn_integrity_ratchets.py::test_stale_sources_with_live_bindings_do_not_regrow -q
```

It measured exactly 7 rows and failed against the intentionally unchanged
ceiling of 3. This is the expected qualified result for this two-row node, not a
reason to raise the ratchet: the four transient rows and three standing
refusals remain visible for their separately governed closures. An initial
invocation without `-m graph` collected no tests (exit 5) because the repository
default marker expression excludes graph tests; it is retained in a separate
log rather than being overwritten.

Durable run artifacts:

- Applying driver:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T110221158402-stalebind/apply_stale_rebind_removal.py`
- Signed authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T110221158402-stalebind/signed-stale-rebind-authority.json`
- Fresh preview:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T110221158402-stalebind/stale-rebind-preview.json`
- Apply receipt with all 9,614 before/after per-row digests:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T110221158402-stalebind/stale-rebind-apply-receipt.json`
- Production log (SHA-256
  `4cf1a248d6e0d11070bde3bb6dad803c24385337ba4835fcd37f51637a125bc2`):
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T110221158402-stalebind/production-apply.log`
- Marker-excluded collection log (SHA-256
  `840462c22f399ae645f6dcd4a637277a7da64534a99313e37520451019f926d9`):
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T110221158402-stalebind/focused-stale-ratchet.log`
- Focused live-graph result log (SHA-256
  `efc41874b21421ee6f7542afa13e808b0cad4711d5266fd1533665f9f0f01c79`):
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T110221158402-stalebind/focused-stale-ratchet-graph.log`
