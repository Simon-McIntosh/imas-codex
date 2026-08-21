NEEDS-HELP: The exact 35-row delete cannot be applied safely because six adjudicated rows are currently quarantined and every target remains connected while the generic delete mutation only accepts relationshipless nodes.

tried: Re-read live plan version 215 and both adjudication records; derived the exact 35 identities as the 36-row release partition minus `parallel_mach_number`; re-measured all 35 against production; inspected the existing `apply_signed_manifest` delete path; and ran the four credentialed graph ratchets. No authority or apply manifest was signed and no graph mutation was attempted.

options: (1) Extend the existing generic signed-manifest `delete` mutation to sign, lock, and remove the complete target-owned relationship closure atomically, then re-run this exact node. (2) Re-adjudicate the six current quarantine rows and explicitly decide whether their quarantine is incidental evidence or a mandatory refusal, then extend the generic operator before applying. (3) Narrow the cohort to 29 currently valid rows, but only under new authority because that would violate the present exact-35 whole-cohort contract.

leaning: Option 2 followed by option 1. The node contract directly requires refusal for any currently quarantined identity, while the earlier adjudication explicitly calls one quarantined predecessor deletable; that semantic conflict must be resolved before an irreversible delete. Independently, the generic operator needs a graph-closure-capable delete branch for every one of the 35 targets.

cost-if-wrong: Treating quarantine as incidental without explicit authority could irreversibly delete six identities contrary to the node guard. Extending only the operator would still leave the exact cohort unauthorized. Narrowing to 29 would require discarding the cohort and receipts and redoing the entire apply under a newly signed count.

# Unsourced standard-name delete apply: blocked preflight

## Material result

The production graph was not mutated. The requested done-when remains **0 of 35 identities deleted**, **0 deletion receipts**, and **0 replay rows**. `parallel_mach_number` was excluded from the derived 35-row set and no manifest containing it was created.

The exact adjudicated identity set is stable: the live unsourced-without-live-child predicate returns the original **36** identities, the held identity is present, and subtracting it yields exactly **35** identities. The ordered 35-identity SHA-256 is `038ac087c80fa0b7aeede908ff7d65017a90bd5cc2192aae512bc7d659c8299b`.

## Fail-closed condition 1: current quarantine state

The applying-node contract says that an identity currently presenting a parse, grammar, quarantine, validation, unit, or resolution condition must be refused and recorded rather than deleted. The live probe found **6 of 35** identities with `validation_status=quarantined`:

| Identity | Name stage | Validation status |
|---|---|---|
| `cross_section_of_flux_surface` | pending | quarantined |
| `line_integrated_electron_density` | drafted | quarantined |
| `magnetic_field_at_pedestal_top_low_field_side_magnitude` | drafted | quarantined |
| `poloidal_straight_field_line_angle` | drafted | quarantined |
| `tendency_of_total_thermal_plasma_internal_energy` | accepted | quarantined |
| `toroidal_line_integrated_impurity_ion_velocity` | drafted | quarantined |

This is not being used as evidence that any row should be deleted. It is the reason the whole exact cohort was refused before signing. The earlier adjudication explicitly records `cross_section_of_flux_surface` as pending and quarantined while assigning it to delete on independent successor evidence. That makes the current node guard and the adjudication materially inconsistent for at least one row; the worker cannot silently choose which authority to weaken.

All 35 targets still satisfy the two live topology measurements that were safe to re-check: **0 producing sources** and **0 live structural children**. Catalog-release provenance was not used to override the quarantine refusal.

## Fail-closed condition 2: generic delete compare-and-set cannot delete these nodes

The live graph reports **3 to 35 relationships per target**, with all **35 of 35** targets connected. The closure includes vocabulary, unit, physics-domain, review, history, component, parent, cluster, and internal-change relationships. The generic signed-manifest delete branch at `imas_codex/standard_names/signed_manifest.py:851` matches only a target satisfying `AND NOT (target)--()` and then requires exactly one changed row. Therefore every current target would make the delete compare-and-set return zero and roll back.

This is not a reason to bypass `apply_signed_manifest` or write a bespoke operator. The repair belongs in that existing generic operator: delete must sign and lock the exact removable relationship closure, preserve the required ledger/history evidence in its receipt, remove the authorized closure atomically, and retain the current manifest/hash/collateral/replay proofs. That code path is outside this node's exclusive write scope.

No preview was promoted into an authority because a 29-row subset would violate the bound-at-35 requirement and a 35-row authority would knowingly include six identities the node contract says to refuse.

## Graph immutability and counters

Every command executed by this node was read-only. The first retained counter census and the final retained counter census agree:

| Counter | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` | 7,706 | 7,706 | 0 |
| `LLMCost` | 27,596 | 27,596 | 0 |

Because no manifest was applied, there is no claimed post-mutation collateral proof and no replay claim. The retained pre-apply probe contains a per-row production snapshot for all 35 targets; the node deliberately does not mislabel an unchanged read-only graph as a successful apply/replay proof.

## Ratchets

The graph-credentialed ratchet module was explicitly selected with `-m graph` and exited **0**: **4 passed, 0 failed, 0 skipped, 0 deselected**. This was a real run, not pytest selection exit 5. It confirms the unchanged production graph remains within all four current ceilings; it does not satisfy the missing deletion measure.

Command:

`UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m graph tests/graph/test_sn_integrity_ratchets.py -q`

## Retained logs

| Artifact | SHA-256 | Headline |
|---|---|---|
| `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T114044597343-orphanapply/preapply-probe.log` | `2fe761daa651fe671cd72e2077efa324476623a1f0366f89191fe6a2184128c6` | 35 rows; zero producers; zero live children; 6 quarantined; 3–35 relationships each |
| `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T114044597343-orphanapply/blocker-census.log` | `b9adb9bafefbe4b40662cb3878b4580e4eee798ad87345e3ecc137b856ef110d` | Global predicate remains the exact original 36; held row present; counters 7,706 / 27,596 |
| `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T114044597343-orphanapply/ratchets.log` | `d56cad41006294fa5ada3adfdd1f34347ef419f0cd4f3153a07cc721cc7134ec` | 4 passed, 0 failed, 0 skipped |
| `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T114044597343-orphanapply/post-blocker-counters.log` | `04fb066925372304b7b6541454c2c925c6e66d1bfbacce59c3fbcf290b788dc3` | Final counters unchanged at 7,706 / 27,596 |

## Required next authority

The coordinator needs two follow-ons before redispatching this node:

1. Resolve whether the six explicitly listed quarantine states are mandatory current refusals or adjudicated incidental state, without using quarantine itself as deletion evidence.
2. Allocate a repair node for `imas_codex/standard_names/signed_manifest.py` and its disposable-graph tests so the existing generic delete mutation can remove a signed complete StandardName closure rather than only an isolated node.

After those land, regenerate the exact 35-row authority from production, require `authority_rows=35`, `admitted=35`, `refused=0`, apply the exact preview hash, verify 35 receipts and unchanged collateral/cost counts, replay as `already_applied` with `changed=0`, and rerun the four ratchets.
