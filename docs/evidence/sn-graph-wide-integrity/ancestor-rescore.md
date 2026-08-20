# Ancestor rescore receipt

Status: **blocked by the 45-minute execution fence**.

The sanctioned `rescore_name` backend was invoked once per identity with the
ordinary name-review pool and `rotation_cap=0`. This prohibited refinement,
renaming, and automatic retry. Two draws completed; the third was interrupted
at the hard time fence after staging but before a quorum result.

| Identity | Before stage / score | After stage / score | Completed run | Attributable pipeline cost | Outcome |
|---|---:|---:|---|---:|---|
| `radial_ion_momentum` | reviewed / 0.675 | reviewed / 0.5875 | `sn-rescore-20260820T141241Z` | $0.240110 | Fresh authoritative-escalation result did not clear; not retried |
| `radial_momentum_flux` | reviewed / 0.93125 | accepted / 0.95625 | `sn-rescore-20260820T142320Z` | $0.429416 | Accepted only through the fresh quorum-consensus score |
| `poloidal_neutral_state_momentum_flux` | reviewed / 0.9 | drafted / none | `sn-rescore-20260820T144208Z` (interrupted) | Not fully receipted | No completed quorum result; unclaimed after interruption |

The completed pipeline outcomes identify `openrouter/x-ai/grok-4.5` as the
canonical name reviewer. The exact per-cycle reviewer-model set and LLMCost
rows were not collected because the third invocation had to be interrupted;
therefore the node's complete evidence condition is not met. Known completed
pipeline cost is $0.669526, below the $150 authorization, but this is not a
final attributable-cost receipt for all three identities.

No completed identity was accepted outside a fresh quorum score.
`radial_ion_momentum` is explicitly preserved as non-clearing and was not
retried. `poloidal_neutral_state_momentum_flux` remains drafted under run
`sn-rescore-20260820T144208Z`; a serialized continuation must complete that
existing staged run or deliberately recover it under fresh authority. It must
not replay either completed identity.

The full captured operation is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T140516310593-sgwi-ancestor-rescore/ancestor-rescore-run.log`.
The process exited 130 after the time-fence interrupt. A post-interrupt live
read confirmed all three identities were unclaimed; their stages were reviewed,
accepted, and drafted respectively.
