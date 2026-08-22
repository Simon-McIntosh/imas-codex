# Residual Standard Name integrity reuse map

## Outcome

At dispatched revision `328a0816bfbbb3ee48c7fb376c751726eac15d8c`, all six requested capabilities have a concrete repository entry point. Four are fit to reuse as-is, while two should be extended at their existing seam. No capability needs a separate mutation framework.

The live plan is decisive about sequencing and authority: release the last structural child before superseding the umbrella; treat reverse-search candidates as evidence rather than attachment authority; keep the extracted-source census read-only and partitioned by lifecycle; and rerun the established integrated verification at the new HEAD.

## One row per required capability

| Capability | Repository path | Invocable symbol or CLI command | Fitness verdict | One-line fitness rationale |
|---|---|---|---|---|
| Parent release | `imas_codex/standard_names/signed_manifest.py:488` | `apply_signed_manifest(authority_path, authority_file_sha256=..., authority_payload_sha256=..., reason=..., apply=False/True, manifest_sha256=...)` | `reuse-as-is` | The closed structural-release program validates exactly one `HAS_PARENT` removal, locks and re-hashes its participants, refuses a mixed structural authority, preserves lifecycle plus the complete producer closure, verifies the parentless postcondition, writes a receipt, and replays write-free. |
| Identity supersede | `imas_codex/standard_names/signed_manifest.py:3944` | `apply_signed_manifest(...)` with a builder-emitted `RepairMutationKind.supersede` row | `reuse-as-is` | This exact signed-envelope path already superseded five legacy spellings and refused `area_of_flux_surface` while it had live children; after parent release makes that count zero, the unchanged authority can be regenerated and applied. |
| Source attachment | `imas_codex/standard_names/graph_ops.py:10254` | `persist_claimed_attachments(attachments)` | `extend` | The kernel already atomically writes source status/scalar, `PRODUCED_NAME`, backing `HAS_STANDARD_NAME`, and target `source_paths`, but it is claim-fenced pipeline persistence rather than independently signed standalone authority; add an unbound ordinary-source attachment program to `apply_signed_manifest` and reuse this write shape plus `guard_source_pairings`. |
| Unsourced-name census | `imas_codex/standard_names/ledger.py:96` | `find_provenance_orphans(gc=graph_client)` | `reuse-as-is` | It is read-only and returns every materialized live, non-error-sibling name lacking an incoming `PRODUCED_NAME` source with identity, stage, and origin, which is the residual unsourced-name cohort the plan must remeasure after each attachment/supersede. |
| Dual-bound census | `imas_codex/standard_names/provenance_lifecycle.py:1002` | `find_semantic_source_invariant_violations(graph_client)` filtered to `len(row["live_targets"]) > 1` | `reuse-as-is` | It already returns the exact composed/attached source identities, complete live target sets, scalar, and upstream projections; the stated filter yields the current dual-bound cohort without mutation or a second query definition. |
| Integrated verification | `docs/evidence/sn-graph-wide-integrity/final-integration-verification.md:13` | `uv run --no-sync pytest -p no:cacheprovider tests/standard_names tests/units` followed by `uv run --no-sync ruff check --no-cache .` and `uv run --no-sync ruff format --check --no-cache .` | `extend` | Reuse the established full-suite, units, generated-model shadow-warning, authority-digest, and Ruff recipe, but rerun it at this HEAD and explicitly record the parent-release suite; the existing evidence is a prior-revision receipt, not current verification. |

## Application boundary

- Parent release and identity supersede remain two single-kind signed invocations. The release authority must be applied and its parent/producer/lifecycle postconditions proved before a fresh supersede authority is derived.
- `persist_claimed_attachments` is implementation evidence, not permission to bypass the signed envelope. The new closed program needs exact-set admission from `guard_source_pairings`, signed accepted-target and DD/version participants, compare-and-set refusal on an already-bound or drifting source, the same four-mirror write, immutable receipt identity, and write-free replay.
- `find_provenance_orphans` counts all materialized live unsourced identities. If the closure report also needs the narrower “no live structural child” partition, join its returned IDs to the live-child predicate already pinned in `tests/graph/test_sn_integrity_ratchets.py`; do not silently change the ledger definition.
- `find_semantic_source_invariant_violations` deliberately includes zero-target, scalar, and projection defects as well as dual-bound rows. Filtering on more than one live target is a read-only projection of its complete result, not a different authority rule.
- The integrated verification row is intentionally `extend`: tests and lint are already invocable, but only a fresh run can establish the plan's required 0 failures, 0 errors, 0 generated-model shadow warnings, four intact committed authority digests, and both Ruff gates after the new release implementation.

## Quantitative anchor check

The captured resolver checked **6 capability rows / 6 repository paths / 6 invocable anchors** at revision `328a0816bfbbb3ee48c7fb376c751726eac15d8c`: **6 resolved, 0 unresolved**. See `logs/anchor-resolution.log` in the worker delivery directory.
