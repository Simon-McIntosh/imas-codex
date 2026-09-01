# Optical-element ordinal identity disposition

## Outcome

`first_local_tangential_back_surface_radius_of_optical_element` was submitted through the sanctioned `sn edit` operator with a mandatory reason that names both the ordinal-index breach and the DD path from which the ordinal arose. The operator reported the plan-authorized **vocabulary-gap** outcome: the exact non-ordinal proposal `local_tangential_back_surface_radius_of_optical_element` fails the public ISN grammar round-trip because `local_tangential_back_surface_radius` matches no physical base or geometry carrier. The nearest grammar vocabulary remains ordinal: `first_local_tangential_coordinate` and `second_local_tangential_coordinate`.

No non-ordinal replacement was produced. No nearest-object, vertical-direction, or other semantic substitution was made. The original identity remains visible and unchanged so the vocabulary gap cannot be mistaken for a completed rename.

## DD provenance and the enumerated index

The ordinal derives from:

```text
spectrometer_visible/channel/optical_element/back_surface/x1_curvature
```

The `x1` leaf enumerates the first of the optical element back surface's two local tangential directions. The corresponding `x2` leaf is:

```text
spectrometer_visible/channel/optical_element/back_surface/x2_curvature
```

Both DD leaves carry unit `m`. The live source census found the X1 source as `dd:spectrometer_visible/channel/optical_element/back_surface/x1_curvature`, currently `status=extracted` and unbound. This is consistent with the plan's classification of the name as ungrounded, while still identifying what the forbidden `first` was enumerating.

## Dry-run gate

An initial dry run tested whether the accepted identity could be steered with `--hint`. The operator refused that mode without writing because an accepted name cannot re-enter name generation from a hint and directed the caller to use a complete `--rename` proposal. That refusal was inspected and no live hint invocation followed.

The final proposed spelling was then previewed with this exact dry-run invocation:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
uv run --no-sync imas-codex sn edit \
  first_local_tangential_back_surface_radius_of_optical_element \
  --rename local_tangential_back_surface_radius_of_optical_element \
  --reason "DD path spectrometer_visible/channel/optical_element/back_surface/x1_curvature uses x1 to enumerate the first local tangential surface direction. The identity carries first as an ordinal-index breach; ordered sample positions belong in DD provenance, so replace it with the non-ordinal local-tangential back-surface radius identity or report the exact vocabulary gap." \
  --scope self --cost-limit 10 --dry-run
```

The dry run exited 2 with the deterministic grammar refusal:

```text
new name fails ISN grammar round-trip: parse failed: ParseError: residue
'local_tangential_back_surface_radius' does not match any physical_base or
geometry_carrier; nearest candidates: ['first_local_tangential_coordinate',
'second_local_tangential_coordinate']
```

This output was inspected before any live invocation. It established that the exact semantic replacement is unavailable in the current public vocabulary and that the expected live outcome was a vocabulary gap rather than a reviewed successor.

## Sanctioned live invocation

At `2026-09-01T16:08:57Z`, the same validated edit was issued live, removing only `--dry-run`:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
uv run --no-sync imas-codex sn edit \
  first_local_tangential_back_surface_radius_of_optical_element \
  --rename local_tangential_back_surface_radius_of_optical_element \
  --reason "DD path spectrometer_visible/channel/optical_element/back_surface/x1_curvature uses x1 to enumerate the first local tangential surface direction. The identity carries first as an ordinal-index breach; ordered sample positions belong in DD provenance, so replace it with the non-ordinal local-tangential back-surface radius identity or report the exact vocabulary gap." \
  --scope self --cost-limit 10
```

The live command exited 2 with the same grammar-round-trip vocabulary gap, before review, persistence, or source migration. This is a successful disposition under the node's two allowed outcomes: the sanctioned path proved that no exact non-ordinal identity exists in the current grammar.

## Graph and no-hand-edit verification

The post-invocation verification used the schema-owned `StandardName.id` key and first proved coverage: **4,675 of 4,675** `StandardName` nodes carry `id`. It then queried the original, proposed, and sibling identities explicitly.

Results:

- Original: `first_local_tangential_back_surface_radius_of_optical_element` still exists with `name_stage=accepted`, `docs_stage=reviewed`, `validation_status=valid`, `origin=pipeline`, and `reviewer_score_name=0.9625`.
- Its provenance remains the pre-existing `run_id=sn-rescore-20260901T135204Z`.
- `edit_mode`, `edit_status`, `edit_reason`, and `edit_run_id` all remain null on the original.
- Proposed non-ordinal node `local_tangential_back_surface_radius_of_optical_element`: **0** nodes.
- New `sn-edit-*` runs since the live invocation began: **0**, against **543 of 543** `SNRun` rows with both `id` and `started_at` covered.

These facts establish that the identity was not hand-edited: the graph carries neither a replacement node nor changed edit metadata, and the only attempted mutation was the recorded `sn edit` command, which failed closed at grammar validation before it could create an edit run.

The source instrument also proved **9,678 of 9,678** `StandardNameSource` rows carry both `id` and `source_id`. The X1 source remains extracted and unbound; no source was manually attached or migrated.

## Second-local sibling disposition

The exact sibling `second_local_tangential_back_surface_radius_of_optical_element` does not exist: the post-invocation graph query returned **0** nodes. It therefore could not receive the same edit treatment and is **out of this node's scope** rather than silently omitted.

The DD-side X2 enumerand does exist as `dd:spectrometer_visible/channel/optical_element/back_surface/x2_curvature`; it is `status=composed` and bound to the accepted identity `vertical_back_surface_curvature_of_optical_element`. Altering that distinct accepted identity or its source binding was not authorized by the live plan's targeted instruction for the X1 ordinal name and would require separate semantic adjudication.

## Cost evidence

The post-invocation `LLMCost` query covered **35,180 of 35,180** rows on `id`, `llm_at`, and `standard_name_ids`, then selected calls at or after `2026-09-01T16:08:57Z` attributable to either the original or proposed identity.

- Attributable LLM calls: **0**
- Attributable run IDs: **0**
- Exact attributable spend: **USD 0.000000**
- Authorized ceiling: **USD 10.000000**
- Ceiling consumed: **0.0%**

The zero spend is expected and verified: grammar validation rejected the proposal before the inline review pool made a model call.

## Final disposition

The optical-element ordinal identity remains visible and unchanged under a named **ISN vocabulary gap**. Disposing it as a non-ordinal identity requires an upstream grammar decision that exposes a non-ordinal local-tangential surface-direction carrier, after which the same sanctioned `sn edit --rename` path can be retried. Codex must not invent that token or substitute the X2/vertical identity.
