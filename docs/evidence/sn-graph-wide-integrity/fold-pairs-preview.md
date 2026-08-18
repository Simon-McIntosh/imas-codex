# Fold-into-ancestor candidate previews

Date: 2026-08-19

## Outcome

The 10 census rows resolve to six unique proposed name folds. Read-only live
adjudication confirmed that every proposal points from the descendant to an
existing `REFINED_FROM` ancestor and that none is self-directed or
reverse-reachable. The landed strict operator admitted only one unique fold,
covering one census row. It refused the other five unique folds, covering nine
census rows, before producing apply authority.

No apply was attempted. `StandardNameChange` remained **7,152 → 7,152**;
`LLMCost` remained **27,467 → 27,467**; and `SNRun` remained **489 → 489**.
Provider calls were **0**.

The input was the exact `fold_into_ancestor_candidate_pairs` class in
`dual-binding-census.json`, whose SHA-256 is
`3e83656f18bf2094ceff95f4bf9f66f8e4832679f2f93377848af19c2a998809`.
The census is candidate evidence, not mutation authority: each unique name
fold was re-read from the live graph and passed independently through
`supersede_into_ancestor(..., apply=False)`.

## One adjudication per census row

Lifecycle fields are shown as
`name_stage / validation_status / reviewer_score_name`.
Repeated source rows sharing one name pair deliberately share one operator
preview: the instrument folds the complete descendant identity and all of its
sources, not an individual DD edge.

| # | DD source | Fold source (descendant) → target (ancestor) | Lineage | Lifecycle and scalar | Instrument result |
|---:|---|---|---|---|---|
| 1 | `dd:camera_ir/channel/camera/direction/x` | `x_image_up_unit_vector_of_camera` → `x_direction_unit_vector_of_camera` | 1 hop; no reverse path | descendant `accepted / valid / 0.9875`; ancestor `accepted / valid / 0.98125`; scalar selects ancestor | **Fold-eligible.** Signed hash `fc12f9ff4e51ac4658d8a3aa68576661f41e30b4fb99bbb3f259624ae5d83d17`; 1 source = 0 retarget + 1 deduplicate + 0 stale detach + 0 stale refusal (`preview-01.log`). |
| 2 | `dd:edge_profiles/ggd/neutral/velocity/phi` | `toroidal_neutral_momentum_convection_velocity` → `toroidal_neutral_velocity` | 1 hop; no reverse path | descendant `accepted / valid / 0.88125`; ancestor `accepted / valid / 0.98125`; scalar selects ancestor | **Refused.** Complete descendant source set cannot migrate exactly: `dd:edge_transport/model/ggd/neutral/momentum/v/phi` also binds `toroidal_momentum_convection_velocity` and the descendant, with its scalar selecting the descendant (`preview-02.log`). |
| 3 | `dd:edge_sources/source/ggd/ion/momentum/r` | `radial_ion_momentum_source` → `radial_ion_momentum` | 1 hop; no reverse path | descendant `accepted / valid / 0.9`; ancestor `reviewed / valid / 0.675`; scalar selects ancestor | **Refused:** `ancestor 'radial_ion_momentum' is not an accepted live name` (`preview-03.log`). |
| 4 | `dd:plasma_profiles/ggd/neutral/velocity/phi` | `toroidal_neutral_momentum_convection_velocity` → `toroidal_neutral_velocity` | 1 hop; no reverse path | descendant `accepted / valid / 0.88125`; ancestor `accepted / valid / 0.98125`; scalar selects ancestor | **Refused** by the same complete-source closure as row 2 (`preview-02.log`). |
| 5 | `dd:plasma_sources/source/ggd/ion/momentum/radial` | `radial_ion_momentum_source` → `radial_ion_momentum` | 1 hop; no reverse path | descendant `accepted / valid / 0.9`; ancestor `reviewed / valid / 0.675`; scalar selects ancestor | **Refused** because the ancestor is not accepted (`preview-03.log`). |
| 6 | `dd:plasma_sources/source/profiles_1d/ion/momentum/radial` | `radial_ion_momentum_source` → `radial_ion_momentum` | 1 hop; no reverse path | descendant `accepted / valid / 0.9`; ancestor `reviewed / valid / 0.675`; scalar selects ancestor | **Refused** because the ancestor is not accepted (`preview-03.log`). |
| 7 | `dd:plasma_transport/model/ggd/momentum/flux/radial` | `radial_momentum` → `radial_momentum_flux` | 2 hops through `radial_neutral_momentum_flux`; no reverse path | descendant `reviewed / valid / 0.50625`; ancestor `reviewed / valid / 0.93125`; scalar selects ancestor | **Refused:** `ancestor 'radial_momentum_flux' is not an accepted live name` (`preview-04.log`). |
| 8 | `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/poloidal` | `poloidal_linear_neutral_internal_state_momentum_flux` → `poloidal_neutral_state_momentum_flux` | 3 hops through `poloidal_neutral_internal_state_linear_momentum_flux` and `poloidal_neutral_internal_state_momentum_flux`; no reverse path | descendant `accepted / valid / 0.9125`; ancestor `reviewed / valid / 0.9`; scalar selects ancestor | **Refused:** `ancestor 'poloidal_neutral_state_momentum_flux' is not an accepted live name` (`preview-05.log`). |
| 9 | `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` | `radial_neutral_internal_state_momentum_flux` → `radial_neutral_state_momentum_flux` | 1 hop; no reverse path | descendant `accepted / valid / 0.95625`; ancestor `accepted / valid / 0.85625`; scalar selects the descendant | **Refused.** The source already binds both identities while its scalar selects the descendant, a shape the exact fold cannot migrate (`preview-06.log`). |
| 10 | `dd:plasma_transport/model/profiles_1d/neutral/state/momentum/flux/poloidal` | `poloidal_linear_neutral_internal_state_momentum_flux` → `poloidal_neutral_state_momentum_flux` | same strict 3-hop path as row 8 | descendant `accepted / valid / 0.9125`; ancestor `reviewed / valid / 0.9`; scalar selects ancestor | **Refused** because the ancestor is not accepted (`preview-05.log`). |

Thus the row accounting is exact: **10 = 1 fold-eligible + 9 refused**. The
unique-name accounting is **6 = 1 fold-eligible + 5 refused**.

## Semantic checks behind the structural result

The eligible camera row is semantically aligned with the source path. The
descendant describes the x component of the camera's **image-up orientation**,
whereas the ancestor describes the x component of its **line-of-sight
direction**. The DD path is `camera/direction/x`, its scalar already selects
`x_direction_unit_vector_of_camera`, and the signed operation removes the one
redundant image-up binding without retargeting a source.

The refusals preserve distinctions that cannot be erased merely because a
lineage path exists:

- `toroidal_neutral_momentum_convection_velocity` is an effective velocity
  convecting aggregate neutral momentum; `toroidal_neutral_velocity` is mean
  translational neutral flow. A third live identity in the descendant's full
  source closure makes a global fold unsafe.
- `radial_ion_momentum_source` describes a force-per-area source term, while
  `radial_ion_momentum` is documented as ion momentum flux. Besides the
  unaccepted ancestor, the source-versus-flux distinction needs semantic
  adjudication rather than lifecycle promotion.
- The reviewed momentum-flux ancestors have not earned accepted lifecycle
  authority. They must pass ordinary review on their own merits; this preview
  does not hand-promote them.
- The radial neutral-state row's scalar still selects the more specific
  internal-state descendant. The exact instrument correctly refuses to infer
  that the scalar should be rewritten.

## Signed logs

All paths below are in the durable worker run envelope
`r-20260818T234327759313-fold-pairs-preview`.

| Log | SHA-256 | Contents |
|---|---|---|
| `ancestry-adjudication.log` | `d6ac8f17e83a33b3834b43097e031f3b27e0e0faf3aa3df9ac1e0cae895b729f` | Ten live source rows, both name identities and lifecycle properties, directed paths, reverse-reachability checks, descriptions, scores, bindings, and scalars. |
| `preview-01.log` | `a1fdcb3a57174a524075be6248b92a9d27d11285b2ed3b8c8cde537681ca3f02` | Complete signed eligible camera manifest. |
| `preview-02.log` | `d7c15cb05b282ea77cd3d84d8e829ca48f353573916170713d69fbb86e89119b` | Toroidal-neutral complete-source refusal. |
| `preview-03.log` | `e32e23bb5fc4cc917888c6bac279e445b4c9932098c757ce44f8f054d16f6fa2` | Radial-ion unaccepted-ancestor refusal. |
| `preview-04.log` | `d992b4a1842976594f771ca984810fba8e9823a211a96e3ad88cb5907225204b` | Radial-momentum-flux unaccepted-ancestor refusal. |
| `preview-05.log` | `8baa74f94ea66903c0fdb8ae9117f774242df1cd59c69b2a34ede2803f399785` | Poloidal-neutral-state unaccepted-ancestor refusal. |
| `preview-06.log` | `8621d03d50f9e97d8ba1344aaad7630bffba1f3cd7c636f7c5cc273a2cb97ca3` | Radial-neutral scalar/binding-shape refusal. |
| `write-counters.log` | `cdd666f26a4688aabad31e248438196a993b217c84c2801f3f497ad8bfb33059` | Flat before/after graph counters proving zero persistent writes. |

## Recommended apply order

1. Independently review and, if approved, apply only
   `x_image_up_unit_vector_of_camera` into
   `x_direction_unit_vector_of_camera` using manifest
   `fc12f9ff4e51ac4658d8a3aa68576661f41e30b4fb99bbb3f259624ae5d83d17`.
   Regenerate the preview immediately before apply and require an exact hash
   match.
2. Do **not** place the other five unique folds in an apply queue. First
   adjudicate the two complete-source/scalar conflicts and route the three
   reviewed ancestors through ordinary lifecycle review without hand
   acceptance. Any changed state requires a new signed preview and review.

This order keeps the one currently exact fold independent and prevents a
structural census heuristic from authorizing semantically lossy or
lifecycle-ineligible mutations.
