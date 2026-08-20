NEEDS-HELP: the exact owner/geometry apply is blocked by a live attachment-guard conflict for the signed reflector row; the one transaction rolled back before commit.

tried: Verified the committed 49-row authority at file SHA-256 `dbb37f7be12ba99d7e85bf13b9d63e6c19cb6c20bd35fe687e590f798e2dc85b` and canonical `jq -cS '.rows'` SHA-256 `4de9c2df481180931a47b7a8bcc76cb69253e23d96e2dfa151bd86edcb76c8cd`; derived the exact 12 write-required ready rows in the production invocation; admitted all 12 with zero preflight refusals; rebuilt the locked transaction manifest and matched it to the preview only by canonical SHA-256 (`e258e1fa99dc72d5518faaba5198c35159df0142dc007d305d4fd2e8576a866e`); then attempted six grouped compare-and-set migrations in one transaction. The attachment guard refused `dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi` → `toroidal_coordinate_of_reflector` because `dd:spectrometer_x_ray_crystal/channel/reflector/sphere_centre/phi` already produces that identity and the guard classifies the two leaves as distinct vector fields. The exception occurred before commit and the transaction rollback path executed. A later independent read was attempted but could not authenticate to Neo4j, so current counters could not be re-read after rollback; the last successful pre-apply read was StandardNameChange 7,496 and LLMCost 27,477 with zero declared receipts.

options: (1) Extend the attachment rule, with focused tests and disposable-graph proof, so an owner's `centre` and parameterized `sphere_centre` named points may share the owner-qualified coordinate identity when the locked geometry-cardinality rule says shape parameterization is provenance rather than identity; then regenerate and apply the same signed 12-row cohort. (2) Re-adjudicate the reflector row to a distinct grammar-valid identity, take it through ordinary review, re-sign the 49-row partition, and apply the resulting exact cohort. (3) Explicitly authorize an 11-row subset and record the reflector as a refusal; this would not meet this node's zero-refusal done-when and would require a new signed subset authority.

leaning: Option 1, because the live plan's locked geometry-cardinality rule says a named point shares its owner's coordinate identity regardless of shape parameterization, while both leaves identify points of the same reflector and the accepted target is explicitly owner-qualified. That conclusion still needs a scoped guard owner to inspect the DD cardinality and encode the rule; this worker may not change `attachment_audit.py` or its tests.

cost-if-wrong: No production rollback is required because the attempted transaction did not commit. A wrong semantic choice would require revising or re-signing the authority partition, rerunning ordinary review if a new identity is introduced, and rerunning the serialized exact apply and all closure/counter/replay proofs.

## Fail-closed outcome

The write boundary was not crossed. Both invocations failed before a graph
commit:

- The first stopped before graph access because Python's compact JSON hash
  omitted the newline emitted by the artifact's declared `jq -cS` method. The
  corrected check reproduces the signed digest exactly.
- The second passed the artifact, partition, live-state, accepted-valid-target,
  counter, and canonical-manifest gates. It failed during the transaction's
  attachment validation, and the encompassing transaction was rolled back.

The exact write cohort contains 12 sources across six surviving targets:

| Signed surviving target | Source rows |
|---|---:|
| `toroidal_coordinate_of_aperture` | 5 |
| `toroidal_coordinate_of_filter_window` | 3 |
| `toroidal_angle_of_coil_conductor_element` | 1 |
| `toroidal_coordinate_of_line_of_sight` | 1 |
| `toroidal_coordinate_of_optical_element` | 1 |
| `toroidal_coordinate_of_reflector` | 1 |
| **Total admitted before guard execution** | **12** |

The blocking row is semantically concrete: the source description binds the
toroidal coordinate at `spectrometer_x_ray_crystal/channel/reflector/centre/phi`
to the accepted-and-valid owner identity `toroidal_coordinate_of_reflector`.
The guard's comparison source is
`spectrometer_x_ray_crystal/channel/reflector/sphere_centre/phi`, already bound
to the same identity. The unresolved question is whether those two named-point
representations are one owner coordinate under the locked cardinality rule or
genuinely distinct vector fields.

## Evidence state

| Gate | Result |
|---|---|
| Signed authority | PASS: 49 rows; 21 ready; 9 already selected; 12 require write |
| Exact live preflight | PASS: 12 admitted, 0 refused |
| Locked transaction preflight | PASS: 12 admitted, 0 refused |
| Manifest comparison | PASS: preview and locked transaction canonical SHA-256 both `e258e1fa…` |
| Pre-apply counters | StandardNameChange 7,496; LLMCost 27,477; declared receipt rows 0 |
| Attachment guard | REFUSED: 1 reflector row |
| Transaction | ROLLED BACK before commit |
| Apply/replay/closure done-when | NOT PRODUCED; no partial 11-row apply was permitted |

Run artifacts:

- `production-owner-geometry-apply.log` — complete gate stream and refusal
  traceback.
- `owner-geometry-preview.json` — derived 12-row manifest and six grouped
  receipt identities.
- `owner-geometry-baseline.json` — pre-apply counters, authority census, and
  9,527 out-of-allowlist row digests captured before the rolled-back attempt.
- `run_owner_geometry_apply.py` — single-invocation preview/apply/replay driver;
  retained for audit and correction, not rerun after the two-failure stop.

All run artifacts are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T130012050116-sgwi-owner-geometry-ready-apply/`.
