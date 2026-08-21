# Owner and geometry residue adjudication

## Outcome

A fresh read of the complete signed 49-row owner/geometry authority found
**5 live residual rows** still producing
`toroidal_angle_of_measurement_position`. This artifact dispositions all five:
**3 exact migrations + 2 named refusals = 5 dispositioned**, with
**0 undispositioned**.

The three migration targets already exist and are both `accepted` and `valid`.
The two refusals preserve semantic distinctions that the current graph cannot
yet encode safely. A refusal does not endorse the old measurement-position
binding; it withholds mutation until an exact, reviewed identity exists.

| Measure | Result |
|---|---:|
| Signed authority rows re-read | 49 |
| Live residual rows derived in this invocation | **5** |
| Exact migrations | **3** |
| Named refusals | **2** |
| Dispositioned | **5** |
| Undispositioned | **0** |
| Configured DD authority | **4.1.1** |
| `StandardNameChange` | **7,756 → 7,756** |
| Production graph mutations | **0** |

## Exact row dispositions

Every row currently has one live target and the matching scalar mirror,
`toroidal_angle_of_measurement_position`. “Current owner” is the physical or
geometric object named by the DD path and its parent documentation, not the
generic incumbent Standard Name.

| DD source | Current owner | Authoritative geometry identity | Exact migration target or verbatim refusal reason |
|---|---|---|---|
| `dd:b_field_non_axisymmetric/time_slice/field_map/grid/phi` | 3D grid of the non-axisymmetric error-field map | Toroidal grid coordinate of the 3D error-field-map discretization | **Named refusal:** `REFUSE_FIELD_MAP_GRID_RETARGET: DD 4.1.1 defines the owner as the '3D grid' of the 'error field map'; field_map_grid is a discretization artifact, not a physics-meaningful Standard Name locus, and the pinned closed grammar rejects toroidal_coordinate_of_field_map_grid. Do not substitute a measurement-position or nearest flux-coordinate identity.` |
| `dd:spectrometer_visible/channel/active_spatial_resolution/centre/phi` | Active spatial resolution zone | Toroidal coordinate of the active spatial resolution zone center | **Migrate exactly to** `toroidal_coordinate_of_active_spatial_resolution_zone` — accepted, valid, name-review score 0.99375. |
| `dd:spectrometer_visible/channel/detector/centre/phi` | Detector | Toroidal coordinate of the detector center | **Migrate exactly to** `toroidal_coordinate_of_detector` — accepted, valid, name-review score 1.00000. |
| `dd:spectrometer_visible/channel/polarizer/centre/phi` | Polarizer | Toroidal coordinate of the polarizer center | **Migrate exactly to** `toroidal_coordinate_of_polarizer` — accepted, valid, name-review score 1.00000. |
| `dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi` | Reflector surface center and local-coordinate origin | Toroidal coordinate of the reflector surface center | **Named refusal:** `REFUSE_REFLECTOR_CENTRE_COLLAPSE: DD 4.1.1 distinguishes reflector/centre as the local-coordinate origin and middle point of the object surface from reflector/sphere_centre as the center of the sphere defining a spherical mirror. The accepted toroidal_coordinate_of_reflector is currently defined and sourced as the sphere center of curvature, so migrating reflector/centre to it would collapse two distinct points. Require an ordinary-reviewed identity split before either binding moves.` |

These are exact migration destinations, not fold authority. A later applying
node must derive the three-row cohort again from live state, use the signed
source-migration operator, and preserve the scalar, live edge, backing
projection, last-producer, and out-of-allowlist closure guards.

## DD 4.1.1 owner evidence

### Error-field grid: policy refusal, not nearest-name substitution

The graph's configured DD version is 4.1.1. Its exact parent documentation is:

> `b_field_non_axisymmetric/time_slice/field_map`: “Description of the error field map”

> `b_field_non_axisymmetric/time_slice/field_map/grid`: “3D grid”

> `b_field_non_axisymmetric/time_slice/field_map/grid/phi`: “Toroidal angle (oriented counter-clockwise when viewing from above)” (`FLT_1D`)

The `phi` values therefore locate a grid in a field-map discretization; they do
not locate a measurement. The pinned closed grammar deliberately has no
`field_map_grid` locus and rejects
`toroidal_coordinate_of_field_map_grid`. The refusal records that absence
without substituting either the incumbent measurement-position identity or a
nearby flux-coordinate identity.

### Reflector: surface center and sphere center are different points

DD 4.1.1 states:

> `spectrometer_x_ray_crystal/channel/reflector/centre`: “Coordinates of the origin of the local coordinate system (X1,X2,X3) describing the object. This origin is located within the object area and should be the middle point of the object surface. If geometry_type=2, it's the centre of the circular object. If geometry_type=3, it's the centre of the rectangular object.”

> `spectrometer_x_ray_crystal/channel/reflector/sphere_centre`: “Position of the center of the sphere which defines the mirror, in the case of a spherical mirror (geometry_type = 2), derived from the above geometric data”

The live graph corroborates the conflict. The sibling source
`dd:spectrometer_x_ray_crystal/channel/reflector/sphere_centre/phi` exclusively
produces `toroidal_coordinate_of_reflector`, whose accepted description is
“Toroidal angular coordinate of the center of the sphere defining a
reflector’s spherical mirror surface, locating that center around the symmetry
axis.” The same identity cannot also receive `reflector/centre/phi`: one is the
surface/local-frame center and the other is the center of curvature of the
defining sphere. The earlier `distinct-vector conflict` was therefore a correct
fail-closed guard result.

Resolution is a governed identity split through ordinary composition and
review. This evidence does not invent the missing spelling or move the sibling
binding. Until that review lands, `reflector/centre/phi` remains a named
refusal, while `sphere_centre/phi` remains unchanged.

### Three exact owners

The other three parent descriptions make their owners explicit:

- `active_spatial_resolution/centre`: “Position of the centre of the spatially resolved zone.”
- `detector`: “Detector description”; its `centre` is the circle center or the
  origin inside the detector/aperture area, depending on geometry type.
- `polarizer`: “Polarizer description”; its `centre` is the circle center or
  local-coordinate origin inside the component area, depending on geometry
  type.

Each accepted target repeats that exact owner in its reviewed description. No
diagnostic/channel provenance, geometry parameterization, or ordinal position
enters the identity.

## Read-only derivation and evidence

The derivation loaded all 49 rows from the committed signed authority, queried
their current `StandardNameSource` scalar and complete live
`PRODUCED_NAME` target set, and selected only rows still carrying the old
target. It then re-read the DD context, target lifecycle, accepted target
descriptions, and reflector sibling binding. Its disposition table is exact-set
checked: any live residual missing from the five enumerated decisions, or any
enumerated decision absent from the live residual, aborts rather than emitting
evidence.

- Source checkout: `c81913747f78ead2c3bf57ab0ea0208543105d94`.
- Machine result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194817469646-ownerresidue/owner-residue.json`
  (SHA-256
  `606ba8b168358d2a245bc9aafeab5c04286d78cbcac6d0bd1639267548fe9633`).
- Complete diagnostic log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194817469646-ownerresidue/owner-residue.log`
  (SHA-256
  `bc8f9d896c7723c7cd7c9e85072d86b3c0a6404b86bd935f07e6615aab9f5fad`).
- The query path used `GraphClient` and read-only `MATCH` statements only. It
  made no provider call and invoked no repair operator.

The first and last live counter reads were identical:
`StandardNameChange` **7,756 before and 7,756 after**. The artifact therefore
proves that adjudication produced semantic authority only and performed zero
production graph mutation.
