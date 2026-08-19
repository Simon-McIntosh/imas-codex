# ISN entity-typed locus-trigger family adjudication

## Recommendation

Decline global adoption of the stricter ISN `entity`-typed locus trigger. Adopt
it only for the `aperture`, `bragg_crystal`, and `passive_structure` locus
families. Retain the shipped hardware-word trigger for the other nine families.

The current strict trigger would newly refuse 134 source/name pairings in 12
families. Only six pairings in three families support a family-wide refusal.
The remaining 128 pairings span nine families where DD structure legitimately
expresses the entity through a concrete subtype, a geometry representation, an
IDS owner, or runtime attachment context without repeating the full ISN locus
token. Several declined families contain real defects, but they also contain
proven good pairings; those defects require semantic row repair rather than a
family-wide lexical gate.

The complete signed recommendation and every measured pairing are in
[`locus-rule-family-adjudication.json`](locus-rule-family-adjudication.json).
The JSON file SHA-256 is
`77b4d9367f6caa117e134c3d34b7fc75db58fc63f3400b028b4d40a11368ac34`.
Its signed `measurements` plus `families` payload SHA-256 is
`9c95341f47a9c4ca6a4ffef2043272a8f329ad44686e53fb4ff9ca4b8f46b574`.

## Live measurement

The read reproduced the attachment audit's graph input exactly: every current
`StandardNameSource-[:PRODUCED_NAME]->StandardName` pairing with its backing
`IMASNode`, including attachment edges that still point to a historical name
stage. Keeping those historical edges is necessary for an apples-to-apples
comparison with the earlier audit figures. The manifest marks every pairing's
name stage; the only all-historical family is `coordinate_system`, and eight of
the 50 `diagnostic_aperture` pairings are historical.

Both triggers start from the same mismatch: the parsed trailing locus and the
expanded DD path share no content token.

- The shipped trigger additionally requires a configured hardware word in the
  locus and requires ISN not to type the locus as a place.
- The strict trigger instead fires for every complete locus token that ISN
  types as `entity`.
- The family key is the complete parsed ISN locus token, not an individual
  Standard Name spelling.

| Measure | Historic | Current live read | Change |
|---|---:|---:|---:|
| Attachment rows read | — | 5,515 | — |
| Shipped-trigger flags | 17 | 16 | −1 |
| Strict entity-trigger flags | 189 | 150 | −39 |
| Strict minus shipped | 172 | 134 | −38 |
| Delta families | — | 12 | — |

The historical `~172` estimate is therefore replaced by an exact current delta
of 134. The delta is a set difference over exact
`(locus, DD path, Standard Name)` triples; the shipped set contains no pairing
outside the strict set in this read.

## Family verdicts

`Adopt` means a later implementation may add that exact locus token to a
selective entity-trigger policy. `Decline` means retain the shipped trigger for
the family; it does not assert that every current pairing is correct.

| ISN entity locus family | Newly refused pairings | Stage counts | Verdict | Deciding live evidence |
|---|---:|---|---|---|
| `aperture` | 4 | 3 accepted, 1 reviewed | **Adopt** | All four DD paths name a different immediate owner: detector, active spatial-resolution zone, optical element, or camera. For example, `spectrometer_uv/channel/detector/centre/phi` is paired with `toroidal_coordinate_of_aperture`, although the owner/geometry authority record assigns the detector owner; `spectrometer_visible/channel/optical_element/geometry/radius` is paired with `radius_of_aperture`, although the path asserts an optical element. |
| `bragg_crystal` | 1 | 1 accepted | **Adopt** | `spectrometer_uv/channel/grating/summit/z` is paired with `vertical_coordinate_of_bragg_crystal`. Live parent documentation says “Description of the grating” and identifies the summit as the grating summit; no Bragg-crystal owner is present. |
| `conductor_cross_section` | 10 | 9 accepted, 1 reviewed | **Decline** | The DD expresses the same entity through `geometry/rectangle` or `geometry/oblique` under conductor hardware. `pf_active/coil/element/geometry/rectangle/r` → `radial_coordinate_of_conductor_cross_section` is a valid structural spelling with no literal locus overlap. |
| `coordinate_system` | 1 | 1 superseded | **Decline** | The only pairing is historical: `distributions/distribution/profiles_2d/grid/r` → `major_radius_of_coordinate_system`. A grid major-radius coordinate structurally belongs to a coordinate system, and there is no live write-boundary benefit to a family trigger. |
| `diagnostic_aperture` | 50 | 36 accepted, 6 reviewed, 8 superseded | **Decline** | Pinhole, detector-opening, antenna-opening, fibre-bundle, filter-window, grating, and polarizer structures are concrete diagnostic-aperture realizations. Representative supported pairings include `camera_ir/channel/camera/pinhole/x` → `x_coordinate_of_diagnostic_aperture` and `reflectometer_profile/channel/antenna_detection/radius` → `radius_of_diagnostic_aperture`. |
| `iron_core_segment` | 3 | 3 accepted | **Decline** | The family is mixed. Both equilibrium separatrix-outline pairings are wrong, but live DD documentation for `ferritic/object/axisymmetric` explicitly says the representation is used “for each iron core segment”; therefore `ferritic/object/axisymmetric/rectangle/height` → `height_of_iron_core_segment` is a supported zero-overlap counterexample. |
| `optical_element` | 44 | 44 accepted | **Decline** | The DD names concrete subclasses—filters, apertures, fibre bundles, gratings, polarizers, mirrors, crystals, reflectors, and detector windows—rather than repeating the superclass. `ec_launchers/mirror/geometry/surface` → `surface_area_of_optical_element` and `spectrometer_x_ray_crystal/channel/crystal/surface` → the same family are representative supported pairings. |
| `passive_structure` | 1 | 1 accepted | **Adopt** | `operational_instrumentation/sensor/length` is paired with `displacement_of_passive_structure`, but the leaf says only “Length measured by a displacement sensor.” The IDS covers sensors on various device parts and uses runtime attachment URIs to identify the attached systems; the schema path does not assert a passive structure. |
| `pellet` | 8 | 8 accepted | **Decline** | The family is mixed. Six fragment and mass-centre velocity paths are legitimate pellet kinematics, for example `spi/injector/fragment/velocity_r` → `radial_velocity_of_pellet`. The two gas-atom pairings are suspicious, but adopting the family trigger would also refuse the six supported rows. |
| `plasma_facing_component` | 3 | 3 accepted | **Decline** | Wall topology supplies the entity class. At minimum, `wall/description_2d/limiter/unit/midplane_thickness` → `thickness_of_plasma_facing_component` is directly supported; a family-wide lexical trigger would reject it. |
| `vacuum_vessel` | 2 | 2 accepted | **Decline** | Live DD documentation for `wall/global_quantities/current_phi` explicitly says “Toroidal current flowing in the vacuum vessel,” supporting `toroidal_current_of_vacuum_vessel`. The lower gas-injection summary likewise denotes injection into the lower vessel region without repeating the owner token. |
| `wave_beam` | 7 | 7 accepted | **Decline** | Diagnostic wavelength, reference-frequency, and phase paths plus Thomson-laser area/wavelength paths describe the propagating beam through instrument context. `interferometer/channel/wavelength/value` → `wavelength_of_wave_beam` and `thomson_scattering/laser/wavelength` → the same identity are supported zero-overlap pairings. |

The verdict partitions close exactly:

- Families: **3 adopt + 9 decline = 12**.
- Pairings: **6 adopt + 128 decline = 134**.

Every family object in the JSON carries exactly one `adopt` or `decline`
verdict, its rationale, representative evidence citations, and the complete
sorted list of newly refused pairings. The list contains the source node ID, DD
path, Standard Name, name stage, origin, and historical-stage marker for each
pairing.

## Interpretation boundary

This is recommendation authority, not graph-mutation or code-change authority.
No pairing was detached, rebound, renamed, accepted, or re-reviewed. If the
recommendation is implemented later, the safe form is a selective policy for
the three adopted locus tokens derived against the live ISN locus universe. It
must not replace the shipped hardware trigger with an all-entity trigger.

Known defects inside declined families remain visible:

- the two separatrix outlines attached to `radial_outline_of_iron_core_segment`;
- the two gas-atom paths attached to `normalized_atomic_count_of_pellet`;
- narrower owner errors within the broad diagnostic-aperture and optical-element
  groups.

Their presence does not justify a family-wide trigger because each family also
contains supported counterexamples. Those rows belong to exact semantic
adjudication and governed repair.

## Read-only proof

Protected node counts were read immediately before and after the live
attachment query and trigger evaluation:

| Node label | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` | 7,451 | 7,451 | 0 |
| `LLMCost` | 27,467 | 27,467 | 0 |

No graph write or provider call occurred. The signed payload can be reproduced
from the JSON with Python `json.dumps` over
`{"measurements": manifest["measurements"], "families": manifest["families"]}`
using `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=True`, UTF-8
encoding, and no trailing newline.
