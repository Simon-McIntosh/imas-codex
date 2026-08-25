# Field-map-grid coordinate base convention

## Decision

Use these three identities when `field_map_grid` becomes available in ISN:

| DD leaf | Identity to add | Why this base is correct |
|---|---|---|
| `b_field_non_axisymmetric/time_slice/field_map/grid/r` | `radial_coordinate_of_field_map_grid` | The DD quantity is cylindrical R. The accepted corpus consistently spells an owning entity's cylindrical-R coordinate with `radial_coordinate_of_*`: 29 accepted identities use that form and none use `major_radius_of_*`. Runtime ISN also registers `radial_coordinate` as a geometric base, whereas it does not register `major_radius` as a base. |
| `b_field_non_axisymmetric/time_slice/field_map/grid/z` | `vertical_coordinate_of_field_map_grid` | Z is the vertical axis of the same three-dimensional support. This preserves the coordinate meaning already established by the DD family rather than naming a physical field or a measurement location. |
| `b_field_non_axisymmetric/time_slice/field_map/grid/phi` | `toroidal_coordinate_of_field_map_grid` | Phi is the toroidal axis of the support. The toroidal-coordinate spelling preserves that geometric axis and avoids the existing, incorrect `toroidal_angle_of_measurement_position` assertion. |

This resolves the earlier asymmetric proposal
`major_radius_of_field_map_grid`, `vertical_coordinate_of_field_map_grid`, and
`toroidal_coordinate_of_field_map_grid`. At the prose level, “major-radius
coordinate” and “radial coordinate” are semantically interchangeable for
cylindrical R in this accepted corpus: the accepted radial-coordinate
descriptions repeatedly define the quantity as perpendicular distance from the
toroidal symmetry axis in the right-handed cylindrical (R, phi, Z) frame. No
separate minor-radius or flux-surface meaning was found for this owner form.

Accordingly, **family symmetry wins the spelling decision without sacrificing
semantic precision**. `radial_coordinate_of_field_map_grid` forms a coherent
radial/vertical/toroidal coordinate family, while still meaning cylindrical R.
The rejected alternative is not semantically wrong in ordinary prose; it is
wrong as an ISN identity choice because `major_radius` is not a registered base,
the live accepted corpus has zero `major_radius_of_*` owners, and every observed
row with that prefix has already been superseded and quarantined.

## Accepted-corpus measurement

The live graph was measured by identity prefix because `StandardName.id` is the
canonical identity. The accepted population sanity count was **2,302
`StandardName` nodes, 2,302 with `id`, and 2,302 with `name_stage`**. Against
that controlled population:

| Candidate owner form | Accepted identities | All lifecycle stages | Control that the prefix scan fires |
|---|---:|---:|---|
| `radial_coordinate_of_*` | **29** | 51 | 29 accepted rows and 22 rows in other stages |
| `major_radius_of_*` | **0** | 6 | Six matching rows exist; all six are `superseded` and `quarantined` |

The sanity count proves that both filter properties exist on every row in the
accepted population. The six non-accepted `major_radius_of_*` matches are an
additional positive control aimed at the same `id STARTS WITH` instrument: the
accepted zero is a lifecycle result, not a plausible empty result from a wrong
property name or a pattern that cannot fire.

Representative accepted cylindrical-R owners include:

- `radial_coordinate_of_electron_cyclotron_launcher_mirror`, sourced from
  `dd:ec_launchers/mirror/geometry/sphere_centre/r`: “Major-radius coordinate
  locating the sphere center that defines an electron-cyclotron launcher mirror
  in the right-handed cylindrical (R, phi, Z) frame.”
- `radial_coordinate_of_control_surface`, sourced from
  `dd:b_field_non_axisymmetric/time_slice/control_surface/outline/r`:
  “Major-radius coordinate of each point on a control-surface outline, measured
  as perpendicular distance from the toroidal symmetry axis in the right-handed
  cylindrical (R, phi, Z) frame.” This is especially close to the target: it is
  another R-coordinate owner inside `b_field_non_axisymmetric`.
- `radial_coordinate_of_magnetic_axis`, sourced from seven DD paths including
  `dd:equilibrium/time_slice/global_quantities/magnetic_axis/r`: “Major-radius
  coordinate locating the magnetic-axis O-point in the right-handed cylindrical
  (R, phi, Z) frame around which nested closed flux surfaces are organized.”

There is **no accepted `major_radius_of_*` identity to cite**. The closest
field-map-specific row is
`major_radius_of_coordinate_system`, whose description says “Major radius (R)
coordinate of the field map grid used for the non-axisymmetric magnetic field.”
It is `superseded` and `quarantined`, as are the other five prefix matches. For
comparison, accepted identities such as `reference_major_radius` and
`flux_surface_averaged_major_radius` do contain the major-radius phrase, but
they are not `major_radius_of_<owner>` coordinate identities; their different
operator/qualifier structures therefore do not establish the proposed owner
form.

## Runtime grammar result

The candidate set came from the installed `imas-standard-names` **0.8.0rc67**
`get_grammar_context()` result, not from remembered vocabulary. The runtime
context reports **175 physical-base tokens** and **35 geometric-base tokens**:

| Token | Physical base | Geometric base |
|---|---:|---:|
| `major_radius` | absent | absent |
| `radial_coordinate` | absent | **present** |
| `radius` | **present** | absent |

ISN therefore does **not** carry `major_radius` and `radial_coordinate` as
separate physical bases. More precisely, `radial_coordinate` is the registered
geometric base for an owner's spatial coordinate, while `major_radius` is not a
registered base in either registry. The generic physical base `radius` exists,
but it denotes a radius quantity; substituting `radius_of_field_map_grid` would
lose the fact that the leaf is the grid's R-coordinate axis rather than a size
or intrinsic radius of the grid.

The corpus likewise does not use the proposed pair to distinguish machine-axis
cylindrical R from minor-radius or flux-surface radial coordinates. Instead,
accepted `radial_coordinate_of_*` descriptions explicitly say “major-radius,”
“cylindrical (R, phi, Z),” or “perpendicular distance from the toroidal symmetry
axis.” Flux-surface coordinates are separately named with registered tokens
such as `normalized_minor_radius`, `normalized_poloidal_flux_coordinate`,
`normalized_toroidal_flux_coordinate`, `poloidal_magnetic_flux_coordinate`,
and `toroidal_flux_coordinate`. The distinction is therefore between
`radial_coordinate` for cylindrical R and those explicit minor-radius/flux
coordinate bases—not between interchangeable `major_radius_of_*` and
`radial_coordinate_of_*` owner spellings.

## DD authority for the target leaf

The target query was aimed at
`IMASNode.id = 'b_field_non_axisymmetric/time_slice/field_map/grid/r'` and
returned **one row with `id` and one row with `description`**. The broader DD
schema control returned **61,366 `IMASNode` candidates and 61,366 with `id`**.
The DD text is exactly:

> “Major radius (R) coordinate of grid.”

Its authoritative `HAS_UNIT` target is `Unit.id = 'm'` with symbol **`m`**.
Thus the leaf is cylindrical major radius R in metres, not minor radius, a
normalized flux coordinate, or a measurement position.

## Scope and raw evidence

This work was read-only with respect to ISN and the graph. It made no vocabulary
change, dependency-pin change, graph mutation, or new `StandardName`.

Raw measurements:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T150357556109-n-basetoken/logs/base-corpus-measurement.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T150357556109-n-basetoken/logs/base-corpus-detail.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T150357556109-n-basetoken/logs/base-prefix-controls.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T150357556109-n-basetoken/logs/base-token-runtime-grammar.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T150357556109-n-basetoken/logs/base-token-runtime-grammar-detail.json`
