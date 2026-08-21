# Live DD producer search for `voltage_of_diagnostic_antenna`

## Result

**No authoritative live DD producer exists for
`voltage_of_diagnostic_antenna`.** The live graph contains exactly one producer
row for the identity, `dd:ece/channel/t_e_voltage`, but that source is
`status='stale'` and its backing DD path is `lifecycle_status='removed'`.
Counting only `composed` or `attached` sources backed by a current DD path gives
**0 live producers**.

The current DD successor is `ece/channel/voltage_t_radiation`. It has the same
unit as the name (`V`) but its path and documentation say that it is the raw
voltage of an **ECE channel**, not a diagnostic antenna. It is already a
`composed` source for the separately reviewed identity
`voltage_of_spectrometer_channel`. The prior exact dual-binding adjudication
also selected that channel-owned identity and removed
`voltage_of_diagnostic_antenna` from this source. Unit agreement alone therefore
does not make the current path authority for the old owner semantics.

The route selected by this evidence is the **sanctioned source-less
rename transition**. The alternative route—an ordinary source-backed rename
after finding an authoritative live producer—is unavailable. A later mutation
node must not migrate the stale `t_e_voltage` source, steal
`voltage_t_radiation` from its channel-owned identity, attach a merely similar
RF/heating-antenna voltage, or hand-promote a successor. This record makes no
graph mutation and grants no acceptance authority; the successor still owes the
sanctioned transition and ordinary review semantics specified by the live plan.

## Quantitative evidence

| Measure | Result |
|---|---:|
| Natural-language DD searches | 6 |
| Raw cluster results inspected | 60 (10 per search) |
| Semantically eligible diagnostic-antenna voltage paths | **0** |
| Independent exact negative eligibility queries | 4 |
| Exact negative queries returning zero hits | **4 of 4** |
| Current `V`-valued quantity leaves mentioning voltage | 31 |
| Current `V` paths with diagnostic-antenna ownership | **0** |
| Producer rows on `voltage_of_diagnostic_antenna` | 1 |
| Live producer rows on `voltage_of_diagnostic_antenna` | **0** |
| Stale producer rows on `voltage_of_diagnostic_antenna` | 1 |
| Current DD rename successors of `ece/channel/t_e_voltage` | 1 |
| Current successor unit agreement | `V` versus `V` — match |
| Current successor owner agreement | channel versus diagnostic antenna — **mismatch** |

An eligible path had to satisfy all of the following:

1. exist in the current DD graph (`lifecycle_status` absent/current or anything
   other than `removed`);
2. be a leaf quantity with DD-authoritative unit `V`;
3. describe a diagnostic antenna as the owner of the voltage, rather than an RF
   heating antenna, probe coil, detector power supply, acquisition channel, or
   spectrometer channel; and
4. for a presently producing source, have `StandardNameSource.status` equal to
   `composed` or `attached` and a live `PRODUCED_NAME` edge.

The broad 31-path voltage census is important because the negative is not an
empty-dictionary artifact. It found real voltage quantities for RF heating
antenna modules, magnetic probes and flux loops, neutron detector supplies,
coils, valves, spectrometers, and the ECE acquisition channel. None carries the
specific owner semantics of `diagnostic_antenna`.

## Authority inputs

- Live plan: `docs/sn-graph-wide-integrity.html`, version 219, SHA-256
  `10c8d1c2acc35fed33d5733c00fcf7c431d5d190713422858a1e1fbfcbb6490b`.
- Worktree source commit: `d00744813a781e7ddf1ee0d93359d26eb8a00b9d`.
- Prior failed rename record:
  `docs/evidence/sn-graph-wide-integrity/carried-name-repairs.md`, SHA-256
  `a95089d73db51dae8c09816f4433a9e56953e77cab5b450f53cf8938fef19a03`.
- Stale-source authority:
  `docs/evidence/sn-graph-wide-integrity/stale-source-lifecycle.json`, SHA-256
  `f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad`.
- Current-path dual-binding adjudication:
  `docs/evidence/sn-graph-wide-integrity/catalog-edit-dual-binding-adjudication.json`,
  SHA-256
  `5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e`.
- Search time: `2026-08-21T15:04:45+02:00`.

## Natural-language search ledger

All six calls used the read-only `search_dd_paths` tool with `dd_version=4`,
`k=20`, `lifecycle_filter='active'`, and `node_category='quantity'`. The tool
returned semantic clusters rather than a flat scalar-path list, so both the raw
cluster count and the number passing the eligibility criteria are recorded.

### Exact identity wording

```text
search_dd_paths(query="voltage of diagnostic antenna", dd_version=4, k=20,
                lifecycle_filter="active", node_category="quantity")
```

Recorded output:

```text
## IMAS Clusters (10 found)
Microwave Diagnostic Antenna Z Unit Vector
Diagnostic Antenna Unit Vector Y
Diagnostic Antenna X1 Unit Vector X-Component
RF Antenna Power and Electrical Measurements
ICRH Antenna Power and Electrical Parameters
Neutron Detector Power Supply Voltage
RAW_HIT_COUNT: 10 clusters
ELIGIBLE_HIT_COUNT: 0 paths
```

The diagnostic-antenna hits are orientation or position fields, not voltage.
The voltage hits belong to RF heating antennas or detector supplies.

### Measurement wording

```text
search_dd_paths(query="diagnostic antenna voltage measurement", dd_version=4,
                k=20, lifecycle_filter="active", node_category="quantity")
```

Recorded output:

```text
## IMAS Clusters (10 found)
Microwave Diagnostic Antenna Z Unit Vector
Diagnostic Antenna Unit Vector Y
Diagnostic Antenna X1 Unit Vector X-Component
RF Antenna Power and Electrical Measurements
Magnetic Probe Field and Voltage Measurements
ICRH Antenna Power and Electrical Parameters
RAW_HIT_COUNT: 10 clusters
ELIGIBLE_HIT_COUNT: 0 paths
```

The voltage clusters describe RF heating modules or magnetic-probe coil
terminals. Neither is a diagnostic antenna voltage.

### Terminal-voltage wording

```text
search_dd_paths(query="antenna terminal voltage in a plasma diagnostic",
                dd_version=4, k=20, lifecycle_filter="active",
                node_category="quantity")
```

Recorded output:

```text
## IMAS Clusters (10 found)
Plasma Profiles Electromagnetic Field Components
Global Plasma Current and Loop Voltage
ICRH Antenna Power and Electrical Parameters
MHD GGD Plasma Field Values
RAW_HIT_COUNT: 10 clusters
ELIGIBLE_HIT_COUNT: 0 paths
```

The antenna voltage result again belongs to ICRH heating hardware.

### Microwave-potential wording

```text
search_dd_paths(query="microwave diagnostic antenna electrical potential",
                dd_version=4, k=20, lifecycle_filter="active",
                node_category="quantity")
```

Recorded output:

```text
## IMAS Clusters (10 found)
Microwave Diagnostic Antenna Z Unit Vector
RF Antenna Power and Electrical Measurements
ICRH Antenna Power and Electrical Parameters
Radial Electric Field Profiles
Flux Surface Electrostatic Potential Profile
Magnetic Probe Field and Voltage Measurements
RAW_HIT_COUNT: 10 clusters
ELIGIBLE_HIT_COUNT: 0 paths
```

No returned voltage path is owned by a microwave diagnostic antenna.

### ECE-channel wording

```text
search_dd_paths(query="electron cyclotron emission channel voltage",
                dd_version=4, k=20, lifecycle_filter="active",
                node_category="quantity")
```

Recorded output:

```text
## IMAS Clusters (10 found)
ECE Electron Temperature Channels
ECE Electron Temperature Radiation
ECE Electron Radiation Temperature Profiles
ECE Channel Voltage Calibration
  ece/channel/calibration_factor
  ece/channel/t_e_voltage
  ece/channel/voltage_t_radiation
  spectrometer_visible/channel/filter_spectrometer/output_voltage
RAW_HIT_COUNT: 10 clusters
ELIGIBLE_HIT_COUNT: 0 paths for diagnostic-antenna ownership
```

This search found the relevant DD lineage, but the current path is explicitly
channel-owned. The removed `t_e_voltage` path is not live.

### Radiometer/spectrometer wording

```text
search_dd_paths(query="radiometer or spectrometer channel voltage",
                dd_version=4, k=20, lifecycle_filter="active",
                node_category="quantity")
```

Recorded output:

```text
## IMAS Clusters (10 found)
Visible Spectrometer Channel Radiance Data
Visible Spectrometer Channel Major Radius Positions
UV Spectrometer Channel Major Radius
Bremsstrahlung Visible Channel Measurements
RAW_HIT_COUNT: 10 clusters
ELIGIBLE_HIT_COUNT: 0 paths for diagnostic-antenna ownership
```

The result confirms the channel/spectrometer semantic family rather than an
antenna-owned voltage.

## Exact path fetch ledger

### Former and current ECE voltage paths

```text
fetch_dd_paths(paths="ece/channel/t_e_voltage ece/channel/voltage_t_radiation",
               dd_version=4, include_version_history=true)
```

Recorded output:

```text
## IMAS Path Details (1 fetched, 1 not found)
ece/channel/voltage_t_radiation
  Raw voltage signal on an ECE channel associated with radiation temperature
  measurement, prior to calibration.
  IDS: ece | Type: STRUCTURE | Units: V
  RENAMED FROM: ece/channel/t_e_voltage
ece/channel/t_e_voltage: NOT FOUND (not_found)
HIT_COUNT: 1 current path
NOT_FOUND_COUNT: 1 removed path
```

### Current path with child preview requested

```text
fetch_dd_paths(paths="ece/channel/voltage_t_radiation", dd_version=4,
               include_version_history=true, include_children=true)
```

Recorded output:

```text
## IMAS Path Details (1 fetched)
ece/channel/voltage_t_radiation
  Raw voltage signal on an ECE channel associated with radiation temperature
  measurement, prior to calibration.
  IDS: ece | Type: STRUCTURE | Units: V
  RENAMED FROM: ece/channel/t_e_voltage
HIT_COUNT: 1
```

## Four exact negative eligibility queries

These are independent zero-hit queries against the current graph. They use
`coalesce(path.lifecycle_status, 'active') <> 'removed'` because current DD
nodes have a null lifecycle marker while removed historical nodes carry the
explicit `removed` value.

### Current source already producing the target identity

```cypher
MATCH (source:StandardNameSource)-[:FROM_DD_PATH]->(path:IMASNode)
MATCH (source)-[:PRODUCED_NAME]->(name:StandardName {id: 'voltage_of_diagnostic_antenna'})
WHERE source.status IN ['composed', 'attached']
  AND coalesce(path.lifecycle_status, 'active') <> 'removed'
RETURN source.id AS source_id, source.status AS source_status,
       path.id AS path, path.lifecycle_status AS dd_lifecycle,
       path.unit AS unit
```

Recorded output:

```text
HIT_COUNT: 0
(no rows)
```

### Current microwave-diagnostic voltage with antenna ownership

```cypher
MATCH (path:IMASNode)
WHERE coalesce(path.lifecycle_status, 'active') <> 'removed'
  AND path.is_leaf = true
  AND path.node_category = 'quantity'
  AND path.unit = 'V'
  AND path.ids IN ['ece', 'reflectometer_profile', 'reflectometer_fluctuation']
  AND (
    toLower(path.id) CONTAINS 'antenna'
    OR toLower(path.id) CONTAINS 'polarizer'
    OR toLower(coalesce(path.documentation, '')) CONTAINS 'antenna'
  )
RETURN path.id AS path, path.ids AS ids, path.unit AS unit,
       path.documentation AS documentation
ORDER BY path.id
```

Recorded output:

```text
HIT_COUNT: 0
(no rows)
```

### Documentation explicitly combining voltage, diagnostic, and antenna

```cypher
MATCH (path:IMASNode)
WHERE coalesce(path.lifecycle_status, 'active') <> 'removed'
  AND path.is_leaf = true
  AND path.node_category = 'quantity'
  AND path.unit = 'V'
  AND toLower(coalesce(path.documentation, '')) CONTAINS 'voltage'
  AND toLower(coalesce(path.documentation, '')) CONTAINS 'diagnostic'
  AND toLower(coalesce(path.documentation, '')) CONTAINS 'antenna'
RETURN path.id AS path, path.ids AS ids, path.unit AS unit,
       path.documentation AS documentation
ORDER BY path.id
```

Recorded output:

```text
HIT_COUNT: 0
(no rows)
```

### Current DD successor of the stale path retaining antenna semantics

```cypher
MATCH (old:IMASNode {id: 'ece/channel/t_e_voltage'})-[:RENAMED_TO]->(path:IMASNode)
WHERE coalesce(path.lifecycle_status, 'active') <> 'removed'
  AND path.is_leaf = true
  AND path.node_category = 'quantity'
  AND path.unit = 'V'
  AND (
    toLower(path.id) CONTAINS 'antenna'
    OR toLower(path.id) CONTAINS 'polarizer'
    OR toLower(coalesce(path.documentation, '')) CONTAINS 'antenna'
  )
RETURN old.id AS renamed_from, path.id AS current_path,
       path.unit AS unit, path.documentation AS documentation
```

Recorded output:

```text
HIT_COUNT: 0
(no rows)
```

## Corroborating graph outputs

### Target identity producer state

```cypher
MATCH (name:StandardName {id: 'voltage_of_diagnostic_antenna'})
OPTIONAL MATCH (name)-[:HAS_UNIT]->(name_unit:Unit)
OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(name)
OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(path:IMASNode)
RETURN name.id AS name, name.unit AS name_unit_property,
       name_unit.symbol AS name_unit_edge, name.name_stage AS name_stage,
       source.id AS source_id, source.status AS source_status,
       source.produced_sn_id AS scalar_target,
       path.id AS path, path.lifecycle_status AS dd_lifecycle,
       path.unit AS path_unit
ORDER BY source_id
```

Recorded output:

```text
HIT_COUNT: 1
{'name': 'voltage_of_diagnostic_antenna',
 'name_unit_property': 'V', 'name_unit_edge': 'V',
 'name_stage': 'accepted',
 'source_id': 'dd:ece/channel/t_e_voltage',
 'source_status': 'stale',
 'scalar_target': 'voltage_of_diagnostic_antenna',
 'path': 'ece/channel/t_e_voltage',
 'dd_lifecycle': 'removed', 'path_unit': 'V'}
```

### Current DD successor and its existing semantic authority

```cypher
MATCH (old:IMASNode {id: 'ece/channel/t_e_voltage'})-[:RENAMED_TO]->(path:IMASNode)
OPTIONAL MATCH (source:StandardNameSource)-[:FROM_DD_PATH]->(path)
OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(name:StandardName)
RETURN old.id AS removed_path, old.lifecycle_status AS old_lifecycle,
       path.id AS current_path, path.lifecycle_status AS current_lifecycle,
       path.unit AS path_unit, path.documentation AS documentation,
       source.id AS current_source, source.status AS source_status,
       source.produced_sn_id AS scalar_target,
       collect(DISTINCT name.id) AS edge_targets,
       collect(DISTINCT name.name_stage) AS target_stages
```

Recorded output:

```text
HIT_COUNT: 1
removed_path: ece/channel/t_e_voltage
old_lifecycle: removed
current_path: ece/channel/voltage_t_radiation
current_lifecycle: null/current
path_unit: V
documentation: Raw voltage measured on each channel, from which the calibrated
               temperature data is then derived
current_source: dd:ece/channel/voltage_t_radiation
source_status: composed
scalar_target: voltage_of_spectrometer_channel
edge_targets: ['voltage_of_spectrometer_channel']
target_stages: ['reviewed']
```

### Current voltage-identity family

```cypher
MATCH (name:StandardName)
WHERE name.id IN [
  'voltage_of_diagnostic_antenna',
  'voltage_of_ece_channel',
  'voltage_of_spectrometer_channel'
]
OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(name)
RETURN name.id AS name, name.name_stage AS name_stage,
       name.validation_status AS validation_status, name.unit AS unit,
       count(DISTINCT CASE WHEN source.status IN ['composed', 'attached']
                           THEN source END) AS live_producer_count,
       collect(DISTINCT CASE WHEN source.status IN ['composed', 'attached']
                             THEN source.id END) AS live_producers,
       collect(DISTINCT source.id) AS all_producers
ORDER BY name
```

Recorded output:

```text
HIT_COUNT: 2 identities
voltage_of_diagnostic_antenna | accepted | valid | V | 0 live producers
  all producers: ['dd:ece/channel/t_e_voltage']
voltage_of_spectrometer_channel | reviewed | valid | V | 2 live producers
  includes: dd:ece/channel/voltage_t_radiation
voltage_of_ece_channel | absent
```

### Broad current voltage-path census

```cypher
MATCH (path:IMASNode)
WHERE coalesce(path.lifecycle_status, 'active') <> 'removed'
  AND path.is_leaf = true
  AND path.node_category = 'quantity'
  AND path.unit = 'V'
  AND (
    toLower(path.id) CONTAINS 'voltage'
    OR toLower(coalesce(path.documentation, '')) CONTAINS 'voltage'
  )
RETURN path.id AS path, path.ids AS ids, path.unit AS unit,
       path.documentation AS documentation
ORDER BY path
```

Recorded output:

```text
HIT_COUNT: 31
coils_non_axisymmetric/coil/conductor/voltage
coils_non_axisymmetric/coil/voltage
core_profiles/global_quantities/v_loop
ece/channel/voltage_t_radiation
equilibrium/time_slice/global_quantities/v_external
gas_injection/valve/response_curve/voltage
gas_injection/valve/voltage
ic_antennas/antenna/module/voltage
ic_antennas/antenna/module/voltage/amplitude
magnetics/b_field_phi_probe/voltage
magnetics/b_field_pol_probe/voltage
magnetics/flux_loop/voltage
magnetics/shunt/voltage
neutron_diagnostic/detector/supply_high_voltage/voltage_out
neutron_diagnostic/detector/supply_high_voltage/voltage_set
neutron_diagnostic/detector/supply_low_voltage/voltage_out
neutron_diagnostic/detector/supply_low_voltage/voltage_set
pf_active/circuit/voltage
pf_active/coil/voltage
pf_active/supply/voltage
plasma_profiles/global_quantities/v_loop
pulse_schedule/flux_control/v_loop
pulse_schedule/pf_active/supply/voltage
spectrometer_mass/channel/photomultiplier_voltage
spectrometer_mass/detector_voltage
spectrometer_uv/channel/supply_high_voltage/voltage_set
spectrometer_visible/channel/filter_spectrometer/output_voltage
spectrometer_visible/channel/filter_spectrometer/photoelectric_voltage
summary/global_quantities/v_loop
tf/coil/conductor/voltage
tf/coil/voltage
ELIGIBLE_HIT_COUNT: 0
```

## Corrected lifecycle-filter probe

An initial exact probe used `path.lifecycle_status = 'active'` and returned zero
ECE/reflectometer voltage rows. Inspection of the exact path then showed that
current DD nodes use a null lifecycle marker while removed nodes use
`removed`. That probe is retained here so no executed search is hidden, but it
is **not** used as absence evidence; every authoritative query above uses the
correct current-path predicate.

```cypher
MATCH (path:IMASNode)
WHERE path.lifecycle_status = 'active'
  AND path.is_leaf = true
  AND path.unit = 'V'
  AND path.ids IN ['ece', 'reflectometer_profile', 'reflectometer_fluctuation']
RETURN path.id AS path, path.ids AS ids, path.data_type AS data_type,
       path.node_category AS node_category, path.unit AS unit,
       path.documentation AS documentation
ORDER BY path
```

Recorded output:

```text
HIT_COUNT: 0
CORRECTION: replace equality-to-active with
            coalesce(path.lifecycle_status, 'active') <> 'removed'
```

The corrected broad census then found the one current ECE channel voltage path,
and the four independent semantic/authority filters still returned zero
eligible diagnostic-antenna producers.
