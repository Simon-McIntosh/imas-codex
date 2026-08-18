# DD resolution authority-evidence export

**Result:** no legacy row is eligible for active typed resolution from machine evidence alone. This export proves exact release/path/raw facts and binds current observational graph evidence, but it supplies no upstream solution identity or governed approval. It makes no adjudication and changes no behavior.

- Base SHA: `32aa552d6f9a69339bb78c1be58b4af6a73905c0`
- Machine payload: `/tmp/reckon-s8-scope/dd-resolution-evidence-export.json` (SHA-256 `93ad2ae36bbb9e322591bbf6f71539b3c170d09672059ca7078f74ed9129512e`; 8,772,933 bytes)
- Prior static transition audit: `/tmp/reckon-s8-scope/dd-resolution-transition-audit.md` (SHA-256 `fd5bb1e70ab0945df501550c38b0b202740eea0046b53d6d17c7b1c88aabbaf1`)
- Prior scope: `/tmp/reckon-s8-scope/dd-resolution-scope.md` (SHA-256 `164bdd74c3a856cbd0510c3594eb84dfd463a7deead18ac9f0e875a467493af4`)
- This Markdown is the human index. The JSON is the complete, non-abbreviated evidence payload: all exact paths, raw unit/type/COCOS/lifecycle facts, graph fact properties, observation IDs and payloads, evidence tokens, relationship destination IDs, collisions, and the review-input manifest.

## 1. Scope and authority boundary

The input registries remain legacy behavior, DDGap remains observational evidence, and the tracked `dd_resolutions.yaml` remains empty. The export does not treat graph scalar values as raw release truth, does not infer approval from registry prose, and does not promote any record. The graph schema itself says DDGap flags are observational (`imas_codex/schemas/standard_name.yaml:2494-2502`), while only active DDResolution records can represent reviewed version-bound interpretation (`:2391-2489`). The packaged manifest is empty (`imas_codex/standard_names/config/dd_resolutions.yaml:1-2`).

The 62 audited legacy rows are 34 `dd_unit_bugs` entries (`imas_codex/units/dd_unit_exceptions.yaml:30-288`) and 28 ordered override/skip entries (`imas_codex/standard_names/config/unit_overrides.yaml:23-225`). Matching semantics are materially different: the unit-bug registry uses `fnmatch`, whose `*` crosses `/`, while the override engine implements segment-aware `*` and recursive `**` (`imas_codex/units/dd_unit_exceptions.py:70-104`; `imas_codex/standard_names/unit_overrides.py:52-83`).

## 2. Method and reproducibility

### 2.1 Schema verification before graph access

The canonical generated context was loaded with:

```python
from imas_codex.graph.schema_context import schema_for
ctx = schema_for("DDGap", "DDGapObservation", "DDResolution",
                 "DDVersion", "IMASNode", include_examples=False)
```

The context digest was `70bc6d4a81b6d70fc9a76b24688bc92669bf177326543284808485e797faece4`. It verified `HAS_OBSERVATION`, `HAS_DD_GAP`, `HAS_RESOLUTION`, `HAS_DD_RESOLUTION`, and `FOR_DD_VERSION`. LinkML defines no direct DDGap→DDVersion or DDGapObservation→DDVersion relationship; observations carry an `observed_dd_version` scalar. DDResolution is the only audited authority type with `FOR_DD_VERSION` (`imas_codex/schemas/standard_name.yaml:2476-2488`).

### 2.2 Read-only graph queries

All graph access used `GraphClient`; every Cypher statement was `MATCH`/`OPTIONAL MATCH`/`UNWIND` plus `RETURN`. No `SET`, `MERGE`, `CREATE`, `DELETE`, procedure call, service action, or graph mutation was executed. The core query shapes were:

```cypher
MATCH (g:DDGap) RETURN g.id, properties(g), keys(g)
MATCH (g:DDGap)-[:HAS_OBSERVATION]->(o:DDGapObservation)
RETURN g.id, o.id, properties(o), keys(o)
MATCH (n:IMASNode)-[r:HAS_DD_GAP]->(g:DDGap)
RETURN g.id, n.id, properties(r), keys(r)
MATCH (r:DDResolution)
OPTIONAL MATCH (r)-[:FOR_DD_VERSION]->(v:DDVersion)
OPTIONAL MATCH (r)-[:SUPPORTED_BY_OBSERVATION]->(o:DDGapObservation)
RETURN r.id, properties(r), collect(DISTINCT v.id), collect(DISTINCT o.id)
UNWIND $paths AS path
OPTIONAL MATCH (n:IMASNode {id:path})-[r]->(v:DDVersion)
WHERE type(r) IN ['INTRODUCED_IN', 'DEPRECATED_IN']
RETURN path, collect(DISTINCT {relationship:type(r), dd_version_id:v.id})
```

`list_dd_gaps()` and `get_dd_gap()` supplied the canonical evidence-token calculation (`imas_codex/standard_names/dd_gaps.py:181-211,2189-2339`). Per-node `keys()` were retained so absent properties are distinguishable. Neo4j cannot store a property with a null value: assigning null removes it. Accordingly, JSON `absent_schema_fields` means absent; every `stored_null_fields` list is empty.

### 2.3 Immutable DD release inspection

The configured version is exact DD `4.1.1` (`pyproject.toml:258-259`). Release inspection used only IMAS/package APIs:

```python
xml_bytes = imas.dd_zip.get_dd_xml("4.1.1")
xml_crc = imas.dd_zip.get_dd_xml_crc("4.1.1")
tree = imas.dd_zip.dd_etree("4.1.1")
factory = imas.IDSFactory("4.1.1")
```

No HDF5 or IMAS data file was read. `IDSFactory` supplied the exact raw unit, data type, rank, and node type for every published path; `dd_etree` supplied global COCOS and exact field/IDS lifecycle and COCOS declarations. No raw value was reconstructed by reversing graph rewrites. These are the same upstream APIs used by the project extractor (`imas_codex/graph/build_dd.py:857-913,1069-1323`).

## 3. Immutable artifact identity

| Fact | Value |
|---|---|
| Configured DD | `4.1.1` |
| `imas-python` | `2.2.0` |
| `imas-data-dictionaries` | `4.1.1` |
| DD XML byte count | `30,799,092` |
| DD XML SHA-256 | `22f6cef9e5937c13a4888d94462fd0f597cf66dca637e85e2a0fcb129335499c` |
| DD XML CRC from API | `2172880527` |
| Published paths / IDSs | `44,150` / `82` |
| Global DD COCOS | `17` |
| Raw unit spellings | `86` |

The full exact release expansion across all 62 rows contains 1,637 unique paths. Raw type distribution is `FLT_0D` 1495, `FLT_1D` 25, `FLT_2D` 13, `INT_0D` 95, `STRUCTURE_0D` 2, `STRUCT_ARRAY_1D` 7. Node type distribution is `constant` 84, `dynamic` 362, `none` 9, `static` 1182. None of the 1,637 matched fields declares a per-field COCOS label or expression in DD 4.1.1; each still inherits global COCOS 17. Field lifecycle is absent on 1,635 paths and explicitly `obsolescent` on 2; schema inheritance yields `active` 204, `alpha` 1431, `obsolescent` 2. Every matched live graph path resolves through its IDS scalar to DD `4.1.1`; exact `INTRODUCED_IN`/`DEPRECATED_IN` destination IDs are retained per path in JSON.

## 4. Live graph evidence snapshot

| Metric | Count |
|---|---:|
| DDGap facts | 35 |
| DDGapObservation nodes | 451 |
| HAS_DD_GAP relationships | 451 |
| Distinct source paths | 450 |
| registered_exception | 34 |
| upstream_issue | 1 |
| unit_defect / self_contradiction / type_wiring | 23 / 11 / 1 |
| DDResolution nodes | 0 |
| FOR_DD_VERSION relationships | 0 |
| DDResolution state changes | 0 |

All 35 facts have `observed_dd_version="4.1.1"`, `evidence_rule="unit_equals_expected"`, and a machine-written `triaged_at`. None has `triage_actor`, `triage_reason`, `status_changed_at/by/reason`, `resolved_dd_version`, or `validation_evidence`. Thirty-four carry `registry_backend="dd_unit_exceptions"`; the separate U21 `type_wiring` fact carries the only upstream URL, issue 272. This is an issue locator, not an exact solution/change ref. There are no state-change receipts. The single identity-change receipt is a registry-sync reclassification of the SPI flow-rate fact from `unit_defect` to `self_contradiction`, not a human approval.

### 4.1 Exact fact catalog

The following tokens are complete, not abbreviated. The JSON attaches every exact observation ID/payload and source relationship to the fact.

| Fact ID | Kind / status | Raw → expected; DD | Sources / observations | Evidence token | Upstream / triage authority |
|---|---|---|---:|---|---|
| `dd_gap:*/direction/[xyz]:unit_defect` | `unit_defect` / `registered_exception` | `m` → `1`; `4.1.1` | 12 / 12 | `dd-gap-evidence:23ae27ea3d8fe15b94537b821299c4c6572f87ddb882cf61f9280e7b6ec0d7d6` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/direction_second/[xyz]:unit_defect` | `unit_defect` / `registered_exception` | `m` → `1`; `4.1.1` | 3 / 3 | `dd-gap-evidence:26a513c95b45b1e7fae1e7d21f59bc074825d616ea1fb301c4a3971c277c4ce0` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/energy_fluxes/kinetic/neutral/incident/values:unit_defect` | `unit_defect` / `registered_exception` | `m^-2.s^-1` → `W.m^-2`; `4.1.1` | 1 / 1 | `dd-gap-evidence:1dbbd167cfd0b0a70c9ef88c8926ef7f133fe1fc352edaaf9a858c0559f95272` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/energy_fluxes/kinetic/neutral/state/incident/values:type_wiring` | `type_wiring` / `upstream_issue` | `m^-2.s^-1` → `W.m^-2`; `4.1.1` | 1 / 1 | `dd-gap-evidence:4ff9e1884eaadf063a4d989e933dbd922aca13fb5aa8e6b78b4998ffe6372187` | https://github.com/iterorganization/IMAS-Data-Dictionary/issues/272; triaged_at present; actor/reason absent |
| `dd_gap:*/energy_fluxes/kinetic/neutral/state/incident/values:unit_defect` | `unit_defect` / `registered_exception` | `m^-2.s^-1` → `W.m^-2`; `4.1.1` | 1 / 1 | `dd-gap-evidence:2c00f044e68dcc172ef52f9248ec19da6c564b83cb8a1fa332a31a78c2e1ef4d` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/energy_fluxes/recombination/neutral/incident/values:unit_defect` | `unit_defect` / `registered_exception` | `m^-2.s^-1` → `W.m^-2`; `4.1.1` | 1 / 1 | `dd-gap-evidence:42a71f26237c334d48c6dd75d5d6d4e6c09aa06ed7740fb19efa08d999cd7591` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/ggd/ion/state/ionisation_potential*:self_contradiction` | `self_contradiction` / `registered_exception` | `e` → `eV`; `4.1.1` | 22 / 22 | `dd-gap-evidence:20e7f6231b31e2894fe9950f14b4fe47576205ffa3c9e61da9a4db6a0ce0332c` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/ggd/ion/state/ionization_potential*:self_contradiction` | `self_contradiction` / `registered_exception` | `e` → `eV`; `4.1.1` | 9 / 9 | `dd-gap-evidence:f68966856c50a7c99cd9793bd98f9d07cfb6963b87547c2bcc5da20498a47fff` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/injection_direction/[xyz]:unit_defect` | `unit_defect` / `registered_exception` | `m` → `1`; `4.1.1` | 3 / 3 | `dd-gap-evidence:546645d705d829d15520a6fb6eb78504fa6428a69f8deb473462d49b6eb42a9c` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/position/psi:self_contradiction` | `self_contradiction` / `registered_exception` | `W` → `Wb`; `4.1.1` | 19 / 19 | `dd-gap-evidence:bfda1749c651f7bf7dce5551beb54f002006f949bd6f0393fdb0fc6185d57f27` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/state/z_max:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 44 / 44 | `dd-gap-evidence:75a1edfa1ad52c829a5531f634839fbb4675b25f81ac8ca62b4371bfd4bd3c11` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/state/z_min:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 44 / 44 | `dd-gap-evidence:d5bf174fb951055b198e75052b68c725750de029a3b666d2468fcc16c112978c` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/unit_vector_major/[xyz]:unit_defect` | `unit_defect` / `registered_exception` | `m` → `1`; `4.1.1` | 3 / 3 | `dd-gap-evidence:59f00b4b8dec7a90c719da034f4ab1be7326a0e5ad0cd8f6a219b2bbfb67d618` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/unit_vector_minor/[xyz]:unit_defect` | `unit_defect` / `registered_exception` | `m` → `1`; `4.1.1` | 3 / 3 | `dd-gap-evidence:2f4bed75f6e075c899c23266eb99c854090cdf27cb0ae5d5065f3078394b2ffa` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/up/[xyz]:unit_defect` | `unit_defect` / `registered_exception` | `m` → `1`; `4.1.1` | 3 / 3 | `dd-gap-evidence:6eddf236a8fdf9c984ac9ee42a8d4c88f460a7400e41092bc8144fc08c65008e` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/vibrational_level:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 69 / 69 | `dd-gap-evidence:bdae28b8decc25b34aad191978c27c990a11340dc1a11984c3a390358504c7f8` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/z_average/values:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 2 / 2 | `dd-gap-evidence:8522a3b7f94bd9d75498a78d8e03c9c56f5386c17241653a5a93d1fd9dc8f6e1` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/z_average:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 8 / 8 | `dd-gap-evidence:cd2215fa6ef1399244ec3114337d7cbc8590dd1852fe1dc6b585176e5434b0da` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/z_ion:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 56 / 56 | `dd-gap-evidence:e27043ace084da8096d185a696ee2aa3d0920cfab718159f5475c6b705711758` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/z_n/value:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 1 / 1 | `dd-gap-evidence:c2f3a87c75bbab927d0bd19f7afb52c05b77aa080b5d681ca773654db6ba7bff` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/z_n:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 106 / 106 | `dd-gap-evidence:759b235a04c1ca696f9f94087e6b53578e3d5419e61842b20f7b104f8ab346f7` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/z_square_average/values:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 2 / 2 | `dd-gap-evidence:dd38276d863ec0468063ab06dcd18bfd79d2cfd994150d3b4c5830649e4543d3` | absent; triaged_at present; actor/reason absent |
| `dd_gap:*/z_square_average:unit_defect` | `unit_defect` / `registered_exception` | `e` → `1`; `4.1.1` | 8 / 8 | `dd-gap-evidence:20741c73ad7bcd663a9c00e554e8fa6b5a02778f17a0d6d095f8c0577835e0c8` | absent; triaged_at present; actor/reason absent |
| `dd_gap:distribution_sources/source/ggd/particles/values:self_contradiction` | `self_contradiction` / `registered_exception` | `m^-6.s^2` → `m^-3.s^-1`; `4.1.1` | 1 / 1 | `dd-gap-evidence:3cce95d5807f4fb7872dc934ae6dc22af58d30747e1fb61fa3c01b23afb3133c` | absent; triaged_at present; actor/reason absent |
| `dd_gap:distributions/distribution/global_quantities/current_tor:unit_defect` | `unit_defect` / `registered_exception` | `N.m` → `A`; `4.1.1` | 1 / 1 | `dd-gap-evidence:b740b119fc36e8c9e29d5cf27c5d5fda1f7dc51aac6d296d83112236632621a6` | absent; triaged_at present; actor/reason absent |
| `dd_gap:ec_launchers/beam/direction/kphi:unit_defect` | `unit_defect` / `registered_exception` | `m^-1` → `1`; `4.1.1` | 1 / 1 | `dd-gap-evidence:3e638605e6634b3efa6f1cb01c256c715851512f69bfcf318388543b9b2ca531` | absent; triaged_at present; actor/reason absent |
| `dd_gap:equilibrium/time_slice/constraints/j_parallel/reconstructed:self_contradiction` | `self_contradiction` / `registered_exception` | `1` → `A.m^-2`; `4.1.1` | 1 / 1 | `dd-gap-evidence:057c9edaba31cd5d9403b3990f4abc99ff03cbb617e74d643152c5d1fe4205aa` | absent; triaged_at present; actor/reason absent |
| `dd_gap:equilibrium/time_slice/constraints/j_phi/reconstructed:self_contradiction` | `self_contradiction` / `registered_exception` | `1` → `A.m^-2`; `4.1.1` | 1 / 1 | `dd-gap-evidence:5925231be697698bb2446508c82e11707295dd2924429bfcfc659dc4c11c2932` | absent; triaged_at present; actor/reason absent |
| `dd_gap:equilibrium/time_slice/constraints/n_e/reconstructed:self_contradiction` | `self_contradiction` / `registered_exception` | `1` → `m^-3`; `4.1.1` | 1 / 1 | `dd-gap-evidence:798f683d5ab163dc99b18fa129fd166e6ea614688be67d471cce592e91cec2d3` | absent; triaged_at present; actor/reason absent |
| `dd_gap:equilibrium/time_slice/constraints/pressure/reconstructed:self_contradiction` | `self_contradiction` / `registered_exception` | `1` → `Pa`; `4.1.1` | 1 / 1 | `dd-gap-evidence:c19363a548ed97958ae7c18118855cbe2b96502f2c3d31ce0e61809fa7c89e6c` | absent; triaged_at present; actor/reason absent |
| `dd_gap:equilibrium/time_slice/constraints/pressure_rotational/reconstructed:self_contradiction` | `self_contradiction` / `registered_exception` | `1` → `Pa`; `4.1.1` | 1 / 1 | `dd-gap-evidence:b4562761852d6d485996b222ff3cf33a974ee56488999a9f28e1fa677b210353` | absent; triaged_at present; actor/reason absent |
| `dd_gap:gyrokinetics_local/*/angle_pol:self_contradiction` | `self_contradiction` / `registered_exception` | `1` → `rad`; `4.1.1` | 3 / 3 | `dd-gap-evidence:7ce0fdbbfe151656c3aa88d2178d4f488729f48ed3b5ed4c1bb8ec336696f987` | absent; triaged_at present; actor/reason absent |
| `dd_gap:spi/injector/*_gas/flow_rate:self_contradiction` | `self_contradiction` / `registered_exception` | `s^-1` → `Pa.m^3.s^-1`; `4.1.1` | 2 / 2 | `dd-gap-evidence:6097c04dbeaaa3f67e9bde0d2e731dcc434b1ab4b222cad82017e973b06b91c7` | absent; triaged_at present; actor/reason absent |
| `dd_gap:wall/global_quantities/power_density_*_target_max:unit_defect` | `unit_defect` / `registered_exception` | `W` → `W.m^-2`; `4.1.1` | 2 / 2 | `dd-gap-evidence:39eac68a3c692360dcbb9388f2153b8029417f0ebca048bfe1788d7115483451` | absent; triaged_at present; actor/reason absent |
| `dd_gap:waves/coherent_wave/*k_perpendicular*:unit_defect` | `unit_defect` / `registered_exception` | `V.m^-1` → `m^-1`; `4.1.1` | 15 / 15 | `dd-gap-evidence:d5db518ace0a17f860d9771cffb70825b3c9775fce30dd3c046356205c857d68` | absent; triaged_at present; actor/reason absent |

### 4.2 Registry-evidence conflicts against immutable DD 4.1.1

The registry backfill expands by path pattern without checking each path’s raw unit (`imas_codex/standard_names/dd_gaps.py:447-480`). That produced 57 observation/source conflicts against the immutable artifact: 35 graph paths are not published in DD 4.1.1, and 22 published paths carry a different raw unit. Exact conflicts and observation IDs are in JSON.

| Row | Graph claims absent from 4.1.1 | Published path raw-unit mismatches |
|---|---:|---:|
| U01 | 3 | 0 |
| U02 | 4 | 0 |
| U08 | 3 | 0 |
| U09 | 3 | 0 |
| U10 | 3 | 0 |
| U19 | 4 | 4 |
| U20 | 9 | 0 |
| U23 | 3 | 2 |
| U30 | 1 | 0 |
| U31 | 0 | 1 |
| U33 | 1 | 15 |
| U34 | 1 | 0 |

Representative conflicts: U20 has nine graph observations for the American-spelling subtree although that subtree is absent from DD 4.1.1; U31 claims `m^-6.s^2`, while the published unit is `(m.s^-1)^-3.m^-3.s^-1`; U33 claims `W` for 19 graph paths, but 15 published paths carry `Wb`, one graph path is absent, and only three DD 4.1.1 paths actually carry `W`. These conflicts invalidate any attempt to turn the patterned observations directly into exact active records.

### 4.3 Current behavior versus immutable raw release

Across the 1,637 unique exact published matches, 26 live graph scalars differ from immutable raw DD because `correct_in_graph` behavior is active. Those same paths’ `HAS_UNIT` edges also carry the rewritten value. A further 18 `m^dimension` paths retain that raw scalar but have no raw unit edge because they are explicit qualification cases. The full 26 + 18 path inventory is in JSON. This is why the graph cannot serve as raw authority.

Representative active rewrites are the two SPI flow rates (`s^-1` raw, `Pa.m^3.s^-1` graph), five equilibrium reconstructed constraints (`1` raw, physical graph units), two gyrokinetic angles (`1` raw, `rad` graph), 14 ionisation-potential fields (`e` raw, `eV` graph), and three position/psi fields (`W` raw, `Wb` graph).

## 5. Row-by-row exact release and graph binding inventory

Every row below is version-scoped to DD 4.1.1. “Raw paths” means exact `(pattern, raw_unit)` matches from the immutable artifact, not graph inference. `Reach` is first-match reachability for ordered O-rules; it is not applicable to U-rules. The path-list SHA-256 is over newline-joined sorted exact paths and binds the concise table to the complete list in JSON. `Conflicts` counts graph observations that disagree with the artifact. All 62 rows lack the four non-machine authority classes: exact upstream solution ref, governed approval receipt, reviewer/approval actor, and governed decision reason.

| Row / source | Legacy tuple and behavior | Pattern paths / raw paths / reach | Raw type(s); lifecycle | Exact-path-list SHA-256 | Bound graph fact(s); obs IDs | Conflicts |
|---|---|---:|---|---|---|---:|
| `U01` `imas_codex/units/dd_unit_exceptions.yaml:30` | `*/z_ion`; `e` → `1`; integrity_suppression_only | 53 / 53 / — | FLT_0D,STRUCTURE_0D; active,alpha | `da5194caa030925441b9ee394bb36246fa28ef6fc80d6acb40b8ab0709a2019d` | dd_gap:*/z_ion:unit_defect; 56 | 3 (3 absent; 0 raw mismatch) |
| `U02` `imas_codex/units/dd_unit_exceptions.yaml:34` | `*/z_n`; `e` → `1`; integrity_suppression_only | 102 / 102 / — | FLT_0D,INT_0D,STRUCTURE_0D; active,alpha | `3188f27aa2314790e12e620b3a0aa2a11139348572662885937d478bc06add22` | dd_gap:*/z_n:unit_defect; 106 | 4 (4 absent; 0 raw mismatch) |
| `U03` `imas_codex/units/dd_unit_exceptions.yaml:38` | `*/z_n/value`; `e` → `1`; integrity_suppression_only | 1 / 1 / — | FLT_0D; active | `a9d3d99b74629e94f4bcb61a83e8fb18ef054f8523bc098b719d10d0daeccd4a` | dd_gap:*/z_n/value:unit_defect; 1 | 0 |
| `U04` `imas_codex/units/dd_unit_exceptions.yaml:42` | `*/z_average`; `e` → `1`; integrity_suppression_only | 8 / 8 / — | FLT_0D,STRUCT_ARRAY_1D; active,alpha | `1bcba69c7abbe1f7e267473c7e7ed3da41bd86de464a7aade3e63668e989daf4` | dd_gap:*/z_average:unit_defect; 8 | 0 |
| `U05` `imas_codex/units/dd_unit_exceptions.yaml:46` | `*/z_average/values`; `e` → `1`; integrity_suppression_only | 2 / 2 / — | FLT_1D; active,alpha | `49b074d64922f58f19c825428e70bcb63718005742692040775ce83ee99a80ce` | dd_gap:*/z_average/values:unit_defect; 2 | 0 |
| `U06` `imas_codex/units/dd_unit_exceptions.yaml:52` | `*/z_square_average`; `e` → `1`; integrity_suppression_only | 8 / 8 / — | FLT_0D,STRUCT_ARRAY_1D; active,alpha | `454f98a74c27facd3b775a0a54707bfce44020665a1f87913d1620698d507d82` | dd_gap:*/z_square_average:unit_defect; 8 | 0 |
| `U07` `imas_codex/units/dd_unit_exceptions.yaml:56` | `*/z_square_average/values`; `e` → `1`; integrity_suppression_only | 2 / 2 / — | FLT_1D; active,alpha | `0ab588cb6b4b9c3258f0de94ca821091940cdae600cf22b5bf7640e5dc25e5ba` | dd_gap:*/z_square_average/values:unit_defect; 2 | 0 |
| `U08` `imas_codex/units/dd_unit_exceptions.yaml:62` | `*/state/z_max`; `e` → `1`; integrity_suppression_only | 41 / 41 / — | FLT_0D; active,alpha | `8640af55ead92ba2ce579cda60ecb42de674cd8a9725b280ae4b404c416e3157` | dd_gap:*/state/z_max:unit_defect; 44 | 3 (3 absent; 0 raw mismatch) |
| `U09` `imas_codex/units/dd_unit_exceptions.yaml:69` | `*/state/z_min`; `e` → `1`; integrity_suppression_only | 41 / 41 / — | FLT_0D; active,alpha | `9db891b28e7c071aba2d9fbb9f693c46aaaed6b16392556abd395ee4f0f5ec27` | dd_gap:*/state/z_min:unit_defect; 44 | 3 (3 absent; 0 raw mismatch) |
| `U10` `imas_codex/units/dd_unit_exceptions.yaml:76` | `*/vibrational_level`; `e` → `1`; integrity_suppression_only | 66 / 66 / — | FLT_0D; active,alpha | `803aad4d71d355b617f1337b7592421132400edba40516afc8142841f3838656` | dd_gap:*/vibrational_level:unit_defect; 69 | 3 (3 absent; 0 raw mismatch) |
| `U11` `imas_codex/units/dd_unit_exceptions.yaml:84` | `*/direction/[xyz]`; `m` → `1`; integrity_suppression_only | 12 / 12 / — | FLT_0D; alpha | `ef53b26db77f3585c9254b693d47a49c14b0bab02c38f1c50baa8c7a4dd365d3` | dd_gap:*/direction/[xyz]:unit_defect; 12 | 0 |
| `U12` `imas_codex/units/dd_unit_exceptions.yaml:88` | `*/direction_second/[xyz]`; `m` → `1`; integrity_suppression_only | 3 / 3 / — | FLT_0D; alpha | `16cf86acb5abb7e2c828d7f676b17f642266fe2535aa176115217b2be5ea881d` | dd_gap:*/direction_second/[xyz]:unit_defect; 3 | 0 |
| `U13` `imas_codex/units/dd_unit_exceptions.yaml:92` | `*/up/[xyz]`; `m` → `1`; integrity_suppression_only | 3 / 3 / — | FLT_0D; alpha | `2babbc6e302104d1dbd5e5a2c30b68e19726e2f4e7f26602fe76e30bc4d14961` | dd_gap:*/up/[xyz]:unit_defect; 3 | 0 |
| `U14` `imas_codex/units/dd_unit_exceptions.yaml:96` | `*/injection_direction/[xyz]`; `m` → `1`; integrity_suppression_only | 3 / 3 / — | FLT_0D; alpha | `efd5c73ea572cf5a9bec7cb04abe6de2f403f315aaf81379abe823801a8eb588` | dd_gap:*/injection_direction/[xyz]:unit_defect; 3 | 0 |
| `U15` `imas_codex/units/dd_unit_exceptions.yaml:100` | `*/unit_vector_major/[xyz]`; `m` → `1`; integrity_suppression_only | 3 / 3 / — | FLT_0D; alpha | `065252ef81cd9e8b356e11514bbe3ce89fd18203d5c2d6e3676ef05cb04d6915` | dd_gap:*/unit_vector_major/[xyz]:unit_defect; 3 | 0 |
| `U16` `imas_codex/units/dd_unit_exceptions.yaml:104` | `*/unit_vector_minor/[xyz]`; `m` → `1`; integrity_suppression_only | 3 / 3 / — | FLT_0D; alpha | `20eac3addb87ed2a7c3e328247e3e6269bfc27c37d3b1fd1121549fb14f0375c` | dd_gap:*/unit_vector_minor/[xyz]:unit_defect; 3 | 0 |
| `U17` `imas_codex/units/dd_unit_exceptions.yaml:108` | `ec_launchers/beam/direction/kphi`; `m^-1` → `1`; integrity_suppression_only | 1 / 1 / — | FLT_1D; alpha | `9a91c1bbcdeaac79359de9c6fe6c0426a5eb9f48657446ec97afa3edb04cbceb` | dd_gap:ec_launchers/beam/direction/kphi:unit_defect; 1 | 0 |
| `U18` `imas_codex/units/dd_unit_exceptions.yaml:118` | `wall/global_quantities/power_density_*_target_max`; `W` → `W.m^-2`; integrity_suppression_only | 2 / 2 / — | FLT_1D; obsolescent | `f92df01cc80be121b9a584d5f016c0b4441618327bcba062da97dfb00a3797eb` | dd_gap:wall/global_quantities/power_density_*_target_max:unit_defect; 2 | 0 |
| `U19` `imas_codex/units/dd_unit_exceptions.yaml:133` | `*/ggd/ion/state/ionisation_potential*`; `e` → `eV`; graph_rewrite_and_integrity_suppression; correct_in_graph | 18 / 14 / — | FLT_1D,FLT_2D,STRUCT_ARRAY_1D; active,alpha | `70cdb2bf28f2db992b214efb0523f3a4b9dfb75b3cf50f9c4b58b0e4c5f429be` | dd_gap:*/ggd/ion/state/ionisation_potential*:self_contradiction; 22 | 8 (4 absent; 4 raw mismatch) |
| `U20` `imas_codex/units/dd_unit_exceptions.yaml:141` | `*/ggd/ion/state/ionization_potential*`; `e` → `eV`; graph_rewrite_and_integrity_suppression; correct_in_graph | 0 / 0 / — | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | dd_gap:*/ggd/ion/state/ionization_potential*:self_contradiction; 9 | 9 (9 absent; 0 raw mismatch) |
| `U21` `imas_codex/units/dd_unit_exceptions.yaml:151` | `*/energy_fluxes/kinetic/neutral/state/incident/values`; `m^-2.s^-1` → `W.m^-2`; integrity_suppression_only | 1 / 1 / — | FLT_1D; active | `4317a4a0e2c0fc607930b7eea68d5768e5bd3ec99df7eda9412d16c7fd735d32` | dd_gap:*/energy_fluxes/kinetic/neutral/state/incident/values:unit_defect<br>dd_gap:*/energy_fluxes/kinetic/neutral/state/incident/values:type_wiring; 2 | 0 |
| `U22` `imas_codex/units/dd_unit_exceptions.yaml:159` | `*/energy_fluxes/kinetic/neutral/incident/values`; `m^-2.s^-1` → `W.m^-2`; integrity_suppression_only | 1 / 1 / — | FLT_1D; active | `42e3acdcf28c8a9567330ebbb5f25952ffce3dedac0e864807d3c85a7ea14edf` | dd_gap:*/energy_fluxes/kinetic/neutral/incident/values:unit_defect; 1 | 0 |
| `U23` `imas_codex/units/dd_unit_exceptions.yaml:169` | `waves/coherent_wave/*k_perpendicular*`; `V.m^-1` → `m^-1`; integrity_suppression_only | 12 / 10 / — | FLT_1D,FLT_2D,STRUCT_ARRAY_1D; alpha | `02b5fbcacb396a5e0ff720e7ac72c35e563d12bd76d52e663cc395122f82c4b7` | dd_gap:waves/coherent_wave/*k_perpendicular*:unit_defect; 15 | 5 (3 absent; 2 raw mismatch) |
| `U24` `imas_codex/units/dd_unit_exceptions.yaml:181` | `spi/injector/*_gas/flow_rate`; `s^-1` → `Pa.m^3.s^-1`; graph_rewrite_and_integrity_suppression; correct_in_graph | 2 / 2 / — | FLT_1D; alpha | `d6049497eeb8084f3b753896430196d76db5f0552de6b43834902df65e57d23c` | dd_gap:spi/injector/*_gas/flow_rate:self_contradiction; 2 | 0 |
| `U25` `imas_codex/units/dd_unit_exceptions.yaml:196` | `equilibrium/time_slice/constraints/pressure/reconstructed`; `1` → `Pa`; graph_rewrite_and_integrity_suppression; correct_in_graph | 1 / 1 / — | FLT_0D; active | `207e13eaaff849d723dba08112c060101040dcd26d867d490e13811c44566c8d` | dd_gap:equilibrium/time_slice/constraints/pressure/reconstructed:self_contradiction; 1 | 0 |
| `U26` `imas_codex/units/dd_unit_exceptions.yaml:203` | `equilibrium/time_slice/constraints/pressure_rotational/reconstructed`; `1` → `Pa`; graph_rewrite_and_integrity_suppression; correct_in_graph | 1 / 1 / — | FLT_0D; active | `580c69a216acf523684c4143e4608077a608b000fc61e221f77e99e547fda93f` | dd_gap:equilibrium/time_slice/constraints/pressure_rotational/reconstructed:self_contradiction; 1 | 0 |
| `U27` `imas_codex/units/dd_unit_exceptions.yaml:210` | `equilibrium/time_slice/constraints/n_e/reconstructed`; `1` → `m^-3`; graph_rewrite_and_integrity_suppression; correct_in_graph | 1 / 1 / — | FLT_0D; active | `cf9ca9b444d16036255bd410ecca299bf0694a35dfb2825cfd8b900c86f2409f` | dd_gap:equilibrium/time_slice/constraints/n_e/reconstructed:self_contradiction; 1 | 0 |
| `U28` `imas_codex/units/dd_unit_exceptions.yaml:217` | `equilibrium/time_slice/constraints/j_phi/reconstructed`; `1` → `A.m^-2`; graph_rewrite_and_integrity_suppression; correct_in_graph | 1 / 1 / — | FLT_0D; active | `87b073e9288e19251b654e003f889324263ce4658ad38d3bed812e53b254432e` | dd_gap:equilibrium/time_slice/constraints/j_phi/reconstructed:self_contradiction; 1 | 0 |
| `U29` `imas_codex/units/dd_unit_exceptions.yaml:224` | `equilibrium/time_slice/constraints/j_parallel/reconstructed`; `1` → `A.m^-2`; graph_rewrite_and_integrity_suppression; correct_in_graph | 1 / 1 / — | FLT_0D; active | `b9920eafe1398452724833c742fed5dd080fec49a9e5be705ecd995a3810771f` | dd_gap:equilibrium/time_slice/constraints/j_parallel/reconstructed:self_contradiction; 1 | 0 |
| `U30` `imas_codex/units/dd_unit_exceptions.yaml:235` | `gyrokinetics_local/*/angle_pol`; `1` → `rad`; graph_rewrite_and_integrity_suppression; correct_in_graph | 2 / 2 / — | FLT_1D; alpha | `53938d79530eb69a5ee4827f6f825457fcafe9ebccaeed2f245790fefb63edc4` | dd_gap:gyrokinetics_local/*/angle_pol:self_contradiction; 3 | 1 (1 absent; 0 raw mismatch) |
| `U31` `imas_codex/units/dd_unit_exceptions.yaml:247` | `distribution_sources/source/ggd/particles/values`; `m^-6.s^2` → `m^-3.s^-1`; graph_rewrite_and_integrity_suppression; correct_in_graph | 1 / 0 / — | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | dd_gap:distribution_sources/source/ggd/particles/values:self_contradiction; 1 | 1 (0 absent; 1 raw mismatch) |
| `U32` `imas_codex/units/dd_unit_exceptions.yaml:254` | `*/energy_fluxes/recombination/neutral/incident/values`; `m^-2.s^-1` → `W.m^-2`; integrity_suppression_only | 1 / 1 / — | FLT_1D; active | `c2f514ce90e2070a37e67aa0db0ebe67aa5aa2e235062a797d857a7b7ce820b0` | dd_gap:*/energy_fluxes/recombination/neutral/incident/values:unit_defect; 1 | 0 |
| `U33` `imas_codex/units/dd_unit_exceptions.yaml:271` | `*/position/psi`; `W` → `Wb`; graph_rewrite_and_integrity_suppression; correct_in_graph | 18 / 3 / — | FLT_1D,FLT_2D; alpha | `0e9c4264bc78fb1993b942c8ef273fd56f9a78dab7b8e9670f1e3c124922ca3f` | dd_gap:*/position/psi:self_contradiction; 19 | 16 (1 absent; 15 raw mismatch) |
| `U34` `imas_codex/units/dd_unit_exceptions.yaml:283` | `distributions/distribution/global_quantities/current_tor`; `N.m` → `A`; integrity_suppression_only | 0 / 0 / — | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | dd_gap:distributions/distribution/global_quantities/current_tor:unit_defect; 1 | 1 (1 absent; 0 raw mismatch) |
| `O01` `imas_codex/standard_names/config/unit_overrides.yaml:23` | `**/element/multiplicity`; `Elementary Charge Unit` → `1`; override | 0 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O02` `imas_codex/standard_names/config/unit_overrides.yaml:32` | `**/ionisation_potential`; `Elementary Charge Unit` → `eV`; override | 5 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O03` `imas_codex/standard_names/config/unit_overrides.yaml:38` | `**/ionization_potential`; `Elementary Charge Unit` → `eV`; override | 3 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O04` `imas_codex/standard_names/config/unit_overrides.yaml:44` | `**/binding_energy`; `Elementary Charge Unit` → `eV`; override | 0 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O05` `imas_codex/standard_names/config/unit_overrides.yaml:54` | `**/z_n`; `Elementary Charge Unit` → `1`; override | 102 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O06` `imas_codex/standard_names/config/unit_overrides.yaml:63` | `**/element/a`; `Atomic Mass Unit` → `u`; override | 89 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O07` `imas_codex/standard_names/config/unit_overrides.yaml:69` | `**/atomic_mass`; `Atomic Mass Unit` → `u`; override | 0 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O08` `imas_codex/standard_names/config/unit_overrides.yaml:75` | `**/a`; `Atomic Mass Unit` → `u`; override | 105 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O09` `imas_codex/standard_names/config/unit_overrides.yaml:84` | `**/z_n`; `e` → `1`; override | 102 / 102 / 102 | FLT_0D,INT_0D,STRUCTURE_0D; active,alpha | `3188f27aa2314790e12e620b3a0aa2a11139348572662885937d478bc06add22` | dd_gap:*/z_n:unit_defect; 106 | 0 |
| `O10` `imas_codex/standard_names/config/unit_overrides.yaml:90` | `**/charge_number`; `e` → `1`; override | 0 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O11` `imas_codex/standard_names/config/unit_overrides.yaml:96` | `**/z_ion`; `e` → `1`; override | 53 / 53 / 53 | FLT_0D,STRUCTURE_0D; active,alpha | `da5194caa030925441b9ee394bb36246fa28ef6fc80d6acb40b8ab0709a2019d` | dd_gap:*/z_ion:unit_defect; 56 | 0 |
| `O12` `imas_codex/standard_names/config/unit_overrides.yaml:102` | `**/z_average`; `e` → `1`; override | 8 / 8 / 8 | FLT_0D,STRUCT_ARRAY_1D; active,alpha | `1bcba69c7abbe1f7e267473c7e7ed3da41bd86de464a7aade3e63668e989daf4` | dd_gap:*/z_average:unit_defect; 8 | 0 |
| `O13` `imas_codex/standard_names/config/unit_overrides.yaml:108` | `**/z_square_average`; `e` → `1`; override | 8 / 8 / 8 | FLT_0D,STRUCT_ARRAY_1D; active,alpha | `454f98a74c27facd3b775a0a54707bfce44020665a1f87913d1620698d507d82` | dd_gap:*/z_square_average:unit_defect; 8 | 0 |
| `O14` `imas_codex/standard_names/config/unit_overrides.yaml:114` | `**/z_min`; `e` → `1`; override | 42 / 42 / 42 | FLT_0D; active,alpha | `a9a3c03ea7a4b3835bfb5f2cae305019e26630360f1cfa9f7ef04bcd91d1a9a9` | dd_gap:*/state/z_min:unit_defect; 44 | 0 |
| `O15` `imas_codex/standard_names/config/unit_overrides.yaml:120` | `**/z_max`; `e` → `1`; override | 42 / 42 / 42 | FLT_0D; active,alpha | `102ac685e780876a897173909e6fededf69111519022f93e9f184e736f472981` | dd_gap:*/state/z_max:unit_defect; 44 | 0 |
| `O16` `imas_codex/standard_names/config/unit_overrides.yaml:126` | `**/vibrational_level`; `e` → `1`; override | 66 / 66 / 66 | FLT_0D; active,alpha | `803aad4d71d355b617f1337b7592421132400edba40516afc8142841f3838656` | dd_gap:*/vibrational_level:unit_defect; 69 | 0 |
| `O17` `imas_codex/standard_names/config/unit_overrides.yaml:134` | `**/ionisation_potential`; `e` → `eV`; override | 5 / 2 / 2 | STRUCT_ARRAY_1D; active,alpha | `563441a75e328f79f9ebd4d37c31ee0b13ce8fcae80654cbba0254f32ccbbbaf` | dd_gap:*/ggd/ion/state/ionisation_potential*:self_contradiction; 22 | 0 |
| `O18` `imas_codex/standard_names/config/unit_overrides.yaml:140` | `**/ionization_potential`; `e` → `eV`; override | 3 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O19` `imas_codex/standard_names/config/unit_overrides.yaml:147` | `**/z`; `e` → `1`; override | 534 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O20` `imas_codex/standard_names/config/unit_overrides.yaml:163` | `**/*unit_vector*/*`; `m` → `1`; override | 1188 / 1188 / 1188 | FLT_0D; active,alpha | `9d4128226e5d34697f798d46e3f37a1c731683456d5cdd94fc3f0a17f8d1abc8` | dd_gap:*/unit_vector_major/[xyz]:unit_defect<br>dd_gap:*/unit_vector_minor/[xyz]:unit_defect; 6 | 0 |
| `O21` `imas_codex/standard_names/config/unit_overrides.yaml:169` | `**/direction/*`; `m` → `1`; override | 47 / 36 / 36 | FLT_0D; alpha | `525e6d1ca260a63f5f8bd18a0e4f7e8bae9c6434c1480f07fcbcb599cf20e44d` | dd_gap:*/direction/[xyz]:unit_defect; 12 | 0 |
| `O22` `imas_codex/standard_names/config/unit_overrides.yaml:175` | `**/direction_second/*`; `m` → `1`; override | 9 / 9 / 9 | FLT_0D; alpha | `c54f8d1153c121261a2da312213c09189db7ec6d4a294adb0525b8a90a3f8b57` | dd_gap:*/direction_second/[xyz]:unit_defect; 3 | 0 |
| `O23` `imas_codex/standard_names/config/unit_overrides.yaml:181` | `**/up/*`; `m` → `1`; override | 9 / 9 / 9 | FLT_0D; alpha | `b8f80b96f6707cc7064e464b4a3f0fb784f1eb4763f936185980a4f0e685cda9` | dd_gap:*/up/[xyz]:unit_defect; 3 | 0 |
| `O24` `imas_codex/standard_names/config/unit_overrides.yaml:187` | `**/injection_direction/*`; `m` → `1`; override | 9 / 9 / 9 | FLT_0D; alpha | `ba2087cc346ed488497c057c460897c04cd157ce6c6922a5077f5792debf3a44` | dd_gap:*/injection_direction/[xyz]:unit_defect; 3 | 0 |
| `O25` `imas_codex/standard_names/config/unit_overrides.yaml:198` | `**`; `m^dimension` → `dd_unit_unresolvable`; skip | 44150 / 18 / 18 | FLT_0D; active,alpha | `a8548fee00ff5b1417ad993fbd87b3e8a0d3d4356d753d7762a8f3a7a39ebc61` | —; 0 | 0 |
| `O26` `imas_codex/standard_names/config/unit_overrides.yaml:209` | `pulse_schedule/**/reference`; `1` → `dd_unit_context_dependent`; skip | 68 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O27` `imas_codex/standard_names/config/unit_overrides.yaml:215` | `pulse_schedule/**/reference/data`; `1` → `dd_unit_context_dependent`; skip | 0 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |
| `O28` `imas_codex/standard_names/config/unit_overrides.yaml:221` | `pulse_schedule/**/reference_waveform/data`; `1` → `dd_unit_context_dependent`; skip | 0 / 0 / 0 | —; — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | —; 0 | 0 |

### 5.1 Exact release findings that change the review input

- Forty-five rows have at least one exact DD 4.1.1 raw tuple. Seventeen do not: `U20`, `U31`, `U34`, `O01`, `O02`, `O03`, `O04`, `O05`, `O06`, `O07`, `O08`, `O10`, `O18`, `O19`, `O26`, `O27`, `O28`.
- The zero-match result has different causes and is not itself a retirement decision. U20, U34, O04, O07, O10, O27 and O28 have no matching published path; U31 and O01-O08/O18/O19/O26 have a path or family but not the legacy raw unit in DD 4.1.1. For example, pulse-schedule `reference` nodes now publish mostly `mixed`, not `1`.
- O20 expands to 1,188 exact DD 4.1.1 `m` paths; only six overlap the narrow U15/U16 graph facts. Machine expansion proves the cohort, not that every member is semantically a direction cosine.
- O21 expands to 36 exact raw tuples but only 12 overlap U11 evidence. O22-O24 each expand to nine exact tuples while each narrow U fact covers three x/y/z paths.
- O25 expands to 18 exact `m^dimension` paths and remains a qualification input; it has no effective DD value and no DDGap resolution binding.
- U24 expands deterministically to `spi/injector/fragmentation_gas/flow_rate` and `spi/injector/propellant_gas/flow_rate`, both raw `s^-1`, `FLT_1D`, dynamic, COCOS 17, inherited alpha lifecycle.
- U33 expands to exactly three raw-`W` paths: `ece/channel/position/psi`, `reflectometer_fluctuation/channel/doppler/position/psi`, and `reflectometer_fluctuation/channel/fluctuations_level/position/psi`.

## 6. Multiplicity and collision checks

| Check | Result |
|---|---:|
| Duplicate DDGap IDs | 0 |
| Duplicate `(path, kind)` DDGap keys | 0 |
| Observation IDs linked to multiple DDGap facts | 0 |
| Source paths linked to multiple DDGap facts | 1 |
| Exact paths matched by multiple legacy rows | 348 |
| Graph source paths absent from immutable DD 4.1.1 | 35 |

The one multi-gap source path is U21, intentionally linked to both the registered unit-defect fact and the separately filed `type_wiring` fact. The 348 exact-path overlaps are fully enumerated in JSON and include the expected U/O overlap plus broad O-rule overlap. No collision was resolved or silently coalesced into authority.

## 7. Authority adjudication result

**Eligible solely from machine evidence: 0 of 62 rows.**

Machine evidence now proves the installed release identity, exact raw tuples, exact type/COCOS/lifecycle facts, current graph bindings, evidence tokens, and collision set. It cannot supply any of the following required non-machine facts:

1. an exact upstream solution/change identity (`upstream_ref`), not merely an issue URL;
2. a durable governed approval receipt;
3. a reviewer/approval actor and approval time;
4. a governed decision reason and positive resolution revision.

The current graph contains zero DDResolution nodes, zero `FOR_DD_VERSION` relationships, zero approval fields, and zero resolution state-change receipts. Machine-written `triaged_at`, registry reasons, `registry_backfill` reporters, and the registry-sync identity event are provenance, not approval. The U21 issue URL is also attached to a separate `type_wiring` fact; it does not authorize a unit resolution and contains no exact upstream solution ref.

## 8. Executable next-action manifest

The JSON contains a non-authoritative `executable_review_manifest` with 1,619 deduplicated exact replacement review tuples, four skip-policy review items, and 17 unmatched legacy rows. Every candidate tuple is marked `NOT_AUTHORIZED_REVIEW_INPUT_ONLY`, carries its source row IDs and any candidate gap/observation/token bindings, and lists the missing required fields. It deliberately does not conform to the packaged active-resolution schema and cannot affect runtime.

The safe review DAG is:

1. Review the 57 graph-vs-release conflicts and govern whether each observation is narrowed, corrected, or rejected. Re-fetch each affected fact afterward because observation changes alter the evidence token.
2. Review the 1,619 exact replacement tuples. Resolve the 348 path overlaps and explicitly decide the broad O20-O24 members; do not inherit narrow U approvals across broader O cohorts.
3. Attach exact upstream URL plus solution/change ref to every selected tuple. Issue 272 alone is insufficient.
4. Record `approved_by`, `approved_at`, `approval_receipt`, governed decision reason, and positive revision against the exact path/version/raw/effective tuple and its fresh observation set.
5. Construct a separate candidate `dd_resolutions.yaml`, then run the existing fail-closed validation/dry-run path. Only successfully validated, explicitly active records may become behavior authority.
6. Keep O25-O28 as qualification-policy review items. A skip is not an effective DD value. Decide the 17 zero-match rows separately as version-scoped obsolete/retire/hold outcomes; zero match is evidence, not adjudication.

Read-only fact refresh commands are executable today:

```bash
uv run imas-codex sn ddgap --list
uv run imas-codex sn ddgap --show 'dd_gap:<exact-pattern>:<kind>'
```

A governed transition, if later authorized, must use the fresh full token and all mandatory CAS fields:

```bash
uv run imas-codex sn ddgap --triage 'dd_gap:<exact-pattern>:<kind>' \
  --expected-status <current-status> \
  --expected-evidence-token 'dd-gap-evidence:<full-sha256>' \
  --to-status <authorized-status> --actor '<human-or-governed-authority>' \
  --reason '<governed-decision-reason>' --apply
```

This command is shown as the existing governed mechanism, not authorization to run it. There is no safe existing command to activate a typed resolution without first supplying the missing upstream and approval fields.

## 9. Inputs and hashes

| Input | SHA-256 |
|---|---|
| `/tmp/reckon-s8-scope/dd-resolution-scope.md` | `164bdd74c3a856cbd0510c3594eb84dfd463a7deead18ac9f0e875a467493af4` |
| `/tmp/reckon-s8-scope/dd-resolution-transition-audit.md` | `fd5bb1e70ab0945df501550c38b0b202740eea0046b53d6d17c7b1c88aabbaf1` |
| `imas_codex/schemas/imas_dd.yaml` | `94b4c8baea23d836795854c078c228c3fb2208b64ff24a000ef6b561eeb3609f` |
| `imas_codex/schemas/standard_name.yaml` | `73e750d38b674e9fd61156550a5c07a24eb27ceb6fe18add1e32a1271345c2c2` |
| `imas_codex/standard_names/config/dd_resolutions.yaml` | `64c20eb0405022f33265e4bc222919c25f51b1c98b00b6e473ff615c963b33cf` |
| `imas_codex/standard_names/config/unit_overrides.yaml` | `13310c50b546cc4860779e641495195f79f24875ae0c058d430141cc8fdb151d` |
| `imas_codex/units/dd_unit_exceptions.yaml` | `81dcd027ff5f14c87a8afc6334e41414e5cc67c14c062d897438dbaaf31bcdf6` |

## 10. Validation receipt

Validation was intentionally limited to static/tracked-data inspection, canonical read-only graph queries, and immutable installed-DD API inspection. No pytest, build, model generation, provider call, pipeline, graph mutation, service control, facility operation, or IMAS data read occurred.

```text
git show --stat 32aa552d6f9a69339bb78c1be58b4af6a73905c0
32aa552d docs: record semantic authority foundations
 docs/plans/llm-context-integrity.html | 10 ++++++++--
 docs/sn-dd-gaps.html                  |  9 ++++++---
 2 files changed, 14 insertions(+), 5 deletions(-)

git status --short --branch
## HEAD (no branch)
```

The detached worktree was clean at the final pre-write check. Only the external Markdown/JSON artifacts were written. No repository commit is expected.
