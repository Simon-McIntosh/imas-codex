# Standard Name release-readiness closure audit

Audited 2026-08-24T17:52:13+02:00 against tree commit
`5dbad32f60e9b1bc44ae1ade4301a10046e7b715` and the live production `codex`
graph. The plan's live HTML at version 41 is the semantic authority. Historical
counts in that plan and its evidence records are treated as observations, not
as substitutes for a fresh graph measurement.

The verdict is strict: a section is **landed** only when its declared
deliverable is present in the tree and its applicable live-graph condition is
satisfied, or when the declared deliverable was explicitly a read-only,
pre-spend census whose durable identity-level record is present. If any
declared part remains open, the whole section is **outstanding**.

| Section | Declared deliverable | Live-tree evidence | Live-graph evidence | Verdict |
|---:|---|---|---|---|
| 2 | Close unscored accepted-entry paths: protect pipeline review provenance, drain the catalog-import cohort, and require every accepted name to carry either its own name score or a durable structural authority. | The symmetric catalog-write guard is present and fail-closed (`imas_codex/standard_names/protection.py:77-127`), and the structural-authority schema/test surface exists (`tests/graph/test_sn_accept_authority.py:80-114`). The live graph invariant test does **not** assert zero overall residual; it only rejects childful bare structural markers and preserves five childless exceptions (`tests/graph/test_sn_accept_authority.py:220-230`). | **Q2:** accepted 2,335; scored 1,894; structural authority 250; overlap 0; neither 191. The residual partitions as 96 catalog edits with no reviewer, 84 derived rows with no reviewer, 5 catalog edits carrying only the structural marker, and 6 rows with a named reviewer but no score or authority. | **outstanding** — the import guard landed, but the plan's accepted-authority invariant is false for 191 live rows. |
| 3 | Make every accepted-to-emitted drop attributable to exactly one identity-bearing reason and fail when `emitted + exclusions != accepted`. | `ExclusionRecord` carries identity, stage, reason, and detail (`imas_codex/standard_names/export.py:130-144`); the report serializes emitted identities and the full exclusion ledger (`imas_codex/standard_names/export.py:196-223`); the gate checks duplicates, overlaps, missing identities, and arithmetic closure (`imas_codex/standard_names/export.py:790-835`). The regression proves both closure and refusal on an unattributed identity (`tests/standard_names/test_export_exclusion_ledger.py:76-134`). | **Q3, actual export path:** 2,030 emitted + 305 exclusions = 2,335 accepted; accounting gate passed with zero issues; all 2,030 emitted identities were serialized. Exclusions were 266 documentation not accepted, 19 invalid validation state, 12 review resolution unrecorded, 3 invalid catalog entries, 2 below score, 2 bound-adjacent, and 1 name-review quorum shortfall. This reproduces the durable run record (`docs/evidence/sn-release-readiness/gate-eligibility.md:83-105`). | **landed**. |
| 4 | Keep supersede lineage internal and prevent successor-bearing deprecation stubs from appearing in a released catalog. | The tree still implements the opposite policy: `_build_stub_entry` emits `status: deprecated`, `superseded_by`, successor prose, and a successor link (`imas_codex/standard_names/export.py:993-1015`), and the export loop still appends those stubs (`imas_codex/standard_names/export.py:1938-1970`). | **Q4:** 1,914 superseded names; 1,402 have reachable `REFINED_FROM` lineage, 17 have a scalar successor, 509 have neither, and 0 have `catalog_approved_at`. The defect is therefore dormant on today's graph but will activate after a first catalog approval. | **outstanding** — the required stub removal is absent. |
| 5 | Drain the accepted-name documentation and validity backlog to a terminal, publishable state rather than letting release filters hide it. | The sanctioned docs campaign record explicitly stopped at its cost fence with unfinished work (`docs/evidence/sn-release-readiness/docs-106-rescore.md:3-17`). The deterministic revalidation repair is present, but its evidence also retains named quarantines rather than declaring them cleared (`docs/evidence/sn-release-readiness/revalidation.md:114-136`). | **Q5:** among 2,335 accepted names, 270 do not have accepted docs: 138 pending, 68 drafted, 52 reviewed, and 12 exhausted. Nineteen accepted names remain quarantined; null validation status is 0. | **outstanding** — both declared backlog axes remain nonzero. |
| 6 | Before spending, partition every non-accepted or exhausted name into redraw-eligible, needs-steering, or correctly-abandoned, with complete counts and source impact. | The durable, identity-level census is present and closes exactly: 144 redraw-eligible + 72 needs-steering + 1 correctly abandoned = 217, residual 0, with 431 distinct stranded sources (`docs/evidence/sn-release-readiness/name-tail-census.md:1-25`). This was explicitly a read-only pre-spend evidence deliverable, not a request to stamp disposition fields onto graph nodes. | **Q6:** the graph has since evolved through the recorded review/refine campaigns and now contains 417 current tail nodes: 5 pending, 21 drafted, 124 reviewed, and 267 exhausted. That drift does not invalidate the time-stamped pre-spend partition; the subsequent campaign records 273 roots terminal and zero eligible work for its exact cohort (`docs/evidence/sn-release-readiness/names-refine-finish.md:3-35`). | **landed** — the declared census, rather than universal tail elimination, is present. |
| 7 | Resolve release-facing metadata gaps: project missing physics domains, dispose missing documentation, confirm null catalog status is harmless, and close the bare-reference detector/normalizer hole. | Catalog serialization unconditionally emits admitted rows as `status: active`, confirming null/draft graph status is publication-safe (`imas_codex/standard_names/export.py:897-915`). A shared math-aware parser exists (`imas_codex/standard_names/doc_links.py:13-20`, `49-88`), but the accept-time normalizer still uses its separate `_BARE_DOC_LINK_RE` (`imas_codex/standard_names/graph_ops.py:9079-9084`, `9181-9215`) and catches normalization failures without refusing acceptance (`imas_codex/standard_names/graph_ops.py:15858-15876`). | **Q7:** among 2,335 accepted names, missing physics domain is 0, but missing documentation is 75; catalog status remains 929 null and 1,406 draft, both safely projected by the exporter. The domain projection landed, but the documentation and fail-closed normalizer conditions do not. | **outstanding**. |
| 8 | Give release authority or an explicit release disposition to the inherited source-less identities, dual-bound sources, source-less antenna rename, and reflector-centre identity rather than letting filter accidents decide them. | The prior closure record preserves the standing-refusal conditions for the source-less identities (`docs/evidence/archive/sn-integrity-residual-closure-landed.html:553-567`), and the reflector evidence records that its source remains bound to `toroidal_angle_of_measurement_position` (`docs/evidence/sn-graph-wide-integrity/owner-geometry-ready-apply.md:45-63`). No tree artifact in this plan supplies the missing release authority. | **Q8:** `neutron_flux_due_to_fusion` and `tendency_of_total_thermal_plasma_internal_energy` remain accepted with 0 sources; one live source remains dual-bound (`dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial`, two live targets); `voltage_of_diagnostic_antenna` is exhausted with no `voltage_of_ece_channel` successor; and `reflector/centre/phi` remains attached to `toroidal_angle_of_measurement_position`. | **outstanding** — the residue has narrowed, but the declared release dispositions are not closed. |

## Quantitative closure

- Sections audited: **7**.
- Landed: **2** (sections 3 and 6).
- Outstanding: **5** (sections 2, 4, 5, 7, and 8).

## Live query record

All queries were read-only. Property coverage was checked before trusting zero
results: the graph contained 4,666 `StandardName` nodes with full coverage of
`name_stage`; 24,789 `StandardNameReview` nodes with full `review_axis`
coverage; 9,639 `StandardNameSource` nodes with full `id` and `status`
coverage; and 250 `HAS_STRUCTURAL_AUTHORITY` edges whose target records all
carried `id`, `accepted_name_id`, `child_ids`, `children`, and `created_at`.

**Q2 — accepted-name authority partition**

```cypher
MATCH (s:StandardName {name_stage: 'accepted'})
OPTIONAL MATCH (s)-[:HAS_STRUCTURAL_AUTHORITY]->(a:StructuralNameAuthority)
WITH s, count(a) > 0 AS has_structural_authority
RETURN count(s) AS accepted,
       count(CASE WHEN s.reviewer_score_name IS NOT NULL THEN 1 END) AS scored,
       count(CASE WHEN has_structural_authority THEN 1 END) AS structural,
       count(CASE WHEN s.reviewer_score_name IS NOT NULL
                   AND has_structural_authority THEN 1 END) AS overlap,
       count(CASE WHEN s.reviewer_score_name IS NULL
                   AND NOT has_structural_authority THEN 1 END) AS residual
```

Result: `accepted=2335, scored=1894, structural=250, overlap=0, residual=191`.

**Q3 — actual export-path accounting**

The production `run_export` path was invoked read-only at `min_score=0.85`,
`skip_gate=True`, `force=True`, and `include_sources=False`. Result:
`accepted_population=2335, emitted=2030, exclusions=305,
accounted_total=2335, accounting_gate_passed=true, issues=[]`, with 2,030
serialized emitted identities.

**Q4 — superseded-lineage state**

```cypher
MATCH (old:StandardName {name_stage: 'superseded'})
WITH old,
     EXISTS { MATCH (:StandardName)-[:REFINED_FROM*1..]->(old) }
       AS has_refined_successor,
     old.superseded_by IS NOT NULL AS has_scalar
RETURN count(old) AS superseded,
       count(CASE WHEN has_refined_successor THEN 1 END) AS edge_reachable,
       count(CASE WHEN has_scalar THEN 1 END) AS scalar,
       count(CASE WHEN NOT has_refined_successor AND NOT has_scalar THEN 1 END)
         AS neither,
       count(CASE WHEN old.catalog_approved_at IS NOT NULL THEN 1 END)
         AS catalog_approved
```

Result: `superseded=1914, edge_reachable=1402, scalar=17, neither=509,
catalog_approved=0`.

**Q5 — accepted documentation and validation backlog**

```cypher
MATCH (s:StandardName {name_stage: 'accepted'})
RETURN count(s) AS accepted,
       count(CASE WHEN s.docs_stage <> 'accepted' OR s.docs_stage IS NULL
                   THEN 1 END) AS docs_not_accepted,
       count(CASE WHEN s.docs_stage = 'pending' THEN 1 END) AS pending,
       count(CASE WHEN s.docs_stage = 'drafted' THEN 1 END) AS drafted,
       count(CASE WHEN s.docs_stage = 'reviewed' THEN 1 END) AS reviewed,
       count(CASE WHEN s.docs_stage = 'exhausted' THEN 1 END) AS exhausted,
       count(CASE WHEN s.validation_status = 'quarantined' THEN 1 END)
         AS quarantined
```

Result: `accepted=2335, docs_not_accepted=270, pending=138, drafted=68,
reviewed=52, exhausted=12, quarantined=19`.

**Q6 — current lifecycle tail**

```cypher
MATCH (s:StandardName)
WHERE s.name_stage IN ['pending', 'drafted', 'reviewed', 'exhausted']
RETURN count(s) AS tail,
       count(CASE WHEN s.name_stage = 'pending' THEN 1 END) AS pending,
       count(CASE WHEN s.name_stage = 'drafted' THEN 1 END) AS drafted,
       count(CASE WHEN s.name_stage = 'reviewed' THEN 1 END) AS reviewed,
       count(CASE WHEN s.name_stage = 'exhausted' THEN 1 END) AS exhausted
```

Result: `tail=417, pending=5, drafted=21, reviewed=124, exhausted=267`.

**Q7 — accepted metadata state**

```cypher
MATCH (s:StandardName {name_stage: 'accepted'})
RETURN count(s) AS accepted,
       count(CASE WHEN s.physics_domain IS NULL OR s.physics_domain = ''
                   THEN 1 END) AS missing_domain,
       count(CASE WHEN s.documentation IS NULL OR s.documentation = ''
                   THEN 1 END) AS missing_documentation,
       count(CASE WHEN s.status IS NULL THEN 1 END) AS null_catalog_status,
       count(CASE WHEN s.status = 'draft' THEN 1 END) AS draft_catalog_status
```

Result: `accepted=2335, missing_domain=0, missing_documentation=75,
null_catalog_status=929, draft_catalog_status=1406`.

**Q8 — inherited residue**

The live graph returned two accepted source-less identities
(`neutron_flux_due_to_fusion`,
`tendency_of_total_thermal_plasma_internal_energy`), one dual-bound live source
(`dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` bound to
`radial_neutral_internal_state_momentum_flux` and
`radial_neutral_state_momentum_flux`), an exhausted
`voltage_of_diagnostic_antenna` with no `voltage_of_ece_channel` node, and
`dd:spectrometer_x_ray_crystal/channel/reflector/centre/phi` still attached to
`toroidal_angle_of_measurement_position`.
