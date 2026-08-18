NEEDS-HELP: The 14-path disposition is complete, but a fresh live-graph read could not be obtained within the read-only fence after two environment fixes failed; no graph write was issued.

tried: Read-only `imas_cx` schema/search calls were cancelled. A scoped `uv run` graph query then failed because uv tried to create `.venv` in the read-only worktree; using the canonical environment against worktree code failed because importing `imas_codex.graph` tried to regenerate missing models in the same read-only tree. The stop-after-two-fixes rule prevents a third workaround. The repository remains clean.

options: (1) Have the orchestrator run the included MATCH-only Cypher through the working read-only MCP/REPL; (2) authorize a disposable editable copy under `/tmp` and run GraphClient there; (3) accept the 2026-08-10 graph projection plus the 2026-08-17 approval refusal as the graph evidence for this disposition.

leaning: Option 1. It gives current graph truth through the sanctioned client, performs no write, and does not weaken the read-only worktree fence.

cost-if-wrong: If the 2026-08-10 graph projection has drifted, the evidence-fact materialization list and fresh approval tokens must be regenerated; the immutable DD 4.1.1 and PR 280 path dispositions below do not change.

# U19 per-path disposition brief

## Decision

The packaged six-path U19 split is **not exactly correct for `sn ddres` exact-path authority**. It identifies the six primary DD fields named by PR 280, but omits eight flattened error-bar fields whose DD 4.1.1 raw unit is also `e` and whose unit is inherited from the same corrected `values` or `coefficients` declaration. The exact governed U19 cohort is therefore **14 paths**, all changing `e` to `eV` under the same PR 280 solution provenance.

Quantitative result:

- DD 4.1.1 U19 raw-tuple matches adjudicated: **14/14**.
- Direct primary fields covered by PR 280: **6/14**.
- Generated error-bar fields covered transitively by the same schema fix: **8/14**.
- U19 matches that are legitimate charge-number fields retained in `e`: **0/14**.
- Adjacent charge-number fields explicitly outside U19 and retained in `e`: **8** representative exact paths (four per IDS), listed below.
- Other/unresolved dispositions inside the 14-path cohort: **0**.

Why 6 and 14 are both observed:

1. PR 280 commit `30a5ddd4b7037b9f93a8f00f7837809403349d99` changes the one `ionization_potential` declaration in `schemas/utilities/dd_support.xsd` from `e` to `eV`. Its commit message describes six affected primary paths: the structure, `values`, and `coefficients` under each of `edge_profiles` and `plasma_profiles`.
2. `generic_grid_scalar` declares `values` and `coefficients` with `units=as_parent` (`schemas/utilities/dd_support.xsd`, around lines 5354-5384).
3. The official flattening transform generates `_error_upper` and `_error_lower` for real-valued fields and copies the source field's appinfo attributes, including resolved units (`dd_data_dictionary.xml.xsl`, around lines 309-380). The official error-bar documentation defines those nodes as absolute errors around the data value (`docs/errorbars.rst`, lines 8-27).
4. Consequently the same schema correction changes the effective flattened unit for each `values_error_{lower,upper}` and `coefficients_error_{lower,upper}` field. They are uncertainty values of an energy, not charge numbers.

The immutable input that exposed the conflict is `/tmp/reckon-s8-scope/dd-resolution-evidence-export.json`: its U19 object records `exact_raw_tuple_path_count=14`, with all fourteen paths below at DD `4.1.1` and raw unit `e`. It separately records four empty-unit index paths and four absent legacy `*_error_index` claims; neither group belongs in the 14-path correction cohort.

## Exact 14-path disposition

Every row has exactly one disposition. “PR 280 inherited unit fix” is not analogy: it follows the DD's own `as_parent` expansion and error-bar-generation transform.

| # | DD 4.1.1 exact path | Raw | Effective | Disposition | Citation |
|---:|---|---:|---:|---|---|
| 1 | `edge_profiles/ggd/ion/state/ionisation_potential` | `e` | `eV` | PR 280 direct unit fix | PR 280 commit `30a5ddd4`; corrected `ionization_potential` declaration; immutable U19 export |
| 2 | `edge_profiles/ggd/ion/state/ionisation_potential/coefficients` | `e` | `eV` | PR 280 direct unit fix | PR 280 commit `30a5ddd4`; `generic_grid_scalar/coefficients` has `units=as_parent` |
| 3 | `edge_profiles/ggd/ion/state/ionisation_potential/coefficients_error_lower` | `e` | `eV` | PR 280 inherited unit fix | `dd_data_dictionary.xml.xsl` generates lower errors and copies the coefficients appinfo/unit; immutable U19 export |
| 4 | `edge_profiles/ggd/ion/state/ionisation_potential/coefficients_error_upper` | `e` | `eV` | PR 280 inherited unit fix | `dd_data_dictionary.xml.xsl` generates upper errors and copies the coefficients appinfo/unit; immutable U19 export |
| 5 | `edge_profiles/ggd/ion/state/ionisation_potential/values` | `e` | `eV` | PR 280 direct unit fix | PR 280 commit `30a5ddd4`; `generic_grid_scalar/values` has `units=as_parent` |
| 6 | `edge_profiles/ggd/ion/state/ionisation_potential/values_error_lower` | `e` | `eV` | PR 280 inherited unit fix | `dd_data_dictionary.xml.xsl` lower-error propagation; `docs/errorbars.rst`; immutable U19 export |
| 7 | `edge_profiles/ggd/ion/state/ionisation_potential/values_error_upper` | `e` | `eV` | PR 280 inherited unit fix | `dd_data_dictionary.xml.xsl` upper-error propagation; `docs/errorbars.rst`; immutable U19 export |
| 8 | `plasma_profiles/ggd/ion/state/ionisation_potential` | `e` | `eV` | PR 280 direct unit fix | PR 280 commit `30a5ddd4`; corrected `ionization_potential` declaration; immutable U19 export |
| 9 | `plasma_profiles/ggd/ion/state/ionisation_potential/coefficients` | `e` | `eV` | PR 280 direct unit fix | PR 280 commit `30a5ddd4`; `generic_grid_scalar/coefficients` has `units=as_parent` |
| 10 | `plasma_profiles/ggd/ion/state/ionisation_potential/coefficients_error_lower` | `e` | `eV` | PR 280 inherited unit fix | `dd_data_dictionary.xml.xsl` lower-error propagation; immutable U19 export |
| 11 | `plasma_profiles/ggd/ion/state/ionisation_potential/coefficients_error_upper` | `e` | `eV` | PR 280 inherited unit fix | `dd_data_dictionary.xml.xsl` upper-error propagation; immutable U19 export |
| 12 | `plasma_profiles/ggd/ion/state/ionisation_potential/values` | `e` | `eV` | PR 280 direct unit fix | PR 280 commit `30a5ddd4`; `generic_grid_scalar/values` has `units=as_parent` |
| 13 | `plasma_profiles/ggd/ion/state/ionisation_potential/values_error_lower` | `e` | `eV` | PR 280 inherited unit fix | `dd_data_dictionary.xml.xsl` lower-error propagation; `docs/errorbars.rst`; immutable U19 export |
| 14 | `plasma_profiles/ggd/ion/state/ionisation_potential/values_error_upper` | `e` | `eV` | PR 280 inherited unit fix | `dd_data_dictionary.xml.xsl` upper-error propagation; `docs/errorbars.rst`; immutable U19 export |

## Explicit non-members and retained charge semantics

The following adjacent fields are **not U19 matches** and must not be added to the candidate. PR 280's commit rationale explicitly says they are charge numbers legitimately annotated `e`:

- `edge_profiles/ggd/ion/state/{z_min,z_max,z_average,z_square_average}` — retain `e`.
- `plasma_profiles/ggd/ion/state/{z_min,z_max,z_average,z_square_average}` — retain `e`.

This is the important semantic boundary: ionisation potential and its absolute errors are energies (`eV`); minimum, maximum, mean, and mean-square charge-number fields remain charge declarations (`e`). The plan's `dd-upstream-provenance-20260810` comment and the upstream commit both forbid generalizing the ionisation-potential fix into charge-row authority.

Also excluded from the 14-path U19 cohort:

- `.../ionisation_potential/grid_index` and `.../grid_subset_index` in both IDSs: DD 4.1.1 publishes empty units, not raw `e`; their integer indices are not energies.
- Four historical `.../{coefficients,values}_error_index` claims: absent from the immutable DD 4.1.1 release and therefore cannot receive a 4.1.1 local resolution.

## U19/O17 overlap

O17 is internally consistent at **2 release matches / 2 exact paths**: the two base structure paths. It is a narrower legacy extraction override that overlaps U19, not evidence that U19 should remain six paths. A single active resolution key must not be approved twice. The minimal canonical ownership is to approve the complete 14-path U19 cohort and treat the two O17 entries as redundant overlap for retirement during cutover; if governance prefers O17 ownership for the two base paths, that choice must be recorded explicitly before approvals because active-key collision is fail-closed.

## Minimal governed steps to clear the refusal

1. **Candidate-data change — repository author + independent reviewer authority.** Replace U19's six `exact_paths` with the fourteen paths above; keep `source_release_match_count: 14`, `dd_version: 4.1.1`, raw `e`, effective `eV`, and upstream commit `30a5ddd4`. Pin tests that the candidate cohort is exactly 14, includes all eight upper/lower error fields, excludes the empty-unit index fields and absent error-index fields, and leaves O17 at 2/2. This changes the candidate-resource digest and requires the normal independent candidate review before integration.
2. **Exact graph evidence — sanctioned DDGap-writer authority.** Materialize/review an exact current DDGap fact and observation set for each of the fourteen paths via the repository's evidence writer, never raw Cypher. Each fact must bind path, `self_contradiction` or other reviewer-selected admissible unit kind, DD `4.1.1`, observed `e`, expected `eV`, and the exact observation identities. The old wildcard backfill fact and its broad observation set are not sufficient for the approval gate's exact-per-path requirement.
3. **Renew approval delegation — lead/governed approver authority.** The 2026-08-17 delegation is bound to candidate SHA-256 `c6ee52ae...`; changing U19 creates a new digest outside that scope. The lead must delegate/approve the reviewed new digest and record the source-row ownership decision for the two U19/O17 overlaps.
4. **Mint per-path authority — delegated `sn ddres approve` authority.** With fresh graph-derived tokens, approve the fourteen U19 records using actor, reason, timestamp, positive revision, exact PR 280 commit provenance, and expected-manifest-digest CAS. PR 280 is open and unreleased, but the plan's locked typed-local-resolution decision permits reviewed local authority before upstream publication; the receipt must preserve that open/unreleased state and proposed fixed boundary rather than claiming a released fix.
5. **Independent verification — reviewer/test authority.** Re-run `sn ddres show U19` and the approval refusal/adversarial suite, prove 14/14 U19 paths attributed to the new U19 candidate digest, prove O17 does not create duplicate active keys, and prove charge-number and index paths remain unchanged. Consumer cutover and retirement of the legacy U19/O17 overrides remain a later separately gated mutation.

## Current graph evidence and remaining blocker

The last successful graph projection available to this node is the read-only 2026-08-10 export. It reports one wildcard `self_contradiction` fact, 22 historical observation identities/source claims, fourteen immutable raw-`e` tuples, four empty-unit index paths, and four absent error-index claims. The 2026-08-17 live approval run then refused all six packaged U19 attempts on the 6-reviewed-vs-14-release-match conflict. Those two records support the candidate correction and predict that exact per-path evidence materialization is the next gate.

The intended fresh read was MATCH-only:

```cypher
MATCH (n:IMASNode)
WHERE n.id STARTS WITH 'edge_profiles/ggd/ion/state/ionisation_potential'
   OR n.id STARTS WITH 'plasma_profiles/ggd/ion/state/ionisation_potential'
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (n)-[:HAS_DD_GAP]->(g:DDGap)
RETURN n.id AS path,
       n.data_type AS data_type,
       n.unit AS unit_scalar,
       collect(DISTINCT u.id) AS unit_edges,
       collect(DISTINCT g.id) AS dd_gap_ids
ORDER BY path
```

It issued no mutation clause. The fresh result is still required before graph evidence is materialized or approval tokens are minted.

## Evidence inputs

- Live plan: `docs/sn-dd-gaps.html`, especially comments `dd-upstream-provenance-20260810` and `ddres-first-approvals-20260817`.
- Packaged candidates: `imas_codex/standard_names/config/dd_resolution_candidates.yaml`, U19 lines 149-165 and O17 lines 262-274.
- Legacy enforcement row: `imas_codex/units/dd_unit_exceptions.yaml`, lines 125-140.
- Immutable DD/reconciled graph export: `/tmp/reckon-s8-scope/dd-resolution-evidence-export.json`, U19 and O17 objects.
- Upstream provenance audit: `/tmp/reckon-s8-scope/dd-upstream-provenance.md`, especially rows 36, 68, 105, 124-127, and 134.
- Official upstream checkout: `/home/ITER/mcintos/Code/data-dictionary`, commit `30a5ddd4b7037b9f93a8f00f7837809403349d99`; `schemas/utilities/dd_support.xsd`; `dd_data_dictionary.xml.xsl`; `docs/errorbars.rst`.

## Fence audit

- Repository modifications: **0**.
- Successful graph writes issued: **0**.
- Graph mutation clauses issued: **0**.
- Fresh successful graph reads: **0** (blocker above).
- Dispositions completed: **14/14**, with **0** unresolved semantic path assignments.
