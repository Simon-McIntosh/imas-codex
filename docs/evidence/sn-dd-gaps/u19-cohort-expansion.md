# Ionisation-potential resolution cohort expansion

Recorded 2026-08-19 against DD 4.1.1 and the live `codex` graph. The official
upstream authority is [IMAS Data Dictionary pull request 280](https://github.com/iterorganization/IMAS-Data-Dictionary/pull/280),
commit `30a5ddd4b7037b9f93a8f00f7837809403349d99`.

## Exact cohort

All rows bridge the published unit `e` to the effective energy unit `eV`.

| Path | Pre-existing exact DDGap | Pre-existing resolution | Pre unit | Post exact DDGap | Post resolution and edges | Post unit |
| --- | --- | --- | --- | --- | --- | --- |
| `edge_profiles/ggd/ion/state/ionisation_potential` | yes | yes | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `edge_profiles/ggd/ion/state/ionisation_potential/coefficients` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `edge_profiles/ggd/ion/state/ionisation_potential/coefficients_error_lower` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `edge_profiles/ggd/ion/state/ionisation_potential/coefficients_error_upper` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `edge_profiles/ggd/ion/state/ionisation_potential/values` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `edge_profiles/ggd/ion/state/ionisation_potential/values_error_lower` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `edge_profiles/ggd/ion/state/ionisation_potential/values_error_upper` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `plasma_profiles/ggd/ion/state/ionisation_potential` | yes | yes | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `plasma_profiles/ggd/ion/state/ionisation_potential/coefficients` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `plasma_profiles/ggd/ion/state/ionisation_potential/coefficients_error_lower` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `plasma_profiles/ggd/ion/state/ionisation_potential/coefficients_error_upper` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `plasma_profiles/ggd/ion/state/ionisation_potential/values` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `plasma_profiles/ggd/ion/state/ionisation_potential/values_error_lower` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |
| `plasma_profiles/ggd/ion/state/ionisation_potential/values_error_upper` | no | no | `eV` | yes | 1 record; 1 bridge, 1 evidence, 1 version | `eV`; one `HAS_UNIT→eV` |

The pre-census showed that the broad build-time correction had already set all
14 graph units to `eV`; the missing work was exact evidence and provenance for
12 descendants. Consequently the apply receipt classified all 12 as
`attached`, not `corrected`. The write boundary nevertheless retains an exact
published-value compare-and-set and refuses any scalar/relationship mismatch.

Excluded from the cohort are index fields, absent error-index claims, and all
charge-number fields. The upstream change intentionally retains charge-number
units as `e`; none were included in the allowlist or touched.

## Receipts and verification

- Pre-census: `/tmp/reckon-u19-cohort/pre-census.log` — 14/14 paths present;
  2 exact DDGap facts and 2 resolution records; all units already `eV`.
- Evidence and port apply: `/tmp/reckon-u19-cohort/apply.log` — 12 facts,
  12 observations, 12 source relationships, 12 resolution writes; receipt hash
  `sha256:648e5b538aab398e7a89210e71b14b6baa093228ab600a79f20947cf40d8e660`.
- Idempotent replay: `/tmp/reckon-u19-cohort/replay.log` — 12 records verified,
  0 writes; receipt hash
  `sha256:560a1a9b3610c09ca90b74ba6c4d35c60e52a8fe10a3da17a15999866e882ca5`.
- Post-census: `/tmp/reckon-u19-cohort/post-census.log` — 14/14 paths each have
  exactly one resolution, one `BRIDGED_BY`, one `EVIDENCED_BY`, one
  `FOR_DD_VERSION`, scalar unit `eV`, and one `HAS_UNIT→eV` edge.
- CAS refusal proof: `/tmp/reckon-u19-cohort/red.log` records the initial missing
  instrument; the focused graph-port suite includes an adversarial `keV`
  scalar/edge mismatch and proves it raises `DDResolutionGraphPortConflict`.
- Schema compliance: `/tmp/reckon-u19-cohort/schema-compliance-graph.log` —
  9 passed, 0 skipped, 0 failed.
- Unit boundary: `/tmp/reckon-u19-cohort/core-tests.log` — 23 passed. The
  expanded 14-path authority makes the legacy graph correction inert for each
  exact path while a non-cohort descendant retains legacy behavior.

The graph mutation was restricted to the 12 missing exact evidence identities,
their 12 observations and source evidence relationships, and their 12
resolution nodes with required edges. Out-of-cohort resolution mutations: 0.
