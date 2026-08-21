# Recommended disposition for `voltage_of_diagnostic_antenna`

## Recommendation

Use the **sanctioned source-less rename transition** from
`voltage_of_diagnostic_antenna` toward the historically intended
`voltage_of_ece_channel`, followed by the ordinary validation and review
pipeline. Do not migrate or detach `dd:ece/channel/t_e_voltage`, do not steal
`dd:ece/channel/voltage_t_radiation` from
`voltage_of_spectrometer_channel`, and do not hand-promote the successor.

This is a disposition recommendation, not mutation authority. A later operator
must use the sanctioned transition, preserve the predecessor and its unique
stale source as ledgered history, and let ordinary review accept, refine, or
refuse the successor. If no such exact transition can preserve those
invariants, retain the identity as a standing recorded refusal rather than
performing a direct fold or detach.

The measured absence of an authoritative live DD producer closes the ordinary
source-backed rename route. It does **not** authorize deletion: the identity
still has one producing `StandardNameSource` in total, and no exact signed
delete manifest exists for it.

## Live authority state

The read-only live-graph command below was run on 2026-08-21 at
`2026-08-21T15:15:10+02:00` through the project MCP `repl()` and canonical
`query()` client. Its aggregate uses the ratchet's nonterminal-child predicate,
counts a producer as live only when it is `composed` or `attached` and backed by
a non-removed DD path, and treats either approved lifecycle or any merged-PR
field as catalog release provenance.

```cypher
MATCH (name:StandardName {id: 'voltage_of_diagnostic_antenna'})
RETURN name.id AS identity,
       name.name_stage AS name_stage,
       name.validation_status AS validation_status,
       COUNT {
         (source:StandardNameSource)-[:PRODUCED_NAME]->(name)
       } AS producing_source_count,
       COUNT {
         (source:StandardNameSource)-[:PRODUCED_NAME]->(name)
         WHERE source.status IN ['composed', 'attached']
           AND EXISTS {
             MATCH (source)-[:FROM_DD_PATH]->(path:IMASNode)
             WHERE coalesce(path.lifecycle_status, 'active') <> 'removed'
           }
       } AS live_producer_count,
       COUNT {
         (child:StandardName)-[:HAS_PARENT]->(name)
         WHERE NOT (child.name_stage IN ['superseded', 'exhausted'])
       } AS live_has_parent_child_count,
       name.catalog_pr_number AS catalog_pr_number,
       name.catalog_pr_url AS catalog_pr_url,
       name.catalog_approved_at AS catalog_approved_at,
       name.catalog_merge_commit_sha AS catalog_merge_commit_sha,
       (name.name_stage = 'approved'
        OR name.catalog_pr_number IS NOT NULL
        OR name.catalog_pr_url IS NOT NULL
        OR name.catalog_approved_at IS NOT NULL
        OR name.catalog_merge_commit_sha IS NOT NULL)
         AS has_catalog_release_provenance
```

Recorded output:

```text
COMMAND: read-only aggregate query for voltage_of_diagnostic_antenna
HIT_COUNT: 1
[{'identity': 'voltage_of_diagnostic_antenna',
  'name_stage': 'accepted',
  'validation_status': 'valid',
  'producing_source_count': 1,
  'live_producer_count': 0,
  'live_has_parent_child_count': 0,
  'catalog_pr_number': None,
  'catalog_pr_url': None,
  'catalog_approved_at': None,
  'catalog_merge_commit_sha': None,
  'has_catalog_release_provenance': False}]
```

The companion family query recorded the only producing row as
`dd:ece/channel/t_e_voltage`, `status='stale'`, with scalar target
`voltage_of_diagnostic_antenna`. It also recorded
`voltage_of_ece_channel` as absent and
`voltage_of_spectrometer_channel` as reviewed, valid, unit `V`, with two
composed producers including `dd:ece/channel/voltage_t_radiation`.

## Delete-authority assessment

Deletion requires all four bounds; one missing bound forbids it. Here exactly
**2 of 4** are present:

| Delete-authority bound | Live result | Assessment |
|---|---:|---|
| No producing `StandardNameSource` at all | **No** — total count is 1 | Fails. A stale source is still a producing source for this bound. |
| No live `HAS_PARENT` child | **Yes** — count is 0 | Passes. The identity is not currently a structural parent. |
| No catalog release provenance | **Yes** — stage is accepted and all four merged-PR fields are null | Passes. Accepted pipeline state is not catalog publication. |
| Exact enumerated signed delete manifest with applying-count equality | **No** | Fails. The producer-search record and this recommendation are evidence, not an apply manifest. |

The additional last-producing-source guard is active: the stale
`dd:ece/channel/t_e_voltage` row is the identity's only producing source. A
detach or ordinary retarget would strip the last producer and has already been
recorded as a standing refusal. The stale source cannot be treated as live DD
authority, but its edge cannot be silently discarded either.

## Three-candidate score

The score has five binary authority checks: the four deletion bounds above plus
the last-producing-source guard. A deletion-bound point means the live evidence
affirmatively supplies that bound; a guard point means the disposition can
proceed without detaching, migrating, or stealing the unique source. Because
deletion requires 4 of 4 bounds and only 2 are present, none of these scores is
deletion authority.

| Candidate disposition | No source at all | No live child | No release provenance | Exact signed delete manifest | Last-source guard preserved | Score | Disposition |
|---|---:|---:|---:|---:|---:|---:|---|
| Supersede directly toward reviewed `voltage_of_spectrometer_channel` | 0 | 1 | 1 | 0 | 0 | **2/5** | Reject. An ordinary fold/retarget would move or strip the unique stale source; leaving it untouched requires the source-less mechanism scored separately. The reviewed channel identity is also authority for channel ownership, not proof that diagnostic-antenna ownership is equivalent. |
| Sanctioned source-less rename transition toward `voltage_of_ece_channel`, then ordinary review | 0 | 1 | 1 | 0 | 1 | **3/5** | **Recommend.** It records the semantic correction without presenting the stale path as live authority, disturbing the reviewed channel binding, deleting the predecessor, or bypassing review. |
| Close as a standing recorded refusal | 0 | 1 | 1 | 0 | 1 | **3/5** | Safe fallback, but not preferred while the plan-selected sanctioned transition is available. It preserves every guard yet leaves the known identity correction unresolved. |

The tied 3/5 authority score is resolved by utility under the live plan: the
source-less transition is the only candidate that both preserves all available
authority and advances the already-selected correction. The standing refusal
becomes final only if the exact sanctioned transition cannot preserve the
predecessor/source ledger or cannot enter ordinary review. The direct
spectrometer-channel fold is not a shortcut to acceptance.

## Evidence inputs and boundary

- Live Reckon plan: `docs/sn-graph-wide-integrity.html`, version **221**,
  SHA-256 `4f643a5312303ce73a5d602c3c87d4530ca4565ad7b623a1b8ba0f1869523a42`.
- Producer search:
  `docs/evidence/sn-graph-wide-integrity/voltage-antenna-producer-search.md`,
  SHA-256 `ccc58cded553aaaedd54e1f0ed99584528831edec346d35f87b3a7d6854dc76f`:
  6 semantic searches, 60 raw clusters, 31 current voltage paths, 4 of 4
  exact negative eligibility queries at zero, and **0 authoritative live DD
  producers**.
- Worktree source commit:
  `5dce2b5ba208609b0db50f2287a25b3339b0b855`.
- Graph effects: **0 writes**, **0 provider calls**, and **0 acceptance
  promotions**. This artifact records a recommendation only.

