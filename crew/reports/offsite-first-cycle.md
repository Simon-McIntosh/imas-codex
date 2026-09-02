NEEDS-HELP: scheduled graph push cannot reach the GitHub API from the SLURM compute node, so the first cycle produced no receipt or recovery point

tried: Submitted `imas-codex graph push --schedule` from this worktree through the repository's shared environment. SLURM job `1261097` ran on the `titan` partition from 2026-09-02 09:51:13 to 09:53:38 CEST, then failed after 145 seconds with exit code `1:0` before census/export because `api.github.com:443` timed out. A diagnostic in the live Neo4j allocation resolved `api.github.com` to `140.82.121.6`, but HTTPS timed out after 12 seconds with exit 124. No HTTP, HTTPS, or all-proxy variable is available on the submitting login node to propagate.

options: (1) provide a supported outbound proxy or network route for `titan` jobs and inject it into this scheduled job, then submit one new immediate cycle; (2) reshape the implementation so the SLURM job performs the graph-local census/export while a login-side step performs the GitHub API and GHCR operations, with one unified receipt; (3) run `graph push --cycle` manually on the login node, which can reach GitHub, but explicitly record that this does not validate the weekly SLURM path and therefore does not satisfy the node's evidence fence.

leaning: Option 2, unless compute-node egress is intentionally available through an existing supported proxy. The graph work remains SLURM-owned while the registry step runs where outbound GitHub access is already demonstrated, and the design does not depend on opening general compute-node internet access.

cost-if-wrong: Retrying the unchanged job consumes about 145 seconds per attempt and schedules another doomed weekly successor without creating a recovery point. Treating a login-node manual cycle as acceptance would require redoing the full scheduled-path measurement later. Building a split coordinator when a supported proxy already exists would add avoidable implementation and integration work.

## Measured outcome

| Evidence | Observed result | Required result | Verdict |
|---|---|---|---|
| Initial scheduled job | `1261097`; `FAILED`; elapsed `00:02:25`; exit `1:0` | Completed cycle | **unmet** |
| Cycle receipt | No JSON file exists under `/home/ITER/mcintos/.local/share/imas-codex/offsite-push/receipts` | Receipt path with `counts_match: true`, `wall_time_seconds`, and `archive_bytes` | **unmet** |
| GitHub Packages API | Latest version remains id `1008109688`, created `2026-07-07T12:27:23Z`, tags `v5.3.0-rc6` and `latest` | New scheduled archive tag | **unmet** |
| Offsite status | `stale`; `4,907,004` seconds behind live data; ref `ghcr.io/simon-mcintosh/imas-codex-graph:v5.3.0-rc6` | `current`, with age in seconds | **unmet** |
| Weekly successor | Job `1261098`; `PENDING (BeginTime)`; start `2026-09-09T09:53:38+02:00`, exactly seven days after submission | Successor pending seven days out | **met** |
| Neo4j availability | Job `1260970` is running on `98dci4-gpu-0002`; GraphClient query succeeded after the failed cycle | Neo4j back and queryable | **met**, although the failed cycle never reached the stop/export operation |
| Census comparison | Live census: `1,614,780` nodes and `4,259,356` relationships | Exact equality with receipt `live_census` | **not comparable** because no receipt exists |

The post-job GraphClient label census was:

```json
{"AgentRun":44,"COCOS":16,"CalibrationEpoch":47,"CodeChunk":271460,"CodeExample":69746,"CodeFile":262307,"DDGap":94,"DDGapIdentityChange":1,"DDGapObservation":512,"DDResolution":49,"DDVersion":35,"DataAccess":11,"DataReference":7817,"DataSource":23,"Diagnostic":109,"DocSource":22,"DocsRevision":3458,"Document":41175,"Facility":4,"FacilityPath":275315,"FacilitySignal":46872,"FacilityUser":4158,"Fanout":6007,"GrammarSegment":1644,"GrammarTemplate":654,"GrammarToken":70392,"GraphMeta":1,"IDS":87,"IMASCoordinateSpec":8,"IMASMapping":53,"IMASNode":61366,"IMASNodeChange":94928,"IMASSemanticCluster":2163,"ISNGrammarVersion":112,"IdentifierSchema":62,"Image":45727,"LLMCost":35340,"Locus":185,"MappingEvidence":1,"Person":3916,"PhysicsDomain":34,"PromotionCandidate":9,"RepairAuthorityDigest":333,"RepairGuard":999,"RepairMutation":346,"RepairParticipant":1047,"RepairRowIdentity":333,"RepairSelection":333,"SNRun":547,"SignalEpoch":48,"SignalNode":89344,"SignalSource":2337,"SoftwareRepo":1751,"StandardName":4683,"StandardNameChange":9994,"StandardNameReview":26829,"StandardNameSource":9675,"StandardNameSourceIdentityRepair":3,"StandardNameSourceRetry":50,"StandardNameSourceSnapshotAdmission":1,"StandardNameSourceSnapshotAdoption":1,"StandardNameSourceSnapshotChange":205,"StandardNameSourceUnitCacheCorrection":2,"StructuralNameAuthority":333,"TDIFunction":189,"Unit":415,"VocabGap":545,"VocabGapEvidence":7,"WikiChunk":130227,"WikiPage":28239}
```

## Durable diagnostics

- SLURM log: `/home/ITER/mcintos/.local/share/imas-codex/services/codex-graph-push.log`
- SLURM log SHA-256: `c5002f63db3eb8278efd95a53b61042c43810a86fac321d7194c498b0aa0a643`
- Failure site: `get_offsite_currency()` calling the GitHub Packages API before `run_offsite_push_cycle()` starts, so no receipt is expected from this failure mode.
- The failed job nevertheless executed its unconditional `sbatch --begin=now+7days "$0"`, producing successor `1261098`.

## Acceptance state

The requested quantitative done-when is not satisfied. The successful evidence count is 2 of 6 material gates: the seven-day successor exists and Neo4j is queryable. The completed-cycle receipt, count equality, measured archive bytes/wall time, new registry tag, and current offsite row are absent.
