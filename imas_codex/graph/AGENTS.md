This file governs the `imas_codex/graph/` subtree — the Neo4j graph client, the
schema-driven DD and facility node models, and every graph/DD operation.



## Schema System

All graph node types, relationships, and properties are defined in LinkML schemas — the single source of truth.

**Schema files:**
- `imas_codex/schemas/facility.yaml` - Facility graph: SourceFile, SignalNode, CodeChunk, FacilityPath, FacilitySignal, etc.
- `imas_codex/schemas/imas_dd.yaml` - DD graph: IMASNode, DDVersion, Unit, IMASCoordinateSpec, Cluster, NodeCategory
- `imas_codex/schemas/standard_name.yaml` - Standard names: StandardName, StandardNameSource, Review, DocsRevision, VocabGap, LLMCost, SNRun
- `imas_codex/schemas/grammar_graph.yaml` - ISN grammar: GrammarSegment, GrammarToken, GrammarTemplate, ISNGrammarVersion
- `imas_codex/schemas/facility_config.yaml` - Per-facility YAML config schema
- `imas_codex/schemas/task_groups.yaml` - Worker task grouping
- `imas_codex/schemas/common.yaml` - Shared enums and mixins

The generic MCP `add_to_graph` tool refuses every concrete class owned by the
`standard_name` and `grammar_graph` schemas. Those nodes are governed state and
must be written through the dedicated Standard Name pipeline. The boundary is
derived through `GraphSchema.get_class_schema_id()` from each LinkML class's
`from_schema` value, so adding another class to either schema automatically
places it behind the same refusal. LinkML `SchemaView` access stays encapsulated
inside `imas_codex/graph/schema.py`.

**Build pipeline:**
- Models auto-generated during `uv sync` via hatch build hook
- Regenerate manually: `uv run build-models --force`
- **CRITICAL — gitignored, auto-generated, never commit (never stage even if `git status` shows them modified/untracked):** `imas_codex/graph/models.py`, `imas_codex/graph/dd_models.py`, `imas_codex/config/models.py`, `agents/schema-reference.md`, `imas_codex/graph/schema_context_data.py`

**PhysicsDomain enum**: Imported from the `imas-standard-names` PyPI package and re-exported from `imas_codex.core.physics_domain`. The canonical vocabulary is maintained in the imas-standard-names project. Contains 32 physics domain values. `imas_codex/core/physics_domain.py` is a hand-written one-line re-export — it IS committed and should NOT be treated as auto-generated.

**NodeCategory enum** (`imas_dd.yaml`): DD node classification — 9 values: `quantity`, `geometry`, `coordinate`, `metadata`, `error`, `structural`, `identifier`, `fit_artifact`, `representation`. Classifier lives in `imas_codex/core/node_classifier.py` (two-pass: Pass 1 attribute-only, Pass 2 graph-relational). Category sets for pipeline participation in `imas_codex/core/node_categories.py`.

**IMASNodeStatus lifecycle** (`imas_dd.yaml`): DD build pipeline `built → enriched → refined → embedded → classified` across seven workers EXTRACT → BUILD → ENRICH → REFINE → EMBED → CLASSIFY → CLUSTER. CLASSIFY (after EMBED) uses `get_model("language")` for three-tier domain assignment — LLM for physics paths; inheritance (`HAS_ERROR`/`HAS_PARENT`) for error/metadata; none for infra metadata (`ids_properties/*`, `code/*`). `"general"` paths retry with expanded cluster context; `--reset-to embedded` re-classifies only (~$2.60, cheapest domain fix).

Always import enums and classes from generated models. Never hardcode status values:

```python
from imas_codex.graph.models import SourceFile, SourceFileStatus, SignalNode

sf = SourceFile(
    id="tcv:/home/codes/liuqe.py",
    facility_id="tcv",
    path="/home/codes/liuqe.py",
    status=SourceFileStatus.discovered,  # Use enum, not string
)
add_to_graph("SourceFile", [sf.model_dump()])
```

**Extending schemas:** Edit LinkML YAML → `uv run build-models --force` → import from `imas_codex.graph.models`. Prefer additive changes, but renames and removals are fine when they improve consistency — the schema must stay clean. When renaming or removing: update all code references, migrate graph data, and rebuild models in a single commit.

**Full schema reference:** [agents/schema-reference.md](agents/schema-reference.md) — auto-generated list of all node labels, properties, vector indexes, relationships, and enums. Rebuilt on `uv sync`.

### Schema Design Guidelines

Conventions when adding classes, properties, or relationships. The build pipeline, `create_nodes()`, and query builder depend on predictable schema structure.

**Dual property + relationship.** Every slot with a class range produces **both** a node property (fast `WHERE` filtering) AND a Neo4j relationship (graph traversal). `create_nodes()` in `client.py` does `SET n += item` then `MERGE (n)-[:REL]->(t:Target {id: item.slot})`. Never remove one side of the dual model.

**Relationship type names.** If the slot has a `relationship_type` annotation, that's used; otherwise the slot name is uppercased (`has_chunk` → `HAS_CHUNK`). Add an explicit annotation when the auto-derived name is unclear. **All `facility_id` slots MUST have `range: Facility` + `annotations: { relationship_type: AT_FACILITY }`** — no exceptions. Prefer verb-based names (`SOURCE_PATH`, `BELONGS_TO_DIAGNOSTIC`).

**Class template:**

```yaml
MyNewNode:
  description: >-
    What this node represents. Include example Cypher queries.
  class_uri: facility:MyNewNode
  attributes:
    id: { identifier: true, required: true, description: "Composite key (e.g., 'tcv:unique_part')" }
    facility_id:
      required: true
      range: Facility
      annotations: { relationship_type: AT_FACILITY }
    status: { range: MyNewNodeStatus, required: true }
    description: { description: "Human-readable, drives semantic search" }
    embedding: { multivalued: true, range: float }
    embedded_at: { range: datetime }
```

**Rules:**
- Use `identifier: true` on exactly one slot per class (always `id`).
- Composite IDs use colon separator: `facility_id:unique_part`. Must be globally unique.
- Nodes with `embedding` + `description` auto-get a vector index `{snake_case_label}_desc_embedding`. Override with `vector_index_name` annotation if needed.
- Status enums live in the same schema file. **Durable states only** — never `scanning`/`processing`. Worker coordination via `claimed_at` timestamps.
- `is_private: true` excludes a slot from the graph (config-only).
- Never hardcode enum values in Python — import from generated models.
- Never skip the `description` field — it drives semantic search.
- Don't use `multivalued: true` on relationship slots unless genuinely many-to-many.

### Schema-Driven Testing

Tests in `tests/graph/` are **parametrized from the schema** — they validate graph data against LinkML declarations. Key modules: `test_schema_compliance.py` (node labels/properties/enums), `test_referential_integrity.py` (relationship types with correct `relationship_type` annotation), `test_data_quality.py` (embedding coverage).

**On test failure, fix the root cause, not the schema.** Three cases: (a) building a new capability → declare in LinkML first, then write code; (b) code writing non-compliant data → fix the code or the bad data; (c) stale data from a prior schema version → migrate/remove the data. Never add schema declarations just to make tests green.

## Dispatching graph work (binding)

A worker that must reach Neo4j needs four things and three of them are decided at
dispatch time, not by the worker. Every line below was measured on this
repository; the dates say when.

**The role decides whether the graph is reachable at all.** `review` and
`investigate` resolve to a read-only sandbox and are `execution_capable: false`,
so they can never open a Neo4j session — a live query routed there fails before
it starts, and the failure looks like a credential problem rather than a routing
one. Route any node that reads or writes the graph to `test` or `implement`.

**The worktree needs the credential, and it is a link.** The main checkout's
`.env` holds `NEO4J_PASSWORD` at mode 600. `NEO4J_URI` is **not** set: while
`GraphClient` documents that the env vars override any profile, with the URI
unset both it and `resolve_neo4j()` resolve the same endpoint from the profile.
So a worktree without the `.env` symlink fails auth before doing any work, and
that is a provisioning fault, not evidence the credential is wrong. Measured
2026-09-05 across eleven live worktrees: every one carries `.env` **and** `.venv`
as symlinks into the main checkout, never copies, so exactly one 600-mode secret
exists on disk. Never copy, print, stage or commit the file.

```bash
ln -sfn /home/ITER/mcintos/Code/imas-codex/.env "$W/.env"   # link, never copy
readlink "$W/.env"                                          # must resolve to the main checkout
```

**Probe through a real client, never through the profile alone.**
`resolve_neo4j()` called outside the application's env loading can resolve a
different path and report a tunnel failure for a host nothing uses. Measured
2026-09-05: a bare probe from the main checkout reported
`Could not resolve hostname titan` and a closed port, while workers were
querying 4,937 identities through the live endpoint at the same moment.
Establish reachability with an ordinary `GraphClient` and read the host and port
it reports.

**Derive the gate from what the change can reach, not from where its code
lives.** `imas_codex/standard_names/graph_ops.py` is Cypher and sits in the
standard-names package, so a change there is gated by `tests/standard_names` —
not by `tests/graph`. Measured 2026-09-05: two commits rewriting 118 SET clauses
were gated on `tests/graph` alone and added 22 failures to `tests/standard_names`
that nothing measured for hours.

**The default markers exclude `graph`, so Cypher you edit is never executed by
the default gate.** The ~445 graph-marked tests need a live Neo4j and are
deselected by `addopts`. A change to query *text* that must be proven to parse
needs `-m graph` against a live database, or an `EXPLAIN` of each modified
statement. A green default run says nothing about whether the statement is valid.

**Append to a SET clause; never prepend.** Several standard-names tests mock
`GraphClient` and dispatch on a literal `SET <alias>.<field>` prefix or an exact
query string. Comma-separated assignments to independent properties are
order-independent in Cypher, so a new assignment goes last. A prepended one
shifts the leading token, misroutes the fakes, and fails assertions that have
nothing to do with the change — the mechanism behind the 22 failures above.

**Read the zero-row rule below before writing any query.** A property the schema
does not declare returns zero rows rather than erroring, so a query can be
confidently wrong and silent.

### Link what must be shared; copy what must diverge

`.env` and `.venv` are symlinked into a worktree. The **generated models are
copied**, and the distinction is load-bearing rather than incidental.

| Resource | Provisioned by | Why |
|---|---|---|
| `.env` | symlink | One 600-mode secret on disk, identical everywhere, never duplicated. |
| `.venv` | symlink | One 69,826-entry environment per repository; a copy is 1.76 GiB of GPFS. |
| `imas_codex/graph/models.py`, `dd_models.py`, `config/models.py`, `graph/schema_context_data.py`, `agents/schema-reference.md` | **`cp -n`** | Derived from *that tree's* schemas. A worktree changing a schema must regenerate them locally, and its copy is then legitimately different from the main checkout's. |

**Never symlink a generated model file.** It fails in both directions. Reading,
the worktree sees the main checkout's copy, which does not reflect the schema the
worktree just edited — so its tests measure the wrong tree. Writing is worse:
`build-models --force` in the worktree would write *through* the symlink into the
main checkout, replacing every peer worker's models with one node's in-progress
schema change. That is the `uv sync`-from-a-worktree hazard in a new costume — a
worktree mutating a shared resource under peers who did not ask for it.

So the rule is: **a worktree that edits `imas_codex/schemas/` must run
`build-models --force` in its own tree before it measures anything.** The copy
placed at creation is a starting point, not a guarantee of currency. Where the
worktree and the main checkout disagree after such an edit, the worktree is right
and the main checkout is behind; refresh the main checkout with
`uv run build-models --force` after integration.

Two corollaries measured 2026-09-05:

- A node that declared one new schema attribute had to regenerate before
  `test_generated_model_currency` would pass, and said so in its manifest. That
  is the expected sequence, not a defect.
- Absent models are a *plausible* cause of a large failure count in a fresh
  worktree, which makes them worth ruling out explicitly rather than assuming.
  When 22 unexplained failures appeared, the node that investigated regenerated
  models in **both** trees first and reproduced the count either way, which is
  what let it attribute the failures to the code change rather than to
  provisioning. Rule the artifact out by measurement before reasoning past it.

### Checkpointing the graph

**One checkpoint is enough to protect a body of work, and it may be taken from
the main checkout.** There is no recurring backup discipline to maintain.

The dump stops the world: `neo4j-admin database dump` refuses while the database
is in use, so `imas-codex graph export` stops the SLURM-hosted server, dumps, and
restarts by default. Every graph-touching worker must be at rest first — a node
holding a read-only session blocks it exactly as much as one writing.

Pass `-o` to place the archive in `BACKUPS_DIR`. A bare `graph export` writes to
`EXPORTS_DIR` while `get_backup_currency` reads `BACKUPS_DIR`, so the archive
otherwise lands where the currency instrument does not look and never registers
as a backup at all.

Do not expect `status: current` with `age_seconds: 0`. The restart writes into
the live tree after the archive is sealed, so a freshly-taken checkpoint reads
`stale` by a few seconds. That is a property of the instrument, not a defect in
the archive. **The archive's existence and its verified contents are the
protection** — a gzip tar carrying a non-empty `graph.dump` member, plus a live
read afterwards returning the same node count as before.

## Graph Operations

**Schema verification:** Before writing Cypher queries, verify property names against `agents/schema-reference.md` (auto-generated) or call `get_graph_schema()`. Common pitfall: WikiChunk/CodeChunk text content is stored in the `text` property.

### A wrong property name returns zero rows, not an error (binding)

Cypher evaluates a missing property to `null` and drops any row whose predicate is
not `true`. So a misspelled or wrongly-guessed property name produces a **clean,
plausible, empty result** — indistinguishable from a real empty set. Four such
failures happened in one session on 2026-08-23, two of them after the rule was
written down and one by its own author; every one was caught by a second agent, none
by the author. **Do not trust a zero until you have proven the key exists.**

```cypher
-- Run this BEFORE believing any zero, and fail closed if with_prop is 0
MATCH (n:<Label>) RETURN count(n) AS candidates, count(n.<prop>) AS with_prop
```

`candidates > 0` with `with_prop = 0` means the property name is wrong, not the set
empty. The same applies to `count(DISTINCT n.<prop>)`, which returns 0 over a
non-existent property without complaint.

**Prove the instrument FIRES, then prove it is AIMED — they are different checks
and only the first is cheap.** Failing closed and carrying a positive control both
establish that a scan can find the thing it reports absent; neither establishes that
it was pointed at the right thing. Measured across two sessions on 2026-08-25, five
scans returned clean, plausible, wrong answers, and every one of them fired
correctly:

| The scan asked about | The question actually asked |
|---|---|
| `imas_codex/ids/graph_ops.py`, 490 lines, by basename | `standard_names/graph_ops.py`, 24,491 lines |
| an invented list of typed roots | reckon's declared `TYPE_ROOTS` |
| `followup-status="open"` | `data-status="open"` |
| a receipt found by operation name | the receipt's own `run_id` + manifest digest |
| a literal `def name` | a public name installed dynamically |

Each returned a number. The basename glob found a real file and counted its real
lines; it answered a real question nobody had asked. **So add a positive control —
one row that must fire, in the same run, against the same pattern — and then say out
loud what the instrument is pointed at.** A control proves it can see; only naming
the target catches an instrument aimed one field, one path, or one file to the left.

Note the direction of the error, because it decides whether review saves you: a
literal-`def` scan reporting six operators absent reads as a *cleaner* result than
the truth. **The failure mode that flatters you is the one that survives review**,
since nobody interrogates good news.

**Standard Name identity keys follow one convention — still check, never
guess.** A `StandardName` uses `id` because it is the identity itself. Every
declared generic foreign key to a `StandardName` uses `standard_name_id` when
scalar and `standard_name_ids` when multivalued:

| Class | Property | Note |
|---|---|---|
| `StandardName` | `id` | the identifier itself; there is **no** `name` property on any SN class |
| `StandardNameSource` | `id` | `dd:<path>` or `signals:<facility>:<signal-id>` |
| `StandardNameReview` | `standard_name_id` | back-reference; authoritative link is `HAS_REVIEW` |
| `DocsRevision` | `standard_name_id` | authoritative link is `DOCS_REVISION_OF` |
| `LLMCost` | `standard_name_ids` | multivalued, and the **only** link — `LLMCost` has no edge to `StandardName` |

`StandardNameReview.review_axis` likewise uses `name`/`docs`, matching the
paired `_name`/`_docs` scalar suffixes (`reviewer_score_name` beside
`reviewer_score_docs`).

Full tables — all 53 relationship directions with LinkML citations, and all 12
foreign-key classes — are in
[`imas_codex/standard_names/AGENTS.md`](imas_codex/standard_names/AGENTS.md#graph-identity-and-joins).
The LinkML declarations remain authoritative when a class uses a semantically
specific relationship slot instead of a generic foreign key.

### Cypher Compatibility — Neo4j 2026

We run **Neo4j 2026.01.x** with `db.query.default_language: CYPHER_5`. The only breaking syntax change that affects this codebase: `x NOT IN [list]` is removed — write `NOT (x IN [list])` instead. `CASE WHEN` is fully supported — use it freely. For "keep old value if new is empty," prefer `SET s.f = coalesce(nullIf(new, ''), old)` over `CASE WHEN`. Test new Cypher against the live graph before committing.

### Neo4j Management

`imas-codex graph <cmd>` (`--help` for the full list): server `start`/`stop`/`status`/`shell`/`profiles`; instances `init`/`switch`/`list`; archives `export`/`load ARCHIVE TARGET`/`fetch`; GHCR `pull TARGET`/`push --dev`/`tags`/`prune --dev-only`; maintenance `clear TARGET`/`secure`; facilities `facility`. `export`, `fetch`, `pull TARGET`, `push`, `tags`, and `prune` accept `-F/--facility`; destructive `load`, `pull`, and `clear` require the explicit `TARGET` selected by the active symlink. `graph status` reports the newest non-empty backup, newest live database file, and their measured lag. Also use `imas-codex tunnel start <host>`/`status` and `config private push` / `config secrets push <host>`.

Never use `DETACH DELETE` on production data without user confirmation. For re-embedding: update nodes in place, don't delete and recreate.

### Graph Migrations

Run migrations as inline Cypher via `imas-codex graph shell` or the MCP `repl()` (`query()`). Never create `scripts/migrate_*.py` or `repair_*.py`. For >10K-node migrations, batch with `LIMIT` to avoid transaction timeouts; verify counts before and after.

### LLMCost Node Properties

`LLMCost` nodes track per-call LLM spend. **All `LLMOperation`-mixin fields are prefixed with `llm_`** — never use bare `cost`, `model`, or `service`. Full property list is in `agents/schema-reference.md`; key fields: `llm_cost`, `llm_model`, `llm_service`, `llm_tokens_{in,out,cached_read,cached_write}`, grouping (`run_id`, `phase`, `pool`, `batch_id`, `for_run`), and `standard_name_ids`.

**Canonical cost queries:**

```cypher
-- Total LLM spend
MATCH (c:LLMCost) RETURN round(sum(c.llm_cost)*100)/100 AS total_usd

-- Per-pool / per-model breakdown
MATCH (c:LLMCost)
RETURN c.pool AS pool, count(c) AS calls, round(sum(c.llm_cost)*100)/100 AS usd
ORDER BY usd DESC

-- Spend for a specific run
MATCH (c:LLMCost {for_run: $run_id}) RETURN sum(c.llm_cost) AS total

-- SNRun budget tracking
MATCH (r:SNRun) RETURN r.cost_spent AS spent, r.cost_limit AS budget, r.stop_reason
ORDER BY r.started_at DESC LIMIT 1
```

`SNRun.cost_spent` / `cost_limit` / `cost_total` are aggregates; `LLMCost.llm_cost` is the per-call source of truth. Embedding costs are always zero — only OpenRouter LLM calls incur cost.

### Neo4j Lock Files — CRITICAL

Neo4j uses several lock file types. Mishandling them **causes data loss**.

| Lock File | Location | Purpose | Safe to Delete? |
|-----------|----------|---------|----------------|
| `store_lock` | `data/databases/` | Coordinates single-writer access | Yes — after confirming Neo4j is stopped |
| `database_lock` | `data/databases/*/` | Per-database writer lock | Yes — after confirming Neo4j is stopped |
| `write.lock` | `data/databases/*/schema/index/*/` | Lucene index segment lock | **NEVER** — deletion corrupts vector indexes |

**Rules:**
1. **Never use `find -name "*.lock"` to clean locks** — this matches Lucene `write.lock` files inside vector index directories.
2. Only remove `store_lock` and `database_lock` explicitly by path, and only after confirming Neo4j has fully stopped.
3. On GPFS/NFS, stale POSIX locks can survive process death. The safe workaround is inode replacement (`cp file file.unlock && mv -f file.unlock file`), not deletion.
4. If Lucene `write.lock` is deleted while Neo4j is running, it triggers `AlreadyClosedException`, checkpoint failure, and potential database reinitialization on next start.

**Never use the Docker entrypoint** (`/startup/docker-entrypoint.sh`) to start Neo4j in Apptainer. It calls `neo4j-admin dbms set-initial-password` and runs `rm -rf conf/*` on every start, which can reinitialize an existing database after a crash. Always use `neo4j console` directly with a host-side `conf/` bind mount.

### Vector Indexes

Nodes with `embedding` + `description` auto-get a quantized cosine vector index (~4× memory savings). `ensure_vector_indexes()` creates them, auto-detects dimension mismatches, and drops/recreates stale indexes — never hand-write `CREATE VECTOR INDEX`. Query with Neo4j 2026.01's native `SEARCH` clause (in-index pre-filtering). Index names and the full list are in `agents/schema-reference.md`.

### Semantic Search & Graph RAG

Use `semantic_search(text, index, k)` in the python REPL:

```python
# Document content (wiki, code)
semantic_search("COCOS sign conventions", index="wiki_chunk_embedding", k=5)

# Descriptive metadata (signals, paths - search by physics meaning)
semantic_search("plasma current measurement", index="facility_signal_desc_embedding", k=10)
```

Combine vector similarity with link traversal via the Cypher 25 `SEARCH` clause
(`MATCH (s:FacilitySignal) SEARCH s IN (VECTOR INDEX <name> FOR $embedding LIMIT
k) SCORE AS score WHERE … WITH s, score MATCH (s)-[:DATA_ACCESS]->…`). Use
`build_vector_search()` from `imas_codex.graph.vector_search` to generate SEARCH
clauses programmatically rather than hand-writing them. All WHERE conditions are
post-filters (in-index pre-filtering requires properties registered as
additional vector index properties).

**Key relationships for traversal:**

| From | Relationship | To |
|------|--------------|-----|
| FacilitySignal | DATA_ACCESS | DataAccess |
| FacilitySignal | HAS_DATA_SOURCE_NODE | SignalNode |
| IMASMapping | SOURCE_PATH | SignalNode |
| IMASMapping | TARGET_PATH | IMASNode |
| WikiChunk | HAS_CHUNK← | WikiPage |
| FacilityPath | AT_FACILITY | Facility |

**Token cost:** Always project specific properties in Cypher (`RETURN n.id, n.name`), never return full nodes. Use Cypher aggregations instead of Python post-processing.

### Batch Operations

Use `UNWIND` for batch graph writes:

```python
query('''
    UNWIND $items AS item
    MERGE (n:Tool {id: item.id})
    SET n += item
    WITH n
    MATCH (f:Facility {id: 'tcv'})
    MERGE (n)-[:AT_FACILITY]->(f)
''', items=tools)
```

## Services

**Neo4j graph and the embedding server are always running** as SLURM jobs on all dev machines (ITER, WSL). Assume both are available. If a service is down, restart it — don't work around it. Always connect via the Python client methods (`GraphClient`, `Encoder`) — never raw HTTP/bolt; they handle SLURM node discovery, tunnel setup, auth from `.env`, and retries.

**SLURM-only rule.** Both services MUST run as SLURM jobs — never bypass with `nohup`, `ssh … &`, `screen`, `tmux`, or anything else. SLURM provides cgroup isolation, clean lifecycle (`scancel`), accounting, and drain cleanup. Rogue processes cause "Duplicate jobid" errors that drain nodes for all users. If SLURM won't schedule, get the node resumed (`scontrol update NodeName=<node> State=RESUME`) — don't work around it.

### Neo4j connection

On ITER login/compute nodes, `GraphClient()` (no args) discovers the SLURM compute node and connects directly — never hardcode `bolt://localhost:7687`:

```python
from imas_codex.graph.client import GraphClient
gc = GraphClient()    # handles SLURM, tunnels, env overrides
```

From WSL/remote, start a tunnel first: `imas-codex tunnel start iter` then `tunnel status`. The profile system auto-tunnels for remote hosts. Override with `export IMAS_CODEX_TUNNEL_BOLT_ITER=17687` if needed.
