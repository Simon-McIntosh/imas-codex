# Coordinator instructions for the review of the existing ITER signal structure

Plan section 2 of `iter-plant-signal-ontology` is the authority for what is
delivered. These instructions add the inputs, the reuse map and the rules.

## Inputs

- The schema: `iter_codex/schemas/facility.yaml` (LinkML). The classes the
  ITER archive work added sit together: `PlantContextProvenance`,
  `PlantStructure`, `PlantFunctionCategory`, `PlantContextDocument`,
  `PlantSignalRow`, `SignalContextLink`, `PlantComponent`, `ControlNode`,
  `ArchiveFieldPattern`, `ArchiveEndpoint`, plus the ITER slots on
  `FacilitySignal` (vocabulary, grouping, component and control-node links,
  group-enrichment state) and the enums near the top of the file. Shared
  classes to compare against include `Document` (discovered, scored, ingested
  workflow), `FacilitySignal`, `SignalNode`, `DataSource`, `DataAccess`,
  `Diagnostic` and the physics-domain and unit links. Read
  `iter_codex/graph/AGENTS.md` for the schema conventions.
- The writers and readers: `iter_codex/discovery/signals/` (plant context,
  archive grouping, field codes, vocabulary, group enrichment, controls
  metadata), `iter_codex/remote/field_code_table.py`,
  `iter_codex/remote/signal_vocabulary.py`, `iter_codex/tools/plant_search.py`,
  the archive scanner `iter_codex/discovery/signals/scanners/codac_uda.py`.
- Known owners the design review named, as a starting census of readers and
  writers (extend it, do not trust it as complete): graph writers
  `plant_context_graph.py`, `archive_grouping_graph.py`,
  `group_enrichment.py`, `field_codes_graph.py`, `vocabulary_graph.py`;
  the shared query filters in `iter_codex/graph/domain_queries.py` (the two
  role axes are separate predicates there); the plant search tools in
  `iter_codex/tools/plant_search.py`; the document pipeline
  `iter_codex/discovery/documents/` (shared `Document`, `DocumentStatus` in
  `iter_codex/schemas/common.yaml`); the document receipt
  `DocumentReceipt` returned by `ICPDocuments.fetch` in
  `iter_codex/discovery/icp/documents.py`, which already carries identity,
  version, digest, size and sheets.
- The field-code table: `iter_codex/discovery/signals/field_codes.py` and
  `field_codes_graph.py` assemble it from graph sources in precedence;
  `iter_codex/remote/field_code_table.py` stores it. The meanings are in
  `field_code_meanings` in the gitignored
  `iter_codex/config/facilities/iter_private.yaml` (69 entries). Copy that file
  into your worktree from the main checkout if it is missing.
- Documentation found so far, private:
  `~/.cache/imas-codex-codac-uda/document-census/signal-lists-sample.tsv`.
  It holds two pages of one title search for signal lists (about 160 rows, the
  search was not exhausted and some rows were withheld by permissions) and the
  first page of a self-description search. It is a sample: say so wherever you
  use it, and count coverage against it as a lower bound.
- The live graph: count ITER-facility nodes and relationships per class and
  type, bounded per label. The project's ICP client reads only plant-breakdown,
  component and function-category records; it cannot search documents.

## Plant networks: the connection source the ontology must accommodate

On 2026-11-23 the coordinator found that ICP4 holds the plant's engineering
source data as structured records, read through Lucy's record tools:

- `SSD Pipe Source Data`: about 122,500 records. Each pipe has a functional
  reference, a function-category code, PBS fields, `from_component_functional_reference`
  and `to_component_functional_reference` (with component uids, guids and port
  guids), `fluid` (empty in the samples read), pipe specification, diameter,
  pressures and temperatures, `network_color`, `electrical_train`.
- `SSD Cable Source Data`: about 253,600 records with the same from/to
  component fields, plus cable family, wire count, gauge, shield, segregation
  code and electrical train. Signal cables are among them, so a sensor can be
  traced to its cubicle.
- Also `SSD Equipment Source Data`, `ICP4 SSD Diagram`, `Component` and
  `Functional Reference` kinds.

Functional references read plant-system code, function-category code and
sequence (for example `00AA00-BJ-1234`, invented here), the same grammar the
archive name parser splits into control path, component category and sequence.
The lead chose (2026-11-23) to design the ontology for these networks now and
import them in the pipeline plan. Settle a typed connection between components
(medium, from, to, source) consistent with the shared schema, and say how an
archive component and a source-data component are identified as the same.
Whether the project's client can read these records by GET is not yet known;
do not probe ICP from this node.

## What the lead has said

- The schema is LinkML and shared; there is one graph. Keep the design
  consistent with the current shared classes; changes to a class another
  facility uses are additive.
- The ontology will evolve as the work goes on; design for extension rather
  than completeness.
- Only the ITER archive parts of the graph will be migrated (that is section 3,
  not this node).
- Asked whether plant structure and the PBS differ; the coordinator's reading
  is that `PlantStructure` is the PBS record under a generic name, and the
  control breakdown (`ControlNode`) is the separate hierarchy parsed from
  archive names. Confirm or correct this from the code and the records.

## Rules

- Read-only on the graph: this node writes no node or relationship.
- No real signal name, document identity, document title or private text in a
  tracked file. Use invented examples or counts.
- Heavy graph reads run as a job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch
  --partition=rigel`, logs on GPFS; each query bounded to one label.
- The figure follows the plot rules in `~/.agents/AGENTS.md`: one SVG under
  `docs/figures/iter-plant-signal-ontology/`, rendered to PNG and looked at
  before committing.
- Commit locally with a conventional subject and a body, and give `commits`
  as a JSON list of full shas. Do not push or merge.
