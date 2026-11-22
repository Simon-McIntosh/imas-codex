---
name: lucy-records
description: >-
  Find ITER documents, records and identities through the Lucy MCP server's
  direct record tools instead of its chat tool. Use for signal lists,
  self-description documents, IDM document identities and versions, PBS/GBS
  breakdown nodes, folder containment, acronym expansion, or any typed search
  over ICP record kinds. Covers which tool answers which question, the traps
  each tool documents (queryable versus readable fields, declared projection
  names, unordered non-census pages, permission-withheld rows, the two-call
  concurrency cap), and worked search recipes.
---

# Find ITER records with Lucy's direct tools

Lucy exposes two ways to find documents and records: a chat tool that searches,
reads documents in full and writes an answer, and a set of typed record tools
that return rows straight from the source. **Reach for the record tools first.**
A chat run costs minutes and returns prose you then have to trust; a typed
search returns the matching records in seconds and lets you count computer
work.

Use the chat tool (`ask_lucy`, or `ask_start` + `ask_progress` where the host
does not render its live panel) only when the question genuinely needs
documents read and an answer written — an NCR's decision, a handover package's
contents, a comparison across several documents. If what you want is *which
records exist*, *the identities*, or *the fields of records you already have*,
use the direct tools.

The tools arrive in a Claude Code session with an `mcp__lucy2__` prefix; the
names below are the part after it.

## Which tool for which question

| Question | Tool |
|---|---|
| What kinds exist, and which fields/operators does one kind accept? | `icp_describe_record_schema` |
| Find records of one kind by typed predicates | `icp_search_records` |
| Read records whose UIDs you already hold | `icp_get_records` |
| One document's current and latest version | `icp_get_document_versions` |
| A record's attachments, comments, signatures, workflow | `icp_get_record_relationships` |
| What a container actually holds | `icp_list_folder_contents` |
| What a container *is* (title, link, parent) before listing it | `icp_get_folder` |
| Does this record really sit under that folder? | `icp_verify_folder_membership` |
| Expand an ITER acronym from the source table | `icp_lookup_abbreviation` |
| The file itself, in this conversation | `icp_get_records` for the metadata; `get_document_file` only when bytes are needed |
| A question that needs documents read and an answer written | `ask_lucy` |

Start every unfamiliar search with `icp_describe_record_schema`, naming the
kind: it returns that kind's `query_fields` (each with its own operators),
`read_properties` (each with its declared name), reference fields and limits.
Call it with an empty `record_type` only when you do not know what kind to
ask, since that returns one overview row for each of several hundred kinds and
is a large answer.

## Traps the tools document

These come from the server's own descriptions. Each one costs a wrong answer
when ignored, and the answer still looks plausible.

- **Only queryable fields filter.** A property you can read is not necessarily
  one you can search. On `Document`, `pbs` is in `read_properties` and absent
  from `query_fields`, so a `pbs` predicate is refused with
  `invalid_query` naming the fields the kind does take. Search the text fields
  and confirm by reading the property back.
- **A projection uses declared property names.** The declared version property
  is `current_version` (and `last_version`, `version_state`,
  `last_version_state`). Projecting `version` is refused with
  `invalid_projection` / `unknown_properties`, the declared list, and
  `source_read_attempted: false` — nothing was read.
- **Pages have no ordering.** ICP pages come in source UID order, which is not
  requestable and not chronological; a page is never "the latest N". Bound a
  date field to a window, sort the page yourself, and state the window. The
  one exception the schema names is a kind whose `source_system` is ICP4,
  which pages newest first.
- **A page is not a census.** Match totals are the service account's view, are
  not disclosed, and the page envelope says so (`total_scope:
  not_disclosed`, `snapshot_consistency: not_established`). Report what a page
  returned, never "there are N".
- **Withheld rows mean rows exist.** `withheld_on_page` counts scanned rows the
  requesting principal may not see. A page of nothing is not a statement that
  nothing matches.
- **Two calls in parallel, at most.** A third concurrent ICP call is refused as
  `capacity_exceeded` (retryable); retry it once after about a call, rather
  than fanning out.
- **A continuation token needs identical arguments.** Same predicates,
  projection and page size; a changed result set invalidates the cursor.
- **Project narrowly.** A page of a hundred documents carrying PBS references
  and titles is large. Ask for the five fields the next step uses, and keep
  `page_size` at the default 20 unless you have a reason. Limits on `Document`:
  100 records, `page_size` 100, 8 predicates.
- **Containment is a field, and it is not a census either.** The containment
  field is the `query_fields` entry whose `source_property` is
  `parent_content` — named `folder` on `Document`, `parent_content` on
  `PBS Node`. A typed containment query includes the named kind's subtypes and
  can miss rows the container listing has; use `icp_list_folder_contents` for
  what a container holds.
- **Naming a `record_type` in `icp_get_records` asserts it.** Leave it empty
  unless the kind is established: each UID is then read under the kind the
  source reports. A UID of another kind comes back `unexpected_record_type`;
  a UID whose kind has no registered schema comes back
  `unregistered_record_type`; a UID simply not returned lists `not_returned`
  in `coverage.outcomes`. The three are different facts. A dotted
  minor-content UID needs an explicit `record_type`; a version-shaped UID is
  an identity and is never split.
- **`follow` reads only one level** and requires an established `record_type`;
  person and file fields are never followed. A link says nothing about status
  or cause.
- **The schema tool is adapter metadata, not record values.** A
  `catalog-derived` property was never observed; an empty value there means
  not observed, never absent. A kind marked `is_navigation` is a container to
  walk, not to search.
- **A document's file is not a field.** `icp_get_records` never returns file
  content; `get_document_file` fetches bytes and spends context on them, so
  call it only when the file itself is needed by you or a next step.
- **`icp_get_document_versions` returns metadata only**, and current and latest
  are separate source facts: latest is not assumed released.
- **A folder listing reports coverage, not population.** A partial listing says
  which MetaTypes were not covered; `recursive` is bounded (depth 4, 300
  containers, 70 s) and names the bound that stopped it.

## Recipes for this project

Identities, numbers and codes below are invented. Take real ones from the
user's request or from the records you read, and do not put them in tracked
files.

### Signal lists

```json
{"record_type": "Document",
 "predicates": [{"field": "title", "operator": "contains", "values": ["Signal List"]}],
 "properties": ["uid", "title", "current_version", "version_state", "file_type", "pbs"],
 "page_size": 100}
```

`title contains` matched titles in mixed case in a live call, so one spelling
covers the corpus's variants, and a selective predicate caps upstream work
better than a page limit does. Follow the continuation with the *same* arguments until `page.has_more` is false; record
each page's `withheld_on_page` so the count you report says what you could not
see. A document's `pbs` property holds the PBS code(s) it names; read it rather
than inferring the system from the title.

### Self-description documents

```json
{"record_type": "Document",
 "predicates": [{"field": "title", "operator": "contains", "values": ["Self-Description"]}],
 "properties": ["uid", "title", "current_version", "file_type", "pbs"],
 "page_size": 50}
```

The corpus also uses the abbreviation "SDD" in titles. Run the two title
searches separately rather than one broad "self" search, and dedupe by `uid`.

### Documents naming one PBS node

`pbs` is not queryable, so search the text and confirm by reading it back:

1. Resolve the node: `icp_search_records` on `PBS Node` with
   `{"field": "code", "operator": "eq", "values": ["00.AA.BB"]}` and
   `properties: ["uid", "code", "title"]`.
2. Find candidates: the same search as the signal-list recipe, with the PBS
   code as the `title contains` value (many titles carry it), or the node's
   title once you have it.
3. Confirm each candidate with `icp_get_records`, projecting `pbs`, and keep
   only the records whose `pbs` includes the code.

Walk the breakdown tree downwards with `parent_content eq <node uid>` on
`PBS Node` to get a node's children; use `icp_list_folder_contents` instead
when you want the IDM folder's members.

### Versions of one document

```json
{"selection": "current_and_latest", "uids": ["AB12CD"]}
```

on `icp_get_document_versions`. Use `icp_get_records` with
`properties: ["uid", "current_version", "version_state", "last_version",
"last_version_state"]` when you also need the record's own fields, and
`icp_get_record_relationships` for the workflow and signatures of a specific
version.

### An acronym or system abbreviation

`icp_lookup_abbreviation` with `{"abbreviation": "PBS"}`. It returns every
meaning the source table holds and says when there was more than one; a term
it does not hold is "not in the observed table", which is not proof the
acronym is unused.

## The project's own read-only client

A pipeline does not go through these conversation tools. It fetches document
bytes through `iter_codex/discovery/icp`:

- `ICPDocuments.fetch(uid, version, storage_dir)` stores the file and returns a
  receipt (size, sha256, page count, extracted text path).
- `ICPRecords` answers typed plant, function-category and component reads.
- The client's access rule is **GET and HEAD only**. Its guarded session
  refuses any other method outside the sign-in exchange, and list-view and
  query POST operations are deliberately absent from the reader.

So: the MCP record tools are for finding and reading metadata in a
conversation; the project client is for fetching bytes and for pipeline reads.
Neither needs the chat tool, and neither may write.

## Keeping this skill true

Every tool name, field name and operator here was checked against the server's
own descriptions and against live calls at the time of writing; the evidence
fragment names where each was confirmed. The server's descriptions are the
authority. When a call is refused, read `error.details` — it names the
declared fields, the candidates, or the meaning it could not resolve, and the
refusal is usually the fastest documentation available.