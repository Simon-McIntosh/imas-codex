"""Every reader of a name's sources answers from the reader that owns them.

The graph holds a name's sources twice: as ``PRODUCED_NAME`` edges from the
source nodes, and as a ``source_paths`` list copied onto the name. The edges are
the authority, so a name whose copied list is empty while its edges are not must
still answer with its sources, and a name whose copied list names a path no edge
gives must answer with none.

The planted graph holds exactly the seven shapes the sourceless count has to
separate, one per case the rule names: an empty copy with no edge, a populated
copy with no edge, a derived name with no edge that holds a live child, a
superseded status, a superseded name stage, a name whose only producer is a
derived record, and a data-dictionary producer with an empty copy. A stale
producer is planted on its own: a name whose only producer is stale reads as
sourceless from search, from ``read_source_paths`` and from the count alike.

The last test walks the whole package for a touch of the copied list, so a
reader added in any module fails it, and every touch that is not a reader is
listed with the role that licenses it. A reader is never on that list: a reader
asks :mod:`imas_codex.standard_names.source_reads` for the enumerated population
or for the edges.

The population is one rule in one place, so a test also walks the package for a
query that states both halves of the rule — the liveness test and the
producerless test — and refuses one outside the owner that is not a stated
write, repair or guard precondition.
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.search import (
    fetch_standard_names,
    search_standard_names,
)
from imas_codex.standard_names.source_paths import merge_source_paths
from imas_codex.standard_names.source_reads import (
    count_live_names_without_source,
    normalize_source_ids,
    read_source_paths,
)

READER_MODULES = (
    "search.py",
    "benchmark_roles.py",
    "judgment_goldens.py",
)

#: The frozen-extraction records a committed corpus artifact is pinned to. Each
#: names the query that produced that artifact; it is kept verbatim so the
#: artifact can be checked against it, and reads nothing at runtime. They are
#: listed by name rather than recognised by suffix, because a live reader that
#: happens to sit in a constant named for a record is exactly what the guard
#: must still refuse.
FROZEN_EXTRACTION_RECORDS = frozenset({"_ADVISORY_CORPUS_EXTRACTION_RECORD"})


def _without_extraction_records(text: str) -> str:
    """Drop the frozen-extraction records a corpus is pinned to.

    Such a record is the query that produced a committed corpus artifact, kept
    verbatim so the artifact can be checked against it; it reads nothing at
    runtime. Everything else in the frozen modules is a live reader.
    """
    skipped: set[int] = set()
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Assign):
            continue
        named = any(
            isinstance(target, ast.Name) and target.id in FROZEN_EXTRACTION_RECORDS
            for target in node.targets
        )
        if named:
            skipped.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return "\n".join(
        line
        for number, line in enumerate(text.splitlines(), 1)
        if number not in skipped
    )


@pytest.fixture()
def planted_graph(disposable_neo4j_uri: str) -> Iterator[GraphClient]:
    """Plant the seven source shapes the count and the readers must separate.

    One name per case the rule names, so the count is asserted by value: three
    are counted and four are not, and the derived parent is counted like the
    rest and carries the attribute a reader exempts it on.
    """
    client = GraphClient(
        uri=disposable_neo4j_uri,
        username="neo4j",
        password=os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", ""),
        graph_name="source-reads-fixture",
    )
    client.query(
        """UNWIND [
          {id:'reads_empty_list', status:'draft', stage:'accepted',
           origin:'pipeline', paths:[]},
          {id:'reads_full_list', status:'draft', stage:'accepted',
           origin:'pipeline', paths:['dd:reads/ghost']},
          {id:'reads_derived_parent', status:'draft', stage:'accepted',
           origin:'derived', paths:[]},
          {id:'reads_status_superseded', status:'superseded', stage:'accepted',
           origin:'pipeline', paths:[]},
          {id:'reads_stage_superseded', status:'draft', stage:'superseded',
           origin:'pipeline', paths:[]},
          {id:'reads_derived_sourced', status:'draft', stage:'accepted',
           origin:'derived', paths:[]},
          {id:'reads_dd_bound', status:'draft', stage:'accepted',
           origin:'pipeline', paths:[]}
        ] AS row
        CREATE (:StandardName {id:row.id, status:row.status,
          name_stage:row.stage, origin:row.origin, source_paths:row.paths})"""
    )
    client.query(
        """UNWIND [
          {id:'derived:reads/source', holder:'reads_derived_sourced'},
          {id:'dd:reads/bound', holder:'reads_dd_bound'}
        ] AS row
        CREATE (source:StandardNameSource {id:row.id, source_id:row.id})
        WITH source, row
        MATCH (holder:StandardName {id:row.holder})
        CREATE (source)-[:PRODUCED_NAME]->(holder)"""
    )
    client.query(
        """MATCH (parent:StandardName {id:'reads_derived_parent'}),
                 (child:StandardName {id:'reads_derived_sourced'})
        CREATE (child)-[:HAS_PARENT]->(parent)"""
    )
    try:
        yield client
    finally:
        client.query(
            "MATCH (node) WHERE node.id STARTS WITH 'reads_' "
            "OR node.id STARTS WITH 'dd:reads/' "
            "OR node.id STARTS WITH 'derived:reads/' DETACH DELETE node"
        )
        client.close()


@pytest.fixture()
def stale_source_graph(disposable_neo4j_uri: str) -> Iterator[GraphClient]:
    """Plant one live name whose only producer is stale, and one live control.

    The control is produced by a source that is not stale, so the pair shows the
    stale edge is what makes the difference and not the absence of every edge.
    """
    client = GraphClient(
        uri=disposable_neo4j_uri,
        username="neo4j",
        password=os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", ""),
        graph_name="source-reads-stale-fixture",
    )
    client.query(
        """MATCH (node) WHERE node.id STARTS WITH 'stales_' DETACH DELETE node"""
    )
    client.query(
        """UNWIND [
          {id:'stales_only', status:'draft', stage:'accepted',
           origin:'pipeline'},
          {id:'stales_control', status:'draft', stage:'accepted',
           origin:'pipeline'}
        ] AS row
        CREATE (:StandardName {id:row.id, status:row.status,
          name_stage:row.stage, origin:row.origin, source_paths:[]})"""
    )
    client.query(
        """MATCH (holder:StandardName {id:'stales_only'})
        CREATE (source:StandardNameSource {id:'dd:stales/retired',
          source_id:'stales/retired', status:'stale'})
        CREATE (source)-[:PRODUCED_NAME]->(holder)"""
    )
    client.query(
        """MATCH (holder:StandardName {id:'stales_control'})
        CREATE (source:StandardNameSource {id:'dd:stales/live',
          source_id:'stales/live'})
        CREATE (source)-[:PRODUCED_NAME]->(holder)"""
    )
    try:
        yield client
    finally:
        client.query(
            "MATCH (node) WHERE node.id STARTS WITH 'stales_' "
            "OR node.id STARTS WITH 'dd:stales/' DETACH DELETE node"
        )
        client.close()


@pytest.mark.graph
def test_reader_answers_from_the_producer_edges(planted_graph: GraphClient) -> None:
    """An empty copy does not hide a name's sources, and a stale one does not
    invent them."""
    assert read_source_paths(planted_graph, "reads_dd_bound") == ["dd:reads/bound"]
    assert read_source_paths(planted_graph, "reads_full_list") == []
    assert read_source_paths(planted_graph, "reads_empty_list") == []


@pytest.mark.graph
def test_name_search_answers_from_the_producer_edges(
    planted_graph: GraphClient,
) -> None:
    """Search is a reader of the same place: the planted producer comes back
    with its path and the planted over-reporting copy comes back with none."""
    rows = fetch_standard_names(
        ["reads_dd_bound", "reads_full_list"],
        include_grammar=False,
        include_documentation=False,
        gc=planted_graph,
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["reads_dd_bound"]["source_paths"] == ["dd:reads/bound"]
    assert by_name["reads_full_list"]["source_paths"] == []

    found = search_standard_names("reads_dd_bound", k=5, gc=planted_graph)
    search_rows = {row["name"]: row for row in found}
    assert search_rows["reads_dd_bound"]["source_paths"] == ["dd:reads/bound"]


@pytest.mark.graph
def test_the_count_reports_three_of_seven_planted_names(
    planted_graph: GraphClient,
) -> None:
    """Seven planted names, one per case: three have no live producer.

    The count is one rule and no exemption, so the derived parent is one of the
    three — listed with ``derived_parent`` and its live children — and the two
    superseded names and the two produced names are not.
    """
    report = count_live_names_without_source(planted_graph)

    assert report["count"] == 3
    assert [entry["id"] for entry in report["names"]] == [
        "reads_derived_parent",
        "reads_empty_list",
        "reads_full_list",
    ]
    derived = report["names"][0]
    assert derived["derived_parent"] is True
    assert derived["live_children"] == 1
    assert derived["origin"] == "derived"
    assert [entry["derived_parent"] for entry in report["names"]] == [
        True,
        False,
        False,
    ]


@pytest.mark.graph
def test_the_count_is_the_owner_s_enumeration(planted_graph: GraphClient) -> None:
    """The command reports what the owner enumerates, by value.

    The command presents the population; it does not enumerate it. The two
    lists are therefore the same names in the same order, so a second query
    re-entering the command cannot answer differently from the invariant.
    """
    from imas_codex.standard_names.source_invariant import (
        find_sourceless_live_name_rows,
    )

    report = count_live_names_without_source(planted_graph)
    owner = find_sourceless_live_name_rows(
        planted_graph, None, derived_parents_exempt=False
    )
    assert [entry["id"] for entry in report["names"]] == [row["id"] for row in owner]
    assert report["count"] == len(owner)


@pytest.mark.graph
def test_a_name_whose_only_producer_is_stale_has_no_source(
    stale_source_graph: GraphClient,
) -> None:
    """Stale reads as no source from search, from read_source_paths and from
    the count alike, while the live-produced control answers with its source."""
    report = count_live_names_without_source(stale_source_graph)
    assert [entry["id"] for entry in report["names"]] == ["stales_only"]
    assert report["count"] == 1

    assert read_source_paths(stale_source_graph, "stales_only") == []
    assert read_source_paths(stale_source_graph, "stales_control") == ["dd:stales/live"]

    rows = fetch_standard_names(
        ["stales_only", "stales_control"],
        include_grammar=False,
        include_documentation=False,
        gc=stale_source_graph,
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["stales_only"]["source_paths"] == []
    assert by_name["stales_control"]["source_paths"] == ["dd:stales/live"]


def test_a_normalized_id_list_has_the_owner_s_order() -> None:
    """The reader normalises with the writer's union, so one list has one order

    wherever it is produced: a reader's answer and a writer's merge must not
    disagree about a name's sources.
    """
    ids = ["dd:b", "dd:a", "dd:b", "", None, "dd:c"]
    given = ["dd:b", "dd:a", "dd:b", "dd:c"]
    assert normalize_source_ids(ids) == merge_source_paths([], given)
    assert normalize_source_ids(ids) == ["dd:a", "dd:b", "dd:c"]
    assert normalize_source_ids(None) == []


_POPULATION = (
    Path(__file__).resolve().parents[2]
    / "docs/evidence/fragments/sn-catalog-consistency"
    / "scc-live-names-without-a-source.json"
)


@pytest.mark.graph
def test_the_recorded_population_carries_the_keys_the_command_emits(
    planted_graph: GraphClient,
) -> None:
    """The committed population is regenerated by the command, so its keys are
    the command's keys — a record transcribed into another shape cannot pass."""
    recorded = json.loads(_POPULATION.read_text(encoding="utf-8"))
    emitted = count_live_names_without_source(planted_graph)
    assert sorted(recorded) == sorted(emitted)


_COPIED_LIST = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.source_paths\b")


@pytest.mark.parametrize("module", READER_MODULES)
def test_each_reader_reads_sources_from_the_one_place(module: str) -> None:
    """A reader that spells the copied list, or the edge pattern, itself fails
    here: sources come from the projection that owns them."""
    package = Path(__file__).resolve().parents[2] / "imas_codex" / "standard_names"
    path = package / module
    assert path.is_file(), f"reader module is missing: {module}"
    text = _without_extraction_records(path.read_text(encoding="utf-8"))

    assert "source_paths_projection" in text, (
        f"{module} does not name the one place that reads a name's sources"
    )
    copied = _COPIED_LIST.search(text)
    assert copied is None, (
        f"{module} reads the copied list: {copied.group(0) if copied else ''}"
    )
    assert "PRODUCED_NAME" not in text, (
        f"{module} spells the producer pattern itself instead of asking "
        "source_paths_projection()"
    )


def test_the_review_pipeline_reads_sources_from_the_one_place() -> None:
    """The review pipeline loads the catalog with its own query, so its
    projection is checked where it sits: it must ask the one place too."""
    cli = Path(__file__).resolve().parents[2] / "imas_codex" / "cli" / "sn.py"
    tree = ast.parse(cli.read_text(encoding="utf-8"))
    load_catalog = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_load_catalog"
        ),
        None,
    )
    assert load_catalog is not None, "the review pipeline no longer loads a catalog"
    body = ast.get_source_segment(cli.read_text(encoding="utf-8"), load_catalog) or ""

    assert "source_paths_projection" in body, (
        "_load_catalog does not name the one place that reads a name's sources"
    )
    copied = _COPIED_LIST.search(body)
    assert copied is None, (
        f"_load_catalog reads the copied list: {copied.group(0) if copied else ''}"
    )


def test_a_reader_in_an_unlisted_record_constant_is_still_refused() -> None:
    """Only a record on the explicit list is skipped: a constant that merely
    ends the sanctioned suffix is a live reader and the guard refuses it."""
    source = (
        '_ADVISORY_CORPUS_EXTRACTION_RECORD = "RETURN sn.source_paths"\n'
        "_UNREVIEWED_CORPUS_EXTRACTION_RECORD = (\n"
        '    "MATCH (sn:StandardName)\\n"\n'
        '    "RETURN sn.source_paths AS source_paths"\n'
        ")\n"
    )
    text = _without_extraction_records(source)

    assert "sn.source_paths" not in text.split("_UNREVIEWED")[0]
    copied = _COPIED_LIST.search(text)
    assert copied is not None, (
        "the unlisted record was skipped although it reads the copied list"
    )
    assert copied.group(0) == "sn.source_paths"


#: What a touch of the copied list may be. A reader is deliberately absent: a
#: reader answers "what are this name's sources" and must ask
#: :func:`imas_codex.standard_names.source_reads.source_paths_projection`
#: instead, so that no two readers can answer differently for one name.
ROLES = {
    "owner": "encodes, parses or unions the property's value",
    "writer": "sets or maintains the list on a name",
    "repair": "reads the list as an input to a repair or migration write",
    "recovery": "reads the list to rebuild provenance an edge lost",
    "record": "a frozen record of the query a committed artifact was made with",
    "inspection": "the list disagreeing with the edges is the subject",
    "guard": "a write precondition compares the list with the bindings",
    "published": "the publication output excludes the property",
    "runtime": "an in-memory item dict whose value came from the routed read",
    "other_label": "the property belongs to another label or a config object",
}

#: Every code-level touch of a name's copied list in the package, by module and
#: by the statement that carries it. Adding a touch anywhere — a query that
#: spawns the property, a row key read, a keyword passed on — fails
#: :func:`test_every_touch_of_the_copied_list_is_classified` until it is listed
#: here with a role, or routed through the one reader.
COPIED_LIST_SITES: dict[tuple[str, str], str] = {
    ("cli/sn.py", "_load_catalog"): "runtime",
    ("cli/sn.py", "_run"): "runtime",
    ("cli/sn.py", "_echo_dd_gap_fact"): "other_label",
    ("graph/sn_cleanup.py", "collect_purge_candidates"): "runtime",
    ("llm/sn_tools.py", "_format_fetch_report"): "runtime",
    ("standard_names/attachment_audit.py", "_DETACH_QUERY"): "writer",
    ("standard_names/attachment_audit.py", "_DETACH_PROJECTION_QUERY"): "writer",
    ("standard_names/attachment_audit.py", "_ATTACH_QUERY"): "writer",
    (
        "standard_names/attachment_audit.py",
        "_TERMINAL_RECOVERY_ELIGIBILITY_QUERY",
    ): "repair",
    ("standard_names/attachment_audit.py", "_TERMINAL_RECOVERY_QUERY"): "writer",
    ("standard_names/attachment_audit.py", "_TERMINAL_RECOVERY_BATCH_QUERY"): "writer",
    (
        "standard_names/attachment_audit.py",
        "_terminal_recovery_preserved_payload",
    ): "repair",
    (
        "standard_names/attachment_audit.py",
        "_terminal_recovery_is_already_current",
    ): "repair",
    ("standard_names/attachment_audit.py", "_terminal_recovery_plan_row"): "repair",
    ("standard_names/audits.py", "vector_family_consistency_check"): "runtime",
    ("standard_names/audits.py", "dd_path_uniqueness_check"): "runtime",
    ("standard_names/benchmark.py", "score_with_reviewer"): "runtime",
    ("standard_names/benchmark.py", "score_descriptions"): "runtime",
    ("standard_names/benchmark.py", "_gen_one"): "runtime",
    ("standard_names/benchmark.py", "_extract_candidates"): "other_label",
    ("standard_names/benchmark.py", "_run_model"): "runtime",
    ("standard_names/benchmark_roles.py", "load_refine_corpus"): "runtime",
    ("standard_names/benchmark_roles.py", "load_breaker_corpus"): "runtime",
    ("standard_names/benchmark_roles.py", "load_docs_sample"): "runtime",
    (
        "standard_names/benchmark_roles.py",
        "_ADVISORY_CORPUS_EXTRACTION_RECORD",
    ): "record",
    ("standard_names/benchmark_roles.py", "build_advisory_paired_corpus"): "runtime",
    ("standard_names/benchmark_roles.py", "load_discrimination_corpus"): "runtime",
    ("standard_names/consolidation.py", "_merge_duplicates"): "runtime",
    ("standard_names/context.py", "fetch_review_neighbours"): "runtime",
    ("standard_names/dd_gaps.py", "_evidence_snapshot"): "other_label",
    ("standard_names/dd_gaps.py", "_registry_sync_plan"): "other_label",
    ("standard_names/dd_gaps.py", "sync_dd_unit_exception_gaps"): "other_label",
    ("standard_names/dd_gaps.py", "transition_dd_gap"): "other_label",
    ("standard_names/dd_gaps.py", "_RECONCILE_QUERY"): "other_label",
    ("standard_names/dd_gaps.py", "reconcile_dd_gaps"): "other_label",
    ("standard_names/dd_gaps.py", "list_dd_gaps"): "other_label",
    ("standard_names/dd_gaps.py", "get_dd_gap"): "other_label",
    ("standard_names/dd_resolutions.py", "_record_from_graph"): "other_label",
    ("standard_names/docs_holdout_eval.py", "_candidate_for"): "runtime",
    ("standard_names/docs_holdout_eval.py", "_generation_item"): "runtime",
    ("standard_names/domain_decision.py", "build_state"): "runtime",
    ("standard_names/domain_decision.py", "build_family_state"): "runtime",
    ("standard_names/edit.py", "_stamp_successor_validation"): "runtime",
    ("standard_names/edit.py", "_FOLD_NAME_MUTATION_QUERY"): "writer",
    ("standard_names/edit.py", "_fold_expected_state"): "runtime",
    ("standard_names/edit.py", "_RECLAIM_NAME_PATHS_QUERY"): "recovery",
    ("standard_names/edit.py", "_reclaim_spelling_case"): "recovery",
    ("standard_names/edit.py", "_reclaim_locator_only_case"): "recovery",
    ("standard_names/edit.py", "_RECLAIM_MUTATION_QUERY"): "writer",
    ("standard_names/edit.py", "_archive_history_for_reclaim"): "recovery",
    ("standard_names/error_siblings.py", "mint_error_siblings"): "writer",
    ("standard_names/export.py", "_PROVENANCE_FIELDS"): "published",
    (
        "standard_names/graph_store/graph_audits.py",
        "reconcile_source_status_liveness",
    ): "repair",
    (
        "standard_names/graph_store/graph_audits.py",
        "reconcile_standard_name_source_paths",
    ): "repair",
    (
        "standard_names/graph_store/graph_audits.py",
        "reconcile_descriptionless_composed_names",
    ): "writer",
    (
        "standard_names/graph_store/graph_audits.py",
        "reconcile_sourceless_pipeline_names",
    ): "writer",
    (
        "standard_names/graph_store/graph_catalog.py",
        "_published_dd_paths_by_name",
    ): "runtime",
    ("standard_names/graph_store/graph_catalog.py", "write_standard_names"): "writer",
    (
        "standard_names/graph_store/graph_catalog.py",
        "persist_generated_name_winners",
    ): "writer",
    (
        "standard_names/graph_store/graph_claim_release.py",
        "release_stranded_refine_claim",
    ): "guard",
    (
        "standard_names/graph_store/graph_parent_lifecycle.py",
        "_UNBACKED_SOURCE_PATH_SCALARS",
    ): "writer",
    (
        "standard_names/graph_store/graph_source_bindings.py",
        "_lock_claimed_name_bindings",
    ): "guard",
    (
        "standard_names/graph_store/graph_source_bindings.py",
        "persist_claimed_attachments",
    ): "writer",
    ("standard_names/graph_store/graph_supersession.py", "_apply"): "writer",
    (
        "standard_names/graph_store/graph_supersession.py",
        "deduplicate_scalar_selected_sources",
    ): "writer",
    (
        "standard_names/graph_store/graph_supersession.py",
        "classify_sourceless_standard_names",
    ): "inspection",
    (
        "standard_names/graph_store/graph_supersession.py",
        "supersede_into_ancestor",
    ): "writer",
    ("standard_names/judgment_goldens.py", "_live_documentation_names"): "runtime",
    ("standard_names/judgment_goldens.py", "build_documentation_sheet"): "runtime",
    ("standard_names/promote.py", "_REVIEW_NODE_FIELDS"): "runtime",
    (
        "standard_names/provenance_lifecycle.py",
        "retarget_standard_name_sources",
    ): "writer",
    ("standard_names/provenance_lifecycle.py", "reset_standard_name_sources"): "writer",
    ("standard_names/provenance_lifecycle.py", "bind_sources_exclusively"): "writer",
    (
        "standard_names/provenance_lifecycle.py",
        "_SEMANTIC_SOURCE_PATH_INSPECTION",
    ): "repair",
    ("standard_names/provenance_lifecycle.py", "_SEMANTIC_SOURCE_PATH_WRITE"): "writer",
    ("standard_names/provenance_rebuild.py", "_fetch_dd_source_paths"): "recovery",
    ("standard_names/release_notes.py", "summarize_dd_gap_facts"): "other_label",
    ("standard_names/review/audits.py", "compute_review_input_hash"): "runtime",
    ("standard_names/review/jev_review.py", "build_docs_state"): "runtime",
    ("standard_names/review/pipeline.py", "_fetch_review_dd_context"): "runtime",
    ("standard_names/search.py", "_enrich_search_rows"): "runtime",
    ("standard_names/search.py", "fetch_standard_names"): "runtime",
    (
        "standard_names/signed_manifest.py",
        "_ordinary_source_migration_refusal",
    ): "inspection",
    (
        "standard_names/signed_manifest.py",
        "_unbound_source_attachment_refusal",
    ): "inspection",
    ("standard_names/signed_manifest.py", "_retarget_supersede_refusal"): "inspection",
    (
        "standard_names/signed_manifest.py",
        "_verify_ordinary_source_migration_postcondition",
    ): "inspection",
    (
        "standard_names/signed_manifest.py",
        "_verify_unbound_source_attachment_postcondition",
    ): "inspection",
    ("standard_names/signed_manifest.py", "_apply_ordinary_source_migration"): "writer",
    ("standard_names/signed_manifest.py", "_apply_unbound_source_attachment"): "writer",
    ("standard_names/signed_manifest.py", "_apply_mutation"): "writer",
    ("standard_names/signed_manifest.py", "_apply_dual_authority_retirement"): "writer",
    (
        "standard_names/signed_manifest.py",
        "_apply_catalog_source_dispositions",
    ): "writer",
    (
        "standard_names/signed_manifest.py",
        "_apply_ineligible_source_retirement",
    ): "writer",
    (
        "standard_names/signed_manifest.py",
        "_verify_source_exchange_postcondition",
    ): "inspection",
    ("standard_names/source_paths.py", "<module>"): "owner",
    ("standard_names/source_reads.py", "source_paths_projection"): "owner",
    ("standard_names/source_reads.py", "read_source_paths_by_name"): "owner",
    (
        "standard_names/worker_tasks/worker_compose_and_validate.py",
        "_compose_batch_body",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_compose_and_validate.py",
        "validate_name_candidate",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_compose_and_validate.py",
        "_get_recent_names",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_compose_batches.py",
        "compose_batch",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_generate_docs.py",
        "_enrich_for_docs_gen",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_refine_names.py",
        "process_refine_name_batch",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_review_docs.py",
        "_bound_docs_review_dd_paths",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_review_docs.py",
        "_set_docs_review_dd_source_paths",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_review_docs.py",
        "_has_dd_source_reference",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_review_docs.py",
        "_enrich_docs_review_dd_context",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_review_docs.py",
        "prepare_docs_review_request",
    ): "runtime",
    (
        "standard_names/worker_tasks/worker_review_names.py",
        "_enrich_name_review_items",
    ): "runtime",
}


#: Auto-generated model modules: ``uv sync`` rebuilds them and no commit stages
#: them, so a hit there is not a site anyone can act on.
GENERATED_MODULES = frozenset(
    {
        "config/models.py",
        "graph/dd_models.py",
        "graph/models.py",
        "graph/schema_context_data.py",
    }
)

#: A property-qualified reference in text: the Cypher form ``sn.source_paths``
#: and the Python attribute form alike.
_QUALIFIED_REFERENCE = re.compile(r"\b[A-Za-z_]\w*\.source_paths\b")


def _carrying_statement(tree: ast.Module, line: int) -> str:
    """The symbol a touch sits in: the innermost def, else a module-level name."""
    best, name = -1, "<module>"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.lineno <= line <= (node.end_lineno or node.lineno):
                if node.lineno > best:
                    best, name = node.lineno, node.name
    if best != -1:
        return name
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.lineno <= line <= (node.end_lineno or node.lineno):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        return target.id
    return name


def _touched_lines(tree: ast.Module) -> list[int]:
    """Lines where the copied list is spawned, projected, read or passed on."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "source_paths":
            lines.append(node.lineno)
        elif isinstance(node, ast.keyword) and node.arg == "source_paths":
            lines.append(node.lineno)
        elif isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and key.value == "source_paths":
                lines.append(node.lineno)
            elif isinstance(key, ast.Name) and key.id == "source_paths":
                lines.append(node.lineno)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value == "source_paths" or _QUALIFIED_REFERENCE.search(value):
                lines.append(node.lineno)
    return lines


def package_touches(root: Path) -> dict[tuple[str, str], set[int]]:
    """Every touch of a name's copied list under ``root``, by module and symbol."""
    touches: dict[tuple[str, str], set[int]] = {}
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).as_posix()
        if module in GENERATED_MODULES or "__pycache__" in module:
            continue
        text = path.read_text(encoding="utf-8")
        if "source_paths" not in text:
            continue
        tree = ast.parse(text)
        for line in _touched_lines(tree):
            site = (module, _carrying_statement(tree, line))
            touches.setdefault(site, set()).add(line)
    return touches


def test_every_touch_of_the_copied_list_is_classified() -> None:
    """A touch of the copied list in any module fails until it is classified.

    The scan walks the whole package, so a reader added in a module no review
    names — the review pipeline included — is caught here rather than by
    whichever caller first asks it for a name's sources.
    """
    package = Path(__file__).resolve().parents[2] / "imas_codex"
    touches = package_touches(package)

    # Positive control: a broken or empty scan would otherwise license every
    # reader by finding nothing to object to.
    assert ("standard_names/source_reads.py", "source_paths_projection") in touches
    assert ("standard_names/search.py", "fetch_standard_names") in touches
    assert len(touches) > 100, f"the scan found only {len(touches)} sites"

    unlisted = sorted(set(touches) - set(COPIED_LIST_SITES))
    assert not unlisted, (
        "these sites touch a name's copied source_paths list and are not listed "
        "with a role. A reader answers with read_source_paths() or by projecting "
        "source_paths_projection(); every other touch takes a role in "
        "COPIED_LIST_SITES:\n" + "\n".join(f"  {m}:{s}" for m, s in unlisted)
    )

    stale = sorted(set(COPIED_LIST_SITES) - set(touches))
    assert not stale, (
        "COPIED_LIST_SITES licenses nothing here: the code moved or was routed "
        "through the one reader, so drop the entry or correct the symbol:\n"
        + "\n".join(f"  {m}:{s}" for m, s in stale)
    )

    unknown = sorted(
        f"{module}:{symbol}: {role}"
        for (module, symbol), role in COPIED_LIST_SITES.items()
        if role not in ROLES
    )
    assert not unknown, "these sites carry a role outside ROLES: " + ", ".join(unknown)


#: The owner of the sourceless population. A statement outside the owner that
#: composes both halves of the rule — the liveness test and the absence of a
#: producer — is a second enumeration of that population, and two enumerations
#: is how a count reported 2,609 on 2026-11-14 while the producer edges held 10.
SOURCE_RULE_OWNER = (
    "standard_names/source_invariant.py",
    "find_sourceless_live_name_rows",
)

#: What a statement that states both halves of the rule may be. ``owner`` is the
#: one enumeration and belongs to the owner module alone; every other role is a
#: statement about names the caller already holds, so it answers a precondition
#: or performs a write rather than enumerating the population.
SOURCE_RULE_ROLES = {
    "owner": "enumerates the population, and only the owner may",
    "writer": "a write whose precondition is that the name it holds has no producer",
    "guard": "a guard that refuses when the name it holds would have no producer",
    "repair": "a repair or migration that selects names by the rule it restores",
    "recovery": "a rebuild that answers with the sources an edge lost",
    "report": "a census of names by status that also reports a producer count, "
    "not the population",
}

#: Every statement in the package that states both halves of the rule, by module
#: and statement, with the role that licenses it. A statement added anywhere
#: fails :func:`test_no_second_enumeration_of_the_sourceless_population` until it
#: asks :func:`imas_codex.standard_names.source_invariant.find_sourceless_live_name_rows`
#: or is listed here with a role that makes it a precondition rather than a
#: population read.
SOURCE_RULE_SITES: dict[tuple[str, str], str] = {
    SOURCE_RULE_OWNER: "owner",
    ("standard_names/source_invariant.py", "_drained_test"): "writer",
    ("cli/sn.py", "sn_status"): "report",
    (
        "standard_names/graph_store/graph_audits.py",
        "reconcile_source_status_liveness",
    ): "repair",
    (
        "standard_names/graph_store/graph_audits.py",
        "reconcile_sourceless_pipeline_names",
    ): "repair",
    (
        "standard_names/graph_store/graph_claim_release.py",
        "_READ_REFINE_CLAIM_RELEASE_QUERY",
    ): "guard",
    (
        "standard_names/graph_store/graph_parent_lifecycle.py",
        "_persist_structural_authority",
    ): "writer",
    (
        "standard_names/graph_store/graph_parent_lifecycle.py",
        "find_orphan_parent_source_candidates",
    ): "repair",
    (
        "standard_names/graph_store/graph_parent_lifecycle.py",
        "reconcile_orphan_parent_sources",
    ): "repair",
    (
        "standard_names/graph_store/graph_parent_lifecycle.py",
        "reconcile_orphan_parent_sources_batched",
    ): "repair",
    (
        "standard_names/graph_store/graph_parent_lifecycle.py",
        "structural_accept_derived_parents",
    ): "writer",
    (
        "standard_names/graph_store/graph_refinement.py",
        "_GENERATED_SUPERSESSION_PREFLIGHT_QUERY",
    ): "guard",
    (
        "standard_names/graph_store/graph_refinement.py",
        "persist_refined_name",
    ): "writer",
    (
        "standard_names/graph_store/graph_source_bindings.py",
        "mark_sources_stale",
    ): "writer",
    ("standard_names/graph_store/graph_supersession.py", "_apply"): "writer",
    (
        "standard_names/graph_store/graph_supersession.py",
        "supersede_exhausted_standard_name_orphans",
    ): "writer",
    (
        "standard_names/graph_store/graph_supersession.py",
        "supersede_into_ancestor",
    ): "writer",
    (
        "standard_names/provenance_lifecycle.py",
        "_SEMANTIC_SOURCE_REPAIR_INSPECTION",
    ): "repair",
    (
        "standard_names/provenance_lifecycle.py",
        "find_semantic_source_invariant_violations",
    ): "repair",
    (
        "standard_names/provenance_rebuild.py",
        "_fetch_change_history_sources",
    ): "recovery",
    (
        "standard_names/signed_manifest.py",
        "_apply_catalog_source_dispositions",
    ): "writer",
    ("standard_names/signed_manifest.py", "_apply_dual_authority_retirement"): "writer",
    (
        "standard_names/signed_manifest.py",
        "_apply_ineligible_source_retirement",
    ): "writer",
    ("standard_names/signed_manifest.py", "_apply_mutation"): "writer",
    ("standard_names/signed_manifest.py", "_apply_source_exchange"): "writer",
    ("standard_names/signed_manifest.py", "_apply_unbound_source_attachment"): "writer",
    ("standard_names/signed_manifest.py", "_dd_residue_release_refusal"): "guard",
    ("standard_names/signed_manifest.py", "_orphan_guard_refusal"): "guard",
    (
        "standard_names/signed_manifest.py",
        "_source_target_reconciliation_refusal",
    ): "guard",
    (
        "standard_names/signed_manifest.py",
        "_structural_source_revival_refusal",
    ): "guard",
    ("standard_names/signed_manifest.py", "_verify_postconditions"): "guard",
}


def _statement_strings(tree: ast.Module) -> list[tuple[int, str]]:
    """Every string constant in a module, an f-string unparsed with its calls.

    The two halves of the rule are interpolated from the owner by most callers,
    so a module that asks for the rule carries neither ``PRODUCED_NAME`` nor
    ``superseded`` of its own. Unparsing the f-string keeps the call it makes
    visible to the scan, so a restored enumeration cannot hide behind an
    interpolation.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append((node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            found.append((node.lineno, ast.unparse(node)))
    return found


def _states_liveness(text: str) -> bool:
    """Does this statement apply the live-name half of the rule?

    Either by naming the owner's predicate, by spelling the superseded status
    and name stage itself, or by excluding terminal statuses through the
    parameter list a restored enumeration would pass in. That last spelling —
    ``NOT sn.status IN $terminal`` with ``'superseded'`` in the parameter — was
    the source audit's own query before it was deleted, so a scan that did not
    recognise it would license the query coming back.
    """
    if "live_name_cypher" in text:
        return True
    if "superseded" in text and ("name_stage" in text or "status" in text):
        return True
    return "status" in text and re.search(r"\$terminal\b", text) is not None


def _states_producerless(text: str) -> bool:
    """Does this statement apply the no-live-producer half of the rule?

    Either by naming the owner's predicate, or by negating the edge itself.
    """
    if "producerless_cypher" in text:
        return True
    return "PRODUCED_NAME" in text and ("NOT EXISTS" in text or "NOT (" in text)


def sourceless_rule_sites(root: Path) -> dict[tuple[str, str], set[int]]:
    """Every statement under ``root`` that states both halves of the rule."""
    sites: dict[tuple[str, str], set[int]] = {}
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).as_posix()
        if module in GENERATED_MODULES or "__pycache__" in module:
            continue
        text = path.read_text(encoding="utf-8")
        if "StandardName" not in text:
            continue
        tree = ast.parse(text)
        for line, value in _statement_strings(tree):
            if "StandardName" not in value:
                continue
            if _states_liveness(value) and _states_producerless(value):
                site = (module, _carrying_statement(tree, line))
                sites.setdefault(site, set()).add(line)
    return sites


def test_the_pool_projection_reads_the_live_child_rule() -> None:
    """``pools.py`` reads the rule rather than restating its stage half."""
    from imas_codex.standard_names.pools import identity_state_expressions
    from imas_codex.standard_names.source_invariant import live_child_cypher

    expression = identity_state_expressions()["has_live_child"]
    assert expression == live_child_cypher()
    assert "status" in expression, (
        "the rule must test the child's status as well as its name stage"
    )


def test_the_planted_source_audit_query_is_refused(tmp_path: Path) -> None:
    """The audit's deleted query is seen by the scan and refused.

    It stated the liveness half as ``NOT sn.status IN $terminal`` with
    ``'superseded'`` in the parameter, so the scan has to recognise a
    parameterised terminal exclusion as well as the literal spelling.
    """
    (tmp_path / "restored_audit.py").write_text(
        'POPULATION = """\n'
        "MATCH (sn:StandardName)\n"
        "WHERE NOT sn.status IN $terminal\n"
        "  AND NOT EXISTS { (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)\n"
        "                   WHERE coalesce(source.status, '') <> 'stale' }\n"
        'RETURN sn.id AS id\n"""\n',
        encoding="utf-8",
    )

    assert sourceless_rule_sites(tmp_path) == {("restored_audit.py", "POPULATION"): {1}}


def test_no_second_enumeration_of_the_sourceless_population() -> None:
    """A statement stating both halves of the rule outside the owner is refused.

    The population has one enumeration. The ``sn sourceless`` command reports
    what ``find_sourceless_live_name_rows`` returns and the source audit
    classifies what that same enumeration names, so no module can answer "which
    live names have no source" with a rule of its own. Before this guard the
    package held three such statements: the count's own query named 9 names on
    2026-11-15, the source audit's own query named 27, and the producer edges
    named 10.
    """
    package = Path(__file__).resolve().parents[2] / "imas_codex"
    sites = sourceless_rule_sites(package)

    # Positive control: the scan must see the owner's own enumeration, and the
    # package holds at least the preconditions listed below, so an empty or
    # broken scan cannot license every second enumeration by finding nothing.
    assert SOURCE_RULE_OWNER in sites, (
        "the scan does not find the owner's enumeration, so it would license anything"
    )
    assert len(sites) > 10, f"the scan found only {len(sites)} sites"

    unlisted = sorted(set(sites) - set(SOURCE_RULE_SITES))
    assert not unlisted, (
        "these statements state both halves of the sourceless rule — a live "
        "name and an absent producer — and are not the owner's enumeration. "
        "Ask find_sourceless_live_name_rows() for the population, or list the "
        "site in SOURCE_RULE_SITES with the role that makes it a precondition:\n"
        + "\n".join(f"  {module}:{symbol}" for module, symbol in unlisted)
    )
    stale = sorted(set(SOURCE_RULE_SITES) - set(sites))
    assert not stale, (
        "SOURCE_RULE_SITES licenses nothing here: the code moved or now asks "
        "the owner, so drop the entry or correct the symbol:\n"
        + "\n".join(f"  {module}:{symbol}" for module, symbol in stale)
    )
    unknown = sorted(
        f"{module}:{symbol}: {role}"
        for (module, symbol), role in SOURCE_RULE_SITES.items()
        if role not in SOURCE_RULE_ROLES
    )
    assert not unknown, (
        "these sites carry a role outside SOURCE_RULE_ROLES: " + ", ".join(unknown)
    )
    owners = sorted(site for site, role in SOURCE_RULE_SITES.items() if role == "owner")
    assert owners == [SOURCE_RULE_OWNER], (
        f"the population has one enumeration; these sites claim to own it: {owners}"
    )
