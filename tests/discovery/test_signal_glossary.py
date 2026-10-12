"""Facility glossary context in EDAS signal enrichment prompts."""

import re
from unittest.mock import patch

import pytest
import yaml

from imas_codex.discovery.base.facility import get_facility
from imas_codex.discovery.signals import parallel
from imas_codex.discovery.signals.models import (
    SignalEnrichmentBatch,
    SignalEnrichmentResult,
)


async def _enrichment_prompt(signals: list[dict]) -> str:
    """Run one enrichment batch over ``signals`` and return its user prompt."""
    config = get_facility("jt-60sa")
    state = parallel.DataDiscoveryState(
        facility="jt-60sa", scanner_types=["edas"], facility_config=config
    )
    claims = iter([signals, []])
    prompts = []

    def claim(*_args, **_kwargs):
        batch = next(claims)
        if not batch:
            state.stop_requested = True
        return batch

    async def llm(*, messages, **_kwargs):
        prompts.append(messages[1]["content"])
        return (
            SignalEnrichmentBatch(
                results=[
                    SignalEnrichmentResult(
                        signal_index=index,
                        physics_domain="general",
                        name="Signal",
                        description="A signal",
                    )
                    for index in range(1, len(signals) + 1)
                ]
            ),
            0.0,
            0,
        )

    with (
        patch.object(parallel, "claim_signals_for_enrichment", side_effect=claim),
        patch.object(parallel, "prepare_signal_sources", return_value=(0, 0, 0)),
        patch.object(parallel, "propagate_units_from_signal_nodes", return_value=0),
        patch.object(parallel, "fetch_tree_context", return_value={}),
        patch.object(parallel, "fetch_epoch_context", return_value={}),
        patch.object(parallel, "fetch_signal_code_refs", return_value={}),
        patch.object(parallel, "_fetch_code_chunks", return_value=[]),
        patch.object(parallel, "mark_signals_enriched"),
        patch.object(parallel, "mark_signals_underspecified"),
        patch.object(parallel, "propagate_source_enrichment", return_value=0),
        patch(
            "imas_codex.discovery.signals.scanners.wiki.fetch_semantic_wiki_context",
            return_value=[],
        ),
        patch("imas_codex.discovery.base.llm.acall_llm_structured", side_effect=llm),
    ):
        await parallel.enrich_worker(state)

    assert len(prompts) == 1
    return prompts[0]


@pytest.mark.asyncio
async def test_a_term_in_the_catalogue_text_reaches_the_prompt():
    command, compressor = (
        await _enrichment_prompt(
            [
                {
                    "id": "zp0",
                    "accessor": "read zp0",
                    "discovery_source": "edas",
                    "data_source_path": "PSRC/comZp0",
                    "name": None,
                    "source_description": "Plasma Vertical Pos. (Zp0) Com.",
                },
                {
                    "id": "compressor",
                    "accessor": "read compressor",
                    "discovery_source": "edas",
                    "data_source_path": "CryoHRS/comPress1",
                    "name": None,
                    "source_description": "Compressor 1 discharge pressure",
                },
            ]
        )
    ).split("### Signal")[1:]
    assert "glossary: Com. — command" in command
    assert "glossary: Com." not in compressor


@pytest.mark.asyncio
async def test_edas_prompt_scopes_category_and_term_context_to_signals():
    signals = [
        {
            "id": "coil-path",
            "accessor": "read coil",
            "discovery_source": "edas",
            "data_source_path": "MMSYS/curUFPLKAT",
            "name": None,
        },
        {
            "id": "coil-name",
            "accessor": "read name",
            "discovery_source": "edas",
            "data_source_path": "MMSYS/other",
            "name": "LFP coil current",
        },
        {
            "id": "other",
            "accessor": "read other",
            "discovery_source": "edas",
            "data_source_path": "PSRC/magFluxLp1",
            "name": None,
        },
    ]
    prompt = await _enrichment_prompt(signals)
    category, coil_path, coil_name, unrelated = prompt.split("### Signal")
    assert "MMSYS" in category
    assert "coil currents" in category
    assert "glossary: UFP — upper fast plasma position control coil" in coil_path
    assert "glossary: LFP — lower fast plasma position control coil" in coil_name
    assert "HiTe" not in unrelated
    assert "FPPCC" not in unrelated
    assert "coil currents" not in unrelated


def test_the_boundary_matcher_holds_a_term_to_whole_words():
    assert parallel.glossary_term_reaches("EFC", "PSSCfst/refvcEFC1")
    assert not parallel.glossary_term_reaches("EFC", "PSSCfst/refcurEF1")
    # The matcher it replaces accepted a case-folded substring, which EFC is
    # inside refcurEF1 ("refcuref1"); keep that difference visible.
    assert "efc" in "PSSCfst/refcurEF1".casefold()


def test_a_term_reaches_across_a_digit_to_letter_turn():
    """A digit-to-letter turn, and a capital run's hand-off, start a word."""
    assert parallel.glossary_term_reaches("HiTe", "MMSYS/curCS1HiTe")
    assert parallel.glossary_term_reaches("HiTe", "MMSYS/curEF2HiTe")
    assert parallel.glossary_term_reaches("HiTe", "MMSYS/curLFPHiTe")
    assert parallel.glossary_term_reaches("HiTe", "MMSYS/curUFPHiTe")
    # An acronym written as capitals may open onto a following word.
    assert parallel.glossary_term_reaches("UFP", "MMSYS/curUFPdisr")
    assert parallel.glossary_term_reaches("LFP", "MMSYS/curLFPdisr")
    # ... but a term inside a longer token is still refused, and so is a
    # capitalised ordinary word that merely begins with the letters.
    assert not parallel.glossary_term_reaches("LFP", "PSC/calFP1")
    assert not parallel.glossary_term_reaches("EFC", "PSSCfst/refcurEF1")
    assert not parallel.glossary_term_reaches("DR", "CRYOHRS/CDPT171")
    assert not parallel.glossary_term_reaches(
        "DR", "CryoHRS/cFT195 Cryo LP By-pass Pressure Drop Flow FE195"
    )
    # A digit inside a token's capital run does not start a word, so the DR of
    # "radial displacement" keeps out of the dump-resistor glossary line.
    assert not parallel.glossary_term_reaches("DR", "CryoMag/mC01DR")


def test_a_source_is_swept_for_its_representative_not_its_member_edge():
    """A representative may itself be a member of a different source.

    Binding the source through the representative's own MEMBER_OF edge then
    skips every source it represents, so the source is located by
    ``representative_id`` instead.
    """
    recorded = []

    class StubGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def query(self, cypher, **params):
            recorded.append(cypher)
            return [{"representative_id": "jt-60sa:general/cs1_temsnur11", "enrichment": {}}]

    with (
        patch.object(parallel, "GraphClient", StubGraph),
        patch.object(parallel, "propagate_source_enrichment", return_value=7) as propagate,
    ):
        assert parallel.sweep_source_enrichment("jt-60sa") == 7

    assert "MATCH (rep:FacilitySignal {id: sg.representative_id})" in recorded[0]
    propagate.assert_called_once_with("jt-60sa:general/cs1_temsnur11", {})


def test_every_propagated_statement_binds_the_parameters_it_names():
    """A repaired MATCH must still be handed the parameter it binds.

    A statement that names ``$rep_id`` and is called without it fails at
    runtime with ParameterMissing, which the pipeline reports only as a
    non-fatal source-preparation warning — so the propagation silently stops
    happening. The stub refuses that call, and counts how many statements ran.
    """
    calls = []

    class BindingGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def query(self, cypher, **params):
            missing = set(re.findall(r"\$(\w+)", cypher)) - set(params)
            assert not missing, f"statement binds {sorted(missing)} but is not given it"
            calls.append(cypher)
            return [{"cnt": 1, "updated": 1}]

    with patch.object(parallel, "GraphClient", BindingGraph):
        updated = parallel.propagate_source_enrichment(
            "jt-60sa:general/cs1_temsnur11",
            {"description": "Switching network unit (SNU)", "diagnostic": "magnetics"},
        )

    assert updated == 1
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_a_coil_abbreviation_reaches_its_own_signal_not_a_lookalike():
    prompt = await _enrichment_prompt(
        [
            {
                "id": "efc",
                "accessor": "read efc",
                "discovery_source": "edas",
                "data_source_path": "PSSCfst/refvcEFC1",
                "name": None,
            },
            {
                "id": "lookalike",
                "accessor": "read lookalike",
                "discovery_source": "edas",
                "data_source_path": "PSSCfst/refcurEF1",
                "name": None,
            },
        ]
    )
    efc, lookalike = prompt.split("### Signal")[1:]
    assert "glossary: EFC — " in efc
    assert "glossary: EFC — " not in lookalike


def test_facility_glossary_is_declared_and_loaded():
    with open("imas_codex/schemas/facility_config.yaml") as stream:
        schema = yaml.safe_load(stream)
    assert "category_descriptions" in schema["classes"]["EDASConfig"]["attributes"]
    assert "signal_glossary" in schema["classes"]["FacilityConfig"]["attributes"]

    config = get_facility("jt-60sa")
    edas = config["data_systems"]["edas"]
    descriptions = {
        entry["code"]: entry["description"] for entry in edas["category_descriptions"]
    }
    glossary = {entry["term"]: entry["meaning"] for entry in config["signal_glossary"]}
    assert descriptions["MMSYS"]
    assert "upper fast plasma position control coil (FPPCC)" in glossary["UFP"]
    assert "lower fast plasma position control coil (FPPCC)" in glossary["LFP"]
    assert "same coil current" in glossary["HiTe"]
    assert "HiTec" in glossary["HiTe"]
    assert "LKAT2" in glossary["LKAT"]
    assert config["signal_member_patterns"]["MMSYS"] == (
        r"^cur(?P<member>.+?)(?:TFLKAT|LKAT|HiTe)$"
    )


def _normalized_statement(cypher: str) -> str:
    """A statement with its whitespace collapsed, so a shape check reads plainly."""
    return " ".join(cypher.split())


def _issued_statements(call) -> list[str]:
    """Run ``call`` against a recording graph and return the statements it issues."""
    recorded: list[str] = []

    class RecordingGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def query(self, cypher, **params):
            recorded.append(_normalized_statement(cypher))
            return [
                {
                    "cnt": 1,
                    "updated": 1,
                    "representative_id": "jt-60sa:general/cs1_temsnur11",
                    "enrichment": {},
                }
            ]

    with patch.object(parallel, "GraphClient", RecordingGraph):
        call()
    return recorded


def test_propagation_binds_the_source_by_representative_id_alone():
    """A group is located by its representative_id, never through a member edge.

    Binding the source through the representative's own ``MEMBER_OF`` edge reaches
    the wrong group whenever the representative is itself a member of another
    source, and skips the group it represents altogether. Every statement the
    propagation issues carries the bound-by-id match instead.
    """
    statements = _issued_statements(
        lambda: parallel.propagate_source_enrichment(
            "jt-60sa:general/cs1_temsnur11",
            {"description": "Switching network unit (SNU)", "diagnostic": "magnetics"},
        )
    )

    assert len(statements) == 4
    for statement in statements:
        assert "MATCH (sg:SignalSource {representative_id: $rep_id})" in statement
        assert "(sg:SignalSource)" not in statement
        assert "-[:MEMBER_OF]->(sg:SignalSource)" not in statement


def test_the_sweep_reads_each_signal_source_by_its_representative_id():
    """The sweep's own lookup names the representative id and no member edge."""
    with patch.object(parallel, "propagate_source_enrichment", return_value=0):
        statements = _issued_statements(
            lambda: parallel.sweep_source_enrichment("jt-60sa")
        )

    assert len(statements) == 1
    statement = statements[0]
    assert "MATCH (rep:FacilitySignal {id: sg.representative_id})" in statement
    assert "sg.representative_id})-[:MEMBER_OF]" not in statement
    assert "-[:MEMBER_OF]->(sg:SignalSource)" not in statement
