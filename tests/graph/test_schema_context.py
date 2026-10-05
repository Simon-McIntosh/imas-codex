"""Tests for auto-generated schema context and schema_for() function.

Tests the build-time generation (gen_schema_context.py) and runtime
schema_for() function (schema_context.py) that provides task-specific
schema slices to agents.
"""

from pathlib import Path

import pytest
import yaml

# =============================================================================
# Task Groups YAML validation
# =============================================================================


class TestTaskGroupsYAML:
    """Validate task_groups.yaml against LinkML schemas."""

    @pytest.fixture(scope="class")
    def task_groups(self):
        yaml_path = (
            Path(__file__).parent.parent.parent
            / "imas_codex"
            / "schemas"
            / "task_groups.yaml"
        )
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def all_schema_labels(self):
        """Get all node labels from both schemas."""
        from imas_codex.graph.schema import GraphSchema

        schemas_dir = Path(__file__).parent.parent.parent / "imas_codex" / "schemas"
        facility = GraphSchema(schemas_dir / "facility.yaml")
        dd = GraphSchema(schemas_dir / "imas_dd.yaml")
        return set(facility.node_labels + dd.node_labels)

    def test_task_groups_file_exists(self):
        path = (
            Path(__file__).parent.parent.parent
            / "imas_codex"
            / "schemas"
            / "task_groups.yaml"
        )
        assert path.exists()

    def test_task_groups_has_required_groups(self, task_groups):
        expected = {"signals", "wiki", "imas", "code", "facility", "data_sources"}
        assert set(task_groups.keys()) == expected

    def test_each_group_has_labels_and_description(self, task_groups):
        for name, group in task_groups.items():
            assert "labels" in group, f"Group '{name}' missing 'labels'"
            assert "description" in group, f"Group '{name}' missing 'description'"
            assert isinstance(group["labels"], list)
            assert len(group["labels"]) > 0

    def test_all_labels_exist_in_schemas(self, task_groups, all_schema_labels):
        """Every label referenced in task groups must exist in LinkML schemas."""
        for group_name, group in task_groups.items():
            for label in group["labels"]:
                assert label in all_schema_labels, (
                    f"Label '{label}' in task group '{group_name}' "
                    f"not found in LinkML schemas"
                )


# =============================================================================
# Schema context generation (gen_schema_context.py)
# =============================================================================


class TestGenSchemaContext:
    """Test the build-time schema context generator."""

    def test_generate_schema_context_produces_valid_python(self, tmp_path):
        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        assert output.exists()
        content = output.read_text()

        # Must be valid Python
        compile(content, str(output), "exec")

    def test_generated_module_has_required_symbols(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert hasattr(mod, "NODE_LABEL_PROPS")
        assert hasattr(mod, "ENUM_VALUES")
        assert hasattr(mod, "RELATIONSHIPS")
        assert hasattr(mod, "VECTOR_INDEXES")
        assert hasattr(mod, "TASK_GROUPS")

    def test_node_label_props_contains_facility(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert "Facility" in mod.NODE_LABEL_PROPS
        assert "FacilitySignal" in mod.NODE_LABEL_PROPS
        # DD labels too
        assert "IMASNode" in mod.NODE_LABEL_PROPS

    def test_relationships_are_tuples(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert len(mod.RELATIONSHIPS) > 0
        for rel in mod.RELATIONSHIPS:
            assert len(rel) == 4  # (from, type, to, cardinality)

    def test_vector_indexes_match_schemas(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert len(mod.VECTOR_INDEXES) > 0
        # Check some known indexes
        assert "wiki_chunk_embedding" in mod.VECTOR_INDEXES
        assert "imas_node_embedding" in mod.VECTOR_INDEXES

    def test_task_groups_loaded(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert "signals" in mod.TASK_GROUPS
        assert "FacilitySignal" in mod.TASK_GROUPS["signals"]

    def test_enum_values_present(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert len(mod.ENUM_VALUES) > 0
        # Check a known enum
        assert "PathStatus" in mod.ENUM_VALUES or "SourceFileStatus" in mod.ENUM_VALUES


# =============================================================================
# Vector index filter properties
# =============================================================================


class TestVectorIndexFilters:
    """Filter properties registered on vector indexes for in-index filtering."""

    @pytest.fixture(scope="class")
    def schemas_dir(self):
        return Path(__file__).parent.parent.parent / "imas_codex" / "schemas"

    def test_schema_exposes_code_chunk_facility_filter(self, schemas_dir):
        from imas_codex.graph.schema import GraphSchema

        schema = GraphSchema(schemas_dir / "facility.yaml")
        filters = schema.vector_index_filters
        assert filters.get("code_chunk_embedding") == ["facility_id"]

    def test_vector_indexes_shape_unchanged(self, schemas_dir):
        """vector_indexes stays a list of 3-tuples."""
        from imas_codex.graph.schema import GraphSchema

        schema = GraphSchema(schemas_dir / "facility.yaml")
        for entry in schema.vector_indexes:
            assert len(entry) == 3
            index_name, label, prop = entry
            assert all(isinstance(x, str) for x in (index_name, label, prop))

    def test_generator_emits_vector_index_filters(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert hasattr(mod, "VECTOR_INDEX_FILTERS")
        assert mod.VECTOR_INDEX_FILTERS["code_chunk_embedding"] == ["facility_id"]

    def test_generated_vector_indexes_shape_unchanged(self, tmp_path):
        """VECTOR_INDEXES stays a mapping to 2-tuples."""
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        for index_name, value in mod.VECTOR_INDEXES.items():
            assert len(value) == 2, index_name

    def test_ddl_registers_filter_property(self):
        from imas_codex.graph.client import _vector_index_ddl

        ddl = _vector_index_ddl(
            "code_chunk_embedding", "CodeChunk", "embedding", 1024, ["facility_id"]
        )
        # The additional-properties WITH clause is Cypher 25 grammar.
        assert ddl.startswith("CYPHER 25 CREATE VECTOR INDEX")
        assert "FOR (n:CodeChunk) ON n.embedding" in ddl
        assert "WITH [n.facility_id]" in ddl
        # WITH must sit between ON and OPTIONS
        assert ddl.index("ON n.embedding") < ddl.index("WITH [n.facility_id]")
        assert ddl.index("WITH [n.facility_id]") < ddl.index("OPTIONS")

    def test_ddl_without_filters_has_no_with_clause(self):
        from imas_codex.graph.client import _vector_index_ddl

        ddl = _vector_index_ddl("imas_node_embedding", "IMASNode", "embedding", 1024)
        assert "WITH [" not in ddl

    def test_ddl_dimensions_and_similarity(self):
        from imas_codex.graph.client import _vector_index_ddl

        ddl = _vector_index_ddl(
            "wiki_chunk_embedding", "WikiChunk", "embedding", 768, ["facility_id"]
        )
        assert "`vector.dimensions`: 768" in ddl
        assert "`vector.similarity_function`: 'cosine'" in ddl


class _FakeNeo4jSession:
    """Records Cypher run against a fake session, modelling live index state."""

    def __init__(self, vector_rows, existing_names=()):
        self.vector_rows = vector_rows
        self.existing = set(existing_names)
        self.statements: list[str] = []
        self.drops: list[str] = []

    def run(self, cypher, **params):
        self.statements.append(cypher)
        if cypher.startswith("SHOW INDEXES YIELD name, type, options, properties"):
            return iter(self.vector_rows)
        if cypher.startswith("SHOW INDEXES YIELD name WHERE name IN"):
            wanted = params.get("names", [])
            return iter([{"name": n} for n in wanted if n in self.existing])
        if cypher.startswith("DROP INDEX"):
            name = cypher.split("`")[1]
            self.drops.append(name)
            self.existing.discard(name)
            return iter([])
        return iter([])


class TestEnsureVectorIndexes:
    """ensure_vector_indexes reconciles live index shape against the schema."""

    def _run(self, client_mod, fake, filters="default"):
        """Run ensure_vector_indexes against the fake session.

        ``filters`` overrides the module's expected filter map.  The sentinel
        ``"default"`` leaves it as imported; passing ``None`` models a checkout
        whose generated module predates the filter surface.
        """
        from contextlib import contextmanager

        @contextmanager
        def fake_session(self):
            yield fake

        original_session = client_mod.GraphClient.session
        original_filters = client_mod.EXPECTED_VECTOR_INDEX_FILTERS
        client_mod.GraphClient.session = fake_session
        if filters != "default":
            client_mod.EXPECTED_VECTOR_INDEX_FILTERS = filters
        try:
            client = object.__new__(client_mod.GraphClient)
            client.ensure_vector_indexes()
        finally:
            client_mod.GraphClient.session = original_session
            client_mod.EXPECTED_VECTOR_INDEX_FILTERS = original_filters
        return fake

    def test_recreates_index_missing_filter_property(self):
        from imas_codex.graph import client as client_mod

        dim = client_mod.get_embedding_dimension()
        fake = _FakeNeo4jSession(
            vector_rows=[
                {"name": "code_chunk_embedding", "dim": dim, "props": ["embedding"]}
            ],
            existing_names=["code_chunk_embedding"],
        )
        self._run(client_mod, fake)

        assert fake.drops == ["code_chunk_embedding"]
        created = [s for s in fake.statements if "CREATE VECTOR INDEX" in s]
        assert any(
            "code_chunk_embedding" in s and "WITH [n.facility_id]" in s for s in created
        )

    def test_keeps_index_with_matching_shape(self):
        from imas_codex.graph import client as client_mod

        dim = client_mod.get_embedding_dimension()
        fake = _FakeNeo4jSession(
            vector_rows=[
                {
                    "name": "code_chunk_embedding",
                    "dim": dim,
                    "props": ["embedding", "facility_id"],
                }
            ],
            existing_names=["code_chunk_embedding"],
        )
        self._run(client_mod, fake)

        assert fake.drops == []
        assert not any(
            "CREATE VECTOR INDEX" in s and "code_chunk_embedding" in s
            for s in fake.statements
        )

    def test_recreates_index_on_dimension_mismatch(self):
        from imas_codex.graph import client as client_mod

        dim = client_mod.get_embedding_dimension()
        fake = _FakeNeo4jSession(
            vector_rows=[
                {
                    "name": "code_chunk_embedding",
                    "dim": dim + 1,
                    "props": ["embedding", "facility_id"],
                }
            ],
            existing_names=["code_chunk_embedding"],
        )
        self._run(client_mod, fake)

        assert fake.drops == ["code_chunk_embedding"]
        assert any(
            "CREATE VECTOR INDEX" in s and "code_chunk_embedding" in s
            for s in fake.statements
        )

    def test_leaves_indexes_the_schema_does_not_own(self):
        from imas_codex.graph import client as client_mod

        fake = _FakeNeo4jSession(
            vector_rows=[
                {"name": "peer_owned_index", "dim": 7, "props": ["embedding"]}
            ],
        )
        self._run(client_mod, fake)

        assert fake.drops == []

    def test_unknown_filter_map_leaves_extra_property_index_alone(self):
        """An unreadable filter map must not justify dropping a live index.

        A checkout whose generated module predates the filter surface cannot
        say which properties an index should carry, so an index that carries
        more than the vector property is not evidence of a mismatch.
        """
        from imas_codex.graph import client as client_mod

        dim = client_mod.get_embedding_dimension()
        fake = _FakeNeo4jSession(
            vector_rows=[
                {
                    "name": "code_chunk_embedding",
                    "dim": dim,
                    "props": ["embedding", "facility_id"],
                }
            ],
            existing_names=["code_chunk_embedding"],
        )
        self._run(client_mod, fake, filters=None)

        assert fake.drops == []
        assert not any(
            "CREATE VECTOR INDEX" in s and "code_chunk_embedding" in s
            for s in fake.statements
        )

    def test_unknown_filter_map_still_drops_on_dimension_mismatch(self):
        from imas_codex.graph import client as client_mod

        dim = client_mod.get_embedding_dimension()
        fake = _FakeNeo4jSession(
            vector_rows=[
                {
                    "name": "code_chunk_embedding",
                    "dim": dim + 1,
                    "props": ["embedding", "facility_id"],
                }
            ],
            existing_names=["code_chunk_embedding"],
        )
        self._run(client_mod, fake, filters=None)

        assert fake.drops == ["code_chunk_embedding"]

    def test_empty_filter_map_is_a_known_absence(self):
        """An empty map is a statement, unlike an unreadable one.

        With the symbol present and empty the schema says no index carries a
        filter property, so a live index that carries one is a real mismatch.
        """
        from imas_codex.graph import client as client_mod

        dim = client_mod.get_embedding_dimension()
        fake = _FakeNeo4jSession(
            vector_rows=[
                {
                    "name": "code_chunk_embedding",
                    "dim": dim,
                    "props": ["embedding", "facility_id"],
                }
            ],
            existing_names=["code_chunk_embedding"],
        )
        self._run(client_mod, fake, filters={})

        assert fake.drops == ["code_chunk_embedding"]


# =============================================================================
# Runtime schema_for() function
# =============================================================================


class TestSchemaFor:
    """Test the runtime schema_for() function."""

    def test_schema_for_overview_returns_string(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="overview")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_schema_for_signals_task(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="signals")
        assert isinstance(result, str)
        assert "FacilitySignal" in result
        assert "DataAccess" in result
        # Should NOT include unrelated labels as section headers
        assert "## WikiPage" not in result
        assert "## IMASNode" not in result

    def test_schema_for_wiki_task(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="wiki")
        assert "WikiPage" in result
        assert "WikiChunk" in result

    def test_schema_for_imas_task(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="imas")
        assert "IMASNode" in result
        assert "DDVersion" in result

    def test_schema_for_specific_labels(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for("Facility", "DataSource")
        assert "Facility" in result
        assert "DataSource" in result
        # Should not contain unrelated labels as section headers
        assert "## WikiChunk" not in result

    def test_schema_for_overview_is_compact(self):
        """Overview should contain all labels but be compact (no full property lists)."""
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="overview")
        assert "Facility" in result
        assert "IMASNode" in result

    def test_schema_for_includes_vector_indexes(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="signals")
        # Should include relevant vector indexes
        assert "embedding" in result.lower() or "vector" in result.lower()

    def test_schema_for_includes_relationships(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="signals")
        # Should include relationships for the labels
        assert "DATA_ACCESS" in result or "BELONGS_TO_DIAGNOSTIC" in result

    def test_schema_for_includes_enums(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="facility")
        assert "discovered" in result or "PathStatus" in result

    def test_schema_for_unknown_task_raises(self):
        from imas_codex.graph.schema_context import schema_for

        with pytest.raises(ValueError, match="Unknown task"):
            schema_for(task="nonexistent")

    def test_schema_for_unknown_label_raises(self):
        from imas_codex.graph.schema_context import schema_for

        with pytest.raises(ValueError, match="Unknown label"):
            schema_for("NonexistentLabel")

    def test_schema_for_token_efficiency(self):
        """Task-specific schema should be smaller than full-detail schema."""
        from imas_codex.graph.schema_context import schema_for

        # Compare signals slice against the full-detail output for ALL labels
        all_labels = schema_for(task="signals")
        # A single-label slice should be significantly smaller
        single = schema_for("FacilitySignal")
        assert len(single) < len(all_labels)
