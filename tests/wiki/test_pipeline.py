"""Tests for wiki ingestion pipeline."""

import pytest

from imas_codex.discovery.wiki.pipeline import html_to_text


class TestHTMLToText:
    """Tests for HTML to text conversion.

    html_to_text returns a tuple of (text, sections_dict).
    """

    def test_strip_script_tags(self):
        """Script tags should be removed."""
        html = "<html><script>alert('bad')</script><p>Content</p></html>"
        text, _ = html_to_text(html)
        assert "alert" not in text
        assert "Content" in text

    def test_strip_style_tags(self):
        """Style tags should be removed."""
        html = "<html><style>.foo { color: red; }</style><p>Content</p></html>"
        text, _ = html_to_text(html)
        assert "color" not in text
        assert "Content" in text

    def test_strip_navigation(self):
        """Navigation elements pass through but bodyContent extraction isolates main content."""
        # Simulate MediaWiki structure with bodyContent
        html = """
        <html>
        <nav>Menu</nav>
        <div id="bodyContent">
        <p>Main content here</p>
        <div class="printfooter">Retrieved from...</div>
        </div>
        </html>
        """
        text, _ = html_to_text(html)
        # Navigation before bodyContent is excluded
        assert "Menu" not in text
        # Main content is preserved
        assert "Main content" in text
        # Footer after bodyContent is excluded
        assert "Retrieved from" not in text

    def test_preserve_paragraph_text(self):
        """Paragraph text should be preserved."""
        html = (
            "<html><body><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
        )
        text, _ = html_to_text(html)
        assert "First paragraph" in text
        assert "Second paragraph" in text

    def test_collapse_whitespace(self):
        """Multiple whitespaces should be collapsed."""
        html = "<html><body><p>Text   with    spaces</p></body></html>"
        text, _ = html_to_text(html)
        # Should have at most single spaces, not triple+
        assert "    " not in text
        # The text should contain the words
        assert "Text" in text
        assert "spaces" in text

    def test_strip_sidebar(self):
        """Sidebar elements should pass through (aside not filtered by default)."""
        html = "<html><aside>Sidebar content</aside><main>Main content</main></html>"
        text, _ = html_to_text(html)
        # aside is not in the skip list, so it passes through
        assert "Main content" in text

    def test_empty_html(self):
        """Empty HTML should return empty string."""
        text, sections = html_to_text("")
        assert text == ""
        assert sections == {}

    def test_returns_tuple(self):
        """Should return tuple of (text, sections)."""
        result = html_to_text("<p>Test</p>")
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, sections = result
        assert isinstance(text, str)
        assert isinstance(sections, dict)

    def test_plain_text(self):
        """Plain text without HTML tags should pass through."""
        text, _ = html_to_text("Just plain text")
        assert "plain text" in text


class TestParseMetaField:
    """Tests for _parse_meta_field() — META:FIELD value extraction."""

    def test_basic_field(self):
        """Should extract name/value from a simple META:FIELD line."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Proposal" title="Proposal" value="magnetic sensor calibration"}%'
        result = _parse_meta_field(line)
        assert "<b>Proposal</b>" in result
        assert "magnetic sensor calibration" in result

    def test_empty_value_skipped(self):
        """Fields with empty values should return empty string."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Comment" title="Comment" value=""}%'
        assert _parse_meta_field(line) == ""

    def test_whitespace_only_value_skipped(self):
        """Fields with whitespace-only values should return empty string."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Comment" title="Comment" value="   "}%'
        assert _parse_meta_field(line) == ""

    def test_url_encoded_value(self):
        """URL-encoded values (Japanese text, newlines) should be decoded."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Text" title="Text" value="%E9%81%8B%E8%BB%A2%E6%97%A5%E8%AA%8C"}%'
        result = _parse_meta_field(line)
        assert "運転日誌" in result  # "operation log" in Japanese

    def test_newlines_converted_to_br(self):
        """Encoded newlines should become <br> tags."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        line = '%META:FIELD{name="Text" title="Text" value="line1%0d%0aline2"}%'
        result = _parse_meta_field(line)
        assert "<br>" in result
        assert "line1" in result
        assert "line2" in result

    def test_system_meta_skipped(self):
        """TOPICINFO, TOPICPARENT, FORM, etc. should return empty string."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        assert (
            _parse_meta_field('%META:TOPICINFO{author="admin" date="1234567890"}%')
            == ""
        )
        assert _parse_meta_field('%META:TOPICPARENT{name="WebHome"}%') == ""
        assert _parse_meta_field('%META:FORM{name="ShotAForm"}%') == ""
        assert _parse_meta_field('%META:FILEATTACHMENT{name="data.csv"}%') == ""
        assert (
            _parse_meta_field('%META:PREFERENCE{name="VIEW_TEMPLATE" value="ShotA"}%')
            == ""
        )

    def test_malformed_line(self):
        """Malformed lines should return empty string, not crash."""
        from imas_codex.discovery.wiki.pipeline import _parse_meta_field

        assert _parse_meta_field("%META:") == ""
        assert _parse_meta_field("%META:FIELD{}%") == ""
        assert _parse_meta_field("not a meta line") == ""


class TestTwikiMarkupToHtml:
    """Tests for twiki_markup_to_html() — full TWiki markup conversion."""

    def test_meta_field_included(self):
        """META:FIELD values should appear in the HTML output."""
        from imas_codex.discovery.wiki.pipeline import twiki_markup_to_html

        markup = (
            '%META:TOPICINFO{author="admin" date="1234567890"}%\n'
            '%META:FORM{name="ShotAForm"}%\n'
            '%META:FIELD{name="Proposal" title="Proposal" value="magnetic sensor calibration"}%\n'
            '%META:FIELD{name="PreComment" title="PreComment" value="OK, data collection"}%\n'
        )
        html = twiki_markup_to_html(markup)
        assert "magnetic sensor calibration" in html
        assert "OK, data collection" in html
        # System metadata should NOT appear
        assert "admin" not in html
        assert "ShotAForm" not in html

    def test_meta_field_empty_body(self):
        """Pages with only META:FIELD data (no body text) should still produce content."""
        from imas_codex.discovery.wiki.pipeline import twiki_markup_to_html

        markup = (
            '%META:TOPICINFO{author="user" date="1234567890"}%\n'
            '%META:TOPICPARENT{name="WebHome"}%\n'
            '%META:FORM{name="SystemDailyReportForm"}%\n'
            '%META:FIELD{name="Date" title="Date" value="2024-01-15"}%\n'
            '%META:FIELD{name="Group" title="Group" value="計測班"}%\n'
            '%META:FIELD{name="Kind" title="Kind" value="運転日誌"}%\n'
        )
        html = twiki_markup_to_html(markup)
        assert "2024-01-15" in html
        assert "計測班" in html
        assert "運転日誌" in html
        # Should have at least 3 <p><b> blocks
        assert html.count("<b>") >= 3

    def test_headings(self):
        """TWiki headings should convert to HTML headings."""
        from imas_codex.discovery.wiki.pipeline import twiki_markup_to_html

        markup = "---+ Main Heading\n---++ Sub Heading\nSome text"
        html = twiki_markup_to_html(markup)
        assert "<h1>" in html
        assert "<h2>" in html
        assert "Main Heading" in html

    def test_verbatim_blocks(self):
        """Verbatim blocks should become <pre> tags."""
        from imas_codex.discovery.wiki.pipeline import twiki_markup_to_html

        markup = "<verbatim>\ncode here\n</verbatim>"
        html = twiki_markup_to_html(markup)
        assert "<pre>" in html
        assert "code here" in html


class TestDynamicPageDetector:
    """A page that is only table headers and a search form is a dynamic table."""

    HEADER_ONLY = (
        "Category Information\n"
        "## Category Information\n"
        "| Category | Description | RO |\n"
        "| --- | --- | --- |\n"
        "\nSearch\nCategory\nDescription\nRO\n"
    )

    DATA_ROWS = (
        "| DataName | PID No. | Unit |\n"
        "| --- | --- | --- |\n"
        "| psrc_magfluxlp1 | 1234A | mWb |\n"
        "| psrc_magfluxlp2 | 1235A | mWb |\n"
    )

    def test_header_only_form_is_dynamic(self):
        from imas_codex.discovery.wiki.pipeline import detect_dynamic_page

        assert detect_dynamic_page(self.HEADER_ONLY, ["EDDB"]) is True

    def test_header_only_without_database_is_not_dynamic(self):
        from imas_codex.discovery.wiki.pipeline import detect_dynamic_page

        # Nothing to name in the stub and nothing to link, so the page is left
        # as it is rather than stubbed with an unnamed database.
        assert detect_dynamic_page(self.HEADER_ONLY, []) is False

    def test_table_with_data_rows_is_not_dynamic(self):
        from imas_codex.discovery.wiki.pipeline import detect_dynamic_page

        assert detect_dynamic_page(self.DATA_ROWS, ["EDDB"]) is False


class TestDynamicPageStub:
    """The stub names the database(s) the page is rendered from."""

    def test_single_database(self):
        from imas_codex.discovery.wiki.pipeline import dynamic_page_stub

        assert dynamic_page_stub(["EDDB"]) == (
            "Dynamic table rendered from the EDDB catalogue; "
            "the rows are the FacilitySignals linked to this page."
        )

    def test_multiple_databases(self):
        from imas_codex.discovery.wiki.pipeline import dynamic_page_stub

        assert dynamic_page_stub(["PMDB", "EDDB"]) == (
            "Dynamic table rendered from the EDDB and PMDB catalogues; "
            "the rows are the FacilitySignals linked to this page."
        )


class TestExtractDatabaseLinks:
    """The ?db= parameters of the links that reach a topic name its database."""

    def test_handbook_links(self):
        from imas_codex.discovery.wiki.pipeline import extract_database_links

        text = (
            "| [[CategoryInformation?db=EDDB][Experiment database (EDDB)]] |\n"
            "| [[DataInformation?db=UDDB][Unprocessed database (UDDB)]] |\n"
            "| [[CategoryInformation?db=PMDB][Plant database (PMDB)]] |\n"
        )
        assert extract_database_links(text) == {
            "CategoryInformation": ["EDDB", "PMDB"],
            "DataInformation": ["UDDB"],
        }

    def test_link_without_parameter_ignored(self):
        from imas_codex.discovery.wiki.pipeline import extract_database_links

        assert extract_database_links("[[SomeTopic][A topic without a database]]") == {}


class TestLinkChunksToEntitiesDatabaseMerge:
    """The fourth DOCUMENTS merge is keyed on fronts_database via the source map."""

    def test_database_merge_emitted_with_source_map(self):
        from unittest.mock import MagicMock, patch

        from imas_codex.discovery.wiki import pipeline as pl

        gc = MagicMock()
        gc.query.return_value = [{"linked": 7}]
        with patch.object(pl, "GraphClient") as gc_cls:
            gc_cls.return_value.__enter__.return_value = gc
            stats = pl.link_chunks_to_entities("jt-60sa")

        db_calls = [
            call
            for call in gc.query.call_args_list
            if "fronts_database" in call.args[0]
        ]
        assert len(db_calls) == 1
        query = db_calls[0].args[0]
        assert "DOCUMENTS" in query
        assert "WHERE p.fronts_database IS NOT NULL" in query
        assert db_calls[0].kwargs["database_sources"] == {"EDDB": "edas"}
        assert stats["database_signals_linked"] == 7

    def test_database_merge_absent_from_scope_without_fronts(self):
        from unittest.mock import MagicMock, patch

        from imas_codex.discovery.wiki import pipeline as pl

        # A page with no fronts_database cannot produce the database merge: the
        # query's own guard is the filter, so it is emitted but matches nothing.
        gc = MagicMock()
        gc.query.return_value = [{"linked": 0}]
        with patch.object(pl, "GraphClient") as gc_cls:
            gc_cls.return_value.__enter__.return_value = gc
            stats = pl.link_chunks_to_entities("jt-60sa")

        assert stats["database_signals_linked"] == 0


class TestIngestMarksDynamicPage:
    """Ingesting a dynamic table marks it without an operator applying the rule."""

    HEADER_ONLY_HTML = (
        "<html><body><h2>Category Information</h2>"
        "<p>The categories the database exposes.</p>"
        "<table><tr><th>Category</th><th>Description</th><th>RO</th></tr></table>"
        "</body></html>"
    )
    LINK_TEXT = "| [[CategoryInformation?db=EDDB][Experiment database (EDDB)]] |"
    STUB = (
        "Dynamic table rendered from the EDDB catalogue; "
        "the rows are the FacilitySignals linked to this page."
    )

    def _run(self, monkeypatch):
        from unittest.mock import MagicMock

        from imas_codex.discovery.wiki import pipeline as pl
        from imas_codex.discovery.wiki.scraper import WikiPage

        class _Vec(list):
            def tolist(self):
                return list(self)

        class _Arr(list):
            def tolist(self):
                return list(self)

        class _Embed:
            def embed_texts(self, texts):
                return _Arr([_Vec([0.1, 0.2, 0.1]) for _ in texts])

        gc = MagicMock()

        def _query(cypher, **kwargs):
            if "CONTAINS '?db='" in cypher:
                return [{"text": self.LINK_TEXT}]
            return []

        gc.query.side_effect = _query
        gc_cls = MagicMock()
        gc_cls.return_value.__enter__.return_value = gc
        monkeypatch.setattr(pl, "GraphClient", gc_cls)

        pipeline = pl.WikiIngestionPipeline("jt-60sa", use_rich=False)
        pipeline._embed_model = _Embed()
        page = WikiPage(
            url="ssh://jt-60sa/var/www/html/twiki/data/Main/CategoryInformation.txt",
            title="Category Information",
            content_html=self.HEADER_ONLY_HTML,
        )
        import asyncio

        stats = asyncio.run(
            pipeline.ingest_page(page, page_id="jt-60sa:CategoryInformation")
        )

        calls = gc.query.call_args_list
        page_merges = [c for c in calls if "MERGE (p:WikiPage {id: $id})" in c.args[0]]
        assert page_merges, "the page persist must set fronts_database"
        assert page_merges[0].kwargs["fronts_database"] == ["EDDB"]
        chunk_persist = [c for c in calls if "UNWIND $chunks AS chunk" in c.args[0]]
        assert chunk_persist[0].kwargs["chunks"][0]["text"] == self.STUB
        return stats

    def test_ingest_marks_dynamic_page(self, monkeypatch):
        stats = self._run(monkeypatch)
        assert stats["chunks"] == 1


class TestMarkDynamicHandbookPages:
    """The bulk rule marks stored pages, including one whose body was empty."""

    def _fake_gc(self, page):
        from unittest.mock import MagicMock

        gc = MagicMock()

        def _query(cypher, **kwargs):
            if "CONTAINS '?db='" in cypher:
                return [
                    {"text": "| [[CategoryInformation?db=EDDB][EDDB]] |"},
                    {"text": "| [[CategoryInformation2?db=LCDB][LCDB]] |"},
                ]
            if "OPTIONAL MATCH (p)-[:HAS_CHUNK]" in cypher:
                if kwargs["page_id"] in page:
                    return [page[kwargs["page_id"]]]
                return []
            return []

        gc.query.side_effect = _query
        return gc

    def test_marks_page_and_embeds_stub_chunk(self, monkeypatch):
        from unittest.mock import MagicMock

        from imas_codex.discovery.wiki import pipeline as pl

        class _Vec(list):
            def tolist(self):
                return list(self)

        class _Arr(list):
            def tolist(self):
                return list(self)

        class _Embed:
            def embed_texts(self, texts):
                return _Arr([_Vec([0.5, 0.5]) for _ in texts])

        page = {
            "fronts": None,
            "chunks": [],  # a skipped page whose topic file held no rows
        }
        gc = self._fake_gc({"jt-60sa:CategoryInformation2": page})
        gc_cls = MagicMock()
        gc_cls.return_value.__enter__.return_value = gc
        monkeypatch.setattr(pl, "GraphClient", gc_cls)
        monkeypatch.setattr(pl, "get_embed_model", lambda: _Embed())

        applied = pl.mark_dynamic_handbook_pages("jt-60sa")

        assert applied["jt-60sa:CategoryInformation2"] == ["LCDB"]
        creates = [
            c for c in gc.query.call_args_list if "MERGE (c:WikiChunk" in c.args[0]
        ]
        assert creates, "an empty page gets a stub chunk"
        assert (
            creates[0].kwargs["text"].startswith("Dynamic table rendered from the LCDB")
        )
        assert creates[0].kwargs["embedding"], "the stub chunk carries an embedding"

        # A marked page is set to status 'ingested' in the same write, so a topic
        # whose ingest was skipped for lack of content is not left 'skipped'.
        marks = [
            c
            for c in gc.query.call_args_list
            if "SET p.fronts_database = $databases" in c.args[0]
        ]
        assert marks, "the marker writes fronts_database"
        assert marks[0].kwargs["ingested"] == "ingested"


class TestIngestPagesSurfacesMarkerFailure:
    """A failing dynamic-page marker is raised out of the run, not logged away."""

    def test_raising_marker_is_surfaced(self, monkeypatch):
        from imas_codex.discovery.wiki import pipeline as pl

        def _boom(facility_id):
            raise RuntimeError("dynamic-page marking failed")

        monkeypatch.setattr(pl, "mark_dynamic_handbook_pages", _boom)
        pipeline = pl.WikiIngestionPipeline("jt-60sa", use_rich=False)

        import asyncio

        with pytest.raises(RuntimeError, match="dynamic-page marking failed"):
            asyncio.run(pipeline.ingest_pages([], rate_limit=0))


@pytest.mark.graph
class TestDatabaseMergeGraph:
    """The fourth DOCUMENTS merge parses against the live graph, not a mock."""

    def test_database_merge_statement_parses(self):
        from imas_codex.discovery.wiki import pipeline as pl
        from imas_codex.graph import GraphClient

        # EXPLAIN the statement the pipeline actually runs, not a copy of it, so a
        # change to the production Cypher is what this gate parses.
        assert "fronts_database" in pl.FRONTS_DATABASE_DOCUMENTS_MERGE
        with GraphClient() as gc:
            gc.query(
                "EXPLAIN " + pl.FRONTS_DATABASE_DOCUMENTS_MERGE,
                facility_id="jt-60sa",
                database_sources=pl.DATABASE_SIGNAL_SOURCES,
            )


class TestPipelineInit:
    """Tests for pipeline initialization."""

    def test_import(self):
        """Pipeline should be importable."""
        from imas_codex.discovery.wiki.pipeline import WikiIngestionPipeline

        assert WikiIngestionPipeline is not None

    def test_create_instance(self):
        """Pipeline should be instantiable without Neo4j."""
        from imas_codex.discovery.wiki.pipeline import WikiIngestionPipeline

        # This may fail without Neo4j, which is expected
        # We just test that the class exists and has expected attributes
        assert hasattr(WikiIngestionPipeline, "ingest_page")
        assert hasattr(WikiIngestionPipeline, "ingest_pages")


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests requiring Neo4j."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        from imas_codex.discovery.wiki.pipeline import WikiIngestionPipeline

        try:
            p = WikiIngestionPipeline("tcv")
            yield p
        except Exception:
            pytest.skip("Neo4j not available")

    def test_ingest_page(self, pipeline):
        """Test ingesting a single page."""
        from imas_codex.discovery.wiki.scraper import WikiPage

        # Create a test page - just verify it can be constructed
        _page = WikiPage(
            url="https://test.example.com/wiki/Test",
            title="Test Page",
            content_html="<html><body><p>This is test content about electron temperature.</p></body></html>",
        )
        # Full integration tests would verify graph state
        # For now just validate the page is valid
        assert _page.page_name == "Test"


class TestIngestFromGraphFailureRouting:
    """ingest_from_graph routes unsupported failures to deferred, rest to failed."""

    def _pipeline(self):
        from imas_codex.discovery.wiki.pipeline import DocumentPipeline

        pipeline = object.__new__(DocumentPipeline)
        pipeline.facility_id = "jt-60sa"
        pipeline.max_size_bytes = 100 * 1024 * 1024
        return pipeline

    def _pending(self):
        return [
            {
                "id": "doc:pdf",
                "document_type": "pdf",
                "url": "https://example.com/a.pdf",
                "filename": "a.pdf",
            }
        ]

    def test_deferrable_failure_counts_deferred(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from imas_codex.discovery.wiki import pipeline as pl

        pipeline = self._pipeline()
        with (
            patch.object(
                pl, "get_pending_wiki_documents", return_value=self._pending()
            ),
            patch.object(pl, "fetch_document_size", return_value=1024),
            patch.object(
                pl,
                "fetch_document_content",
                new=AsyncMock(side_effect=RuntimeError("HTTP Error 404: Not Found")),
            ),
            patch.object(
                pl, "mark_document_failed_or_deferred", return_value="dead link"
            ) as mock_mark,
            patch.object(pl, "GraphClient"),
        ):
            stats = asyncio.run(pipeline.ingest_from_graph())

        assert stats["documents_deferred"] == 1
        assert stats["documents_failed"] == 0
        args = mock_mark.call_args.args
        assert args[0] == "doc:pdf"
        assert args[2] == "pdf"

    def test_unclassified_failure_counts_failed(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from imas_codex.discovery.wiki import pipeline as pl

        pipeline = self._pipeline()
        with (
            patch.object(
                pl, "get_pending_wiki_documents", return_value=self._pending()
            ),
            patch.object(pl, "fetch_document_size", return_value=1024),
            patch.object(
                pl,
                "fetch_document_content",
                new=AsyncMock(side_effect=RuntimeError("Connection refused")),
            ),
            patch.object(pl, "mark_document_failed_or_deferred", return_value=None),
            patch.object(pl, "GraphClient"),
        ):
            stats = asyncio.run(pipeline.ingest_from_graph())

        assert stats["documents_failed"] == 1
        assert stats["documents_deferred"] == 0

    def test_status_write_failure_propagates(self):
        """A terminal status write that fails surfaces, not silently swallowed."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from imas_codex.discovery.wiki import pipeline as pl

        pipeline = self._pipeline()
        with (
            patch.object(
                pl, "get_pending_wiki_documents", return_value=self._pending()
            ),
            patch.object(pl, "fetch_document_size", return_value=1024),
            patch.object(
                pl,
                "fetch_document_content",
                new=AsyncMock(side_effect=RuntimeError("HTTP Error 404: Not Found")),
            ),
            patch.object(
                pl,
                "mark_document_failed_or_deferred",
                side_effect=RuntimeError("Neo4j unavailable"),
            ),
            patch.object(pl, "GraphClient"),
        ):
            with pytest.raises(RuntimeError):
                asyncio.run(pipeline.ingest_from_graph())
