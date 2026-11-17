"""Document identity, read-only transport and file-content boundaries."""

import io
from unittest.mock import Mock

import pytest
import requests

from imas_codex.discovery.icp import ICPClient, ICPRequestRefused, client as module

RECORDS = "https://records.example.test"
DOCUMENTS = "https://documents.example.test"
IDENTITY = "https://identity.example.test"
GATEWAY = "https://gateway.example.test"
DOCUMENT_URL = DOCUMENTS + "/?uid=SAMPLE&version=v1.2&action=get_document"


@pytest.fixture
def document_client(monkeypatch):
    config = {
        "origin": RECORDS,
        "description_path": "/openapi.json",
        "readable_origins": [RECORDS, DOCUMENTS],
        "document_origin": DOCUMENTS,
        "document_path": "/",
        "document_identities": ["SAMPLE"],
        "credential_service": "synthetic",
        "identity_origins": [IDENTITY],
        "sign_in_origins": [IDENTITY, GATEWAY, RECORDS, DOCUMENTS],
        "gateway_sign_in_paths": ["/gateway/sign-in"],
    }
    monkeypatch.delenv("IMAS_CODEX_ICP_ORIGIN", raising=False)
    monkeypatch.delenv("IMAS_CODEX_ICP_DESCRIPTION_PATH", raising=False)
    monkeypatch.setattr(
        module, "get_facility_infrastructure", lambda _: {"icp": config}
    )
    manager = Mock()
    manager.get_stored_credentials.return_value = ("reader@example.test", "secret")
    monkeypatch.setattr(module, "CredentialManager", lambda: manager)
    sent = []

    def send(adapter, request, **kwargs):
        sent.append(request)
        response = requests.Response()
        response.status_code = 200
        response.url = request.url
        response.request = request
        response.headers["Content-Type"] = "application/pdf"
        response._content = b"%PDF-synthetic"
        response.raw = io.BytesIO(response._content)
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with ICPClient(facility="synthetic", min_interval=0) as client:
        client._authenticated = True
        yield client, sent, config


@pytest.mark.parametrize("entry", ["client", "session", "prepared", "base", "adapter"])
def test_unlisted_document_identity_sends_zero_requests(document_client, entry):
    client, sent, _ = document_client
    url = DOCUMENT_URL.replace("SAMPLE", "UNLISTED")
    request = requests.Request("GET", url).prepare()
    with pytest.raises(ICPRequestRefused, match="identity"):
        if entry == "client":
            client.get(url)
        elif entry == "session":
            client.session.get(url)
        elif entry == "prepared":
            client.session.send(request)
        elif entry == "base":
            requests.Session.send(client.session, request)
        else:
            client.session.get_adapter(url).send(request)
    assert sent == []


def test_allowlisted_document_is_readable(document_client):
    client, sent, _ = document_client
    assert client.get(DOCUMENT_URL).status_code == 200
    assert len(sent) == 1


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize(
    "entry",
    [
        "client",
        "session",
        "convenience",
        "prepared",
        "base-request",
        "base-send",
        "adapter",
    ],
)
def test_document_write_refused_before_transport(document_client, method, entry):
    client, sent, _ = document_client
    request = requests.Request(method, DOCUMENT_URL).prepare()
    with pytest.raises(ICPRequestRefused, match="only GET and HEAD"):
        if entry == "client":
            client.request(method, DOCUMENT_URL)
        elif entry == "session":
            client.session.request(method, DOCUMENT_URL)
        elif entry == "convenience":
            getattr(client.session, method.lower())(DOCUMENT_URL)
        elif entry == "prepared":
            client.session.send(request)
        elif entry == "base-request":
            requests.Session.request(client.session, method, DOCUMENT_URL)
        elif entry == "base-send":
            requests.Session.send(client.session, request)
        else:
            client.session.get_adapter(DOCUMENT_URL).send(request)
    assert sent == []


@pytest.mark.parametrize("state", ["before", "exchange", "signed-in"])
@pytest.mark.parametrize("path", ["/api/items", "/docs/openapi.json", "/"])
def test_document_writes_stay_refused_during_sign_in(document_client, state, path):
    from contextlib import nullcontext

    client, sent, _ = document_client
    client._authenticated = state == "signed-in"
    context = client._sign_in_exchange() if state == "exchange" else nullcontext()
    with context, pytest.raises(ICPRequestRefused):
        client.session.post(DOCUMENTS + path, data={"SAMLResponse": "synthetic"})
    assert sent == []


@pytest.mark.parametrize("entry", ["client", "session", "prepared", "adapter"])
@pytest.mark.parametrize(
    "url",
    [
        "https://unlisted.example.test/",
        "http://documents.example.test/",
        "https://documents.example.test.evil.test/",
        "https://documents.example.test:444/",
        "https://reader@documents.example.test/",
        "//unrelated.example.test/",
    ],
)
def test_document_other_host_refused(document_client, entry, url):
    client, sent, _ = document_client
    url += "?uid=SAMPLE&version=v1.2&action=get_document"
    with pytest.raises((ICPRequestRefused, requests.exceptions.RequestException)):
        if entry == "client":
            client.get(url)
        elif entry == "session":
            client.session.get(url)
        elif entry == "prepared":
            client.session.send(requests.Request("GET", url).prepare())
        else:
            client.session.get_adapter(DOCUMENTS).send(
                requests.Request("GET", url).prepare()
            )
    assert sent == []


@pytest.mark.parametrize(
    "suffix",
    [
        "/",
        "/?uid=SAMPLE",
        "/other?uid=SAMPLE&version=v1.2&action=get_document",
        "/?uid=SAMPLE&uid=UNLISTED&version=v1.2&action=get_document",
        "/?uid=UNLISTED&uid=SAMPLE&version=v1.2&action=get_document",
        "/?uid=SAMPLE&version=v1.2&action=delete_document",
        "/?uid=SAMPLE&version=v1.2&action=get_document&extra=yes",
        "/?uid=SAMPLE&version=latest&action=get_document",
    ],
)
def test_document_form_is_exact(document_client, suffix):
    client, sent, _ = document_client
    with pytest.raises(ICPRequestRefused):
        client.get(DOCUMENTS + suffix)
    assert sent == []


def test_document_params_checked_before_authentication(document_client):
    client, sent, _ = document_client
    client._authenticated = False
    with pytest.raises(ICPRequestRefused, match="identity"):
        client.get(
            DOCUMENTS,
            params={"uid": "UNLISTED", "version": "v1.2", "action": "get_document"},
        )
    assert sent == []


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_document_denial_stops_without_sign_in_retry(
    document_client, monkeypatch, status
):
    client, sent, _ = document_client

    def denied(adapter, request, **kwargs):
        sent.append(request)
        result = requests.Response()
        result.status_code = status
        result.url = request.url
        result._content = b"error"
        return result

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", denied)
    with pytest.raises(requests.HTTPError):
        client.get(DOCUMENT_URL)
    with pytest.raises(ICPRequestRefused, match="stopped"):
        client.session.get(DOCUMENT_URL)
    assert len(sent) == 1
    assert client._policy.sign_in_requests == 0


def test_document_request_budget_is_per_identity(document_client):
    client, sent, _ = document_client
    for _ in range(5):
        client.session.head(DOCUMENT_URL)
    with pytest.raises(ICPRequestRefused, match="budget"):
        client.session.get(DOCUMENT_URL)
    assert len(sent) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, content_type",
    [
        (b"<html><form>sign in</form></html>", "text/html"),
        (b'{"error":"not permitted"}', "application/json"),
        (b"%PDF-broken", "application/pdf"),
        (b"PK\x03\x04not-a-workbook", "application/octet-stream"),
    ],
)
@pytest.mark.parametrize("expected_type", ["pdf", "xlsx"])
async def test_sign_in_and_error_bodies_store_nothing(
    document_client, monkeypatch, tmp_path, body, content_type, expected_type
):
    from imas_codex.discovery.icp.documents import ICPDocuments

    client, sent, _ = document_client

    def send(adapter, request, **kwargs):
        assert kwargs.get("verify") is True
        sent.append(request)
        result = requests.Response()
        result.status_code = 200
        result.url = request.url
        result.headers.update(
            {"Content-Type": content_type, "Content-Length": str(len(body))}
        )
        result._content = body if request.method == "GET" else b""
        return result

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    receipt = await ICPDocuments(client).fetch(
        "SAMPLE", "v1.2", tmp_path / "private", expected_type=expected_type
    )
    assert receipt.outcome == "invalid-content"
    assert receipt.path is None
    assert receipt.sha256 is None
    assert not list(tmp_path.iterdir())
    assert [request.method for request in sent] == ["HEAD", "GET"]


@pytest.fixture
def file_transport(document_client, monkeypatch):
    client, sent, _ = document_client

    def install(body, content_type, *, head_status=200, get_status=200):
        def send(adapter, request, **kwargs):
            sent.append(request)
            result = requests.Response()
            result.status_code = head_status if request.method == "HEAD" else get_status
            result.url = request.url
            result.request = request
            result.headers.update(
                {"Content-Type": content_type, "Content-Length": str(len(body))}
            )
            result._content = body if request.method == "GET" else b""
            result.raw = io.BytesIO(result._content)
            return result

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)

    return client, sent, install


@pytest.mark.asyncio
async def test_pdf_stored_only_after_signature_and_page_validation(
    file_transport, tmp_path
):
    import hashlib

    from pypdf import PdfWriter

    from imas_codex.discovery.icp.documents import ICPDocuments

    client, sent, install = file_transport
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    writer.write(stream)
    body = stream.getvalue()
    install(body, "application/pdf")
    receipt = await ICPDocuments(client).fetch(
        "SAMPLE", "v1.2", tmp_path, expected_type="pdf"
    )
    assert receipt.outcome == "fetched"
    assert receipt.page_count == 1
    assert receipt.signature == "%PDF-"
    assert receipt.size_bytes == len(body)
    assert receipt.sha256 == hashlib.sha256(body).hexdigest()
    assert (tmp_path / "SAMPLE-v1.2.pdf").read_bytes() == body
    assert [request.method for request in sent] == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_workbook_uses_existing_extraction_and_counts_all_sheets(
    file_transport, monkeypatch, tmp_path
):
    from openpyxl import Workbook

    from imas_codex.discovery.icp.documents import ICPDocuments
    from imas_codex.discovery.wiki import excel

    client, _, install = file_transport
    workbook = Workbook()
    workbook.active.title = "Signals"
    workbook.active.append(["Name", "Description"])
    workbook.active.append(["synthetic", "A synthetic sensor"])
    numeric = workbook.create_sheet("Numeric")
    numeric.append([1, 2])
    workbook.create_sheet("Empty")
    stream = io.BytesIO()
    workbook.save(stream)
    body = stream.getvalue()
    extractor = Mock(wraps=excel.extract_excel_full)
    monkeypatch.setattr(excel, "extract_excel_full", extractor)
    install(body, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    receipt = await ICPDocuments(client).fetch(
        "SAMPLE", "v1.2", tmp_path, expected_type="xlsx"
    )
    assert receipt.outcome == "fetched"
    assert receipt.sheets == (("Signals", 2), ("Numeric", 1), ("Empty", 1))
    extractor.assert_called_once_with(body)
    assert (tmp_path / "SAMPLE-v1.2.txt").read_text() == excel.extract_excel_full(body)
    assert "A synthetic sensor" in (tmp_path / "SAMPLE-v1.2.txt").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 500, 503])
async def test_failed_size_probe_stores_nothing_and_sends_no_get(
    file_transport, tmp_path, status
):
    from imas_codex.discovery.icp.documents import ICPDocuments

    client, sent, install = file_transport
    install(b"error", "text/plain", head_status=status)
    receipt = await ICPDocuments(client).fetch(
        "SAMPLE", "v1.2", tmp_path, expected_type="pdf"
    )
    assert receipt.outcome in {"not-permitted", "error"}
    assert receipt.status == status
    assert receipt.content_type == "text/plain"
    assert receipt.path is None
    assert list(tmp_path.iterdir()) == []
    assert [request.method for request in sent] == ["HEAD"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, outcome",
    [
        (302, "redirect-or-sign-in"),
        (404, "not-found"),
        (403, "not-permitted"),
        (500, "error"),
    ],
)
async def test_failed_get_has_explicit_outcome_and_no_file(
    file_transport, tmp_path, status, outcome
):
    from imas_codex.discovery.icp.documents import ICPDocuments

    client, sent, install = file_transport
    install(b"<html>error</html>", "text/html", get_status=status)
    receipt = await ICPDocuments(client).fetch(
        "SAMPLE", "v99.9", tmp_path, expected_type="pdf"
    )
    assert receipt.outcome == outcome
    assert receipt.status == status
    assert receipt.content_type == "text/html"
    assert receipt.path is None
    assert list(tmp_path.iterdir()) == []
    assert [request.method for request in sent] == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_document_identity_refused_before_authentication_or_size_probe(
    document_client, tmp_path
):
    from imas_codex.discovery.icp.documents import ICPDocuments

    client, sent, _ = document_client
    client._authenticated = False
    with pytest.raises(ICPRequestRefused, match="identity"):
        await ICPDocuments(client).fetch(
            "UNLISTED", "v1.2", tmp_path, expected_type="pdf"
        )
    assert sent == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_size_probe_prevents_oversized_get(file_transport, tmp_path):
    from imas_codex.discovery.icp.documents import ICPDocuments

    client, sent, install = file_transport
    install(b"too large", "application/pdf")
    receipt = await ICPDocuments(client).fetch(
        "SAMPLE", "v1.2", tmp_path, expected_type="pdf", max_size_bytes=1
    )
    assert receipt.outcome == "too-large"
    assert list(tmp_path.iterdir()) == []
    assert [request.method for request in sent] == ["HEAD"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_origin", "https://unlisted.example.test"),
        ("document_origin", RECORDS),
        ("document_origin", DOCUMENTS + "/other"),
        ("document_path", "//other"),
        ("document_path", "/?action=other"),
        ("document_identities", []),
        ("document_identities", "SAMPLE"),
        ("document_identities", ["../other"]),
    ],
)
def test_document_private_config_fails_closed(document_client, field, value):
    from imas_codex.discovery.icp import ICPConfigurationError

    _, sent, config = document_client
    config[field] = value
    with pytest.raises(ICPConfigurationError, match="document"):
        ICPClient(facility="synthetic", min_interval=0)
    assert sent == []


def test_total_document_budget_applies_across_identities(document_client):
    client, sent, _ = document_client
    from imas_codex.discovery.icp.client import _DocumentAccess

    identities = frozenset(f"SAMPLE{i}" for i in range(9))
    client._policy.documents = _DocumentAccess(DOCUMENTS, "/", identities)
    for uid in sorted(identities)[:8]:
        for _ in range(5):
            client.session.head(client.document_url(uid, "v1.2"))
    with pytest.raises(ICPRequestRefused, match="budget"):
        client.session.head(client.document_url(sorted(identities)[8], "v1.2"))
    assert len(sent) == 40


def test_session_budget_includes_records_and_sign_in_reads(document_client):
    client, sent, _ = document_client
    for _ in range(39):
        client.session.get(RECORDS + "/items")
    client.session.head(DOCUMENT_URL)
    with pytest.raises(ICPRequestRefused, match="session request budget"):
        client.session.get(DOCUMENT_URL)
    assert len(sent) == 40


def test_records_only_session_can_read_records_beyond_document_budget(document_client):
    _, sent, _ = document_client
    with ICPClient(facility="synthetic", min_interval=0, records_only=True) as client:
        client._authenticated = True
        for _ in range(45):
            client.get("/items")
        assert len(sent) == 45
        assert client._policy.document_requests == {}


@pytest.mark.parametrize("entry", ["client", "session", "prepared", "adapter"])
def test_records_only_session_refuses_documents_before_transport(
    document_client, entry
):
    _, sent, _ = document_client
    with ICPClient(facility="synthetic", min_interval=0, records_only=True) as client:
        client._authenticated = True
        request = requests.Request("GET", DOCUMENT_URL).prepare()
        with pytest.raises(ICPRequestRefused, match="records-only"):
            if entry == "client":
                client.get(DOCUMENT_URL)
            elif entry == "session":
                client.session.get(DOCUMENT_URL)
            elif entry == "prepared":
                client.session.send(request)
            else:
                client.session.get_adapter(DOCUMENT_URL).send(request)
        assert sent == []


@pytest.mark.asyncio
async def test_probe_network_failure_stops_without_a_get(
    document_client, monkeypatch, tmp_path
):
    from imas_codex.discovery.icp.documents import ICPDocuments

    client, sent, _ = document_client

    def failed(adapter, request, **kwargs):
        sent.append(request)
        raise requests.Timeout("synthetic timeout")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", failed)
    receipt = await ICPDocuments(client).fetch(
        "SAMPLE", "v1.2", tmp_path, expected_type="pdf"
    )
    assert receipt.outcome == "error"
    assert receipt.status is None
    assert receipt.path is None
    assert [request.method for request in sent] == ["HEAD"]
    assert list(tmp_path.iterdir()) == []
