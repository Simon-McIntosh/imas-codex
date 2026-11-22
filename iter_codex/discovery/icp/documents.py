"""Fetch any well-formed document identity through the existing guarded session."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

from iter_codex.discovery.base.transfer import TransferClient
from iter_codex.discovery.icp.client import ICPClient, ICPRequestRefused, RequestRecord


@dataclass(frozen=True)
class DocumentReceipt:
    """Private metadata for a fetch; failed responses have no stored file."""

    uid: str = field(repr=False)
    version: str
    outcome: str
    status: int | None
    content_type: str | None
    size_bytes: int | None = None
    sha256: str | None = None
    path: str | None = field(default=None, repr=False)
    signature: str | None = None
    page_count: int | None = None
    sheets: tuple[tuple[str, int], ...] = field(default=(), repr=False)
    extracted_text_path: str | None = field(default=None, repr=False)


def _outcome(status: int | None) -> str:
    if status == 404:
        return "not-found"
    if status in {401, 403}:
        return "not-permitted"
    if status is not None and 300 <= status < 400:
        return "redirect-or-sign-in"
    return "error"


class ICPDocuments:
    """Reuse one client's sign-in, transport, identity guard and request pacer."""

    def __init__(self, client: ICPClient) -> None:
        self.client = client

    def _last_request(self, before: int, method: str) -> RequestRecord:
        records = self.client.request_log[before:]
        if len(records) != 1 or records[0].method != method:
            raise ICPRequestRefused(
                "The document transfer did not send one guarded read"
            )
        return records[0]

    async def fetch(
        self,
        uid: str,
        version: str,
        storage_dir: Path,
        *,
        expected_type: str,
        max_size_bytes: int = 50 * 1024 * 1024,
    ) -> DocumentReceipt:
        """Probe size, validate the entire file, then store it outside the source tree.

        Redirects are returned as explicit outcomes. A document denial is never
        retried through sign-in, and a refused probe cannot trigger a download.
        Spreadsheet text uses the shared extraction, including its truncation
        markers; sheet row counts include every row in the workbook dimensions.
        """
        if expected_type not in {"pdf", "xlsx"} or max_size_bytes < 1:
            raise ValueError("Use a supported document type and positive size limit")
        storage_dir = storage_dir.resolve()
        if storage_dir.is_relative_to(Path(__file__).resolve().parents[3]):
            raise ValueError("Document storage must be outside the source tree")
        url = self.client.document_url(uid, version)
        self.client.authenticate()
        transfer = TransferClient(
            session=self.client.session, verify_ssl=True, allow_redirects=False
        )
        before = len(self.client.request_log)
        size = await transfer.get_size(url, timeout=self.client.timeout)
        probe = self._last_request(before, "HEAD")
        if probe.status is None or probe.status in {401, 403} or probe.status >= 500:
            return DocumentReceipt(
                uid, version, _outcome(probe.status), probe.status, probe.content_type
            )
        if size is not None and size > max_size_bytes:
            return DocumentReceipt(
                uid, version, "too-large", probe.status, probe.content_type, size
            )
        before = len(self.client.request_log)
        result = await transfer.download(
            url, timeout=self.client.timeout, expected_type=expected_type
        )
        read = self._last_request(before, "GET")
        if not result.success:
            outcome = _outcome(read.status) if read.status != 200 else "invalid-content"
            return DocumentReceipt(
                uid, version, outcome, read.status, read.content_type, result.size_bytes
            )
        content = result.content
        assert content is not None
        if len(content) > max_size_bytes:
            return DocumentReceipt(
                uid, version, "too-large", read.status, read.content_type, len(content)
            )
        content_type = (read.content_type or "").split(";", 1)[0].lower().strip()
        if content_type in {
            "text/html",
            "application/json",
            "text/plain",
            "text/xml",
            "application/xml",
        }:
            return DocumentReceipt(
                uid,
                version,
                "invalid-content",
                read.status,
                read.content_type,
                len(content),
            )
        page_count = None
        sheets = ()
        extracted_text = None
        try:
            if expected_type == "pdf":
                from pypdf import PdfReader

                if not content.startswith(b"%PDF-"):
                    raise ValueError("The PDF signature is missing")
                reader = PdfReader(io.BytesIO(content), strict=True)
                page_count = len(reader.pages)
                if page_count < 1:
                    raise ValueError("The PDF has no pages")
                signature = "%PDF-"
            else:
                from openpyxl import load_workbook

                from iter_codex.discovery.wiki.excel import extract_excel_full

                workbook = load_workbook(
                    io.BytesIO(content), read_only=True, data_only=True
                )
                try:
                    sheets = tuple(
                        (sheet.title, sheet.max_row or 0) for sheet in workbook
                    )
                    if not sheets:
                        raise ValueError("The workbook has no sheets")
                finally:
                    workbook.close()
                extracted_text = extract_excel_full(content)
                signature = "PK\\x03\\x04"
        except Exception:
            return DocumentReceipt(
                uid,
                version,
                "invalid-content",
                read.status,
                read.content_type,
                len(content),
            )
        storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = storage_dir / f"{uid}-{version}.{expected_type}"
        with path.open("xb") as handle:
            path.chmod(0o600)
            handle.write(content)
        text_path = None
        if extracted_text is not None:
            text_path = path.with_suffix(".txt")
            with text_path.open("x") as handle:
                text_path.chmod(0o600)
                handle.write(extracted_text)
        return DocumentReceipt(
            uid,
            version,
            "fetched",
            read.status,
            read.content_type,
            len(content),
            hashlib.sha256(content).hexdigest(),
            str(path),
            signature,
            page_count,
            sheets,
            str(text_path) if text_path is not None else None,
        )
