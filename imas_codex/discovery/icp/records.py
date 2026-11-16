"""Typed plant, function-category and component reads on the guarded client.

All collection reads use GET search pages. Tree branches are returned as one
server-defined page, whose loaded flag is checked before claiming completeness.
List-view and query POST operations are deliberately absent from this reader.
Record values and source identities are private data; callers choose their own
private storage rather than persisting them as repository fixtures.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from imas_codex.discovery.icp.client import ICPClient

_BASE = "/api/machine-component"


class ICPNotFound(LookupError):
    """No record matches the requested identity or exact code."""


class ICPPermissionDenied(PermissionError):
    """The account is not permitted to read a requested record."""


class ICPResponseError(ValueError):
    """A response cannot establish record identity or collection completeness."""


class ICPReadLimit(RuntimeError):
    """The caller's finite page or node budget was exhausted."""


@dataclass(frozen=True)
class RecordSource:
    """The configured instance, GET endpoint and native record identity."""

    origin: str
    endpoint: str
    uid: str
    content_type: str


@dataclass(frozen=True)
class ContentReference:
    uid: str
    reference: str
    content_type: str
    source: RecordSource


@dataclass(frozen=True)
class ReferencePage:
    items: tuple[ContentReference, ...]
    start: int
    have_more: bool


@dataclass(frozen=True)
class PlantNode:
    uid: str
    code: str
    title: str
    parent_uid: str | None
    parent_code: str | None
    source: RecordSource


@dataclass(frozen=True)
class FunctionCategory:
    uid: str
    code: str
    title: str
    component_type: str | None
    source: RecordSource


@dataclass(frozen=True)
class ComponentRecord:
    uid: str
    reference: str
    function_reference: str | None
    plant_uid: str | None
    function_category_uid: str | None
    source: RecordSource


@dataclass(frozen=True)
class PlantSubtree:
    """All advertised nodes were visited; denied reads are counted separately.

    The API can omit records invisible to the account. This receipt establishes
    completeness of the returned tree, not a count of silently hidden records.
    """

    nodes: tuple[PlantNode, ...]
    denied_uids: tuple[str, ...]
    branch_reads: int


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ICPResponseError("A required record identity or text field is missing")
    return result


def _reference_uid(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ICPResponseError("A content reference is not an object")
    if value.get("isNullOrEmpty") is True:
        return None
    uid = value.get("uid")
    if uid is None:
        return None
    return _text(value, "uid")


def _optional_text(value: dict[str, Any], key: str) -> str | None:
    result = value.get(key)
    if result is not None and not isinstance(result, str):
        raise ICPResponseError("An optional record text field is not a string")
    return result


class ICPRecords:
    """Read typed records using an existing ICPClient and its sole transport.

    A reader keeps plant contents in memory to resolve parent codes without
    parsing display labels. Construct another reader when a fresh view is
    needed; the client session and its pacing can still be reused.
    """

    def __init__(self, client: ICPClient) -> None:
        self.client = client
        self._plants: dict[str, dict[str, Any]] = {}

    def _get(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.client.get(endpoint, **kwargs)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                raise ICPNotFound("The requested record does not exist") from None
            if status == 403:
                raise ICPPermissionDenied(
                    "The account cannot read this record"
                ) from None
            raise
        try:
            value = response.json()
        except ValueError:
            raise ICPResponseError("The record response is not JSON") from None
        if not isinstance(value, dict):
            raise ICPResponseError("The record response is not an object")
        return value

    def _source(self, endpoint: str, uid: str, kind: str) -> RecordSource:
        return RecordSource(self.client.origin, endpoint, uid, kind)

    def _reference(self, value: Any, endpoint: str) -> ContentReference:
        if not isinstance(value, dict):
            raise ICPResponseError("A collection item is not a content reference")
        uid = _text(value, "uid")
        content_type = value.get("contentType")
        if not isinstance(content_type, dict):
            raise ICPResponseError("A content reference has no type")
        kind = _text(content_type, "name")
        return ContentReference(
            uid, _text(value, "referenceText"), kind, self._source(endpoint, uid, kind)
        )

    def _pages(
        self, domain: str, query: str, *, page_size: int, max_pages: int
    ) -> Iterator[ReferencePage]:
        if not 1 <= page_size <= 100 or max_pages < 1:
            raise ValueError("Use a page size of 1 to 100 and a positive page budget")
        endpoint = f"{_BASE}{domain}/type/search"
        start = 0
        seen: set[str] = set()
        for _ in range(max_pages):
            value = self._get(
                endpoint, params={"query": query, "start": start, "tot": page_size}
            )
            raw = value.get("items")
            more = value.get("haveMore")
            if not isinstance(raw, list) or type(more) is not bool:
                raise ICPResponseError(
                    "The search response has no explicit page outcome"
                )
            if len(raw) > page_size or (more and not raw):
                raise ICPResponseError(
                    "The search page exceeds its bound or makes no progress"
                )
            items = tuple(self._reference(item, endpoint) for item in raw)
            uids = [item.uid for item in items]
            if len(set(uids)) != len(uids) or seen.intersection(uids):
                raise ICPResponseError(
                    "The search repeated a record instead of advancing"
                )
            seen.update(uids)
            yield ReferencePage(items, start, more)
            if not more:
                return
            start += len(items)
        raise ICPReadLimit("The search page budget was exhausted")

    def search_plants(
        self, query: str, *, page_size: int = 50, max_pages: int = 20
    ) -> Iterator[ReferencePage]:
        return self._pages("/pbs", query, page_size=page_size, max_pages=max_pages)

    def search_components(
        self, query: str, *, page_size: int = 50, max_pages: int = 20
    ) -> Iterator[ReferencePage]:
        return self._pages("", query, page_size=page_size, max_pages=max_pages)

    def search_function_categories(
        self, query: str = "", *, page_size: int = 50, max_pages: int = 20
    ) -> Iterator[ReferencePage]:
        return self._pages("/ttt-code", query, page_size=page_size, max_pages=max_pages)

    def _plant_content(self, uid: str) -> dict[str, Any]:
        if uid not in self._plants:
            endpoint = f"{_BASE}/pbs/content/{quote(uid, safe='')}"
            value = self._get(endpoint)
            if _text(value, "uid") != uid:
                raise ICPResponseError(
                    "The returned plant identity differs from the request"
                )
            _text(value, "pbsId")
            _text(value, "title")
            self._plants[uid] = value
        return self._plants[uid]

    def plant_by_uid(self, uid: str) -> PlantNode:
        value = self._plant_content(uid)
        parent_uid = _reference_uid(value.get("parent"))
        if parent_uid == uid:
            raise ICPResponseError("A plant node cannot be its own parent")
        parent_code = (
            _text(self._plant_content(parent_uid), "pbsId") if parent_uid else None
        )
        endpoint = f"{_BASE}/pbs/content/{quote(uid, safe='')}"
        return PlantNode(
            uid,
            _text(value, "pbsId"),
            _text(value, "title"),
            parent_uid,
            parent_code,
            self._source(endpoint, uid, "Pbs"),
        )

    def plant_by_code(
        self, code: str, *, page_size: int = 50, max_pages: int = 20
    ) -> PlantNode:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("A nonempty exact plant code is required")
        match: PlantNode | None = None
        for page in self.search_plants(code, page_size=page_size, max_pages=max_pages):
            for item in page.items:
                value = self._plant_content(item.uid)
                if value["pbsId"] == code:
                    if match is not None and match.uid != item.uid:
                        raise ICPResponseError(
                            "An exact plant code has multiple identities"
                        )
                    match = self.plant_by_uid(item.uid)
        if match is None:
            raise ICPNotFound("No plant node has the requested exact code")
        return match

    def _children(self, uid: str) -> tuple[tuple[ContentReference, bool], ...]:
        endpoint = f"{_BASE}/pbs/tree/{quote(uid, safe='')}"
        value = self._get(endpoint)
        target = value.get("navigationTarget") or {}
        root_ref = value.get("content") or target.get("content")
        if self._reference(root_ref, endpoint).uid != uid:
            raise ICPResponseError("The tree branch belongs to a different plant")
        raw = value.get("children")
        has_children = value.get("hasChildren")
        if value.get("childrenLoaded") is not True or type(has_children) is not bool:
            raise ICPResponseError("The tree branch was not completely loaded")
        if raw is None and has_children is False:
            raw = []
        if not isinstance(raw, list) or bool(raw) != has_children:
            raise ICPResponseError("The tree branch has inconsistent child metadata")
        children = []
        for child in raw:
            if (
                not isinstance(child, dict)
                or type(child.get("hasChildren")) is not bool
            ):
                raise ICPResponseError("The tree child has no explicit branch outcome")
            target = child.get("navigationTarget") or {}
            ref = child.get("content") or target.get("content")
            reference = self._reference(ref, endpoint)
            if reference.content_type != "Pbs":
                raise ICPResponseError("The plant tree returned another content type")
            children.append((reference, child["hasChildren"]))
        return tuple(children)

    def children(self, uid: str) -> tuple[PlantNode, ...]:
        """Read one complete branch and the typed contents of its children."""
        return tuple(self.plant_by_uid(ref.uid) for ref, _ in self._children(uid))

    def subtree(self, uid: str, *, max_nodes: int = 200) -> PlantSubtree:
        """Visit every advertised node up to the caller's explicit node budget."""
        if max_nodes < 1:
            raise ValueError("A positive node budget is required")
        queue = deque([(uid, True)])
        seen: set[str] = set()
        nodes = []
        denied = set()
        branch_reads = 0
        while queue:
            current, expand = queue.popleft()
            if current in seen:
                raise ICPResponseError("The tree repeats a node or contains a cycle")
            if len(seen) >= max_nodes:
                raise ICPReadLimit("The subtree node budget was exhausted")
            seen.add(current)
            try:
                nodes.append(self.plant_by_uid(current))
                if expand:
                    branch_reads += 1
                    for ref, has_children in self._children(current):
                        queue.append((ref.uid, has_children))
            except ICPPermissionDenied:
                denied.add(current)
        return PlantSubtree(tuple(nodes), tuple(sorted(denied)), branch_reads)

    def function_category(self, uid: str) -> FunctionCategory:
        endpoint = f"{_BASE}/ttt-code/content/{quote(uid, safe='')}"
        value = self._get(endpoint)
        if _text(value, "uid") != uid:
            raise ICPResponseError(
                "The function-category identity differs from the request"
            )
        return FunctionCategory(
            uid,
            _text(value, "tttCodeId"),
            _text(value, "title"),
            _optional_text(value, "componentType"),
            self._source(endpoint, uid, "TttCode"),
        )

    def component_by_reference(self, reference: str) -> ComponentRecord:
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("A nonempty component reference is required")
        endpoint = f"{_BASE}/content/by-reference/{quote(reference, safe='')}"
        value = self._get(endpoint)
        uid = _text(value, "uid")
        if _text(value, "referenceText") != reference:
            raise ICPResponseError(
                "The returned component reference differs from the request"
            )
        return ComponentRecord(
            uid,
            reference,
            _optional_text(value, "functionReference"),
            _reference_uid(value.get("pbs")),
            _reference_uid(value.get("tttCode")),
            self._source(endpoint, uid, "MachineComponent"),
        )

    def components_for_plant(
        self, uid: str, *, page_size: int = 50, max_pages: int = 20
    ) -> Iterator[ComponentRecord]:
        """Filter a complete bounded GET search by the component's actual PBS link.

        Search text alone is not a PBS filter. Exhausting the page budget raises
        explicitly, so a partial list cannot silently stand in for completeness.
        """
        for page in self.search_components(
            "", page_size=page_size, max_pages=max_pages
        ):
            for ref in page.items:
                record = self.component_by_reference(ref.reference)
                if record.plant_uid == uid:
                    yield record
