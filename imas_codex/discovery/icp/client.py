"""Read ICP through a guarded session and the existing gateway sign-in exchange.

Select an application origin from the private readable_origins allowlist with
IMAS_CODEX_ICP_ORIGIN or the constructor's origin argument. Neither selector
grants access to an unlisted host. Set IMAS_CODEX_ICP_DESCRIPTION_PATH to its
description path at runtime. Select the facility with
IMAS_CODEX_ICP_FACILITY or the constructor's facility argument. Its ignored
private configuration holds the icp credential service, identity origins,
sign-in origins and exact gateway sign-in paths. API paths resolve from the
origin's root.
Credentials come only from the system keyring. Cookies live in this session;
they are not flattened into a cross-host persistent cookie dictionary.
"""

from __future__ import annotations

import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit

import requests

from imas_codex.discovery.base.facility import get_facility_infrastructure
from imas_codex.discovery.base.request_pacer import RequestPacer
from imas_codex.discovery.wiki.auth import CredentialManager, CredentialStoreUnavailable
from imas_codex.discovery.wiki.confluence import ConfluenceClient

_READ_METHODS = frozenset({"GET", "HEAD"})


class ICPConfigurationError(ValueError):
    """The private runtime configuration does not name an HTTPS origin."""


class ICPRequestRefused(ValueError):
    """The request violates the method, origin, or sign-in boundary."""


class ICPAuthenticationError(RuntimeError):
    """Stored credentials or a usable authenticated session are unavailable."""


def _origin(url: str) -> str:
    try:
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.port not in (None, 443)
        ):
            raise ValueError
    except ValueError:
        raise ICPRequestRefused(
            "Only the configured HTTPS origins are allowed"
        ) from None
    return f"https://{parts.hostname}"


def _description(response: requests.Response) -> dict[str, Any] | None:
    if response.status_code != 200:
        return None
    try:
        value = response.json()
    except ValueError:
        return None
    if (
        isinstance(value, dict)
        and isinstance(value.get("paths"), dict)
        and value["paths"]
        and (value.get("openapi") or value.get("swagger"))
    ):
        return value
    return None


@dataclass(frozen=True)
class RequestRecord:
    """A private receipt with origin, never query or authentication data."""

    method: str
    target: str
    status: int | None
    phase: str
    origin: str | None = field(default=None, repr=False)
    path: str | None = None
    started_at: float | None = None
    content_type: str | None = None


@dataclass(frozen=True)
class _DocumentAccess:
    origin: str
    path: str
    identities: frozenset[str] = field(repr=False)

    def check(self, request: requests.PreparedRequest) -> str:
        parts = urlsplit(request.url)
        try:
            pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True)
        except ValueError:
            raise ICPRequestRefused("The document query is invalid") from None
        query = dict(pairs)
        if (
            parts.path != self.path
            or parts.fragment
            or len(pairs) != len(query)
            or query.keys() != {"uid", "version", "action"}
            or query["action"] != "get_document"
            or not re.fullmatch(r"v[0-9]+(?:\.[0-9]+)*", query["version"])
        ):
            raise ICPRequestRefused("Only the configured document GET form is allowed")
        if query["uid"] not in self.identities:
            raise ICPRequestRefused("The document identity is not privately listed")
        return query["uid"]


class _RequestPolicy:
    def __init__(
        self,
        readable_origins: frozenset[str],
        interval: float,
        identity_origins: frozenset[str],
        sign_in_origins: frozenset[str],
        gateway_paths: frozenset[str],
        description_path: str,
        documents: _DocumentAccess | None = None,
        records_only: bool = False,
    ) -> None:
        self.readable_origins = readable_origins
        self.pacer = RequestPacer(interval)
        self.identity_origins = identity_origins
        self.sign_in_origins = sign_in_origins
        self.gateway_paths = gateway_paths
        self.description_path = description_path
        self.documents = documents
        self.records_only = records_only
        self.document_requests: dict[str, int] = {}
        self.document_stopped = False
        self.authenticating = False
        self.signed_in = False
        self.records: list[RequestRecord] = []
        self.sign_in_requests = 0
        self.max_sign_in_requests = 40

    def check(self, request: requests.PreparedRequest) -> str:
        """Refuse a method or origin before any transport can send it."""
        origin = _origin(request.url or "")
        if self.records_only and self.documents and origin == self.documents.origin:
            raise ICPRequestRefused(
                "Document access is disabled for records-only reads"
            )
        host_header = request.headers.get("Host")
        if host_header and host_header.lower() not in {
            urlsplit(origin).netloc,
            urlsplit(origin).netloc + ":443",
        }:
            raise ICPRequestRefused("A different Host header is not allowed")
        method = request.method
        if origin in self.readable_origins:
            if method not in _READ_METHODS:
                path = unquote(urlsplit(request.url).path)
                if (
                    method != "POST"
                    or path not in self.gateway_paths
                    or path == self.description_path
                    or path.lower().split("/")[1] in {"api", "docs", "swagger"}
                ):
                    raise ICPRequestRefused(
                        "ICP accepts only GET and HEAD outside sign-in paths"
                    )
                self._check_sign_in(request, origin)
            elif self.documents and origin == self.documents.origin:
                self.documents.check(request)
            return "application"
        self._check_sign_in(request, origin)
        return (
            "identity-provider"
            if origin in self.identity_origins
            else "sign-in-gateway"
        )

    def _check_sign_in(self, request: requests.PreparedRequest, origin: str) -> None:
        if not self.authenticating or self.signed_in:
            raise ICPRequestRefused(
                "Sign-in requests require an active unsigned exchange"
            )
        if origin not in self.sign_in_origins:
            raise ICPRequestRefused("The request origin is not allowed")
        if request.method in _READ_METHODS:
            return
        if request.method != "POST":
            raise ICPRequestRefused("Only sign-in form POSTs are allowed")
        self._check_sign_in_body(request, origin in self.identity_origins)

    @staticmethod
    def _check_sign_in_body(request: requests.PreparedRequest, identity: bool) -> None:
        """Allow the shared login forms and standard assertion relay bodies only."""
        import json

        body = request.body
        if isinstance(body, bytes):
            try:
                body = body.decode("utf-8")
            except UnicodeError:
                raise ICPRequestRefused(
                    "The sign-in body is not a UTF-8 form"
                ) from None
        if not isinstance(body, str) or len(body) > 2_000_000:
            raise ICPRequestRefused("The sign-in body is not a bounded form")
        content_type = request.headers.get("Content-Type", "").split(";", 1)[0]
        if content_type == "application/json" and identity:
            try:
                data = json.loads(body)
            except ValueError:
                raise ICPRequestRefused("The identity lookup body is invalid") from None
            allowed = {
                "username",
                "isOtherIdpSupported",
                "checkPhones",
                "isRemoteNGCSupported",
                "isCookieBannerShown",
                "isFidoSupported",
                "forceotclogin",
                "isExternalFederationDisallowed",
                "isRemoteConnectSupported",
                "federationFlags",
                "isSignup",
                "flowToken",
            }
            if (
                urlsplit(request.url).path.endswith("/GetCredentialType")
                and isinstance(data, dict)
                and {"username", "flowToken"} <= data.keys() <= allowed
            ):
                return
            raise ICPRequestRefused("Only the identity lookup JSON is allowed")
        if content_type != "application/x-www-form-urlencoded":
            raise ICPRequestRefused("Only encoded sign-in forms are allowed")
        try:
            pairs = parse_qsl(body, keep_blank_values=True, strict_parsing=True)
        except ValueError:
            raise ICPRequestRefused("The sign-in form is invalid") from None
        fields = dict(pairs)
        keys = fields.keys()
        if not keys or len(fields) != len(pairs):
            raise ICPRequestRefused("The sign-in form has missing or duplicate fields")
        if not identity:
            if "Authorization" in request.headers:
                raise ICPRequestRefused(
                    "Credentials may go only to the identity provider"
                )
            relay_forms = (
                ({"SAMLResponse"}, {"SAMLResponse", "RelayState"}),
                ({"id_token", "state"}, {"id_token", "code", "state", "session_state"}),
                ({"code", "state"}, {"id_token", "code", "state", "session_state"}),
                ({"wresult"}, {"wa", "wresult", "wctx"}),
            )
            if any(required <= keys <= allowed for required, allowed in relay_forms):
                return
            raise ICPRequestRefused("Only assertion relay forms may reach a gateway")
        allowed = {
            "i13",
            "login",
            "loginfmt",
            "type",
            "LoginOptions",
            "lrt",
            "lrtPartition",
            "hisRegion",
            "hisScaleUnit",
            "passwd",
            "ps",
            "psRNGCDefaultType",
            "psRNGCEntropy",
            "psRNGCSLK",
            "canary",
            "ctx",
            "hpgrequestid",
            "flowToken",
            "PPSX",
            "NewUser",
            "FoundMSAs",
            "fspost",
            "i21",
            "CookieDisclosure",
            "IsFidoSupported",
            "isSignupPost",
            "i19",
        }
        if not {"ctx", "flowToken"} <= keys <= allowed:
            raise ICPRequestRefused("Only identity-provider login forms are allowed")

    def before_send(self, request: requests.PreparedRequest) -> None:
        if self.documents and not self.records_only and len(self.records) >= 40:
            raise ICPRequestRefused("The document session request budget was exhausted")
        if (
            self.documents
            and _origin(request.url) == self.documents.origin
            and request.method in _READ_METHODS
        ):
            uid = self.documents.check(request)
            if self.document_stopped:
                raise ICPRequestRefused(
                    "Document reads stopped after a refused response"
                )
            if (
                self.document_requests.get(uid, 0) >= 5
                or sum(self.document_requests.values()) >= 40
            ):
                raise ICPRequestRefused("The document request budget was exhausted")
            self.document_requests[uid] = self.document_requests.get(uid, 0) + 1
        if self.authenticating:
            self.sign_in_requests += 1
            if self.sign_in_requests > self.max_sign_in_requests:
                raise ICPRequestRefused("The sign-in request budget was exhausted")

    def throttle(self) -> None:
        self.pacer.wait()


class _GuardedAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, policy: _RequestPolicy) -> None:
        self._policy = policy
        super().__init__(max_retries=0)

    def send(
        self, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        target = self._policy.check(request)
        if "verify" in kwargs and not kwargs["verify"]:
            raise ICPRequestRefused("TLS verification is required")
        self._policy.before_send(request)
        self._policy.throttle()
        started_at = time.monotonic()
        signing_in = self._policy.authenticating
        response = None
        try:
            response = super().send(request, **kwargs)
        finally:
            if (
                self._policy.documents
                and _origin(request.url) == self._policy.documents.origin
                and response is not None
                and (response.status_code in {401, 403} or response.status_code >= 500)
            ):
                self._policy.document_stopped = True
            self._policy.records.append(
                RequestRecord(
                    request.method or "",
                    target,
                    response.status_code if response is not None else None,
                    "sign-in" if signing_in else "read",
                    _origin(request.url),
                    urlsplit(request.url).path if signing_in else None,
                    started_at,
                    response.headers.get("Content-Type")
                    if response is not None
                    else None,
                )
            )
        return response


class _GuardedSession(requests.Session):
    def __init__(self, policy: _RequestPolicy) -> None:
        self._policy = policy
        self._sealed = False
        super().__init__()
        self.adapters.clear()
        self.mount("https://", _GuardedAdapter(policy))
        self._sealed = True
        self.trust_env = False
        self.max_redirects = 20

    def mount(self, prefix: str, adapter: requests.adapters.BaseAdapter) -> None:
        if self._sealed:
            raise ICPRequestRefused("The guarded transport cannot be replaced")
        super().mount(prefix, adapter)

    def send(
        self, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        # Validate prepared requests as well as adapter calls and redirects.
        self._policy.check(request)
        kwargs.setdefault("timeout", 30)
        return super().send(request, **kwargs)


class _APILogin(ConfluenceClient):
    def __init__(
        self,
        origin: str,
        description_url: str,
        session: _GuardedSession,
        credentials: CredentialManager,
        credential_service: str,
        timeout: float,
    ) -> None:
        # Keep the shared exchange on the guarded transport throughout sign-in.
        self.base_url = origin
        self.timeout = timeout
        self.credential_service = credential_service
        self._session = session
        self._creds = credentials
        self._authenticated = False
        self.description_url = description_url

    def _verify_sso_session(self, session: requests.Session) -> bool:
        response = session.get(
            self.description_url, timeout=self.timeout, allow_redirects=False
        )
        return _description(response) is not None

    def _handle_mfa(self, session: requests.Session, config: dict) -> None:
        raise ICPAuthenticationError("Interactive authentication is not available")


class ICPClient:
    """Read privately allowlisted application hosts, renewing stale sessions.

    Runtime selectors choose an origin without widening the private allowlist.
    Only the sign-in exchange may contact the identity provider.
    The exposed session uses the same policy for convenience methods, prepared
    requests, redirects and direct adapter calls.
    """

    def __init__(
        self,
        *,
        facility: str | None = None,
        origin: str | None = None,
        timeout: float = 30,
        min_interval: float = 0.5,
        records_only: bool = False,
    ) -> None:
        facility = facility or os.environ.get("IMAS_CODEX_ICP_FACILITY")
        if not facility:
            raise ICPConfigurationError("Configure the ICP facility")
        config = get_facility_infrastructure(facility).get("icp", {})
        configured = origin or os.environ.get(
            "IMAS_CODEX_ICP_ORIGIN", config.get("origin", "")
        )
        path = os.environ.get(
            "IMAS_CODEX_ICP_DESCRIPTION_PATH", config.get("description_path", "")
        )
        try:
            allowed = config.get("readable_origins", [config.get("origin", "")])
            if not isinstance(allowed, list) or not allowed:
                raise ValueError
            for value in [configured, *allowed]:
                parts = urlsplit(value)
                if parts.path not in ("", "/") or parts.query or parts.fragment:
                    raise ValueError
                _origin(value)
            readable_origins = frozenset(_origin(value) for value in allowed)
            origin = _origin(configured)
            if origin not in readable_origins:
                raise ValueError
        except (ICPRequestRefused, ValueError, TypeError, AttributeError):
            raise ICPConfigurationError(
                "Select an HTTPS origin from the private readable origins"
            ) from None
        if not path.startswith("/") or path.startswith("//"):
            raise ICPConfigurationError("Configure an absolute description path")
        documents = None
        if any(key.startswith("document_") for key in config):
            try:
                document_origin = _origin(config["document_origin"])
                parts = urlsplit(config["document_origin"])
                document_path = config["document_path"]
                identities = config["document_identities"]
                if (
                    parts.path not in ("", "/")
                    or parts.query
                    or parts.fragment
                    or document_origin not in readable_origins
                    or document_origin == origin
                    or not isinstance(document_path, str)
                    or not document_path.startswith("/")
                    or document_path.startswith("//")
                    or urlsplit(document_path).path != document_path
                    or not isinstance(identities, list)
                    or not identities
                    or any(
                        not isinstance(uid, str)
                        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", uid)
                        for uid in identities
                    )
                ):
                    raise ValueError
                documents = _DocumentAccess(
                    document_origin, document_path, frozenset(identities)
                )
            except (KeyError, ValueError, TypeError, AttributeError):
                raise ICPConfigurationError(
                    "Configure a separate listed document origin, path and identities"
                ) from None
        self._origin = origin
        self._description_url = urljoin(origin + "/", path)
        if _origin(self._description_url) != origin:
            raise ICPConfigurationError(
                "The description must use the configured origin"
            )
        if timeout <= 0 or min_interval < 0:
            raise ICPConfigurationError("Timeout and request interval must be valid")
        try:
            identity_origins = frozenset(
                _origin(url) for url in config["identity_origins"]
            )
            sign_in_origins = frozenset(
                _origin(url) for url in config["sign_in_origins"]
            )
            gateway_paths = frozenset(config.get("gateway_sign_in_paths", []))
            service = config["credential_service"]
        except (KeyError, TypeError, ICPRequestRefused):
            raise ICPConfigurationError(
                "Configure the private sign-in allowlist"
            ) from None
        if (
            not identity_origins
            or not identity_origins <= sign_in_origins
            or readable_origins & identity_origins
        ):
            raise ICPConfigurationError(
                "Identity origins must belong to the sign-in allowlist"
            )
        if not isinstance(service, str) or not service:
            raise ICPConfigurationError("Configure the stored credential service")
        if any(not isinstance(p, str) or not p.startswith("/") for p in gateway_paths):
            raise ICPConfigurationError("Gateway sign-in paths must be absolute paths")
        self.timeout = timeout
        self._creds = CredentialManager()
        try:
            self._credentials = self._creds.get_stored_credentials(service)
        except CredentialStoreUnavailable as exc:
            raise ICPAuthenticationError(str(exc)) from exc
        if not self._credentials:
            raise ICPAuthenticationError("No stored credentials for the ICP sign-in")
        if "@" not in self._credentials[0]:
            raise ICPAuthenticationError("The stored username must be an email")
        self._policy = _RequestPolicy(
            readable_origins,
            min_interval,
            identity_origins,
            sign_in_origins,
            gateway_paths,
            urlsplit(self._description_url).path,
            documents,
            records_only,
        )
        self._session = _GuardedSession(self._policy)
        self._login = _APILogin(
            origin, self._description_url, self._session, self._creds, service, timeout
        )
        self._authenticated = False
        self._auth_lock = threading.RLock()

    @property
    def origin(self) -> str:
        """The selected application origin attached to record identities."""
        return self._origin

    @property
    def session(self) -> requests.Session:
        """The guarded session; direct use retains the same refusal policy."""
        return self._session

    @property
    def request_log(self) -> tuple[RequestRecord, ...]:
        """Return redacted receipts for completed transport calls."""
        return tuple(self._policy.records)

    @contextmanager
    def _sign_in_exchange(self):
        if self._policy.signed_in:
            raise ICPRequestRefused(
                "A signed-in session cannot open an implicit exchange"
            )
        self._policy.sign_in_requests = 0
        self._policy.authenticating = True
        try:
            yield
        finally:
            self._policy.authenticating = False

    @property
    def _authenticated(self) -> bool:
        return self._policy.signed_in

    @_authenticated.setter
    def _authenticated(self, value: bool) -> None:
        self._policy.signed_in = value

    def authenticate(self, *, force: bool = False) -> bool:
        """Verify the API description or renew the gateway session once."""
        with self._auth_lock:
            if self._authenticated and not force:
                return True
            if not force:
                probe = self.session.get(
                    self._description_url, timeout=self.timeout, allow_redirects=False
                )
                if _description(probe) is not None:
                    self._authenticated = True
                    return True
            self._authenticated = False
            self.session.cookies.clear()
            with self._sign_in_exchange():
                success = self._login._authenticate_sso(*self._credentials)
            if not success:
                raise ICPAuthenticationError(
                    "The guarded sign-in did not verify API access"
                )
            self._authenticated = True
            return True

    def request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        """Read an origin-rooted API endpoint, retrying once after session expiry."""
        url = urljoin(self._origin + "/", endpoint)
        prepared = requests.Request(
            method, url, headers=kwargs.get("headers"), params=kwargs.get("params")
        ).prepare()
        self._policy.check(prepared)
        self.authenticate()
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("allow_redirects", False)
        response = self.session.request(method, url, **kwargs)
        is_document = (
            self._policy.documents is not None
            and _origin(url) == self._policy.documents.origin
        )
        if not is_document and self._stale(response):
            self.authenticate(force=True)
            response = self.session.request(method, url, **kwargs)
            if self._stale(response):
                self._authenticated = False
                raise ICPAuthenticationError("The renewed session was refused")
        response.raise_for_status()
        return response

    def document_url(self, uid: str, version: str) -> str:
        """Build and validate a privately listed document GET before sign-in."""
        access = self._policy.documents
        if access is None:
            raise ICPConfigurationError("Configure private document access")
        query = urlencode({"uid": uid, "version": version, "action": "get_document"})
        url = access.origin + access.path + "?" + query
        self._policy.check(requests.Request("GET", url).prepare())
        return url

    @staticmethod
    def _stale(response: requests.Response) -> bool:
        return (
            response.status_code == 401
            or (
                response.is_redirect
                and any(
                    marker in response.headers.get("Location", "").lower()
                    for marker in ("login", "/my.policy", "/my.logout.php3")
                )
            )
            or (
                "text/html" in response.headers.get("Content-Type", "")
                and any(
                    marker in response.text
                    for marker in ("window.parent.location.reload()", "$Config=")
                )
            )
        )

    def get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        """Read an endpoint with GET."""
        return self.request("GET", endpoint, **kwargs)

    def head(self, endpoint: str, **kwargs: Any) -> requests.Response:
        """Read response headers with HEAD."""
        return self.request("HEAD", endpoint, **kwargs)

    def read_description(self) -> dict[str, Any]:
        """Read and validate the API's machine-readable description."""
        response = self.get(self._description_url)
        description = _description(response)
        if description is None:
            raise ICPAuthenticationError("The API description was not returned")
        return description

    def close(self) -> None:
        """Close this client's connections and clear its in-memory cookies."""
        self._authenticated = False
        self.session.cookies.clear()
        self._session.close()

    def __enter__(self) -> ICPClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
