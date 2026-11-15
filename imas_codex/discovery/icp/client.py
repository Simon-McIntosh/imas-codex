"""Read ICP through a guarded session and the existing gateway sign-in exchange.

Set IMAS_CODEX_ICP_ORIGIN to the authorised integration origin and
IMAS_CODEX_ICP_DESCRIPTION_PATH to its description path at runtime. Neither
setting has a production fallback. API paths resolve from the origin's root.
Credentials come only from the system keyring. Cookies live in this session;
they are not flattened into a cross-host persistent cookie dictionary.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests

from imas_codex.discovery.wiki.auth import CredentialManager
from imas_codex.discovery.wiki.confluence import ConfluenceClient

_IDENTITY_ORIGIN = "https://login.microsoftonline.com"
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
    """An outbound request receipt with no host, query, headers or body."""

    method: str
    target: str
    status: int


class _RequestPolicy:
    def __init__(self, origin: str, interval: float) -> None:
        self.origin = origin
        self.interval = interval
        self.authenticating = False
        self.records: list[RequestRecord] = []
        self._last_send: float | None = None
        self._lock = threading.Lock()

    def check(self, request: requests.PreparedRequest) -> str:
        """Refuse a method or origin before any transport can send it."""
        origin = _origin(request.url or "")
        host_header = request.headers.get("Host")
        if host_header and host_header.lower() not in {
            urlsplit(origin).netloc,
            urlsplit(origin).netloc + ":443",
        }:
            raise ICPRequestRefused("A different Host header is not allowed")
        method = request.method
        if origin == self.origin:
            if method not in _READ_METHODS:
                raise ICPRequestRefused("ICP accepts only GET and HEAD")
            return "integration"
        if origin == _IDENTITY_ORIGIN and self.authenticating:
            if method not in _READ_METHODS | {"POST"}:
                raise ICPRequestRefused("Unsupported identity-provider method")
            return "identity-provider"
        raise ICPRequestRefused("The request origin is not allowed")

    def throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_send is not None:
                time.sleep(max(0.0, self.interval - (now - self._last_send)))
            self._last_send = time.monotonic()


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
        self._policy.throttle()
        response = super().send(request, **kwargs)
        self._policy.records.append(
            RequestRecord(request.method or "", target, response.status_code)
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
        timeout: float,
    ) -> None:
        # Keep the shared exchange on the guarded transport throughout sign-in.
        self.base_url = origin
        self.timeout = timeout
        self.credential_service = "iter"
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
    """Read a privately configured integration instance, renewing stale sessions.

    The runtime origin is the authority for the sole allowed application host.
    Requests to other application origins, including a production origin, are
    refused. Only the sign-in exchange may contact the identity provider.
    The exposed session uses the same policy for convenience methods, prepared
    requests, redirects and direct adapter calls.
    """

    def __init__(self, *, timeout: float = 30, min_interval: float = 0.5) -> None:
        configured = os.environ.get("IMAS_CODEX_ICP_ORIGIN", "")
        path = os.environ.get("IMAS_CODEX_ICP_DESCRIPTION_PATH", "")
        try:
            origin = _origin(configured)
        except ICPRequestRefused:
            raise ICPConfigurationError(
                "Configure the private integration origin"
            ) from None
        parts = urlsplit(configured)
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise ICPConfigurationError("The integration origin must have no path")
        if origin == _IDENTITY_ORIGIN:
            raise ICPConfigurationError("The integration origin must be an application")
        if not path.startswith("/") or path.startswith("//"):
            raise ICPConfigurationError("Configure an absolute description path")
        self._origin = origin
        self._description_url = urljoin(origin + "/", path)
        if _origin(self._description_url) != origin:
            raise ICPConfigurationError(
                "The description must use the configured origin"
            )
        if timeout <= 0 or min_interval < 0:
            raise ICPConfigurationError("Timeout and request interval must be valid")
        self.timeout = timeout
        self._creds = CredentialManager()
        self._credentials = self._creds.get_stored_credentials("iter")
        if not self._credentials:
            raise ICPAuthenticationError("No stored credentials for the ICP sign-in")
        if "@" not in self._credentials[0]:
            raise ICPAuthenticationError("The stored username must be an email")
        self._policy = _RequestPolicy(origin, min_interval)
        self._session = _GuardedSession(self._policy)
        self._login = _APILogin(
            origin, self._description_url, self._session, self._creds, timeout
        )
        self._authenticated = False
        self._auth_lock = threading.RLock()

    @property
    def session(self) -> requests.Session:
        """The guarded session; direct use retains the same refusal policy."""
        return self._session

    @property
    def request_log(self) -> tuple[RequestRecord, ...]:
        """Return redacted receipts for completed transport calls."""
        return tuple(self._policy.records)

    @contextmanager
    def _identity_exchange(self):
        self._policy.authenticating = True
        try:
            yield
        finally:
            self._policy.authenticating = False

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
            with self._identity_exchange():
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
            method, url, headers=kwargs.get("headers")
        ).prepare()
        self._policy.check(prepared)
        self.authenticate()
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("allow_redirects", False)
        response = self.session.request(method, url, **kwargs)
        if self._stale(response):
            self.authenticate(force=True)
            response = self.session.request(method, url, **kwargs)
            if self._stale(response):
                self._authenticated = False
                raise ICPAuthenticationError("The renewed session was refused")
        response.raise_for_status()
        return response

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
