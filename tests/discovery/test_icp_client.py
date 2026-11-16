"""Guard each application transport, session renewal and keyring-only sign-in."""

import io
import json
import logging
import sys
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import requests

from imas_codex.discovery.icp import (
    ICPAuthenticationError,
    ICPClient,
    ICPConfigurationError,
    ICPRequestRefused,
    client as module,
)
from imas_codex.discovery.wiki.auth import CredentialManager, CredentialStoreUnavailable
from imas_codex.discovery.wiki.confluence import ConfluenceClient

ORIGIN = "https://integration.example.test"
APPLICATION_ORIGINS = (ORIGIN, "https://production.example.test")
IDENTITY = "https://login.microsoftonline.com"
GATEWAY = "https://gateway.example.test"
DESCRIPTION = {"openapi": "3.0.0", "paths": {"/items": {"get": {}}}}
WRITES = ("POST", "PUT", "PATCH", "DELETE")


@pytest.fixture(autouse=True, params=APPLICATION_ORIGINS)
def private_config(monkeypatch, request):
    monkeypatch.setattr(sys.modules[__name__], "ORIGIN", request.param)
    config = {
        "origin": ORIGIN,
        "readable_origins": list(APPLICATION_ORIGINS),
        "credential_service": "iter",
        "identity_origins": [IDENTITY],
        "sign_in_origins": [IDENTITY, GATEWAY, *APPLICATION_ORIGINS],
        "gateway_sign_in_paths": ["/gateway/sign-in"],
    }
    monkeypatch.setenv("IMAS_CODEX_ICP_FACILITY", "test-facility")
    monkeypatch.setattr(
        module, "get_facility_infrastructure", lambda facility: {"icp": config}
    )
    return config


def response(request, status=200, body=None, headers=None):
    result = requests.Response()
    result.status_code = status
    result.url = request.url
    result.request = request
    result.headers.update(headers or {"Content-Type": "application/json"})
    result._content = json.dumps(body if body is not None else DESCRIPTION).encode()
    result.raw = io.BytesIO(result._content)
    return result


@pytest.fixture
def transport(monkeypatch):
    sent = []

    def send(adapter, request, **kwargs):
        sent.append(request)
        return response(request)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return sent


@pytest.fixture
def client(monkeypatch, transport):
    monkeypatch.setenv("IMAS_CODEX_ICP_ORIGIN", ORIGIN)
    monkeypatch.setenv("IMAS_CODEX_ICP_DESCRIPTION_PATH", "/openapi.json")
    manager = Mock(spec=CredentialManager)
    manager.get_stored_credentials.return_value = ("reader@example.test", "test-secret")
    monkeypatch.setattr(module, "CredentialManager", lambda: manager)
    with ICPClient(min_interval=0) as instance:
        yield instance


@pytest.mark.parametrize("method", WRITES)
@pytest.mark.parametrize(
    "entry",
    (
        "client.request",
        "session.request",
        "session.convenience",
        "session.send",
        "base.request",
        "base.send",
        "adapter.send",
    ),
)
def test_write_refused_before_transport(client, transport, entry, method):
    url = ORIGIN + "/items"
    prepared = requests.Request(method, url).prepare()
    with pytest.raises(ICPRequestRefused, match="only GET and HEAD"):
        match entry:
            case "client.request":
                client.request(method, url)
            case "session.request":
                client.session.request(method, url)
            case "session.convenience":
                getattr(client.session, method.lower())(url)
            case "session.send":
                client.session.send(prepared)
            case "base.request":
                requests.Session.request(client.session, method, url)
            case "base.send":
                requests.Session.send(client.session, prepared)
            case "adapter.send":
                client.session.get_adapter(url).send(prepared)
    assert transport == []


@pytest.mark.parametrize("method", WRITES)
@pytest.mark.parametrize("entry", ("get", "head", "read_description", "authenticate"))
def test_fixed_method_entry_points_reject_method_override(
    client, transport, entry, method
):
    args = ("/items",) if entry in {"get", "head"} else ()
    with pytest.raises(TypeError):
        getattr(client, entry)(*args, method=method)
    assert transport == []


@pytest.mark.parametrize(
    "url",
    (
        "https://unlisted.example.test/items",
        "https://unrelated.example.test/items",
        "http://integration.example.test/items",
        "https://integration.example.test.evil.test/items",
        "https://integration.example.test:444/items",
        "https://reader@integration.example.test/items",
        "https://login.microsoftonline.com.evil.test/items",
        "//unrelated.example.test/items",
    ),
)
@pytest.mark.parametrize("entry", ("client", "session", "prepared", "adapter"))
def test_other_origin_refused(client, transport, url, entry):
    with pytest.raises((ICPRequestRefused, requests.exceptions.RequestException)):
        if entry == "client":
            client.get(url)
        elif entry == "session":
            client.session.get(url)
        elif entry == "prepared":
            client.session.send(requests.Request("GET", url).prepare())
        else:
            request = requests.Request("GET", url).prepare()
            client.session.get_adapter(ORIGIN).send(request)
    assert transport == []


@pytest.mark.parametrize("method", ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"))
def test_identity_provider_is_sign_in_only(client, transport, method):
    with pytest.raises(ICPRequestRefused):
        client.session.request(method, IDENTITY + "/login")
    assert transport == []


def test_sign_in_exception_does_not_allow_application_writes(client, transport):
    with client._sign_in_exchange():
        client.session.post(
            IDENTITY + "/login", data={"ctx": "test", "flowToken": "test"}
        )
        with pytest.raises(ICPRequestRefused):
            client.session.post(ORIGIN + "/saml", data={"assertion": "test"})
        with pytest.raises(ICPRequestRefused):
            client.session.delete(IDENTITY + "/login")
    assert [(r.method, r.url) for r in transport] == [("POST", IDENTITY + "/login")]


@pytest.mark.parametrize(
    "target", (ORIGIN + "/items", "https://other.example.test/items")
)
def test_preserved_post_redirect_refused(client, transport, monkeypatch, target):
    def redirect(adapter, request, **kwargs):
        transport.append(request)
        return response(request, status=307, headers={"Location": target})

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", redirect)
    with client._sign_in_exchange(), pytest.raises(ICPRequestRefused):
        client.session.post(
            IDENTITY + "/login", data={"ctx": "test", "flowToken": "test"}
        )
    assert len(transport) == 1
    assert transport[0].url.startswith(IDENTITY)


def test_read_redirect_to_other_host_refused(client, transport, monkeypatch):
    def redirect(adapter, request, **kwargs):
        transport.append(request)
        return response(
            request,
            status=302,
            headers={"Location": "https://other.example.test/items"},
        )

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", redirect)
    with pytest.raises(ICPRequestRefused):
        client.session.get(ORIGIN + "/redirect")
    assert len(transport) == 1


@pytest.mark.parametrize(
    "expired", ("unauthorized", "reload", "identity-redirect", "gateway-redirect")
)
def test_stale_session_is_renewed_once(client, transport, monkeypatch, expired):
    calls = 0

    def send(adapter, request, **kwargs):
        nonlocal calls
        transport.append(request)
        calls += 1
        if calls == 1:
            if expired == "unauthorized":
                return response(request, status=401)
            if expired == "identity-redirect":
                return response(request, status=302, headers={"Location": IDENTITY})
            if expired == "gateway-redirect":
                return response(request, status=302, headers={"Location": "/my.policy"})
            result = response(request, headers={"Content-Type": "text/html"})
            result._content = b"window.parent.location.reload()"
            return result
        return response(request, body={"items": ["synthetic"]})

    def sign_in(*credentials):
        assert client.session.cookies.get("stale") is None
        client.session.cookies.set(
            "renewed", "synthetic", domain="integration.example.test"
        )
        return True

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    login = Mock(side_effect=sign_in)
    monkeypatch.setattr(client._login, "_authenticate_sso", login)
    client._authenticated = True
    client.session.cookies.set("stale", "synthetic")
    assert client.get("/items").json() == {"items": ["synthetic"]}
    login.assert_called_once()
    assert [r.method for r in transport] == ["GET", "GET"]
    assert client.session.cookies.get("renewed") == "synthetic"
    assert not client._policy.authenticating


def test_repeated_expiry_refuses_without_retry_loop(client, transport, monkeypatch):
    def send(adapter, request, **kwargs):
        transport.append(request)
        return response(request, status=401)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    login = Mock(return_value=True)
    monkeypatch.setattr(client._login, "_authenticate_sso", login)
    client._authenticated = True
    with pytest.raises(ICPAuthenticationError, match="renewed session"):
        client.get("/items")
    assert len(transport) == 2
    login.assert_called_once()
    assert not client._authenticated


def test_api_verification_uses_description_instead_of_confluence_endpoint(
    client, transport, monkeypatch
):
    def send(adapter, request, **kwargs):
        transport.append(request)
        return response(request, status=404 if "/rest/api/" in request.url else 200)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    confluence = ConfluenceClient.__new__(ConfluenceClient)
    confluence.base_url = ORIGIN
    confluence.timeout = 30
    confluence.credential_service = "iter"
    confluence._session = client.session
    confluence._creds = Mock()
    assert confluence._verify_sso_session(client.session) is False
    assert transport[-1].url == ORIGIN + "/rest/api/user/current"
    assert client._login._verify_sso_session(client.session) is True
    assert transport[-1].url == ORIGIN + "/openapi.json"
    confluence._creds.set_session.assert_not_called()


def test_shared_sign_in_exchange_stays_on_guarded_transport(
    client, transport, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="imas_codex.discovery.wiki.confluence")

    def send(adapter, request, **kwargs):
        transport.append(request)
        if request.url == ORIGIN + "/openapi.json":
            if any(r.method == "POST" for r in transport):
                return response(request)
            return response(
                request, status=302, headers={"Location": IDENTITY + "/login"}
            )
        if request.url == ORIGIN + "/":
            return response(
                request, status=302, headers={"Location": IDENTITY + "/login"}
            )
        if request.method == "GET" and request.url == IDENTITY + "/login":
            result = response(request, headers={"Content-Type": "text/html"})
            result._content = (
                b'$Config={"sFT":"synthetic","sCtx":"context","urlPost":"/login"};'
            )
            return result
        if request.url.startswith(IDENTITY + "/common/GetCredentialType"):
            return response(request, body={"IfExistsResult": 0})
        if request.url == IDENTITY + "/login" and request.method == "POST":
            return response(
                request,
                status=302,
                headers={"Location": ORIGIN + "/callback?state=private-redirect-value"},
            )
        return response(request)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    assert client.authenticate()
    assert client.read_description() == DESCRIPTION
    assert any(r.method == "POST" and r.url.startswith(IDENTITY) for r in transport)
    assert all(
        r.method in {"GET", "HEAD"} for r in transport if r.url.startswith(ORIGIN)
    )
    assert not client._policy.authenticating
    assert "reader@example.test" not in caplog.text
    assert "test-secret" not in caplog.text
    assert "private-redirect-value" not in caplog.text
    assert ORIGIN not in caplog.text


def test_description_read_and_origin_rooted_paths(client, transport):
    assert client.read_description() == DESCRIPTION
    client.get("api/items")
    client.head("/api/items")
    assert [r.url for r in transport][-2:] == [ORIGIN + "/api/items"] * 2
    assert {r.method for r in client.request_log} == {"GET", "HEAD"}
    assert {r.target for r in client.request_log} == {"application"}
    assert not any(ORIGIN in str(r) for r in client.request_log)


def test_missing_stored_credentials_refuses_before_any_request(monkeypatch, transport):
    monkeypatch.setenv("IMAS_CODEX_ICP_ORIGIN", ORIGIN)
    monkeypatch.setenv("IMAS_CODEX_ICP_DESCRIPTION_PATH", "/openapi.json")
    monkeypatch.setenv("ITER_USERNAME", "environment@example.test")
    monkeypatch.setenv("ITER_PASSWORD", "environment-secret")
    manager = Mock(spec=CredentialManager)
    manager.get_stored_credentials.return_value = None
    monkeypatch.setattr(module, "CredentialManager", lambda: manager)
    with pytest.raises(ICPAuthenticationError, match="No stored credentials"):
        ICPClient()
    manager.get_stored_credentials.assert_called_once_with("iter")
    manager.get_credentials.assert_not_called()
    assert transport == []


@pytest.mark.parametrize("entry", ("client", "session", "adapter"))
def test_host_header_cannot_route_to_another_application(client, transport, entry):
    headers = {"Host": "unlisted.example.test"}
    with pytest.raises(ICPRequestRefused):
        if entry == "client":
            client.get("/items", headers=headers)
        elif entry == "session":
            client.session.get(ORIGIN, headers=headers)
        else:
            request = requests.Request("GET", ORIGIN, headers=headers).prepare()
            client.session.get_adapter(ORIGIN).send(request)
    assert transport == []


def test_guarded_transport_cannot_be_replaced_or_tls_disabled(client, transport):
    with pytest.raises(ICPRequestRefused):
        client.session.mount("https://", requests.adapters.HTTPAdapter())
    with pytest.raises(ICPRequestRefused):
        client.session.get(ORIGIN, verify=False)
    assert transport == []


@pytest.mark.parametrize("verify", (None, False, 0, ""))
@pytest.mark.parametrize("entry", ("session", "adapter"))
def test_low_level_tls_verification_cannot_be_disabled(
    client, transport, entry, verify
):
    request = requests.Request("GET", ORIGIN).prepare()
    with pytest.raises(ICPRequestRefused):
        if entry == "session":
            client.session.send(request, verify=verify)
        else:
            client.session.get_adapter(ORIGIN).send(request, verify=verify)
    assert transport == []


def test_request_rate_is_applied_at_transport_boundary(client, transport, monkeypatch):
    clock = Mock(side_effect=(10.0, 10.0, 10.1, 10.5, 10.5))
    sleep = Mock()
    monkeypatch.setattr(module.time, "monotonic", clock)
    monkeypatch.setattr(module.time, "sleep", sleep)
    client._policy.pacer.interval = 0.5
    client.session.get(ORIGIN)
    client.session.head(ORIGIN)
    assert sleep.call_args.args[0] == pytest.approx(0.4)
    assert len(transport) == 2


@pytest.mark.parametrize(
    "setting", ("IMAS_CODEX_ICP_ORIGIN", "IMAS_CODEX_ICP_DESCRIPTION_PATH")
)
def test_no_implicit_application_configuration(
    client, monkeypatch, setting, transport, private_config
):
    monkeypatch.delenv(setting)
    if setting == "IMAS_CODEX_ICP_ORIGIN":
        private_config.pop("origin")
    with pytest.raises(ICPConfigurationError):
        ICPClient()
    assert transport == []


def test_each_private_application_origin_is_readable(client, transport):
    client._authenticated = True
    for origin in APPLICATION_ORIGINS:
        client.get(origin + "/items")
        client.head(origin + "/items")
    assert [r.method for r in transport] == ["GET", "HEAD", "GET", "HEAD"]


@pytest.mark.parametrize("selector", ("environment", "constructor"))
def test_selector_cannot_authorise_an_unlisted_host(
    client, monkeypatch, transport, selector
):
    kwargs = {}
    if selector == "environment":
        monkeypatch.setenv("IMAS_CODEX_ICP_ORIGIN", "https://unlisted.example.test")
    else:
        kwargs["origin"] = "https://unlisted.example.test"
    with pytest.raises(ICPConfigurationError, match="private readable"):
        ICPClient(**kwargs)
    assert transport == []


@pytest.mark.parametrize(
    "origins",
    (
        [],
        "https://integration.example.test",
        ["http://other.example.test"],
        ["https://other.example.test/path"],
        ["https://other.example.test?query=yes"],
        ["https://other.example.test#fragment"],
        [IDENTITY],
        [None],
    ),
)
def test_invalid_private_readable_allowlist_refused(
    client, private_config, transport, origins
):
    private_config["readable_origins"] = origins
    with pytest.raises(ICPConfigurationError):
        ICPClient()
    assert transport == []


def test_keyring_lookup_does_not_use_environment_or_prompt(monkeypatch):
    manager = CredentialManager.__new__(CredentialManager)
    manager._keyring_available = False
    monkeypatch.setenv("ITER_USERNAME", "environment@example.test")
    monkeypatch.setenv("ITER_PASSWORD", "environment-secret")
    with pytest.raises(CredentialStoreUnavailable, match="keyring is unavailable"):
        manager.get_stored_credentials("iter")
    manager._keyring_available = True
    monkeypatch.setattr(
        manager,
        "_keyring_op",
        lambda operation, **kwargs: json.dumps(
            {"username": "stored@example.test", "password": "test"}
        ),
    )
    assert manager.get_stored_credentials("iter") == ("stored@example.test", "test")


def test_unreadable_credential_store_reports_unavailable_before_http(
    monkeypatch, transport
):
    monkeypatch.setenv("IMAS_CODEX_ICP_ORIGIN", ORIGIN)
    monkeypatch.setenv("IMAS_CODEX_ICP_DESCRIPTION_PATH", "/openapi.json")
    manager = Mock(spec=CredentialManager)
    manager.get_stored_credentials.side_effect = CredentialStoreUnavailable(
        "System keyring is unavailable; credential storage could not be read"
    )
    monkeypatch.setattr(module, "CredentialManager", lambda: manager)
    with pytest.raises(ICPAuthenticationError, match="keyring is unavailable"):
        ICPClient()
    assert transport == []


@pytest.mark.parametrize(
    "value", ("invalid-json", "[]", "{}", '{"username":0,"password":1}')
)
def test_malformed_keyring_credentials_refuse(monkeypatch, value):
    manager = CredentialManager.__new__(CredentialManager)
    manager._keyring_available = True
    monkeypatch.setattr(manager, "_keyring_op", lambda operation, **kwargs: value)
    assert manager.get_stored_credentials("iter") is None


@pytest.mark.parametrize("state", ("before", "exchange", "signed-in"))
@pytest.mark.parametrize(
    "path",
    (
        "/api/items",
        "/api/query/execute",
        "/api/type/list-contents",
        "/docs/openapi.json",
        "/openapi.json",
    ),
)
def test_api_post_refused_in_every_authentication_state(client, transport, state, path):
    client._authenticated = state == "signed-in"
    context = client._sign_in_exchange() if state == "exchange" else nullcontext()
    with context, pytest.raises(ICPRequestRefused, match="only GET and HEAD"):
        client.session.post(ORIGIN + path, data={"SAMLResponse": "synthetic"})
    assert transport == []


@pytest.mark.parametrize(
    "target", (IDENTITY + "/login", GATEWAY + "/assertion", ORIGIN + "/gateway/sign-in")
)
@pytest.mark.parametrize("state", ("before", "signed-in", "inconsistent"))
def test_sign_in_post_refused_outside_unsigned_exchange(
    client, transport, target, state
):
    client._authenticated = state != "before"
    client._policy.authenticating = state == "inconsistent"
    form = (
        {"ctx": "test", "flowToken": "test"}
        if target.startswith(IDENTITY)
        else {"SAMLResponse": "synthetic"}
    )
    with pytest.raises(ICPRequestRefused, match="active unsigned exchange"):
        client.session.post(target, data=form)
    assert transport == []


def test_unlisted_origin_post_refused_during_exchange(client, transport):
    with client._sign_in_exchange(), pytest.raises(ICPRequestRefused):
        client.session.post(
            "https://unlisted.example.test/assertion",
            data={"SAMLResponse": "synthetic"},
        )
    assert transport == []


@pytest.mark.parametrize(
    ("target", "expected_origin", "expected_path"),
    (
        (GATEWAY + "/assertion", GATEWAY, "/assertion"),
        (ORIGIN + "/gateway/sign-in", ORIGIN, "/gateway/sign-in"),
    ),
)
def test_observed_gateway_form_allowed_only_during_sign_in(
    client, transport, target, expected_origin, expected_path
):
    with client._sign_in_exchange():
        client.session.post(
            target, data={"SAMLResponse": "synthetic", "RelayState": "test"}
        )
    assert len(transport) == 1
    record = client.request_log[0]
    assert record.phase == "sign-in"
    assert record.origin == expected_origin
    assert record.path == expected_path
    assert "synthetic" not in str(record)
    client._authenticated = True
    with pytest.raises(ICPRequestRefused):
        client.session.post(target, data={"SAMLResponse": "synthetic"})
    assert len(transport) == 1


@pytest.mark.parametrize(
    "target", (GATEWAY + "/assertion", ORIGIN + "/gateway/sign-in")
)
@pytest.mark.parametrize(
    "form",
    (
        {"passwd": "test"},
        {"SAMLResponse": "synthetic", "passwd": "test"},
        {"arbitrary": "test"},
    ),
)
def test_gateway_refuses_credentials_and_non_login_forms(
    client, transport, target, form
):
    with client._sign_in_exchange(), pytest.raises(ICPRequestRefused):
        client.session.post(target, data=form)
    assert transport == []


def test_identity_provider_refuses_arbitrary_post_bodies(client, transport):
    with client._sign_in_exchange():
        with pytest.raises(ICPRequestRefused):
            client.session.post(IDENTITY + "/login", data={"arbitrary": "test"})
        with pytest.raises(ICPRequestRefused):
            client.session.post(IDENTITY + "/login", json={"arbitrary": "test"})
    assert transport == []


def test_sign_in_has_one_total_transport_budget(client, transport):
    client._policy.max_sign_in_requests = 2
    with client._sign_in_exchange():
        client.session.get(IDENTITY + "/login")
        client.session.post(
            IDENTITY + "/login", data={"ctx": "test", "flowToken": "test"}
        )
        with pytest.raises(ICPRequestRefused, match="budget"):
            client.session.get(IDENTITY + "/login")
    assert len(transport) == 2


def test_api_path_cannot_be_configured_as_a_gateway_exception(client, transport):
    client._policy.gateway_paths = frozenset(
        {"/api/query/execute", "/docs/openapi.json"}
    )
    with client._sign_in_exchange():
        for path in client._policy.gateway_paths:
            with pytest.raises(ICPRequestRefused):
                client.session.post(ORIGIN + path, data={"SAMLResponse": "synthetic"})
    assert transport == []


def test_selected_gateway_has_identical_unsigned_boundary(client, transport):
    target = ORIGIN + "/gateway/sign-in"
    with client._sign_in_exchange():
        client.session.post(target, data={"SAMLResponse": "synthetic"})
        with pytest.raises(ICPRequestRefused):
            client.session.post(target, data={"passwd": "synthetic"})
    client._authenticated = True
    with pytest.raises(ICPRequestRefused):
        client.session.post(target, data={"SAMLResponse": "synthetic"})
    assert len(transport) == 1
    assert client.request_log[0].origin == ORIGIN


@pytest.mark.parametrize("listed", (True, False))
def test_redirect_checks_the_destination_host(client, transport, monkeypatch, listed):
    destination = (
        next(origin for origin in APPLICATION_ORIGINS if origin != ORIGIN)
        if listed
        else "https://unlisted.example.test"
    )

    def send(adapter, request, **kwargs):
        transport.append(request)
        if request.url == ORIGIN + "/redirect":
            return response(
                request, status=302, headers={"Location": destination + "/items"}
            )
        return response(request)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    client._authenticated = True
    if listed:
        client.get("/redirect", allow_redirects=True)
        assert len(transport) == 2
        assert client.request_log[-1].origin == destination
    else:
        with pytest.raises(ICPRequestRefused):
            client.get("/redirect", allow_redirects=True)
        assert len(transport) == 1


def test_private_single_origin_configuration_remains_readable(
    client, private_config, transport
):
    private_config.pop("readable_origins")
    with ICPClient(min_interval=0) as single:
        single._authenticated = True
        single.get("/items")
        other = next(origin for origin in APPLICATION_ORIGINS if origin != ORIGIN)
        with pytest.raises(ICPRequestRefused):
            single.get(other + "/items")
    assert len(transport) == 1
