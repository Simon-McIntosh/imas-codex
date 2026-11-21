"""Credential forwarding waits for SSH and distinguishes an unreadable store."""

import io
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from iter_codex.discovery.wiki import auth


@pytest.fixture
def forward(monkeypatch, tmp_path):
    clock = [0.0]
    socket = tmp_path / "dbus-forward.sock"

    def sleep(seconds):
        clock[0] += seconds

    def path(value):
        if str(value) == f"/run/user/{auth.os.getuid()}":
            return tmp_path
        return Path(value)

    proc = Mock(pid=12345, returncode=None, stderr=io.BytesIO())
    proc.poll.return_value = None
    monkeypatch.setattr(auth, "Path", path)
    monkeypatch.setattr(auth, "_dbus_forward_socket", None)
    monkeypatch.setattr(auth, "_dbus_forward_pid", None)
    monkeypatch.setattr(auth.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(auth.time, "sleep", sleep)
    monkeypatch.setattr(auth.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(auth.atexit, "register", Mock())
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    return clock, socket, proc


def test_socket_appearing_after_slow_ssh_is_used(monkeypatch, forward):
    clock, socket, proc = forward

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] >= 5:
            socket.touch()

    monkeypatch.setattr(auth.time, "sleep", sleep)
    assert auth._forward_dbus_socket("login.example.test") is True
    assert 5 <= clock[0] <= 15
    assert auth.os.environ["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={socket}"
    assert auth._dbus_forward_pid == proc.pid
    proc.terminate.assert_not_called()


def test_forward_reports_early_ssh_error_without_waiting(forward, caplog):
    clock, socket, proc = forward
    proc.poll.return_value = 255
    proc.returncode = 255
    proc.stderr = io.BytesIO(b"ssh: connection refused\n")
    with caplog.at_level(logging.WARNING):
        assert auth._forward_dbus_socket("login.example.test") is False
    assert clock[0] == 0
    assert "D-Bus forward SSH failed (exit 255): ssh: connection refused" in caplog.text
    assert auth._dbus_forward_socket is None
    assert "DBUS_SESSION_BUS_ADDRESS" not in auth.os.environ
    assert not socket.exists()


def test_forward_timeout_is_bounded(forward, caplog):
    clock, socket, proc = forward
    with caplog.at_level(logging.WARNING):
        assert auth._forward_dbus_socket("login.example.test") is False
    assert 14.9 <= clock[0] <= 15.1
    assert "socket did not appear" in caplog.text
    proc.terminate.assert_called_once()
    assert not socket.exists()


def test_unavailable_keyring_is_not_an_empty_credential_store():
    manager = auth.CredentialManager.__new__(auth.CredentialManager)
    manager._keyring_available = False
    with pytest.raises(RuntimeError, match="keyring is unavailable"):
        manager.get_stored_credentials("test-facility")


def test_failed_keyring_read_is_not_an_empty_credential_store(monkeypatch):
    import keyring

    manager = auth.CredentialManager.__new__(auth.CredentialManager)
    manager._keyring_available = True

    def fail(*args):
        raise OSError("session bus disconnected")

    monkeypatch.setattr(keyring, "get_password", fail)
    with pytest.raises(RuntimeError, match="keyring could not be read"):
        manager.get_stored_credentials("test-facility")


def test_readable_empty_keyring_returns_no_stored_credentials(monkeypatch):
    import keyring

    manager = auth.CredentialManager.__new__(auth.CredentialManager)
    manager._keyring_available = True
    monkeypatch.setattr(keyring, "get_password", lambda *args: None)
    assert manager.get_stored_credentials("test-facility") is None
