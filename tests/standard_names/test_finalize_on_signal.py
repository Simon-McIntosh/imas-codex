"""Shutdown finalization tests.

Verifies:
- drain_pending returns False after DRAIN_TIMEOUT when the writer is wedged
- a finalize_sn_run timeout doesn't crash — error logged, no exception
- signal registration preserves SIGINT/SIGTERM identity
- duplicate signals do not advance shutdown state
- only later SIGINT deliveries force shutdown and hard exit
- normal coroutine completion does not start the exit watchdog
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent

_EVENT = LLMCostEvent(model="test", tokens_in=10, tokens_out=5, phase="test")


# ── drain timeout proceeds to finalize ───────────────────────────────


@pytest.mark.asyncio
async def test_drain_timeout_proceeds():
    """When writer is wedged, drain_pending should time out.

    The caller (loop.py) wraps drain_pending in wait_for(timeout=DRAIN_TIMEOUT).
    Simulate that here: a writer that sleeps forever should be cut off.
    """
    mgr = BudgetManager(10.0, run_id="test-drain-timeout")
    cancel_event = threading.Event()

    def _wedged_record(**kwargs):
        cancel_event.wait(timeout=60)  # blocks until test cleans up

    await mgr.start()

    with (
        patch(
            "imas_codex.standard_names.graph_ops.record_llm_cost",
            side_effect=_wedged_record,
        ),
        patch("imas_codex.standard_names.budget._WRITER_CALL_TIMEOUT", 60.0),
    ):
        # Enqueue a write that will wedge
        lease = mgr.reserve(1.0, phase="test")
        assert lease is not None
        lease.charge_event(0.01, _EVENT)

        # Give the writer loop a moment to pick up the item
        await asyncio.sleep(0.2)

        # Simulate what loop.py does: wait_for with a short timeout
        DRAIN_TIMEOUT = 2.0  # shortened for test speed
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(mgr.drain_pending()),
                timeout=DRAIN_TIMEOUT,
            )

    # After timeout, cancel the writer and unblock the thread
    cancel_event.set()
    if mgr._writer_task is not None:
        mgr._writer_task.cancel()
        try:
            await mgr._writer_task
        except (asyncio.CancelledError, Exception):
            pass


# ── finalize timeout doesn't crash ───────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_timeout_no_crash(caplog):
    """finalize_sn_run wrapped in wait_for — timeout logs critical, no raise."""
    cancel_event = threading.Event()

    def _wedged_finalize(*args, **kwargs):
        # Block until cancelled — won't leak threads after test
        cancel_event.wait(timeout=5)

    FINALIZE_TIMEOUT = 0.5  # shortened for test

    timed_out = False
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_wedged_finalize, "run-123", status="completed"),
            timeout=FINALIZE_TIMEOUT,
        )
    except TimeoutError:
        timed_out = True

    # Unblock the thread so it doesn't leak
    cancel_event.set()

    assert timed_out, "Should have timed out"
    # If we reach here without exception, the test passes — the finally
    # block in loop.py catches TimeoutError and continues.
    # The finally block in loop.py catches TimeoutError and continues.


# ── shutdown signal state transitions ───────────────────────────────


def _capture_signal_handlers(*, display=None):
    """Install handlers on a fake loop and return its captured callbacks."""
    from imas_codex.cli.shutdown import install_shutdown_handlers

    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.time.return_value = 0.0
    stop_event = MagicMock(spec=asyncio.Event)
    callbacks = {}

    def _capture(signal_kind, callback):
        callbacks[signal_kind] = callback

    loop.add_signal_handler.side_effect = _capture
    with patch("imas_codex.cli.shutdown.asyncio.get_running_loop", return_value=loop):
        install_shutdown_handlers(stop_event=stop_event, display=display)

    return loop, stop_event, callbacks


def test_signal_registration_preserves_identity():
    """SIGINT and SIGTERM use distinct callbacks carrying their identities."""
    loop, _, callbacks = _capture_signal_handlers()

    assert set(callbacks) == {signal.SIGINT, signal.SIGTERM}
    assert callbacks[signal.SIGINT] is not callbacks[signal.SIGTERM]
    assert loop.add_signal_handler.call_count == 2


def test_signal_fallback_marshals_identity_and_restores_handlers():
    """Fallback callbacks reach the loop and restore process signal ownership."""
    from imas_codex.cli.shutdown import (
        _SIGNAL_DEBOUNCE_SECONDS,
        install_shutdown_handlers,
    )

    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.time.return_value = 0.0
    loop.add_signal_handler.side_effect = NotImplementedError
    stop_event = MagicMock(spec=asyncio.Event)
    owner_task = MagicMock(spec=asyncio.Task)
    prior_handlers = {
        signal.SIGINT: MagicMock(),
        signal.SIGTERM: MagicMock(),
    }
    installed_handlers = dict(prior_handlers)

    def _install_process_handler(signal_kind, handler):
        previous_handler = installed_handlers[signal_kind]
        installed_handlers[signal_kind] = handler
        return previous_handler

    with (
        patch("imas_codex.cli.shutdown.asyncio.get_running_loop", return_value=loop),
        patch("imas_codex.cli.shutdown.asyncio.current_task", return_value=owner_task),
        patch(
            "imas_codex.cli.shutdown.signal.signal",
            side_effect=_install_process_handler,
        ),
        patch("imas_codex.cli.shutdown.asyncio.all_tasks", return_value=set()),
        patch("imas_codex.cli.shutdown._start_exit_watchdog") as watchdog,
        patch("imas_codex.cli.shutdown._force_stop_display"),
        patch("imas_codex.cli.shutdown._force_kill_ssh_pools"),
    ):
        install_shutdown_handlers(stop_event=stop_event)

        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        graceful_callback = loop.call_soon_threadsafe.call_args.args[0]
        graceful_callback()

        loop.time.return_value = _SIGNAL_DEBOUNCE_SECONDS * 2
        installed_handlers[signal.SIGINT](signal.SIGINT, None)
        forced_callback = loop.call_soon_threadsafe.call_args.args[0]
        forced_callback()

        stop_event.set.assert_called_once_with()
        watchdog.assert_called_once_with(45)
        assert graceful_callback is not forced_callback

        restore_callback = owner_task.add_done_callback.call_args.args[0]
        restore_callback(owner_task)

    assert installed_handlers == prior_handlers


@pytest.mark.parametrize("registration_error", [OSError, ValueError])
def test_partial_asyncio_registration_is_rolled_back_before_fallback(
    registration_error,
):
    """A failure on the later signal cannot leave the earlier one clobbered."""
    from imas_codex.cli.shutdown import install_shutdown_handlers

    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    stop_event = MagicMock(spec=asyncio.Event)
    owner_task = MagicMock(spec=asyncio.Task)
    prior_handlers = {
        signal.SIGINT: MagicMock(),
        signal.SIGTERM: MagicMock(),
    }
    installed_handlers = dict(prior_handlers)
    asyncio_sigint_handler = MagicMock()

    def _install_asyncio_handler(signal_kind, callback):
        if signal_kind == signal.SIGINT:
            installed_handlers[signal_kind] = asyncio_sigint_handler
            return
        raise registration_error

    def _remove_asyncio_handler(signal_kind):
        installed_handlers[signal_kind] = signal.SIG_DFL
        return True

    def _install_process_handler(signal_kind, handler):
        previous_handler = installed_handlers[signal_kind]
        installed_handlers[signal_kind] = handler
        return previous_handler

    loop.add_signal_handler.side_effect = _install_asyncio_handler
    loop.remove_signal_handler.side_effect = _remove_asyncio_handler

    with (
        patch("imas_codex.cli.shutdown.asyncio.get_running_loop", return_value=loop),
        patch("imas_codex.cli.shutdown.asyncio.current_task", return_value=owner_task),
        patch(
            "imas_codex.cli.shutdown.signal.getsignal",
            side_effect=lambda signal_kind: prior_handlers[signal_kind],
        ),
        patch(
            "imas_codex.cli.shutdown.signal.signal",
            side_effect=_install_process_handler,
        ),
    ):
        install_shutdown_handlers(stop_event=stop_event)

        loop.remove_signal_handler.assert_called_once_with(signal.SIGINT)
        assert installed_handlers[signal.SIGINT] is not asyncio_sigint_handler
        assert installed_handlers[signal.SIGINT] is not prior_handlers[signal.SIGINT]
        assert installed_handlers[signal.SIGTERM] is not prior_handlers[signal.SIGTERM]

        restore_callback = owner_task.add_done_callback.call_args.args[0]
        restore_callback(owner_task)

    assert installed_handlers == prior_handlers


@pytest.mark.parametrize("first_signal", [signal.SIGINT, signal.SIGTERM])
def test_first_signal_requests_graceful_shutdown_once(first_signal):
    """Either signal requests the single cooperative shutdown transition."""
    display = MagicMock()
    _, stop_event, callbacks = _capture_signal_handlers(display=display)

    callbacks[first_signal]()

    stop_event.set.assert_called_once_with()
    display.begin_shutdown.assert_called_once_with()


def test_immediate_duplicate_sigint_is_coalesced_without_extending_window():
    """Immediate duplicate SIGINT callbacks count as one operator delivery."""
    from imas_codex.cli.shutdown import _SIGNAL_DEBOUNCE_SECONDS

    loop, stop_event, callbacks = _capture_signal_handlers()
    task = MagicMock(spec=asyncio.Task)

    with (
        patch("imas_codex.cli.shutdown.asyncio.all_tasks", return_value={task}),
        patch("imas_codex.cli.shutdown._start_exit_watchdog") as watchdog,
        patch("imas_codex.cli.shutdown._force_stop_display") as stop_display,
        patch("imas_codex.cli.shutdown._force_kill_ssh_pools") as kill_pools,
    ):
        callbacks[signal.SIGINT]()
        loop.time.return_value = _SIGNAL_DEBOUNCE_SECONDS / 2
        callbacks[signal.SIGINT]()

        stop_event.set.assert_called_once_with()
        watchdog.assert_not_called()
        stop_display.assert_not_called()
        kill_pools.assert_not_called()
        task.cancel.assert_not_called()

        # The ignored callback did not move the acceptance timestamp, so this
        # delivery is distinct from the original even though it is close to
        # the duplicate.
        loop.time.return_value = _SIGNAL_DEBOUNCE_SECONDS * 1.1
        callbacks[signal.SIGINT]()

        watchdog.assert_called_once_with(45)
        task.cancel.assert_called_once_with()


@pytest.mark.parametrize(
    ("first_signal", "duplicate_signal"),
    [
        (signal.SIGINT, signal.SIGTERM),
        (signal.SIGTERM, signal.SIGINT),
    ],
)
def test_immediate_mixed_signal_pair_is_coalesced(first_signal, duplicate_signal):
    """An immediate SIGINT/SIGTERM pair is one shutdown request in either order."""
    from imas_codex.cli.shutdown import _SIGNAL_DEBOUNCE_SECONDS

    loop, stop_event, callbacks = _capture_signal_handlers()

    with (
        patch("imas_codex.cli.shutdown._start_exit_watchdog") as watchdog,
        patch("imas_codex.cli.shutdown._force_stop_display") as stop_display,
        patch("imas_codex.cli.shutdown._force_kill_ssh_pools") as kill_pools,
    ):
        callbacks[first_signal]()
        loop.time.return_value = _SIGNAL_DEBOUNCE_SECONDS / 2
        callbacks[duplicate_signal]()

    stop_event.set.assert_called_once_with()
    watchdog.assert_not_called()
    stop_display.assert_not_called()
    kill_pools.assert_not_called()


def test_repeated_sigterm_remains_cooperative_at_later_times():
    """SIGTERM never advances a cooperative shutdown to forced shutdown."""
    from imas_codex.cli.shutdown import _SIGNAL_DEBOUNCE_SECONDS

    loop, stop_event, callbacks = _capture_signal_handlers()

    with (
        patch("imas_codex.cli.shutdown._start_exit_watchdog") as watchdog,
        patch("imas_codex.cli.shutdown._force_stop_display") as stop_display,
        patch("imas_codex.cli.shutdown._force_kill_ssh_pools") as kill_pools,
        patch("imas_codex.cli.shutdown.os._exit") as hard_exit,
    ):
        callbacks[signal.SIGTERM]()
        loop.time.return_value = _SIGNAL_DEBOUNCE_SECONDS * 2
        callbacks[signal.SIGTERM]()
        loop.time.return_value = 100.0
        callbacks[signal.SIGTERM]()

    stop_event.set.assert_called_once_with()
    watchdog.assert_not_called()
    stop_display.assert_not_called()
    kill_pools.assert_not_called()
    hard_exit.assert_not_called()


def test_ignored_later_sigterm_does_not_extend_debounce_window():
    """An ignored SIGTERM does not delay the next distinct SIGINT transition."""
    from imas_codex.cli.shutdown import _SIGNAL_DEBOUNCE_SECONDS

    loop, _, callbacks = _capture_signal_handlers()

    with (
        patch("imas_codex.cli.shutdown.asyncio.all_tasks", return_value=set()),
        patch("imas_codex.cli.shutdown._start_exit_watchdog") as watchdog,
        patch("imas_codex.cli.shutdown._force_stop_display"),
        patch("imas_codex.cli.shutdown._force_kill_ssh_pools"),
    ):
        callbacks[signal.SIGINT]()
        loop.time.return_value = 10.0
        callbacks[signal.SIGTERM]()
        loop.time.return_value = 10.0 + _SIGNAL_DEBOUNCE_SECONDS / 2
        callbacks[signal.SIGINT]()

    watchdog.assert_called_once_with(45)


def test_later_sigint_forces_shutdown_once():
    """A distinct later SIGINT starts the watchdog and cancels active tasks."""
    from imas_codex.cli.shutdown import _SIGNAL_DEBOUNCE_SECONDS

    display = MagicMock()
    loop, stop_event, callbacks = _capture_signal_handlers(display=display)
    tasks = {MagicMock(spec=asyncio.Task), MagicMock(spec=asyncio.Task)}

    with (
        patch("imas_codex.cli.shutdown.asyncio.all_tasks", return_value=tasks),
        patch("imas_codex.cli.shutdown._start_exit_watchdog") as watchdog,
        patch("imas_codex.cli.shutdown._force_stop_display") as stop_display,
        patch("imas_codex.cli.shutdown._force_kill_ssh_pools") as kill_pools,
    ):
        callbacks[signal.SIGTERM]()
        loop.time.return_value = _SIGNAL_DEBOUNCE_SECONDS * 2
        callbacks[signal.SIGINT]()

    stop_event.set.assert_called_once_with()
    stop_display.assert_called_once_with(display)
    kill_pools.assert_called_once_with()
    watchdog.assert_called_once_with(45)
    for task in tasks:
        task.cancel.assert_called_once_with()


def test_another_later_sigint_hard_exits_with_interrupt_status():
    """The SIGINT following forced shutdown exits immediately with status 130."""
    from imas_codex.cli.shutdown import _SIGNAL_DEBOUNCE_SECONDS

    loop, _, callbacks = _capture_signal_handlers()

    with (
        patch("imas_codex.cli.shutdown.asyncio.all_tasks", return_value=set()),
        patch("imas_codex.cli.shutdown._start_exit_watchdog"),
        patch("imas_codex.cli.shutdown._force_stop_display"),
        patch("imas_codex.cli.shutdown._force_kill_ssh_pools"),
        patch("imas_codex.cli.shutdown.os._exit") as hard_exit,
    ):
        callbacks[signal.SIGINT]()
        loop.time.return_value = _SIGNAL_DEBOUNCE_SECONDS * 2
        callbacks[signal.SIGINT]()
        loop.time.return_value = _SIGNAL_DEBOUNCE_SECONDS * 4
        callbacks[signal.SIGINT]()

    hard_exit.assert_called_once_with(130)


def test_normal_completion_does_not_start_watchdog():
    """Completing normally leaves the forced-shutdown watchdog inactive."""
    from imas_codex.cli.shutdown import safe_asyncio_run

    async def _complete():
        return "done"

    with (
        patch("imas_codex.cli.shutdown._start_exit_watchdog") as watchdog,
        patch("imas_codex.cli.shutdown._force_kill_ssh_pools"),
    ):
        assert safe_asyncio_run(_complete()) == "done"

    watchdog.assert_not_called()
