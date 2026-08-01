"""Graceful shutdown for discovery CLI commands.

Provides cooperative shutdown for both SIGINT (Ctrl+C) and SIGTERM
with rich shutdown progress tracking:

  First SIGINT/SIGTERM: Signal workers to stop, switch display to
                        shutdown mode showing per-group drain progress.
                        Workers finish their current batch and exit
                        cleanly.
  Later SIGINT:         Stops Rich display, cancels all async tasks.
                        Starts a 45s watchdog to outlast drain +
                        finalize timeouts.
  Another later SIGINT: Immediate process exit (os._exit).

SIGTERM is cooperative (same as first SIGINT).  Force-kill
requires ``kill -9`` (SIGKILL).

Usage in discovery CLIs::

    from imas_codex.cli.shutdown import install_shutdown_handlers, safe_asyncio_run

    with MyProgressDisplay(...) as display:
        async def run_with_display():
            stop_event = asyncio.Event()
            install_shutdown_handlers(
                stop_event=stop_event,
                display=display,  # BaseProgressDisplay subclass
            )
            try:
                return await run_parallel_discovery(
                    ..., stop_event=stop_event,
                )
            finally:
                ...  # cancel refresh/ticker tasks

        result = safe_asyncio_run(run_with_display())
        display.print_summary()  # display still alive here

The stop_event is wired into the discovery state's should_stop() check
by each parallel runner via a watcher task.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


# Default timeout for the executor shutdown after asyncio.run() completes.
# Threads blocked on SSH subprocesses or LLM HTTP calls may outlive the
# event loop; a short timeout prevents the process from hanging.
_EXECUTOR_SHUTDOWN_TIMEOUT = 5

# PTYs and process launchers can deliver the same operator interrupt through
# more than one signal path.  Treat callbacks within this interval as one
# delivery so a single Ctrl+C cannot immediately force shutdown.
_SIGNAL_DEBOUNCE_SECONDS = 0.1


def safe_asyncio_run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine, suppressing 'Event loop is closed' on cleanup.

    ``asyncio.run()`` closes the event loop after the coroutine returns.
    If any ``asyncio.create_subprocess_exec()``-spawned transports are
    garbage-collected *after* the loop is closed, their ``__del__``
    methods raise ``RuntimeError: Event loop is closed`` — an ugly but
    harmless traceback on stderr.

    This wrapper installs a temporary ``sys.unraisablehook`` that
    silences only that specific error, then restores the original hook.

    It also ensures the default executor is shut down with a short
    timeout so that leaked threads (from ``asyncio.to_thread()`` calls
    to SSH subprocesses or LLM HTTP calls) do not prevent the process
    from exiting.
    """
    old_hook = sys.unraisablehook

    def _suppress_closed_loop(unraisable: sys.UnraisableHookArgs) -> None:
        if (
            isinstance(unraisable.exc_value, RuntimeError)
            and str(unraisable.exc_value) == "Event loop is closed"
        ):
            return
        old_hook(unraisable)

    sys.unraisablehook = _suppress_closed_loop
    try:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(coro)
        finally:
            try:
                # Cancel any straggling tasks (with timeout so we
                # don't hang on threads blocked in to_thread)
                _cancel_remaining_tasks(loop)
            finally:
                # Shut down the executor with a short timeout so leaked
                # threads from to_thread() don't block exit.
                loop.run_until_complete(
                    loop.shutdown_default_executor(timeout=_EXECUTOR_SHUTDOWN_TIMEOUT)
                )
                # Force-kill SSH subprocess pools to unblock any
                # remaining executor threads waiting on SSH I/O, so
                # the thread.join() in Python's atexit handler returns.
                _force_kill_ssh_pools()
                loop.close()

        return result
    finally:
        sys.unraisablehook = old_hook


def _start_exit_watchdog(grace_seconds: float) -> None:
    """Start a daemon thread that force-exits after a grace period.

    Python's ``concurrent.futures.thread._python_exit`` atexit handler
    joins all executor threads.  If an SSH subprocess is still running,
    ``thread.join()`` blocks forever.  This watchdog ensures the process
    exits cleanly after the CLI has printed its output.
    """

    def _watchdog() -> None:
        threading.Event().wait(timeout=grace_seconds)
        # If we're still alive, flush and hard-exit.  Call os._exit FIRST
        # before _force_kill_ssh_pools which could deadlock on import lock.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(130)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()


# Timeout for _cancel_remaining_tasks — must be shorter than the
# exit watchdog grace period so we proceed to executor shutdown
# rather than hanging on unkillable to_thread tasks.
_CANCEL_TASKS_TIMEOUT = 5


def _cancel_remaining_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel remaining tasks on the loop with a timeout.

    Tasks stuck in ``asyncio.to_thread()`` cannot be cancelled until
    the underlying thread finishes.  A bounded ``asyncio.wait()``
    prevents the process from hanging indefinitely.
    """
    tasks = asyncio.all_tasks(loop)
    if not tasks:
        return
    for task in tasks:
        task.cancel()

    async def _wait() -> None:
        _, still_pending = await asyncio.wait(tasks, timeout=_CANCEL_TASKS_TIMEOUT)
        if still_pending:
            logger.warning(
                "%d task(s) still pending after %ss cancel timeout "
                "— forcing SSH pool shutdown",
                len(still_pending),
                _CANCEL_TASKS_TIMEOUT,
            )
            _force_kill_ssh_pools()

    loop.run_until_complete(_wait())


def install_shutdown_handlers(
    *,
    stop_event: asyncio.Event,
    display: object | None = None,
) -> None:
    """Install SIGINT and SIGTERM handlers on the running asyncio event loop.

    Replaces asyncio's default SIGINT handling (which raises
    KeyboardInterrupt and requires multiple presses) with a
    cooperative shutdown:

    1. First SIGINT/SIGTERM: Sets stop_event, switches the progress
       display to shutdown mode (yellow border, live worker-drain
       tracker).  Workers finish their current batch and exit.
    2. A later SIGINT: Stops Rich Live display, cancels all async tasks
       so the coroutine chain unwinds.  Starts a 45s watchdog to
       outlast drain_pending (30s) + finalize_sn_run (10s) + buffer.
    3. Another later SIGINT: ``os._exit(130)`` (hard exit).

    SIGTERM triggers the same cooperative shutdown as the first SIGINT.
    Further SIGTERM deliveries are idempotent; only later SIGINT deliveries
    advance to forced shutdown and hard exit.

    Args:
        stop_event: asyncio.Event that parallel runners watch to
            set ``state.stop_requested = True``.
        display: Optional BaseProgressDisplay instance.  Switched to
            shutdown mode on the graceful transition and stopped on
            the forced transition.
    """
    loop = asyncio.get_running_loop()
    shutdown_level = 0
    last_accepted_signal_at: float | None = None

    def _handle_shutdown_signal(signal_kind: signal.Signals) -> None:
        nonlocal last_accepted_signal_at, shutdown_level
        now = loop.time()
        if (
            last_accepted_signal_at is not None
            and now - last_accepted_signal_at < _SIGNAL_DEBOUNCE_SECONDS
        ):
            return

        if shutdown_level == 0:
            last_accepted_signal_at = now
            shutdown_level = 1
            logger.info("Graceful shutdown requested (%s)", signal_kind.name)
            stop_event.set()
            # Switch display to shutdown mode
            if display is not None and hasattr(display, "begin_shutdown"):
                try:
                    display.begin_shutdown()
                except Exception:
                    pass
            return

        # SIGTERM remains cooperative no matter how many times it is delivered.
        # It is not accepted as a state transition and therefore cannot extend
        # the debounce window for a subsequent SIGINT.
        if signal_kind == signal.SIGTERM:
            return

        last_accepted_signal_at = now
        if shutdown_level == 1:
            shutdown_level = 2
            logger.warning("Forced shutdown (later Ctrl+C)")
            _force_stop_display(display)
            # Force-kill SSH worker pools so leaked threads don't
            # block process exit.
            _force_kill_ssh_pools()
            # Start exit watchdog NOW — must outlast DRAIN_TIMEOUT (30s)
            # + FINALIZE_TIMEOUT (10s) + buffer so finalize_sn_run lands.
            _start_exit_watchdog(45)
            # Cancel all running tasks from the event loop so the
            # coroutine chain unwinds.
            for task in asyncio.all_tasks(loop):
                task.cancel()
        else:
            # Hard exit -- reached when graceful cancel didn't work
            os._exit(130)

    def _handle_sigint() -> None:
        _handle_shutdown_signal(signal.SIGINT)

    def _handle_sigterm() -> None:
        _handle_shutdown_signal(signal.SIGTERM)

    signal_callbacks = {
        signal.SIGINT: _handle_sigint,
        signal.SIGTERM: _handle_sigterm,
    }
    previous_process_handlers = {
        signal_kind: signal.getsignal(signal_kind) for signal_kind in signal_callbacks
    }
    registered_signals: list[signal.Signals] = []
    try:
        for signal_kind, callback in signal_callbacks.items():
            loop.add_signal_handler(signal_kind, callback)
            registered_signals.append(signal_kind)
    except (NotImplementedError, OSError, RuntimeError, ValueError) as exc:
        cleanup_succeeded = True
        for signal_kind in registered_signals:
            try:
                removed = loop.remove_signal_handler(signal_kind)
                if not removed:
                    cleanup_succeeded = False
                    continue
                signal.signal(signal_kind, previous_process_handlers[signal_kind])
            except (NotImplementedError, OSError, RuntimeError, ValueError):
                cleanup_succeeded = False
        if not cleanup_succeeded:
            logger.warning(
                "Asyncio signal registration failed and could not be rolled back; "
                "signal fallback was not installed",
                exc_info=exc,
            )
            return
        _install_signal_fallback(loop, signal_callbacks, exc)


def _install_signal_fallback(
    loop: asyncio.AbstractEventLoop,
    callbacks: dict[signal.Signals, Callable[[], None]],
    registration_error: Exception,
) -> None:
    """Install task-scoped ``signal.signal`` wrappers when asyncio cannot."""
    try:
        owner_task = asyncio.current_task(loop=loop)
    except RuntimeError:
        owner_task = None
    if owner_task is None:
        logger.warning(
            "Asyncio signal registration failed and no task owns the handler "
            "lifecycle; existing process handlers were preserved",
            exc_info=registration_error,
        )
        return

    previous_handlers: dict[signal.Signals, signal.Handlers] = {}
    restored = False

    def _restore_handlers(_completed_task: object | None = None) -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        for signal_kind, previous_handler in previous_handlers.items():
            try:
                signal.signal(signal_kind, previous_handler)
            except (OSError, RuntimeError, ValueError):
                logger.warning(
                    "Could not restore the previous %s handler",
                    signal_kind.name,
                    exc_info=True,
                )

    try:
        for signal_kind, callback in callbacks.items():

            def _marshal_to_loop(
                _signum: int,
                _frame: object,
                *,
                callback: Callable[[], None] = callback,
            ) -> None:
                try:
                    loop.call_soon_threadsafe(callback)
                except RuntimeError:
                    logger.warning("Ignoring shutdown signal after event loop closure")

            previous_handlers[signal_kind] = signal.signal(
                signal_kind, _marshal_to_loop
            )
        owner_task.add_done_callback(_restore_handlers)
    except (OSError, RuntimeError, ValueError):
        _restore_handlers()
        logger.warning(
            "Process signal fallback could not be installed; existing handlers "
            "were preserved",
            exc_info=True,
        )


def _force_stop_display(display: object | None) -> None:
    """Stop Rich display immediately so terminal is usable."""
    live = getattr(display, "_live", None) if display else None
    if live is not None and getattr(live, "is_started", False):
        try:
            live.stop()
        except Exception:
            pass


def _force_kill_ssh_pools() -> None:
    """Synchronously force-kill all SSH worker pools.

    Called on forced shutdown to ensure leaked threads from
    ``asyncio.to_thread()`` SSH subprocess calls don't block
    process exit.
    """
    try:
        from imas_codex.remote.ssh_worker import force_kill_all_pools

        force_kill_all_pools()
    except Exception:
        pass


async def watch_stop_event(
    stop_event: asyncio.Event,
    state: object,
) -> None:
    """Watch a stop event and set state.stop_requested when triggered.

    Spawned as a task inside each parallel runner to bridge the CLI
    signal handler (which sets the event) to the discovery state
    (which workers poll via should_stop()).

    Args:
        stop_event: Event set by the shutdown signal handler.
        state: Discovery state object with a ``stop_requested`` attribute.
    """
    await stop_event.wait()
    state.stop_requested = True
    logger.info("Stop event received -- workers will finish current batch")


def force_exit() -> None:
    """Force process exit, cleaning up SSH resources.

    Called after all CLI output has been printed to prevent the
    process from hanging on leaked executor threads (non-daemon
    threads spawned by ``asyncio.to_thread()`` for SSH subprocess
    calls). Python's atexit handler waits for ALL threads to
    ``join()`` which blocks indefinitely if a subprocess is still
    running on a remote host.

    This function:
    1. Force-kills all SSH worker pool subprocesses
    2. Flushes stdout/stderr
    3. Calls ``os._exit(0)`` to bypass atexit thread joins
    """
    _force_kill_ssh_pools()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)
