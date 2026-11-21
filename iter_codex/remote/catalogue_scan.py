"""Run a resumable private metadata scan without loading graph infrastructure.

Invoke with ``uv run python -m iter_codex.remote.catalogue_scan --help``.
Endpoints, exclusions and interpreter setup are read from private configuration.
The process emits bounded progress and an atomic terminal receipt on every exit
it can handle; a time budget drains in-flight batches before a scheduler limit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from iter_codex.config.codac_uda import (
    resolve_archive_endpoint,
    validate_codac_uda_config,
)
from iter_codex.remote.archive_catalogue import ArchiveCatalogue
from iter_codex.remote.environment import resolve_remote_environment
from iter_codex.remote.script_source import load_script_source


def write_receipt(path: Path, data: dict) -> None:
    """Publish a complete receipt with a same-directory atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(data, indent=2))
    pending.replace(path)


class MetadataReaders:
    """One isolated native reader per calling thread, at most eight per scan."""

    def __init__(
        self,
        config: dict,
        *,
        timeout: float = 300,
        max_retries: int = 3,
        backoff: float = 1,
    ):
        if (
            not 0 <= max_retries <= 5
            or not 0 <= backoff <= 30
            or not 0 < timeout <= 300
        ):
            raise ValueError("metadata timeout and retry bounds are invalid")
        self.settings = validate_codac_uda_config(config["data_systems"]["codac_uda"])
        for role in self.settings.readable_roles:
            resolve_archive_endpoint(self.settings, role)
        self.environment = resolve_remote_environment(config, "codac_uda")
        self.source = load_script_source("codac_uda.py")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.local = threading.local()
        self.sessions = []
        self.lock = threading.Lock()

    def _open(self, role: str):
        endpoint = resolve_archive_endpoint(self.settings, role)
        excluded = [
            item.host for item in self.settings.archives if item.excluded and item.host
        ]
        environment = {
            **os.environ,
            "CODAC_UDA_HOST": endpoint.host,
            "CODAC_UDA_PORT": str(endpoint.port),
            "CODAC_UDA_OPERATOR_HOSTS": json.dumps(excluded),
            "CODAC_UDA_METADATA_STREAM": "1",
        }
        command = "; ".join(
            [
                *self.environment.setup_commands,
                "exec "
                + self.environment.python_command
                + " -u -c "
                + shlex.quote(self.source),
            ]
        )
        process = subprocess.Popen(
            ["bash", "-lc", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=environment,
        )
        session = {"role": role, "process": process, "buffer": bytearray()}
        with self.lock:
            self.sessions.append(session)
        self.local.session = session
        if self._reply(session).get("ready") is not True:
            raise RuntimeError("native metadata reader did not acknowledge startup")
        return session

    def _reply(self, session: dict) -> dict:
        process, buffer = session["process"], session["buffer"]
        deadline = time.monotonic() + self.timeout
        while True:
            boundary = buffer.find(b"\n")
            if boundary >= 0:
                line = bytes(buffer[:boundary]).decode(errors="replace")
                del buffer[: boundary + 1]
                if line.startswith("CODAC_METADATA "):
                    return json.loads(line[len("CODAC_METADATA ") :])
                # Native diagnostics are private job-log data, never receipts.
                print(line, file=sys.stderr, flush=True)
                continue
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([process.stdout], [], [], remaining)[0]
            ):
                raise TimeoutError("native metadata reader exceeded its batch deadline")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(
                    "native metadata reader exited before a batch receipt"
                )
            buffer.extend(chunk)

    def fetch(self, role: str, names: list[str], cached: dict) -> dict:
        """Restart and split stalled ranges within one shared retry budget.

        Exhaustion is a fetch failure, not an archive metadata refusal. Returned
        successes and failed subranges together account for the original range.
        """
        pending = deque([names])
        entries, failures = [], []
        counts = dict.fromkeys(
            (
                "timeouts",
                "restarts",
                "submitted_full_requests",
                "submitted_type_requests",
                "full_requests",
                "type_requests",
            ),
            0,
        )
        attempts = 0
        last_error = ""
        while pending:
            current = pending.popleft()
            if attempts >= self.max_retries + 1:
                for remaining in [current, *pending]:
                    failures.append(
                        {
                            "names": remaining,
                            "error": last_error,
                            "error_type": "TimeoutError",
                            "attempts": attempts,
                        }
                    )
                break
            selected_cache = {name: cached[name] for name in current if name in cached}
            attempts += 1
            counts["submitted_full_requests"] += len(current)
            counts["submitted_type_requests"] += len(current) - len(selected_cache)
            try:
                result = self._fetch_once(role, current, selected_cache)
            except TimeoutError as exc:
                last_error = str(exc)
                counts["timeouts"] += 1
                self._restart()
                counts["restarts"] += 1
                if attempts >= self.max_retries + 1:
                    for remaining in [current, *pending]:
                        failures.append(
                            {
                                "names": remaining,
                                "error": str(exc),
                                "error_type": "TimeoutError",
                                "attempts": attempts,
                            }
                        )
                    break
                half = max(1, len(current) // 2)
                parts = [current[:half]] + ([current[half:]] if current[half:] else [])
                pending.extendleft(reversed(parts))
                time.sleep(min(30, self.backoff * 2 ** (counts["timeouts"] - 1)))
            else:
                entries.extend(result["entries"])
                counts["full_requests"] += len(current)
                counts["type_requests"] += len(current) - len(selected_cache)
        return {
            "entries": entries,
            "range_failures": failures,
            "request_counts": counts,
        }

    def _restart(self) -> None:
        session = getattr(self.local, "session", None)
        self.local.session = None
        if session is None:
            return
        process = session["process"]
        print(
            f"Restarting timed-out owned metadata reader pid={process.pid}",
            file=sys.stderr,
            flush=True,
        )
        if process.poll() is None:
            process.terminate()
        try:
            self._close(session, expected_stop=True)
        finally:
            with self.lock:
                self.sessions = [item for item in self.sessions if item is not session]

    def _fetch_once(self, role: str, names: list[str], cached: dict) -> dict:
        session = getattr(self.local, "session", None)
        if session is None or session["role"] != role:
            if session is not None:
                self._close(session)
            session = self.local.session = self._open(role)
        request = {
            "operation": "metadata",
            "names": names,
            "metadata_batch_size": len(names),
            "type_documents": cached,
        }
        process = session["process"]
        process.stdin.write((json.dumps(request) + "\n").encode())
        process.stdin.flush()
        receipt = self._reply(session)
        if receipt.get("ok") is not True:
            raise RuntimeError(
                "native metadata request failed: " + str(receipt.get("error"))
            )
        return receipt["result"]

    @staticmethod
    def _close(session: dict, *, expected_stop: bool = False) -> None:
        process = session["process"]
        if process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass  # The child closed its input; still reap it and read its status.
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(
                f"Stopping owned metadata reader pid={process.pid}",
                file=sys.stderr,
                flush=True,
            )
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if process.stdout:
            process.stdout.close()
        if process.returncode and not expected_stop:
            raise RuntimeError(
                f"metadata reader exited with status {process.returncode}"
            )

    def close(self) -> None:
        errors = []
        for session in self.sessions:
            try:
                self._close(session)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                "metadata reader cleanup failed: " + str(errors[0])
            ) from errors[0]


def run_scan(
    *,
    store_path: Path,
    config: dict,
    receipt_path: Path,
    workers: int = 1,
    batch_size: int = 100,
    max_batches: int | None = None,
    max_seconds: float = 82800,
    reader_factory=MetadataReaders,
    stop: threading.Event | None = None,
) -> dict:
    """Execute one bounded job and retain enough receipt data for a later job."""
    started = time.monotonic()
    receipt = {
        "status": "running",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "started_at": datetime.now(UTC).isoformat(),
        "store_path": str(store_path),
        "workers": workers,
        "batch_size": batch_size,
        "data_reads": 0,
        "graph_operations": 0,
    }
    store = readers = lease = None
    progress_path = receipt_path.with_name(receipt_path.stem + "-progress.json")
    last_progress = 0

    def progress(stats):
        nonlocal last_progress
        now = time.monotonic()
        if now - last_progress >= 60:
            last_progress = now
            data = {
                **stats,
                "at": datetime.now(UTC).isoformat(),
                "job_id": receipt["job_id"],
            }
            write_receipt(progress_path, data)
            print(json.dumps(data), flush=True)

    try:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        lease = store_path.with_suffix(store_path.suffix + ".lock").open("a")
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another catalogue job owns this store") from exc
        readers = reader_factory(config)
        store = ArchiveCatalogue(store_path)
        census = store.census()
        if set(census["counts"]) != set(readers.settings.readable_roles):
            raise ValueError(
                "census archive roles differ from configured readable roles"
            )
        receipt["before"] = store.summary()
        write_receipt(
            receipt_path.with_name(receipt_path.stem + "-start.json"), receipt
        )
        result = store.scan(
            readers.fetch,
            workers=workers,
            batch_size=batch_size,
            max_batches=max_batches,
            max_seconds=max_seconds,
            progress=progress,
            stop=stop,
        )
        receipt.update(
            status="complete" if result["complete"] else "paused", result=result
        )
        if result["complete"]:
            repeat = store.scan(readers.fetch, workers=workers, batch_size=batch_size)
            if repeat["metadata_requests"] != 0 or repeat["rows_added"] != 0:
                raise RuntimeError("completed catalogue repeat changed the store")
            receipt["repeat"] = repeat
        receipt["integrity"] = store.db.execute("PRAGMA quick_check").fetchone()[0]
        if receipt["integrity"] != "ok":
            raise RuntimeError("private catalogue integrity check failed")
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if store is not None:
            try:
                receipt["result"] = store.summary()
            except Exception as snapshot_error:
                receipt["snapshot_error"] = str(snapshot_error)
        raise
    finally:
        cleanup_errors = []
        for resource in (readers, store, lease):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors:
            receipt["cleanup_errors"] = [str(exc) for exc in cleanup_errors]
            if receipt["status"] != "failed":
                receipt.update(
                    status="failed",
                    error_type=type(cleanup_errors[0]).__name__,
                    error=str(cleanup_errors[0]),
                )
        receipt.update(
            elapsed_seconds=time.monotonic() - started,
            completed_at=datetime.now(UTC).isoformat(),
        )
        write_receipt(receipt_path, receipt)
        if cleanup_errors and sys.exc_info()[0] is None:
            raise cleanup_errors[0]
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-seconds", type=float, default=82800)
    parser.add_argument("--batch-timeout", type=float, default=60)
    parser.add_argument("--retry-limit", type=int, choices=range(6), default=3)
    args = parser.parse_args(argv)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    try:
        import yaml

        run_scan(
            store_path=args.store,
            config=yaml.safe_load(args.config.read_text()),
            receipt_path=args.receipt,
            workers=args.workers,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            max_seconds=args.max_seconds,
            stop=stop,
            reader_factory=lambda config: MetadataReaders(
                config, timeout=args.batch_timeout, max_retries=args.retry_limit
            ),
        )
        return 0
    except Exception as exc:
        if not args.receipt.exists():
            write_receipt(
                args.receipt,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
