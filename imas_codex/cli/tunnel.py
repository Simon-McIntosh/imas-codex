"""Unified SSH tunnel management CLI.

Manages SSH tunnels to all remote services (Neo4j graph, embedding server)
from a single top-level command group.  Uses ``autossh`` when available for
automatic reconnection, otherwise falls back to plain ``ssh -f -N``.

All requested port forwards are batched into **one SSH connection** per host
(single ProxyJump, single auth).

Persistent tunnels are managed via systemd user services with autossh,
providing automatic reconnection after network interruptions, VPN drops,
or SSH keepalive timeouts.

Usage::

    imas-codex tunnel start [HOST]           # All services
    imas-codex tunnel start [HOST] --neo4j   # Just graph ports
    imas-codex tunnel start [HOST] --embed   # Just embed port
    imas-codex tunnel start [HOST] --llm     # Just LLM proxy port
    imas-codex tunnel start [HOST] --docs    # Just docs-server port
    imas-codex tunnel stop [HOST]
    imas-codex tunnel status
    imas-codex tunnel service install [HOST]  # Persistent autossh via systemd
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import click

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ============================================================================
# Helpers
# ============================================================================


def _record_tunnel_pid(host: str, first_local_port: int) -> None:
    """Find and record the PID of the autossh/ssh process we just started."""
    from imas_codex.remote.tunnel import _find_and_record_pid

    _find_and_record_pid(host, first_local_port)


def _discover_compute_node(host: str) -> str | None:
    """SSH to host and discover the SLURM compute node for imas-codex services.

    Delegates to the shared ``discover_compute_node`` in the tunnel module.
    """
    from imas_codex.remote.tunnel import discover_compute_node

    return discover_compute_node(host)


def _discover_vllm_node(host: str) -> str | None:
    """Discover the SLURM node running an ambix-serve vLLM job.

    SSHes to *host* and queries ``squeue`` for running ``ambix-serve-*``
    jobs, returning the first compute node found.
    """
    try:
        result = subprocess.run(
            [
                "ssh",
                host,
                "-o",
                "ConnectTimeout=10",
                'squeue -u "$USER" -t R -o "%N %j" --noheader 2>/dev/null'
                " | grep 'ambix-serve-' | head -1 | awk '{print $1}'",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            node = result.stdout.strip()
            if node:
                return node
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _resolve_host(host: str | None) -> str:
    """Resolve HOST from argument, graph profile, or fail."""
    if host:
        return host

    try:
        from imas_codex.graph.profiles import resolve_neo4j

        profile = resolve_neo4j(auto_tunnel=False)
        if profile.host:
            return profile.host
    except Exception:
        pass

    raise click.ClickException(
        "No HOST specified and the active graph profile has no remote host.\n"
        "Provide a host: imas-codex tunnel start <host>"
    )


def _get_tunnel_ports(
    host: str,
    neo4j: bool,
    embed: bool,
    llm: bool = False,
    vllm: bool = False,
    docs: bool = False,
    ink: bool = False,
    clipboard: bool = False,
    *,
    emit_status: bool = True,
) -> list[tuple[int, int, str, str, str]]:
    """Return list of (remote_port, local_port, label, bind_addr, direction) for requested services.

    ``direction`` is ``"L"`` for a forward tunnel (``-L local:bind:remote``) or
    ``"R"`` for a reverse tunnel (``-R remote:bind:local``).  For forward tunnels
    ``bind_addr`` is the host on the remote side; for reverse tunnels it is the
    address on the local (SSH-client) side that the remote port is forwarded to.

    When no service flag is given, returns all known ports.
    For SLURM-backed services, ``bind_addr`` is set to the compute node hostname.
    """
    from imas_codex.remote.tunnel import TUNNEL_OFFSET

    ports: list[tuple[int, int, str, str, str]] = []
    all_services = (
        not neo4j
        and not embed
        and not llm
        and not vllm
        and not docs
        and not ink
        and not clipboard
    )

    # Discover SLURM compute node if any service uses a compute location.
    # All services share the same allocation, so one lookup suffices.
    compute_node: str | None = None
    try:
        from imas_codex.graph.profiles import get_graph_location
        from imas_codex.remote.locations import resolve_location
        from imas_codex.settings import get_embedding_location, get_llm_location

        locations = [get_graph_location(), get_embedding_location(), get_llm_location()]
        if any(resolve_location(loc).scheduler == "slurm" for loc in locations):
            compute_node = _discover_compute_node(host)
            if compute_node and emit_status:
                click.echo(f"  SLURM compute node: {compute_node}")
    except Exception:
        pass

    if neo4j or all_services:
        try:
            from imas_codex.graph.profiles import resolve_neo4j

            profile = resolve_neo4j(auto_tunnel=False)
            bind = compute_node or "127.0.0.1"
            ports.append(
                (
                    profile.bolt_port,
                    profile.bolt_port + TUNNEL_OFFSET,
                    f"neo4j-bolt ({profile.name})",
                    bind,
                    "L",
                )
            )
            ports.append(
                (
                    profile.http_port,
                    profile.http_port + TUNNEL_OFFSET,
                    f"neo4j-http ({profile.name})",
                    bind,
                    "L",
                )
            )
        except Exception:
            # Fallback: use base ports from convention constants
            from imas_codex.graph.profiles import BOLT_BASE_PORT, HTTP_BASE_PORT

            bind = compute_node or "127.0.0.1"
            ports.append(
                (
                    BOLT_BASE_PORT,
                    BOLT_BASE_PORT + TUNNEL_OFFSET,
                    "neo4j-bolt",
                    bind,
                    "L",
                )
            )
            ports.append(
                (
                    HTTP_BASE_PORT,
                    HTTP_BASE_PORT + TUNNEL_OFFSET,
                    "neo4j-http",
                    bind,
                    "L",
                )
            )

    if embed or all_services:
        from imas_codex.settings import get_embed_server_port

        embed_port = get_embed_server_port()
        # When a SLURM compute node is active, forward through it.
        # Otherwise fall back to localhost (embed on login node).
        remote_bind = compute_node or "127.0.0.1"
        # Embed uses same-port forwarding (no offset)
        ports.append((embed_port, embed_port, "embed", remote_bind, "L"))

    if llm or all_services:
        from imas_codex.settings import get_llm_proxy_port

        llm_port = get_llm_proxy_port()
        # LLM proxy runs on login node (needs outbound internet)
        ports.append((llm_port, llm_port, "llm", "127.0.0.1", "L"))

    if vllm or all_services:
        from imas_codex.settings import get_vllm_port

        vllm_port = get_vllm_port()
        vllm_node = _discover_vllm_node(host)
        if vllm_node:
            ports.append((vllm_port, vllm_port, "vllm", vllm_node, "L"))
        elif vllm and not all_services:
            # Explicit --vllm with no running job — fail clearly
            click.echo(
                "⚠ No running ambix-serve SLURM job found. "
                "Start one with: ambix agent serve"
            )

    if docs or all_services:
        from imas_codex.settings import get_docs_server_port

        docs_port = get_docs_server_port()
        # docs-server runs on the login node (~/docs-server/serve.py)
        ports.append((docs_port, docs_port, "docs", "127.0.0.1", "L"))

    if ink or all_services:
        from imas_codex.settings import get_ink_display_port

        ink_port = get_ink_display_port()
        # ink display server runs on the login node (uv run efit-ink)
        ports.append((ink_port, ink_port, "ink", "127.0.0.1", "L"))

    if clipboard or all_services:
        from imas_codex.settings import get_wsl_clip_port

        clip_port = get_wsl_clip_port()
        # wsl-clip-server: image + text clipboard HTTP relay on WSL.
        # paste-img on iter POSTs the path back via /copy to return it to Windows clipboard.
        ports.append((clip_port, clip_port, "wsl-clip", "localhost", "R"))

    return ports


def _service_name(host: str) -> str:
    return f"imas-codex-tunnel-{host}"


def _service_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def _service_file(host: str) -> Path:
    return _service_dir() / f"{_service_name(host)}.service"


def _requested_services(
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool,
    vllm_only: bool = False,
    docs_only: bool = False,
    ink_only: bool = False,
    clipboard_only: bool = False,
) -> set[str]:
    if (
        not neo4j_only
        and not embed_only
        and not llm_only
        and not vllm_only
        and not docs_only
        and not ink_only
        and not clipboard_only
    ):
        return {"neo4j", "embed", "llm", "vllm", "docs", "ink", "wsl-clip"}
    selected: set[str] = set()
    if neo4j_only:
        selected.add("neo4j")
    if embed_only:
        selected.add("embed")
    if llm_only:
        selected.add("llm")
    if vllm_only:
        selected.add("vllm")
    if docs_only:
        selected.add("docs")
    if ink_only:
        selected.add("ink")
    if clipboard_only:
        selected.add("wsl-clip")
    return selected


# Inline manifest comment that lists the services a systemd unit forwards.
# Written by _build_systemd_service_content; read by _service_selected_services.
# Lets the CLI detect "supports request" precisely instead of inferring from
# --X flag presence — older units (installed before a new service was added)
# need explicit reinstall to claim the new service, so the CLI no longer
# silently starts a stale unit when the user asks for a newly-added service.
SERVICE_MANIFEST_PREFIX = "# tunnel-services: "

# Service set tracked by units installed before SERVICE_MANIFEST_PREFIX
# existed. New services added after this list ship MUST trigger a service
# reinstall — they are intentionally absent from this frozen fallback so
# `_installed_service_supports_request` returns False for them.
_LEGACY_SERVICE_SET: frozenset[str] = frozenset({"neo4j", "embed", "llm", "vllm"})


def _service_selected_services(service_text: str) -> set[str]:
    # Prefer the manifest comment when present (units installed by current
    # or future CLI versions).
    for line in service_text.splitlines():
        if line.startswith(SERVICE_MANIFEST_PREFIX):
            return set(line[len(SERVICE_MANIFEST_PREFIX) :].split())
    # Fallback for legacy units: detect explicit flags in ExecStart, or
    # assume the pre-manifest service set if the unit was installed
    # flagless ("all services as of install time").
    selected = {
        service
        for flag, service in (
            ("--neo4j", "neo4j"),
            ("--embed", "embed"),
            ("--llm", "llm"),
            ("--vllm", "vllm"),
            ("--docs", "docs"),
            ("--ink", "ink"),
            ("--clipboard", "wsl-clip"),
        )
        if flag in service_text
    }
    return selected or set(_LEGACY_SERVICE_SET)


def _installed_service_supports_request(
    host: str,
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool,
    vllm_only: bool = False,
    docs_only: bool = False,
    ink_only: bool = False,
    clipboard_only: bool = False,
) -> bool:
    service_file = _service_file(host)
    if not service_file.exists():
        return False
    try:
        service_text = service_file.read_text()
    except OSError:
        return False
    requested = _requested_services(
        neo4j_only, embed_only, llm_only, vllm_only, docs_only, ink_only, clipboard_only
    )
    installed = _service_selected_services(service_text)
    return requested.issubset(installed)


def _run_systemctl_user(
    args: list[str], *, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=check,
        capture_output=not check,
        text=True,
    )


def _build_foreground_tunnel_command(
    host: str,
    ports: list[tuple[int, int, str, str, str]],
) -> tuple[list[str], dict[str, str]]:
    autossh = shutil.which("autossh")
    if not autossh:
        raise click.ClickException(
            "autossh not found. Install with: sudo apt install autossh"
        )

    from imas_codex.remote.tunnel import SSH_TUNNEL_OPTS, local_forward_spec

    forward_args: list[str] = []
    for remote_port, local_port, _label, bind_addr, direction in ports:
        if direction == "R":
            # Reverse tunnel: bind remote_port on the SSH server (iter),
            # forward connections back to bind_addr:local_port on the client.
            forward_args.extend(["-R", f"{remote_port}:{bind_addr}:{local_port}"])
        else:
            # Forward tunnel (default): expose remote service locally.
            forward_args.extend(
                ["-L", local_forward_spec(local_port, bind_addr, remote_port)]
            )

    cmd = [
        autossh,
        "-M",
        "0",
        "-N",
        *SSH_TUNNEL_OPTS,
        *forward_args,
        host,
    ]
    env = {
        **os.environ,
        "AUTOSSH_GATETIME": "0",
        "AUTOSSH_POLL": "30",
        "AUTOSSH_PORT": "0",
    }
    return cmd, env


def _terminate_tunnel_process(child: subprocess.Popen | None) -> None:
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            # A task in uninterruptible kernel sleep cannot be reaped. Do not
            # wedge the supervisor too; leave it orphaned and start a healthy
            # replacement when possible.
            click.echo(f"Tunnel process {child.pid} could not be reaped; continuing")
    except ProcessLookupError:
        pass


def _run_service_supervisor(
    host: str,
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool,
    vllm_only: bool = False,
    docs_only: bool = False,
    ink_only: bool = False,
    clipboard_only: bool = False,
) -> None:
    from imas_codex.remote.tunnel import is_tunnel_active

    stop_requested = False
    child: subprocess.Popen | None = None
    current_signature: tuple[tuple[int, int, str], ...] | None = None

    def _handle_signal(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not stop_requested:
            ports = _get_tunnel_ports(
                host,
                neo4j_only,
                embed_only,
                llm_only,
                vllm_only,
                docs_only,
                ink_only,
                clipboard_only,
                emit_status=False,
            )
            if not ports:
                raise click.ClickException("No services selected for tunneling.")

            signature = tuple(
                (remote_port, local_port, bind_addr)
                for remote_port, local_port, _label, bind_addr, _dir in ports
            )

            if (
                child is None
                or child.poll() is not None
                or signature != current_signature
            ):
                if current_signature is None:
                    click.echo(f"Starting persistent tunnel supervisor for {host}")
                elif signature != current_signature:
                    click.echo(f"Tunnel targets changed for {host}; restarting autossh")
                else:
                    click.echo(f"autossh exited for {host}; restarting")

                _terminate_tunnel_process(child)
                cmd, env = _build_foreground_tunnel_command(host, ports)
                child = subprocess.Popen(
                    cmd,
                    env=env,
                    start_new_session=True,
                )
                current_signature = signature

            for _ in range(15):
                if stop_requested:
                    break
                time.sleep(2)
                if child is None or child.poll() is not None:
                    break
                missing_ports = [
                    local_port
                    for _remote, local_port, _label, _bind, direction in ports
                    if direction == "L" and not is_tunnel_active(local_port)
                ]
                if missing_ports:
                    click.echo(
                        f"Tunnel listeners missing for {host}: "
                        + ", ".join(str(port) for port in missing_ports)
                        + "; restarting autossh"
                    )
                    _terminate_tunnel_process(child)
                    child = None
                    break
                latest_ports = _get_tunnel_ports(
                    host,
                    neo4j_only,
                    embed_only,
                    llm_only,
                    vllm_only,
                    docs_only,
                    ink_only,
                    clipboard_only,
                    emit_status=False,
                )
                latest_signature = tuple(
                    (remote_port, local_port, bind_addr)
                    for remote_port, local_port, _label, bind_addr, _dir in latest_ports
                )
                if latest_signature != current_signature:
                    break
    finally:
        _terminate_tunnel_process(child)


def _build_systemd_service_content(
    host: str,
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool,
    vllm_only: bool = False,
    docs_only: bool = False,
    ink_only: bool = False,
    clipboard_only: bool = False,
) -> str:
    uv = shutil.which("uv")
    if not uv:
        raise click.ClickException("uv not found on PATH")

    flag_args = []
    if neo4j_only:
        flag_args.append("--neo4j")
    if embed_only:
        flag_args.append("--embed")
    if llm_only:
        flag_args.append("--llm")
    if vllm_only:
        flag_args.append("--vllm")
    if docs_only:
        flag_args.append("--docs")
    if ink_only:
        flag_args.append("--ink")
    if clipboard_only:
        flag_args.append("--clipboard")
    flags = " ".join(flag_args)

    services = _requested_services(
        neo4j_only, embed_only, llm_only, vllm_only, docs_only, ink_only, clipboard_only
    )
    manifest_line = SERVICE_MANIFEST_PREFIX + " ".join(sorted(services))

    log_dir = Path.home() / ".local" / "share" / "imas-codex" / "logs"
    autossh_log = log_dir / f"autossh-{host}.log"

    exec_start = (
        f"{uv} run --project {_REPO_ROOT} imas-codex tunnel service-run {host}"
        + (f" {flags}" if flags else "")
    )

    return f"""\
{manifest_line}
[Unit]
Description=IMAS Codex SSH tunnels to {host}
Documentation=https://github.com/iterorganization/imas-codex
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=10

[Service]
Type=simple
WorkingDirectory={_REPO_ROOT}

Environment=\"AUTOSSH_GATETIME=0\"
Environment=\"AUTOSSH_POLL=30\"
Environment=\"AUTOSSH_LOGFILE={autossh_log}\"
Environment=\"AUTOSSH_PORT=0\"
Environment=\"HOME={Path.home()}\"

ExecStart={exec_start}

Restart=always
RestartSec=5
KillMode=control-group
TimeoutStopSec=15

[Install]
WantedBy=default.target
"""


def _start_tunnels(
    host: str,
    ports: list[tuple[int, int, str, str, str]],
    use_autossh: bool,
) -> int:
    """Start port-forward tunnels in a single SSH connection.

    All requested forwards are passed as ``-L`` or ``-R`` flags to one
    ``autossh``/``ssh`` process so there is only one ProxyJump negotiation.
    Forward tunnels (``-L``) whose local port is already bound are skipped.
    Reverse tunnels (``-R``) are always included (cannot check remote bind).

    Returns the number of active tunnels (including pre-existing ones).
    """
    import time

    from imas_codex.remote.tunnel import is_tunnel_active

    already_active: list[tuple[int, str]] = []
    to_start: list[tuple[int, int, str, str, str]] = []

    for remote_port, local_port, label, bind_addr, direction in ports:
        if direction == "L" and is_tunnel_active(local_port):
            already_active.append((local_port, label))
        else:
            to_start.append((remote_port, local_port, label, bind_addr, direction))

    for local_port, label in already_active:
        click.echo(f"  {label}: already active (localhost:{local_port})")

    if not to_start:
        return len(already_active)

    from imas_codex.remote.tunnel import local_forward_spec

    # Build -L / -R flags for all ports that need tunneling
    forward_args: list[str] = []
    for remote_port, local_port, _label, bind_addr, direction in to_start:
        if direction == "R":
            forward_args.extend(["-R", f"{remote_port}:{bind_addr}:{local_port}"])
        else:
            forward_args.extend(
                ["-L", local_forward_spec(local_port, bind_addr, remote_port)]
            )

    # Use shared SSH options from tunnel module (keepalives, no ControlMaster)
    from imas_codex.remote.tunnel import SSH_TUNNEL_OPTS

    if use_autossh:
        cmd = [
            "autossh", "-M", "0", "-f", "-N",
            *SSH_TUNNEL_OPTS, *forward_args, host,
        ]  # fmt: skip
        env = {
            **os.environ,
            "AUTOSSH_GATETIME": "0",
            "AUTOSSH_POLL": "30",
        }
    else:
        cmd = [
            "ssh", "-f", "-N",
            *SSH_TUNNEL_OPTS,
            *forward_args,
            host,
        ]  # fmt: skip
        env = None

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else str(e)
        click.echo(f"  FAILED: {stderr}")
        return len(already_active)
    except subprocess.TimeoutExpired:
        click.echo("  TIMEOUT waiting for SSH connection")
        return len(already_active)

    # Record the autossh/ssh PID for targeted cleanup later.
    _record_tunnel_pid(host, to_start[0][1])

    # Wait for forward tunnels to bind. Reverse tunnels bind on the remote
    # side so we can't check them locally; count them as successful.
    time.sleep(0.5)
    new_ok = 0
    for _attempt in range(5):
        new_ok = 0
        all_up = True
        for _remote_port, local_port, _label, _bind_addr, direction in to_start:
            if direction == "R" or is_tunnel_active(local_port):
                new_ok += 1
            else:
                all_up = False
        if all_up:
            break
        time.sleep(1.0)

    for remote_port, local_port, label, bind_addr, direction in to_start:
        if direction == "R":
            click.echo(
                f"  {label}: {host}:{remote_port} ← {bind_addr}:{local_port} (reverse)"
            )
        elif is_tunnel_active(local_port):
            click.echo(
                f"  {label}: localhost:{local_port} → {host}→{bind_addr}:{remote_port}"
            )
        else:
            click.echo(f"  {label}: FAILED (port {local_port} not reachable)")

    return len(already_active) + new_ok


# ============================================================================
# Keyring D-Bus Forwarding
# ============================================================================


def _start_keyring_forward(host: str) -> None:
    """Forward the D-Bus socket from host for keyring access.

    Sets up a unix socket forward from ``host:/run/user/<uid>/bus``
    to a local socket, enabling SecretService keyring access on
    nodes without their own gnome-keyring-daemon.
    """
    from imas_codex.discovery.wiki.auth import (
        _dbus_forward_socket,
        _forward_dbus_socket,
    )

    if _dbus_forward_socket:
        click.echo(f"  keyring: already forwarded ({_dbus_forward_socket})")
        return

    click.echo(f"Forwarding D-Bus socket from {host}...")
    if _forward_dbus_socket(host):
        click.echo(f"✓ Keyring D-Bus forwarded via {host}")
        click.echo(f"  Socket: {_dbus_forward_socket}")

        # Verify keyring works
        try:
            import keyring

            backend = keyring.get_keyring()
            backend_module = type(backend).__module__ or ""
            if "fail" not in backend_module and "null" not in backend_module:
                click.echo(f"  Backend: {type(backend).__name__}")
            else:
                click.echo("  ⚠ Keyring backend still not functional after forwarding")
        except Exception as e:
            click.echo(f"  ⚠ Keyring check failed: {e}")
    else:
        click.echo("✗ D-Bus forwarding failed")
        click.echo(f"  Verify SSH access: ssh -o BatchMode=yes {host} echo ok")
        raise SystemExit(1)


def _get_dbus_forward_status() -> str | None:
    """Check if a D-Bus forward socket exists and return its path."""
    import os
    from pathlib import Path

    uid = os.getuid()
    candidates = [
        Path(f"/run/user/{uid}/dbus-forward.sock"),
        Path(f"/tmp/dbus-forward-{uid}.sock"),  # noqa: S108
    ]
    for sock in candidates:
        if sock.exists():
            return str(sock)
    return None


# ============================================================================
# Command Group
# ============================================================================


@click.group()
def tunnel() -> None:
    """Manage SSH tunnels to remote services.

    Forward remote Neo4j and embedding server ports to localhost
    for transparent access from your workstation.

    \b
      imas-codex tunnel start [HOST]           Start tunnels (all services)
      imas-codex tunnel start HOST --neo4j     Just graph ports
      imas-codex tunnel start HOST --embed     Just embedding port
      imas-codex tunnel start HOST --llm       Just LLM proxy port
      imas-codex tunnel start HOST --vllm      Just vLLM inference port
      imas-codex tunnel start HOST --docs      Just docs-server port
      imas-codex tunnel start HOST --ink       Just ink display port
      imas-codex tunnel start HOST --clipboard  Reverse clipboard tunnel (WSL→Windows)
      imas-codex tunnel start HOST --keyring   D-Bus socket for keyring
      imas-codex tunnel stop [HOST]            Stop tunnels
      imas-codex tunnel status                 Show active tunnels
      imas-codex tunnel keyring HOST           Forward keyring D-Bus socket
      imas-codex tunnel service install        Persistent autossh via systemd
    """
    pass


@tunnel.command("start")
@click.argument("host", required=False)
@click.option("--neo4j", "neo4j_only", is_flag=True, help="Tunnel Neo4j ports only")
@click.option("--embed", "embed_only", is_flag=True, help="Tunnel embedding port only")
@click.option("--llm", "llm_only", is_flag=True, help="Tunnel LLM proxy port only")
@click.option(
    "--vllm", "vllm_only", is_flag=True, help="Tunnel vLLM inference port only"
)
@click.option("--docs", "docs_only", is_flag=True, help="Tunnel docs-server port only")
@click.option("--ink", "ink_only", is_flag=True, help="Tunnel ink display port only")
@click.option(
    "--clipboard",
    "clipboard_only",
    is_flag=True,
    help="Reverse-tunnel clipboard port (wsl-clip-server on WSL → Windows)",
)
@click.option(
    "--keyring",
    "keyring_only",
    is_flag=True,
    help="Forward D-Bus socket for keyring access (SLURM compute nodes)",
)
def tunnel_start(
    host: str | None,
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool = False,
    vllm_only: bool = False,
    docs_only: bool = False,
    ink_only: bool = False,
    clipboard_only: bool = False,
    keyring_only: bool = False,
) -> None:
    """Start SSH tunnels to remote services.

    Uses autossh when available for automatic reconnection, otherwise
    falls back to plain ssh.  HOST defaults to the active graph profile's
    configured host.

    The --clipboard flag adds a reverse tunnel (-R 2490) so that
    ``curl localhost:2490/paste`` on the remote host reads the Windows clipboard
    via wsl-clip-server running on WSL.  Also supports POST /copy to write
    text back to the Windows clipboard.

    The --keyring flag forwards the D-Bus socket from HOST to the local
    machine, enabling keyring access on SLURM compute nodes that lack
    their own SecretService daemon.

    \b
    Examples:
      imas-codex tunnel start iter             # All services
      imas-codex tunnel start iter --neo4j     # Just graph
      imas-codex tunnel start iter --embed     # Just embedding
      imas-codex tunnel start iter --llm       # Just LLM proxy
      imas-codex tunnel start iter --vllm      # Just vLLM inference
      imas-codex tunnel start iter --docs      # Just docs-server
      imas-codex tunnel start iter --ink       # Just ink display
      imas-codex tunnel start iter --clipboard # Clipboard reverse tunnel
      imas-codex tunnel start iter --keyring   # D-Bus for keyring
    """
    target = _resolve_host(host)

    if (
        _installed_service_supports_request(
            target,
            neo4j_only,
            embed_only,
            llm_only,
            vllm_only,
            docs_only,
            ink_only,
            clipboard_only,
        )
        and not keyring_only
    ):
        click.echo(f"Starting systemd tunnel service for {target}:")
        _run_systemctl_user(["start", _service_name(target)])
        click.echo(f"✓ Tunnel service active: {_service_name(target)}")
        return

    # Handle keyring-only mode separately (unix socket, not TCP port)
    if keyring_only:
        _start_keyring_forward(target)
        return

    # Warn if a stale systemd unit exists that doesn't cover this request —
    # the user almost certainly wants to reinstall instead of running an
    # ad-hoc tunnel alongside the persistent one.
    if _service_file(target).exists():
        click.echo(
            f"  ⚠ Installed systemd service '{_service_name(target)}' does "
            "not cover the requested service(s).\n"
            "    Starting an ad-hoc tunnel in parallel — for persistent "
            f"coverage rerun: imas-codex tunnel service install {target}"
        )

    use_autossh = bool(shutil.which("autossh"))

    if use_autossh:
        click.echo(f"Starting tunnels to {target} (autossh):")
    else:
        click.echo(
            f"Starting tunnels to {target} (ssh — install autossh for auto-reconnect):"
        )

    ports = _get_tunnel_ports(
        target,
        neo4j_only,
        embed_only,
        llm_only,
        vllm_only,
        docs_only,
        ink_only,
        clipboard_only,
    )
    ok = _start_tunnels(target, ports, use_autossh)

    if ok == len(ports):
        click.echo(f"✓ All {ok} tunnel(s) active")
    elif ok > 0:
        click.echo(f"⚠ {ok}/{len(ports)} tunnel(s) active")
    else:
        raise click.ClickException("No tunnels could be established.")


@tunnel.command("stop")
@click.argument("host", required=False)
@click.option("--all", "stop_all", is_flag=True, help="Stop tunnels to all hosts")
def tunnel_stop(host: str | None, stop_all: bool) -> None:
    """Stop SSH tunnels.

    Without HOST, stops tunnels to all configured remote hosts.
    With HOST, stops only tunnels to that specific host.

    \b
    Examples:
      imas-codex tunnel stop           # Stop all tunnels
      imas-codex tunnel stop iter      # Stop only iter tunnels
    """
    from imas_codex.graph.profiles import _get_all_hosts
    from imas_codex.remote.tunnel import stop_tunnel

    if host:
        # Explicit host — stop just that one
        service_file = _service_file(host)
        if service_file.exists():
            subprocess.run(
                ["systemctl", "--user", "stop", _service_name(host)],
                capture_output=True,
                text=True,
            )
        if stop_tunnel(host):
            click.echo(f"✓ Tunnels to {host} stopped")
        elif service_file.exists():
            click.echo(f"✓ Tunnel service to {host} stopped")
        else:
            click.echo(f"No active tunnels to {host} found")
        return

    # No host specified — stop all configured remote hosts
    hosts = list(_get_all_hosts().values())
    if not hosts:
        click.echo("No remote hosts configured")
        return

    stopped: list[str] = []
    for h in sorted(set(hosts)):
        service_file = _service_file(h)
        if service_file.exists():
            subprocess.run(
                ["systemctl", "--user", "stop", _service_name(h)],
                capture_output=True,
                text=True,
            )
        if stop_tunnel(h):
            stopped.append(h)
        elif service_file.exists():
            stopped.append(h)

    if stopped:
        click.echo(f"✓ Tunnels stopped: {', '.join(stopped)}")
    else:
        click.echo("No active tunnels found")


@tunnel.command("status")
def tunnel_status() -> None:
    """Show active SSH tunnels across all services.

    Scans known Neo4j and embedding server ports (both direct and
    tunneled) for anything listening.  Ports bound by an SSH process
    are labeled accordingly.
    """
    from imas_codex.remote.tunnel import TUNNEL_OFFSET, is_tunnel_active
    from imas_codex.settings import (
        get_docs_server_port,
        get_embed_server_port,
        get_ink_display_port,
        get_llm_proxy_port,
        get_vllm_port,
        get_wsl_clip_port,
    )

    embed_port = get_embed_server_port()
    llm_port = get_llm_proxy_port()
    vllm_port = get_vllm_port()
    docs_port = get_docs_server_port()
    ink_port = get_ink_display_port()
    wsl_clip_port = get_wsl_clip_port()

    # Build port→label map using the same resolution as tunnel_start
    # so that labels match (graph name "codex", not location "iter").
    known_ports: dict[int, str] = {}
    try:
        from imas_codex.graph.profiles import resolve_neo4j

        profile = resolve_neo4j(auto_tunnel=False)
        name = profile.name
        known_ports[profile.bolt_port] = f"neo4j-bolt ({name})"
        known_ports[profile.bolt_port + TUNNEL_OFFSET] = (
            f"neo4j-bolt ({name}, tunneled)"
        )
        known_ports[profile.http_port] = f"neo4j-http ({name})"
        known_ports[profile.http_port + TUNNEL_OFFSET] = (
            f"neo4j-http ({name}, tunneled)"
        )
    except Exception:
        from imas_codex.graph.profiles import BOLT_BASE_PORT, HTTP_BASE_PORT

        known_ports[BOLT_BASE_PORT] = "neo4j-bolt"
        known_ports[BOLT_BASE_PORT + TUNNEL_OFFSET] = "neo4j-bolt (tunneled)"
        known_ports[HTTP_BASE_PORT] = "neo4j-http"
        known_ports[HTTP_BASE_PORT + TUNNEL_OFFSET] = "neo4j-http (tunneled)"
    known_ports[embed_port] = "embed"
    known_ports[llm_port] = "llm"
    known_ports[vllm_port] = "vllm"
    known_ports[docs_port] = "docs"
    known_ports[ink_port] = "ink"
    # wsl-clip-server is a reverse tunnel — bound on the remote (iter) side.
    known_ports[wsl_clip_port] = "wsl-clip (reverse)"

    # Build port→location map for SSH-forwarded ports so we can show
    # "iter" (or whichever host) instead of a generic "(ssh)" marker.
    port_host: dict[int, str] = {}
    try:
        host = profile.host  # type: ignore[possibly-undefined]
        for p in known_ports:
            if p >= TUNNEL_OFFSET:
                port_host[p] = host
        # Same-port forwards (no TUNNEL_OFFSET offset) — embed, llm, vllm
        # already land above TUNNEL_OFFSET (e.g. 18765, 18400), but docs
        # and ink live at 8765/8766 by default and would fall through to "(ssh)".
        for p in (embed_port, llm_port, vllm_port, docs_port, ink_port, wsl_clip_port):
            port_host[p] = host
    except Exception:
        pass

    # Identify configured SSH forwards from process command lines. The
    # liveness probe below ensures stale/wedged processes are not reported.
    from imas_codex.remote.tunnel import _ssh_forwarded_local_ports

    ssh_ports = set(known_ports).intersection(_ssh_forwarded_local_ports())

    # Check all known ports for liveness
    tunnels: list[tuple[int, str, bool]] = []
    for port, label in sorted(known_ports.items()):
        if is_tunnel_active(port):
            is_ssh = port in ssh_ports
            tunnels.append((port, label, is_ssh))

    if tunnels:
        click.echo("Active tunnels:")
        for port, label, is_ssh in tunnels:
            if is_ssh:
                location = port_host.get(port, "ssh")
                marker = f" ({location})"
            else:
                marker = ""
            click.echo(f"  :{port}  {label}{marker}")
    else:
        click.echo("No active tunnels on known service ports")

    # Check D-Bus keyring forwarding
    dbus_sock = _get_dbus_forward_status()
    if dbus_sock:
        click.echo(f"\nD-Bus keyring forward: {dbus_sock}")


# ============================================================================
# Systemd Service Subcommand
# ============================================================================


@tunnel.command("service")
@click.argument(
    "action",
    type=click.Choice(["install", "uninstall", "status", "start", "stop", "logs"]),
)
@click.argument("host", required=False)
@click.option("--neo4j", "neo4j_only", is_flag=True, help="Tunnel Neo4j ports only")
@click.option("--embed", "embed_only", is_flag=True, help="Tunnel embedding port only")
@click.option("--llm", "llm_only", is_flag=True, help="Tunnel LLM proxy port only")
@click.option(
    "--vllm", "vllm_only", is_flag=True, help="Tunnel vLLM inference port only"
)
@click.option("--docs", "docs_only", is_flag=True, help="Tunnel docs-server port only")
@click.option("--ink", "ink_only", is_flag=True, help="Tunnel ink display port only")
@click.option(
    "--clipboard",
    "clipboard_only",
    is_flag=True,
    help="Reverse-tunnel clipboard port (wsl-clip-server on WSL → Windows)",
)
def tunnel_service(
    action: str,
    host: str | None,
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool = False,
    vllm_only: bool = False,
    docs_only: bool = False,
    ink_only: bool = False,
    clipboard_only: bool = False,
) -> None:
    """Manage persistent SSH tunnels via systemd + autossh.

    Installs a systemd user service that maintains reconnecting SSH
    tunnels to the specified HOST for Neo4j, embedding, LLM, vLLM,
    docs-server, and/or wsl-clip-server clipboard services.

    \b
    Examples:
      imas-codex tunnel service install iter           # All services
      imas-codex tunnel service install iter --neo4j   # Just graph
      imas-codex tunnel service install iter --embed   # Just embedding
      imas-codex tunnel service install iter --llm     # Just LLM proxy
      imas-codex tunnel service install iter --vllm    # Just vLLM inference
      imas-codex tunnel service install iter --docs    # Just docs-server
      imas-codex tunnel service install iter --ink     # Just ink display
      imas-codex tunnel service install iter --clipboard # Clipboard reverse tunnel
      imas-codex tunnel service start iter
      imas-codex tunnel service status iter
      imas-codex tunnel service logs iter
    """
    import platform

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")
    if not shutil.which("systemctl"):
        raise click.ClickException("systemctl not found")

    target = _resolve_host(host)
    service_name = _service_name(target)
    service_dir = _service_dir()
    service_file = _service_file(target)

    if action == "install":
        if not shutil.which("autossh"):
            raise click.ClickException(
                "autossh not found. Install with: sudo apt install autossh"
            )

        ports = _get_tunnel_ports(
            target,
            neo4j_only,
            embed_only,
            llm_only,
            vllm_only,
            docs_only,
            ink_only,
            clipboard_only,
        )
        if not ports:
            raise click.ClickException("No services selected for tunneling.")
        log_dir = Path.home() / ".local" / "share" / "imas-codex" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        service_content = _build_systemd_service_content(
            target,
            neo4j_only,
            embed_only,
            llm_only,
            vllm_only,
            docs_only,
            ink_only,
            clipboard_only,
        )
        service_dir.mkdir(parents=True, exist_ok=True)
        service_file.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", service_name], check=True)
        subprocess.run(["systemctl", "--user", "restart", service_name], check=True)
        click.echo(f"✓ Tunnel service installed: {service_name}")
        click.echo(f"  Host: {target}")
        click.echo("  Ports:")
        for remote_port, local_port, label, bind_addr, direction in ports:
            if direction == "R":
                click.echo(
                    f"    {label}: {target}:{remote_port} ← {bind_addr}:{local_port} (reverse)"
                )
            else:
                click.echo(
                    f"    {label}: localhost:{local_port} → {bind_addr}:{remote_port}"
                )
        click.echo(f"  Log: {log_dir / f'autossh-{target}.log'}")
        click.echo(f"  Service started: imas-codex tunnel service logs {target}")

        # Check loginctl linger status
        try:
            result = subprocess.run(
                ["loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Linger=no" in result.stdout:
                click.echo(
                    "\n  ⚠ User linger is disabled. The service will stop on logout."
                )
                click.echo("  Enable persistence: sudo loginctl enable-linger $USER")
        except Exception:
            pass  # Best-effort linger check

    elif action == "uninstall":
        if not service_file.exists():
            click.echo(f"Service {service_name} not installed")
            return
        subprocess.run(
            ["systemctl", "--user", "stop", service_name], capture_output=True
        )
        subprocess.run(
            ["systemctl", "--user", "disable", service_name], capture_output=True
        )
        service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        click.echo(f"Service {service_name} uninstalled")

    elif action == "status":
        if not service_file.exists():
            click.echo(f"Service {service_name} not installed")
            click.echo(f"  Install: imas-codex tunnel service install {target}")
            return
        result = subprocess.run(
            ["systemctl", "--user", "status", service_name],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr)

    elif action == "start":
        if not service_file.exists():
            raise click.ClickException(
                f"Service not installed. Run: imas-codex tunnel service install {target}"
            )
        subprocess.run(["systemctl", "--user", "start", service_name], check=True)
        click.echo(f"Tunnel service to {target} started")

    elif action == "stop":
        subprocess.run(["systemctl", "--user", "stop", service_name], check=True)
        click.echo(f"Tunnel service to {target} stopped")

    elif action == "logs":
        subprocess.run(["journalctl", "--user", "-u", service_name, "-n", "50", "-f"])


@tunnel.command("service-run", hidden=True)
@click.argument("host", required=False)
@click.option("--neo4j", "neo4j_only", is_flag=True, help="Tunnel Neo4j ports only")
@click.option("--embed", "embed_only", is_flag=True, help="Tunnel embedding port only")
@click.option("--llm", "llm_only", is_flag=True, help="Tunnel LLM proxy port only")
@click.option(
    "--vllm", "vllm_only", is_flag=True, help="Tunnel vLLM inference port only"
)
@click.option("--docs", "docs_only", is_flag=True, help="Tunnel docs-server port only")
@click.option("--ink", "ink_only", is_flag=True, help="Tunnel ink display port only")
@click.option(
    "--clipboard",
    "clipboard_only",
    is_flag=True,
    help="Reverse-tunnel clipboard port",
)
def tunnel_service_run(
    host: str | None,
    neo4j_only: bool,
    embed_only: bool,
    llm_only: bool,
    vllm_only: bool = False,
    docs_only: bool = False,
    ink_only: bool = False,
    clipboard_only: bool = False,
) -> None:
    """Run the persistent systemd tunnel supervisor in the foreground."""
    target = _resolve_host(host)
    _run_service_supervisor(
        target,
        neo4j_only,
        embed_only,
        llm_only,
        vllm_only,
        docs_only,
        ink_only,
        clipboard_only,
    )
