# The architecture document's deployment instruction and every other path it asserts

`docs/architecture/services.md` told a reader that compute-node deployment runs
`sbatch /tmp/codex-embed.sh`. That path does not exist on disk today. It was
never a reader-runnable artifact either: in the code the document describes,
the temporary script was written and deleted inside a single remote command, so
the file was gone before the document's own next step could be followed. The
instruction named the one path in the flow that could not survive being read.

This record states what the document now says instead, quotes the surviving
command from the source that defines it, and checks every other path and
command the same document asserts — one inaccurate line in a file is weak
evidence that the rest of it is accurate.

## The defect, with the reproduction

The submission path before the repair (commit `03837c2b7`,
`fix(services): preserve submitted launch scripts`) built a remote command
whose last clause removed the script it had just submitted:

```
f"printf '%s' {script_quoted} > /tmp/{job_name}.sh && "
f"sbatch /tmp/{job_name}.sh && "
f"rm -f /tmp/{job_name}.sh"
```

So `/tmp/codex-embed.sh` existed for the lifetime of that one remote shell and
no longer. The document's diagram showed exactly this path as the submission
step, which is why following the document could not reproduce a launch. The
same commit deleted the tracked `slurm/codex-embed.sh` launcher, so after it
there was no path the old instruction could be rescued from.

The removal clause is gone. The current submission path
(`imas_codex/cli/services.py:556`, `:557`, `:591`-`:593`) writes the script
beside the service log, where it persists:

```
services_dir_abs = f"{remote_home}/.local/share/imas-codex/services"
script_path = f"{services_dir_abs}/{job_name}.sh"
...
f"mkdir -p {safe_services_dir} && "
f"printf '%s' {script_quoted} > {script_path} && "
f"sbatch {script_path}"
```

with `#SBATCH --output={services_dir_abs}/{job_name}.log` (`services.py:573`).
The job names are `_NEO4J_JOB = "codex-neo4j"` and `_EMBED_JOB = "codex-embed"`
(`services.py:41`-`:42`), so each service's script sits beside its own log:
`codex-embed.sh` / `codex-embed.log`, `codex-neo4j.sh` / `codex-neo4j.log`.
Both files are present on this workstation today.

Confirmed directly: the string `/tmp/codex-` no longer occurs anywhere in the
package (`logs/audit_paths.log`, line 56 row, `ABSENT`).

## The replacement, established from the code rather than assumed

The command that owns compute-node deployment is the service's own `start`. It
is defined at `imas_codex/cli/embed.py:37` (`@embed.command("start")`) and
`imas_codex/cli/graph.py` for Neo4j; Click generates each usage line, so the
verbatim form is obtained by asking the command rather than by reading it out
of a docstring:

```
Usage: imas-codex embed start [OPTIONS]      # embedding server
Usage: imas-codex graph start [OPTIONS]      # Neo4j
```

Captured to `logs/usage_embed_start.log` and `logs/usage_graph_start.log`, both
exit 0. The document now quotes these lines, states that the command owns
deployment and needs no separate submission step, and describes the generated
script as durable — readable to see what was submitted, and re-submittable
verbatim to reproduce a launch without going through `start` again.

## Every other path and command the document asserts

Checked as a whole, not at one line. `logs/audit_paths.log` (exit 0) carries
the full table; the result:

| Document line | Assertion | Observed today |
|---|---|---|
| 39 | `pyproject.toml` sets the deploy location | exists — `location = "titan"` at `pyproject.toml:266` (embedding) and `:246` (graph) |
| 40 | `resolve_location` searches facility YAMLs | exists — `imas_codex/remote/locations.py:65` |
| 41 | `iter.yaml` defines `compute_locations.titan` | exists — `imas_codex/config/facilities/iter.yaml` |
| 42 | the resolution returns a `LocationInfo` | exists — `imas_codex/remote/locations.py:36` |
| 53 | `_is_compute_target` selects SLURM vs systemd | exists — `imas_codex/cli/services.py:213` |
| 54 | `_submit_service_job` performs the submission | exists — `imas_codex/cli/services.py:497` |
| 56 | the submission runs `sbatch /tmp/codex-embed.sh` | **ABSENT** — the defect above |
| 56 | the script handed to `sbatch` | exists — `~/.local/share/imas-codex/services/codex-embed.sh` |
| 57 | `_wait_for_job` waits for the allocation | exists — `imas_codex/cli/services.py:1272` |
| 76 | `_wait_for_health` waits for the endpoint | exists — `imas_codex/cli/services.py:1431` |
| 74 | `llm start` installs a systemd user unit | exists — `~/.config/systemd/user/imas-codex-llm.service` |
| 74 | the unit name the code installs | exists — `imas_codex/cli/llm_cli.py:670` |
| 92 | `SLURM_JOB_ID` selects foreground mode | exists — `imas_codex/cli/embed.py` |
| 184 | the embed log the troubleshooting step reads | exists — `~/.local/share/imas-codex/services/codex-embed.log` |
| 197 | the Neo4j log the troubleshooting step reads | exists — `~/.local/share/imas-codex/services/codex-neo4j.log` |
| 196 | Neo4j heap is set in `neo4j.conf` | exists — generated by `imas_codex/graph/remote.py` |
| 213 | rich output is forced by `IMAS_CODEX_RICH` | exists — `imas_codex/cli/rich_output.py:30` |
| 189 | the proxy needs `OPENROUTER_API_KEY` from `.env` | exists — read in `imas_codex/embeddings/` |
| 201 | the default ports are configurable in `pyproject.toml` | exists — `pyproject.toml:233` |

The document's invocation set resolves as a whole: `graph`/`embed`/`llm`
`start`, `status`, `stop`, `logs`, `restart`, and `service <action>` all exist
as commands, `service` accepting `install`/`uninstall`/`status`/`start`/`stop`
in both groups; the flags the document uses (`-f`, `--gpu`, `-g`, `--port`) are
present on the commands it attaches them to. The external tools the
troubleshooting steps invoke (`sbatch`, `squeue`, `scancel`, `curl`,
`systemctl`) are all on `PATH`.

Two lines in the document are captured output rather than assertions about
disk — the Titan resource header (line 112) and the status samples (lines 127,
142-150). They are recorded here as not-a-path rather than as unchecked, so the
audit does not read as complete coverage of them.

## A second inaccuracy, found by checking the rest

The port table asserted an LLM proxy port of `18790`. The port is resolved by
`get_llm_proxy_port()` (`imas_codex/settings.py:823`), which is
`LLM_BASE_PORT = 18400` (`settings.py:711`) plus the facility's index in
`tool.imas-codex.locations` (`settings.py:840`). `iter` is index 0, and the
document's own table is titled "default ports", so the resolved value is
`18400`:

```
llm_location = iter
llm_port     = 18400
embed_server = 18765
offset(iter) = 0
```

`pyproject.toml:764` states the same convention in prose ("Port base = 18400,
offset by location index"), and the group's own help prints
`imas-codex llm status --url http://remote:18400` (`llm_cli.py:296`) — so the
value is corroborated in three independent places and `18790` appears nowhere
in the package. The table row now reads `18400`, and the table says the numbers
are base ports that a facility with a non-zero index offsets.

`18790` was reported by nobody before this check: the count of places it occurs
in the repository was exactly one, the document line itself.

## Method and limits

- Probes are read-only; this node mutated no file outside its write fence and
  ran no graph query.
- Each claim was checked against current disk and current source at base
  `4ba57a134b15849389a85ae5e9e115423b81c728`. A path that exists today is not a
  promise: `sbatch`-generated artifacts are per-launch, so the two service
  scripts reflect the last launch on this workstation.
- Rich rendering was not exercised. The document's claim that the spinner
  degrades to plain text without a TTY (`IMAS_CODEX_RICH` forces it) was checked
  only as far as the symbol's existence, not by running a deployment.
- This gate measures this node's own change. Verifying the merged document, and
  a full unit gate over the package, belongs to a separately dispatched node.
- The service unit template `imas_codex/config/services/imas-codex-embed.service`
  line 12 already directs a reader to `imas-codex embed start`; it is outside
  this node's write scope and was left untouched.

## Reproduction

| Log | Carries | Exit |
|---|---|---|
| `logs/audit_paths.log` | the per-claim existence table, the CLI surface, the external tools | 0 |
| `logs/usage_embed_start.log` | verbatim `imas-codex embed start --help` | 0 |
| `logs/usage_graph_start.log` | verbatim `imas-codex graph start --help` | 0 |

Scripts: `audit_paths.py` produces the first log from the claim list embedded in
it; each claim names the document line it came from. The CLI surface it prints
comes from walking the Click group (`/tmp/cmd_audit.json`, 168 commands) and
the port figures from resolving `get_llm_proxy_port()` in the worktree with the
main checkout's environment.