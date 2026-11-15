# Retire the LLM proxy

## Goal

No language-model call in the project can route through a LiteLLM proxy, and
the proxy's configuration, settings, probes and route table are gone.

## Authority

The lead stated on 2026-11-15 that the proxy is not used and should not be.

## What is known

- No proxy is running: no scheduler job, no process and no user service on the
  login node, and the proxy's health port does not answer on the login node or
  the fleet node.
- The project is still configured as if one existed. `pyproject.toml` sets the
  LLM location to a facility, declares a proxy port base, and carries a comment
  warning that a seat without the right prefix silently routes through the
  proxy. `imas_codex/settings.py` resolves a proxy host, port and URL.
  `imas_codex/discovery/base/services.py` probes the proxy's health and has a
  canary that still requests a retired model through it.
  `imas_codex/discovery/base/progress.py` displays a proxy service row.
  `imas_codex/config/litellm_config.yaml` is a tracked proxy route table,
  including two routes no seat names.
- The model seats route in three ways today: the local model through its own
  router, OpenRouter directly, and the provider decisions endpoint directly.
  Those stay exactly as they are.

## Rules

- Find every reader of the proxy settings before deleting them; delete the
  setting and its readers together. Leave no setting that resolves a proxy
  host, port or URL, and no environment override for one.
- Remove the proxy health probe, the proxy canary and the proxy row in the
  progress display. Do not remove the checks for the local model router, the
  embedding service or the graph.
- Remove the proxy route table file and the code that reads it. If a price or
  context-window value was read from it, show that the pricing table in
  `pyproject.toml` already carries it, or move it there.
- Keep the `litellm` library dependency if the call layer uses it as a client
  library; this node removes the proxy server path, not the client.
- Remove proxy start and stop commands from the CLI and their documentation in
  the `AGENTS.md` files, replacing each mention with what is true now.
- A seat whose route would have depended on the proxy must fail loudly at
  resolution, naming the seat, not fall back silently.
- Change no model, no price and no seat's route.

## Done when

- A search recorded in the evidence finds no reference to a proxy host, port,
  URL, health probe or route table under `imas_codex/`, `tests/` or the
  configuration, other than the client library's own name.
- A test resolves every declared seat and asserts each resolves to the local
  router, to OpenRouter or to the decisions endpoint, and none to a proxy.
- The focused tests for configuration, the call layer and the discovery
  service checks show no added failure against the base revision, with base
  and after failure ids recorded.
- Under the opt-in live marker, one local seat and one OpenRouter seat each
  answer a request.
- The net change in lines is stated and is negative.
- Negative control: a scratch seat with no recognised route fails resolution
  with an error naming the seat.

## Scope

Write paths: `pyproject.toml`, `imas_codex/settings.py`, `imas_codex/config/`,
`imas_codex/discovery/base/`, `imas_codex/cli/`, `imas_codex/llm/`,
`imas_codex/remote/locations.py`, `tests/`, the `AGENTS.md` files that mention
the proxy, and the evidence fragment
`docs/evidence/fragments/iter-signal-context-and-enrichment/retire-the-llm-proxy.html`.

Tests run from the worktree against the shared environment with `--no-sync`;
anything beyond a focused test file runs on a compute partition with
`env -u SLURM_JOB_ID -u SLURM_JOBID srun --partition=all_debug --time=00:59:00`.
Commit locally with a conventional subject and a body. Do not push or merge.
