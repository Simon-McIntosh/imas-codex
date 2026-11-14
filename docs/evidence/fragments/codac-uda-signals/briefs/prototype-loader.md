# Brief: prototype loader for CODAC UDA data, with a design assessment

## Why

The graph is being extended from IMAS science data to ITER plant data, and the
project is to be renamed from `imas-codex` to `iter-codex`. The first new data
source is the CODAC UDA archive. Search tools in the MCP server (the lead's
working short name for the renamed server is `iter-cx`) will find a signal; a
loader must then read it so an agent or a person can plot it and do basic
analysis. This node builds a throwaway prototype of that loader and reports
what it learned. A plan for the scanner, the loader and the rename is written
from your report, so the report matters as much as the code.

## Goal

One working prototype that lists CODAC UDA variables with unit and description,
lists pulses, and reads one signal over a time window or pulse, delivered with a
written assessment of how it should be built properly in this repository.

## Prior art to port from (read these first)

- `/home/ITER/mcintos/Code/iter-uda/src/iter_uda/uda.py`, `catalogue.py`,
  `windows.py` and its `AGENTS.md`: the consolidated reader. Variable listing,
  pulses, pulse categories, raw / envelope / binned reads, and the merge of
  UDA's own unit and description with the controls-metadata service.
- `/home/ITER/mcintos/Code/mctb/docs/figures/uda_servers.py`,
  `docs/figures/htscl-thermal/htscl_data.py`,
  `docs/figures/discharge-fit/discharge_data.py`,
  `docs/figures/uda-access/discharge_records.py`: the access code in use, with
  the per-query cache and the structured-record reads.
- `/home/ITER/mcintos/Code/mctb/docs/research/mctb-uda-access.html`: servers,
  query grammar, pulse naming, and the read traps (silent decimation when
  paging without `decSamples=-1`, an unreliable `isDecimated()`, timestamp type
  following `tsFormat`, structured records needing `/field`).

These repositories are outside this one and are read-only for you.

## Constraints

- The UDA client is not pip-installable. It comes from
  `ml m-uda-client/7.4-iimkl-2025b` (Python 3.13) or `ml MINT`. This project
  requires Python 3.12. Do not add the client to `pyproject.toml`. Find the
  bridge: `imas_codex/remote/` already runs scripts under a different
  interpreter, and the existing scanners in
  `imas_codex/discovery/signals/scanners/` are the pattern to assess.
- This repository is on GitHub. No CODAC host name and no real channel or pulse
  name goes into any tracked file: code, tests, report or commit message. Read
  hosts from the gitignored `imas_codex/config/facilities/iter_private.yaml`
  (add the keys you need there; it is never staged) or from the environment.
  Tests use a fake client and invented names.
- One archive host is reserved for operators: `OPERATOR_HOSTS` in mctb's
  `uda_servers.py` names it. Never connect to it, and keep that refusal in the
  port.
- Reads are network I/O and light. Keep any live read to a few seconds of one
  or two signals. Run only your own focused test file. If a step becomes heavy,
  submit it to `all_debug` with `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch`.
- Plots and data from live reads are ITER-internal. Write them under
  `~/.cache/imas-codex-codac-uda/` and name the paths in the report; do not
  commit them. Follow the plot style rules in `~/.agents/AGENTS.md`, and look
  at each figure you render.
- Use the shared environment: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv
  PYTHONPATH="$PWD" uv run --no-sync ...`. Do not create a virtualenv.

## What to build

Under `scripts/codac_uda_prototype/` only:

1. A reader script that runs under the UDA client's interpreter and answers
   three requests as JSON or a compact binary file: list variables matching a
   pattern with unit and description, list pulses, and read one variable over a
   time window or pulse in raw or envelope mode. Port the trap handling from
   `iter-uda`.
2. A loader callable from this project's Python 3.12 environment that invokes
   the reader and returns arrays with their metadata (name, unit, description,
   time base, whether decimated, the query used).
3. A small command or function that plots one loaded signal and prints basic
   statistics (count, time span, min, max, mean, sample interval).
4. A test file driven by a fake client that covers the three requests and the
   decimation trap.

Then attempt each of the three requests live, once. If the UDA hosts are not
reachable from the node you run on, do not work around it: record the exact
error and the node name, and say what you would need.

## The assessment (the report)

Write `docs/evidence/fragments/codac-uda-signals/prototype-report.md`. Lead with
what worked and what did not, with the live results (counts, timings, sizes;
no real names). Then give your own view, with file references, on each of:

1. **Placement and reuse.** Where a CODAC UDA scanner and a loader belong in
   this codebase, which existing machinery they should extend (scanner base
   class, remote executor, facility config, MCP tool registration), and what
   must not be duplicated.
2. **The interpreter gap.** The best way to bridge Python 3.12 and the
   module-provided 3.13 client for (a) a batch scan and (b) an interactive
   read from an MCP tool.
3. **Port or depend.** Whether to port the `iter-uda` reader into this
   repository or depend on it, given that `iter-uda` has no remote.
4. **Schema fit.** How CODAC variables, pulses, pulse categories and the
   archive servers map onto the existing LinkML classes in
   `imas_codex/schemas/` (for example the facility signal and data access
   classes), and what is missing. The variable naming convention encodes a
   plant component and a control function; say where that decoding should
   attach so a variable can later be joined to a plant breakdown component.
5. **The rename.** An inventory of every surface that carries the name
   `imas-codex` or `imas_codex` (distribution, package, CLI entry points, MCP
   server names, container and registry names, config and cache directories,
   environment variables, docs, the GitHub repositories, anything a running
   job or a stored path resolves through), with counts, and the order you would
   change them in so that nothing running breaks. Say what should keep the
   `imas` name because it really is about IMAS.
6. **Anything you disagree with** in this approach, and what you would do
   instead.

## Done when

- The focused test file passes; the log is at
  `~/.cache/imas-codex-codac-uda/prototype-tests.log` and its last line is in
  the manifest.
- Each of the three live requests has a recorded outcome in the report: a
  result with count and timing, or the exact error.
- The report answers all six assessment points with file references.
- No host name, variable prefix or pulse name from mctb's `uda_servers.py` or
  the `iter-uda` sources appears in any file of your commit. State the check
  you ran and its output in the manifest.

## Scope

Write only `scripts/codac_uda_prototype/` and
`docs/evidence/fragments/codac-uda-signals/prototype-report.md`. Change no
existing source, schema, config or dependency file; anything you think should
change goes in the report.
