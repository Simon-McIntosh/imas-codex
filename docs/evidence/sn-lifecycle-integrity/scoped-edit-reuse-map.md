# The scoped-edit surface: every hop from `sn edit` to `run_sn_pools`

Scouted 2026-09-18 at HEAD `9705954c2`, read from the worktree. The subject is
the followup's fourth hole: *"sn edit inline runner | launches run_sn_pools
with no --skip-global-maintenance control | Observed twice on 2026-09-09. A
fenced single-identity edit silently escalates into global sourceless-name,
attachment-consistency, source-ledger and derived-parent reconciliation."*

## Verdict: DOES-REPRODUCE

The forwarding is absent at every frame on the edit path, and the absence is
structural rather than a default value: no symbol on the edit path even
*accepts* the flag. `run_sn_pools`'s parameter defaults to `False`, so an edit
launch runs the full global maintenance set.

**Static receipt.** `imas_codex/standard_names/edit.py:3740-3747` is the only
launch of `run_sn_pools` on the edit path and its keyword list is complete:

```python
        return await run_sn_pools(
            cost_limit=cost_limit,
            min_score=min_score,
            rotation_cap=rotation_cap,
            scope_run_id=run_id,
            skip_generate=skip_generate,
            pending_fn=pending_fn,
        )
```

`skip_global_maintenance` is absent, and `loop.py:1303` declares it
`skip_global_maintenance: bool = False`. Nothing upstream can supply it:
neither `sn_edit` (`cli/sn.py:6410`), nor `run_inline_review`
(`edit.py:3752`), nor `_run_scoped_pipeline` (`edit.py:3711`) declares the
parameter or a CLI option — the `sn edit` command carries no
`--skip-global-maintenance` at all (the string appears in `cli/sn.py` only
within the `sn run` command, lines 1574-2181).

**Dynamic receipt.** A signature probe run from the worktree source
(`PYTHONPATH` pinned to the worktree, `edit.__file__` confirmed to resolve
there — log: `logs/probe-signatures.log` in the run directory) shows the
instrument sees a known-present seam before it reports the absence:

```
loop.run_sn_pools: has skip_global_maintenance param = True
edit._run_scoped_pipeline: has skip_global_maintenance param = False
edit.run_inline_review: has skip_global_maintenance param = False
cli._run_sn_cmd: has skip_global_maintenance param = True
cli.sn_edit: has skip_global_maintenance param = False
```

(first and fourth lines are the known-present controls — `is True` where the
run command's forwarding already exists; the False lines are the finding).

**The reproduction, quoted.** Two `sn edit` runs on 2026-09-09 escalated into
global reconciliation; the recorded recovery was "interrupt, then `sn run
--scope-run-id <id> --docs-only --skip-global-maintenance`", the staged edit
surviving with `edit_status='open'`. That is the behaviour of the code above:
the edit path supplies no bypass, so `run_sn_pools` proceeds through its
graph-wide startup and post-drain maintenance blocks.

## The call chain, hop by hop

Every hop from the CLI entry point to the pool loop, as read at HEAD:

| # | Symbol | file:line | What happens here |
|---|---|---|---|
| 1 | `@sn.command("edit")` → `def sn_edit(` | `cli/sn.py:6287`, `:6410` | CLI entry point; no `--skip-global-maintenance` option exists here |
| 2 | `plan = apply_edit(...)` | `cli/sn.py:6517` → `standard_names/edit.py:887` | staging hop: validates and stages the edit, mints `plan.run_id`; does not launch pools |
| 3 | `_pending_fn` closure | `cli/sn.py:6552-6564` | progress callback; scopes its count to `plan.run_id` — evidence the edit already thinks of itself as scoped |
| 4 | `outcome = run_inline_review(plan, cost_limit=cost_limit, pending_fn=_pending_fn)` | `cli/sn.py:6570` → `edit.py:3752` | the call that launches the pools for the staged edit |
| 5 | `skip_generate = plan.entry in ("review_name", "review_docs")` | `edit.py:3784` | mode selection only |
| 6 | `summary = _run_scoped_pipeline(...)` | `edit.py:3786-3793` | forwards `run_id=plan.run_id`; does not forward a bypass |
| 7 | `def _run_scoped_pipeline(` | `edit.py:3711` | signature: `run_id, skip_generate, cost_limit, min_score, rotation_cap, pending_fn` — no `skip_global_maintenance` parameter to forward |
| 8 | `edit.py:3735` `import asyncio`, `:3739` `async def _main()`, `:3749` `return asyncio.run(_main())` | `edit.py:3735-3749` | bridges to the async loop |
| 9 | `return await run_sn_pools(...)` | `edit.py:3740-3747` | **the omission** — keyword list reproduced above |
| 10 | `async def run_sn_pools(` / `skip_global_maintenance: bool = False,` | `loop.py:1269`, `:1303` | the loop's own seam, already present; default False |
| 11 | guard block | `loop.py:1411-1419` | `1411` auto-implies the bypass for `drain_scope_id`; `1412-1414` requires a scope; `1416-1418` refuses maintenance-only modes |
| 12 | `_global_maintenance_call` | `loop.py:1570-1579` | the single choke point every graph-wide writer passes through |
| 13 | bypass guards | `loop.py:1625, 1885, 2286, 2722, 2777` | call-site guards for the writers invoked outside the choke point (`reconcile_attachment_consistency` at `1885`, `embed_description_worker` at `2286`, the derived-parent and doc-link block at `2722`/`2777`) |
| 14 | `"skip_global_maintenance": skip_global_maintenance,` | `loop.py:1538` | carries the value into the run summary/dispatch surface |

## The loop needs no change

The repair is confined to the edit and CLI frames. `run_sn_pools` already
requires `scope_run_id` under the bypass (`loop.py:1412-1414`), and the edit
path supplies exactly that by construction — `edit.py:3744` passes
`scope_run_id=run_id` — so the guard at `1412` is already satisfied for every
edit launch, and the incompatibility guard at `1416` never applies (the edit
path passes neither `attach_only` nor `reconcile_only`). Adding the flag on
the edit side is sufficient; no loop-side condition changes.

## Reuse inventory: what a repair would reuse, with file:line

| Symbol | file:line | Role in a repair |
|---|---|---|
| `run_sn_pools` parameter + docstring | `loop.py:1303`, `:1381-1384` | the receiving seam; already documents the contract ("Bypass graph-wide startup, background, and post-drain maintenance") |
| bypass guard block | `loop.py:1411-1419` | enforces the flag's preconditions; already satisfied by the edit path's `scope_run_id` |
| `drain_scope_id` auto-imply precedent | `loop.py:1411` | shows the codebase's established shape: a scoped launch implies the bypass |
| `_global_maintenance_call` | `loop.py:1570-1579` | the choke point the bypass silences; nothing to change |
| call-site guards | `loop.py:1625, 1885, 2286, 2722, 2777` | the explicit-flag read paths; nothing to change |
| `_run_sn_cmd` param + forward | `cli/sn.py:658`, `:919` | the run command's forwarding pattern to mirror: declare the parameter, pass it through |
| `sn run` option declaration | `cli/sn.py:1574-1587` (`is_flag=True, default=False`) | the option shape to copy for `sn edit` |
| `sn run` validation guards | `cli/sn.py:2054-2057` (rename), `:2139-2172` (scope-requirement and incompatible-combination map) | the guard vocabulary a new edit option would reuse; note the `2139-2172` block requires one of `--focus/--batch, --name, or --scope-run-id` — an edit launch is `--name`-equivalent by construction |
| CLI forwarding wiring tests | `tests/standard_names/test_scoped_global_maintenance.py:473-496`, `:538-553`, `:366-370` | existing True/False/refusal test shapes the edit path's new test would mirror |

## Blind spots: why the missing forward is invisible today

The edit path's tests mock the very seam that would carry a forwarded flag:

- `tests/standard_names/test_edit_inline_review.py` patches `run_inline_review`
  (lines 294, 326, 355, 384, 428, 452, 474) — the call at `cli/sn.py:6570` is
  verified to have happened, never with which kwargs it reached the real
  function.
- `tests/standard_names/test_pinned_rename_refine.py` patches
  `_run_scoped_pipeline` (lines 320, 356, 404, 445, 479) — the frame that
  would forward the flag is replaced, so its argument list is never read.

A repair's test must call the real seam (as `test_edit_inline_review.py`'s
non-mocked cases at 101-255 do for `apply_edit`) or assert the forwarded kwarg
on the mocked boundary, in the shape of
`test_cli_wires_bypass_to_existing_pool_orchestrator`.

## Proposed exclusive write-path set

The forward-the-flag repair touches three files, none of them shared with the
loop:

| Path | Change |
|---|---|
| `imas_codex/cli/sn.py` | add `--skip-global-maintenance` to `sn edit` (shape from `:1574-1587`), forward into `run_inline_review` |
| `imas_codex/standard_names/edit.py` | thread the parameter through `run_inline_review` → `_run_scoped_pipeline` → the `run_sn_pools` call at `:3740-3747` |
| `tests/standard_names/test_scoped_global_maintenance.py` | wiring test mirroring `:473-496`: edit path forwards True, and the default stays False |

Placing the parameter on `_run_scoped_pipeline` (`edit.py:3711`) rather than
duplicating the launch is the one-source-of-truth shape: both of its callers
are then covered by one change.

**The design fork, for the lead.** The followup itself offers two shapes:
"*`sn edit` should forward the flag, or default to the scope it was given.*"
Forwarding (above) preserves the current behaviour for callers that want the
global reconcile and makes the bypass opt-in; default-to-scope inverts the
default at `edit.py:3740` and would make a global reconcile from the edit path
the explicit choice. Either way the flag's *name* and semantics are already
fixed by `loop.py:1303` and the `sn run` help text (`cli/sn.py:1578`); the
fork is only about which default the edit CLI carries, and it is a lifecycle
decision rather than a mechanical one.

## Adjacent: the second caller of the same seam

`rescore_name` (`edit.py:3813`) calls `_run_scoped_pipeline` at
`edit.py:3893-3900` with the same omission. Repairing
`_run_scoped_pipeline`'s signature covers it in one change; whether
`sn rescore` should default to the scoped bypass is the same fork as above and
belongs to the same decision.

## Why a table and not a figure

The relationship here is a call chain carrying exact `file:line` addresses at
every hop; the table above is the faithful rendering, and a drawn diagram
would restate it with less precision. No figure is attached because none would
add information the table does not already carry.