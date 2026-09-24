# The sessions imas-codex should be handed to, and what blocks them

Successor sessions receive this repository entirely through the planning layer, so
this record exists because the coordinator report that produced it was a message,
and a message is not a record. Measured 2026-09-24.

## Two sessions, not one, and not three

### `codex-sn` — the standard-names and catalog stream

| | |
|---|---|
| working directory | `/home/ITER/mcintos/Code/imas-codex` |
| entry | `/reckon-ship migration-project-state` |

Finish the state assessment and act on it: §2 truths plan status against the code,
§3 names the thin-interface refactors, §4 orders the work. Then triage
`sn-pipeline-hardening` — **archived at impl 0.55 with 29 open followups**, which by
this repository's own rule is 29 items of hidden work, because a followup on a
completed plan appears in no pending-work query.

**§2 gates everything downstream.** Nine plans are mis-declared, so a successor
prioritising against current impl figures is prioritising against fiction.

First nodes: merge the two member evidence records (feature depth,
shared-infrastructure coupling); give the three statusless plans a status and the
four active plans an impl; then the 29-followup triage.

### `codex-host` — agent-process observability

| | |
|---|---|
| working directory | `/home/ITER/mcintos/Code/imas-codex` |
| entry | `/reckon-ship host-dashboard-covers-every-agent-process` |

S15, `active`, **impl absent**, one followup. Every process on the login and fleet
nodes, grouped by owner, with correct CPU, memory and filesystem percentages.

First nodes: declare its impl; then cover the orphan classes measured this week — a
three-day `crew watch` producer at 21 % CPU, an unbounded `find / … | head` at ppid 1
running 10 h 49 m, followers alive 7× past a declared `--lifetime 29m`, and 68
retained worktrees across two sessions. **The dashboard's value is exactly that none
of those was visible to anyone**, and each was found by a person looking rather than
by an instrument.

Gates nothing, gated by nothing. That independence is why it is a separate session
rather than a section of the first.

## Two reckon crew defects that block reconciliation, neither owned here

**A stale run cannot be reconciled at all.** One in-flight run has landed work and
only its ledger row is stuck. `crew complete` refuses on a stray-write check naming
**85 paths** — `AGENTS.md`, `uv.lock`, twelve plans, forty test files — none of them
the run's edits. Its worktree is gone and its base is seven days old, so the check
compares today's `main` against a week of every other session's landed work and
attributes all of it to that run. The only remedy offered is one `--accept-path` per
path, which would mean **writing 85 false claims**; `--waive-boundary-refusal` does
not cover it, since that flag is for an uncommitted stray edit. A boundary check with
no surviving worktree and a stale base has no referent and should say so rather than
list the repository. Recorded upstream as `f-srr-stale-run-cannot-be-reconciled`.

**Dispatch requires a follower registered to the dispatching session, and the
preflight reveals that one condition at a time.** Three successive refusals, each
with a different detail: the watcher absent; then a file-redirected follower rejected
because *"its follower writes to a file, which nothing reads until the command exits
— and a follower does not exit"*; then, with a harness-primitive follower armed and
`followers_live: 1`, still refused at `session_attached: False` because the project
watcher was delivering to another session's name. The preflight knows all three and
checks them serially, so each fix reveals the next — about 25 minutes of a 105-minute
window. Stating the three together, with the one required follower form, would cost
nothing.

## Sprints: what was done and what was deliberately not

`S9`, `S11`, `S12`, `S13` hold **zero active plans** and are stale by definition.
`S10` (5 active), `S14` (4 active) and `S15` (2 active) hold the live work.

**The consolidation to a single active sprint was proposed, not executed.** It
reassigns eleven live plans across three sprints, which reshapes the owner's planning
on a relayed instruction rather than a direct one. The proposal: move `S10` and
`S14`'s nine active plans into `S15` beside the two already there, and close
`S9`/`S11`/`S12`/`S13` as empty. One command per plan, pending the owner's word.

## State at handover

| | |
|---|---|
| worktrees registered | **0** — 39 reaped by this session, 29 by the member; `git worktree list` returns the main checkout alone |
| live pointers | 0 for this session; 1 project-wide, the unreconcilable run above |
| graph | checkpointed 2026-09-23, 2,369.8 MB, `gzip -t` verified, 19 s behind live at the time |
| services | `codex-neo4j` and `codex-embed` restored after an external `scancel` and healthy since |

## The four decisions that are rulings, not work

Recorded as `f-usb-four-questions-await-the-lead` on `unbound-source-backlog` and not
repeated here, because one record is better than two that can disagree. A successor
should treat all four as open: the graph checkpoint cadence, whether the WEST review
PR ships, which contract `semantic_similarity_check` is under, and worktree
back-pressure.

## Lane admission: how this repository actually resolves the local lane

Checked 2026-09-24 in answer to a direct question from the imas-ambix coordinator,
because a claim that this repository bypasses the router's admission gate was raised
by a member survey and **does not hold at the configuration level**. The configured
route is the router itself — `pyproject.toml:310`, `api-base =
"http://98dci4-gpu-0003:18802/v1"` — so this project's LLM traffic is admitted by the
gate and observes a declared pause.

Two specific resolution paths were asked about. Both measured:

| path | result |
|---|---|
| any code reading `~/public/imas-ambix/endpoints.json` | **none** — zero references anywhere under `imas_codex/` |
| a hardcoded port constant | `VLLM_PORT = 18800` at `settings.py:716`, exposed by `get_vllm_port()` at `:719` |

**The port constant is live, not dead**, which is the part worth a successor's
attention. `get_vllm_port()` is called at **`imas_codex/cli/tunnel.py:227`** and
**`imas_codex/cli/tunnel.py:1178`**, so the tunnel CLI forwards **18800** — an older
lane — rather than the router's **18802**. That is not an LLM-traffic bypass, because
the model calls go through the configured `api-base`; it is a stale forwarding target
in the tunnel path. A successor pointing it at the router should change the default
and keep the `IMAS_CODEX_VLLM_PORT` override.

**The engine-level exposure is upstream and not ours.** The lane document advertises
the engine directly and the engine binds `0.0.0.0`, so any consumer resolving the
lane through that document rather than through its own configured router address
bypasses both the gate and a declared pause. This repository is not such a consumer.

### The provenance repair is tracked twice; `f-mps-provenance-is-repo-relative` is the survivor

Two followups were raised for one repair within a minute of each other, by a
coordinator and a member working the same finding:

| followup | plan | done-when |
|---|---|---|
| `f-srr-minted-from-names-a-directory-that-is-gone` | `sn-release-readiness` | repo-relative storage, a test from outside the main checkout, repair of the twelve, a negative control |
| **`f-mps-provenance-is-repo-relative`** | `migration-project-state` | **the same, plus a test asserting a minted value contains no absolute path at all** |

**The second is the survivor, and the reason is the extra clause rather than the
location.** `sn-release-readiness` is the better home — it owns
`catalog_release.py` — but a stronger done-when beats a better filing. Asserting
that a minted value contains *no absolute path* is what keeps the repair correct;
rewriting the twelve existing values and adding a test for the known case leaves a
correct implementation with nothing preventing the next absolute path. That is the
defect class this sprint kept hitting once the obvious fail-opens closed, and it is
precisely why the original defect survived three weeks unnoticed.

A successor should implement `f-mps-provenance-is-repo-relative` and treat the
`sn-release-readiness` followup as its duplicate.

**One citation hazard this collision exposed, worth more than the duplication.** A
`git push` in a shared checkout reports a range whose endpoint is **the branch tip,
not your commit**. Two sessions committing seconds apart both see a range ending at
whichever landed last. A coordinator read that tip as its own SHA and cited a peer's
commit in an upward report. The commit that is yours is the one `git log` attributes
to your subject line, not the one the push range ends at — and an index-lock
collision between the two `git add` calls is the only reason anyone looked.
