# Zellij Session Quickstart

Organize work using **project-scoped sessions** with multi-tab agent coordination. Connect from WSL via the `cx` command. Sessions persist across SSH disconnects.

## Architecture

```
WSL Client
 └─ cx                    → SSH → iter → zellij session "codex"      (infrastructure)
 └─ cx iter imas-codex    → SSH → iter → zellij session "imas-codex" (agents + mcp + git)
 └─ cx iter efitpp        → SSH → iter → zellij session "efitpp"     (agent + build + run)
 └─ cx iter tcv           → SSH → iter → zellij session "tcv"        (facility discovery)
```

All sessions run on the **same iter login node**. The `cx` command handles SSH + PTY + focus-reporting fixes for Windows Terminal/WSL.

## Session Types

### Project Sessions

| Session        | Layout       | Use Case                    |
|----------------|--------------|-----------------------------|
| `imas-codex`   | project.kdl  | Codex development           |
| `efitpp`       | project.kdl  | EFIT++ development          |
| `codex`        | codex.kdl    | Infrastructure services     |

Project sessions start with a single shell tab. Add tabs as needed — they persist across reconnects via `session_serialization`.

### Facility Sessions (discovery pipelines)

| Session    | Layout        | Tabs                                            | Use Case           |
|------------|---------------|-------------------------------------------------|--------------------|
| `tcv`      | facility.kdl  | `wiki`, `paths`, `code`, `docs`, `signals`, `map` | TCV discovery    |
| `jet`      | facility.kdl  | (same tabs)                                     | JET discovery      |
| `jt-60sa`  | facility.kdl  | (same tabs)                                     | JT-60SA discovery  |

### Layout Resolution

The `cx` script auto-selects layouts:

1. **Exact match:** `~/.config/zellij/layouts/$SESSION.kdl` (e.g., `codex.kdl`)
2. **Known facility:** `facility.kdl` for tcv, jet, jt-60sa, iter
3. **Generic fallback:** `project.kdl` (single shell tab, add more as needed)

## Install & Sync

```bash
# Sync all layouts, config, cx, and tools to remote host
cx sync iter
```

This copies `~/.config/zellij/layouts/*.kdl`, themes, config (with path rewriting), and `~/.local/bin/{cx,focus-filter,plan,view,glow}` to the remote host.

## Daily Workflow

### 1. Start infrastructure (once)

```bash
cx                          # Attach to "codex" session on iter
```

In the `codex` session tabs:

| Tab       | Command                                              |
|-----------|------------------------------------------------------|
| `graph`   | `uv run imas-codex graph start`                      |
| `embed`   | `uv run imas-codex embed start`                      |
| `tunnel`  | `uv run imas-codex tunnel start iter`                |
| `llm`     | general purpose shell / llm monitoring               |

### 2. Open a project session

```bash
# Each command opens a separate SSH + zellij session
cx iter imas-codex     # Codex development
cx iter efitpp         # EFIT++ work
```

First launch creates a single shell tab. Add tabs (`Alt+t` → `n`) as your workflow grows — they persist across SSH drops and reconnects.

### 3. Open facility discovery sessions

```bash
cx iter tcv
cx iter jet
cx iter jt-60sa
```

In the `tcv` session (for example):

| Tab        | Command                                         |
|------------|--------------------------------------------------|
| `wiki`     | `uv run imas-codex discover tcv --only wiki`      |
| `paths`    | `uv run imas-codex discover tcv --only paths`     |
| `code`     | `uv run imas-codex discover tcv --only code`      |
| `signals`  | `uv run imas-codex discover tcv --only signals`   |
| `docs`     | `uv run imas-codex discover tcv --only documents` |
| `map`      | `uv run imas-codex discover tcv --only candidates,mapping` |

### 4. Monitor progress

```bash
# Watch live stats in any tab
watch -n 10 uv run imas-codex discover status tcv

# Or check logs
tail -f ~/.local/share/imas-codex/logs/paths_tcv.log
```

## Session Navigation

### Switching Between Sessions

| Action                          | Key / Command                                 |
|---------------------------------|-----------------------------------------------|
| Detach (keeps session alive)    | `Ctrl+q`                                      |
| Reattach from WSL               | `cx iter imas-codex`                          |
| Switch session from inside      | `cx iter tcv` (from within any zellij session)|
| List sessions (from iter shell) | `zellij list-sessions`                        |
| Kill a session                  | `zellij kill-session tcv`                     |

### Within a Session

| Action              | Key                        |
|---------------------|----------------------------|
| Tab mode            | `Alt+t`                    |
| Next/prev tab       | `h`/`l` (in tab mode)     |
| Go to tab N         | `1`-`9` (in tab mode)     |
| New tab             | `n` (in tab mode)         |
| Rename tab          | `r` (in tab mode)         |
| Pane mode           | `Alt+p`                   |
| Move focus (panes)  | `Alt+h/j/k/l`             |
| Scroll mode         | `Alt+s`                   |
| Search in scroll    | `s` (in scroll mode)      |
| Lock (pass-through) | `Alt+g`                   |

### Recommended: Multi-Tab Windows Terminal

Open multiple WSL terminal tabs, one per session:

```
WT Tab 1:  cx                   → codex session (infra)
WT Tab 2:  cx iter imas-codex   → imas-codex session (agents)
WT Tab 3:  cx iter tcv          → tcv session (discovery)
WT Tab 4:  cx iter jet          → jet session (discovery)
```

Each WSL tab maps to one remote zellij session. Close and reopen any WSL tab without losing state.

## Session Persistence

- **Atomic creation:** `cx` uses `zellij attach --create` to atomically attach or create sessions, preventing the race condition where a dropped SSH connection causes a duplicate server that orphans the original.
- **Session serialization:** Enabled (`session_serialization true`) so sessions survive server crashes.
- **SSH drops:** Just run `cx iter imas-codex` again — it reattaches to the existing session.
- **Orphan cleanup:** `cx` automatically cleans dead sessions and stale resurrection data on startup.
- **Version upgrades:** `cx` detects and kills orphan zellij servers from old versions that use incompatible socket paths.

## Tips

- **Sessions are persistent.** SSH drops, laptop sleep, network changes — just run `cx` again to reattach.
- **Don't nest zellij.** If inside zellij with 2+ args (e.g., `cx iter tcv`), `cx` switches sessions locally instead of nesting.
- **Logs over pipes.** Never pipe `imas-codex` CLI output. Check `~/.local/share/imas-codex/logs/` instead.
- **New tabs on the fly.** `Alt+t` → `n` to add a tab, `r` to rename it.
- **Clean slate.** `zellij kill-session <name>` or `zellij delete-all-sessions` to start fresh.
