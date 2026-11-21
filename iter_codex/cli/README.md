# CLI Architecture

The ITER Codex CLI follows a hierarchical structure with domain-specific subgroups.

## Discovery Commands

The `discover` command runs facility discovery in stage order. Select domains with `--only`:

```
iter-codex discover
├── <facility>                  # Run the full discovery sequence
│   ├── --only <domain>         # Run selected domains
│   └── --flush                 # Drain work already seeded in the graph
├── status <facility>           # All domains
├── clear <facility>            # All domains (nuclear reset)
├── seed <facility>             # Seed root paths
└── inspect <facility>          # Debug view
```

## Design Principles

1. **One sequence**: `discover <facility>` runs the domains in dependency order.
2. **Domain selection**: `--only <domain>` limits work to named domains.
3. **State commands**: `discover status` and `discover clear` inspect and reset discovery data.

## Examples

```bash
# Show status for all domains
iter-codex discover status tcv

# Show status for a specific domain
iter-codex discover status tcv -d wiki

# Clear all discovery data
iter-codex discover clear tcv

# Clear only wiki data
iter-codex discover clear tcv -d wiki

# Run wiki discovery
iter-codex discover tcv --only wiki --cost-limit 5.0
```

## Status Output

`discover status <facility>` shows stats for all domains:
- **Paths**: discovered/scanned/scored counts, purpose distribution, high-value paths
- **Wiki**: pages/chunks/artifacts counts, accumulated LLM cost
- **Signals**: signal/enrichment counts

Use `-d/--domain` to filter: `discover status tcv -d wiki`
