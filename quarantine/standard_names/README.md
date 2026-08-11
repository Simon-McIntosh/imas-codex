# Quarantined standard-name reconciliation modules

These modules are retained for reference only. They are **inoperable**: each
one is driven by a `--manifest` file whose builder is referenced from nowhere a
shipped command can reach, so no path in the CLI or the pool loop can construct
the input they require.

They live outside the `imas_codex` package on purpose — nothing can import them,
and the live sweeps they each duplicate remain the single authority. Read them
for prior art; do not wire them back in without rebuilding the manifest
producer they depend on.
