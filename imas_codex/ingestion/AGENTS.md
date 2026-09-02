This file governs the `imas_codex/ingestion/` subtree — the source-file ingestion pipeline
and the lifecycle of the source files it consumes.



### SourceFile Lifecycle

```
discovered → ingested | failed | stale
```

Ingestion is interrupt-safe — rerun to continue. Already-ingested files are skipped.
