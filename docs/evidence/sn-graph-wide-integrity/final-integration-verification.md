# Final integration verification

Date: 2026-08-22

Tested HEAD: `2a06f062c5473ff483436b791e8188eb73734e1d`

Verdict: **PASS**. The integrated grammar pin, dimensional-unit veto, and
watchdog determinism changes pass together. The complete Standard Names suite,
the complete units suite, ten consecutive watchdog repetitions, the six
established focused integration files, the public grammar/unit assertions, the
lock provenance scan, and both Ruff gates all completed with zero failures.

## Quantitative gates

| Gate | Result | Exit |
|---|---:|---:|
| `tests/standard_names/` | **6,576 passed**, 8 skipped, 286 deselected, **0 failed, 0 errors** | 0 |
| `tests/units` | **27 passed**, **0 failed, 0 errors** | 0 |
| Younger-replica watchdog test, ten sequential invocations | **10 passed**, **0 failed** | 0 |
| Six focused integration files | **19 passed**, 17 skipped, **0 failed, 0 errors** | 0 |
| Public unit and grammar assertions | **7 assertions passed**, **0 failed** | 0 |
| Previous rc66 commit scan in `uv.lock` | **0 references** | 0 |
| `ruff check --no-cache .` | All checks passed | 0 |
| `ruff format --check --no-cache .` | 1,178 files already formatted | 0 |

The full suite's eight skips are the repository's existing environment-marked
cases under its default marker policy. The six focused files were also run as a
separate explicit group. Their graph-backed cases require the disposable Neo4j
endpoint used by the earlier integration verification; this node did not
provision that external fixture, so 17 graph cases skipped while all 19
locally executable cases passed. The result claimed here is therefore exactly
the requested **zero failures per focused file**, not a claim that every graph
case re-executed.

## Six focused integration files

| File | Passed | Skipped | Failed |
|---|---:|---:|---:|
| `test_source_target_reconciliation.py` | 1 | 3 | **0** |
| `test_structural_source_revival.py` | 1 | 2 | **0** |
| `test_ordinary_source_migration.py` | 1 | 5 | **0** |
| `test_repair_authority_builder.py` | 14 | 0 | **0** |
| `test_geometry_base_projection.py` | 2 | 0 | **0** |
| `test_normalization_peel_unit_repair_graph.py` | 0 | 7 | **0** |

## Unit semantics and grammar pin

The public assertion capture verifies dimensional equivalence is necessary but
not sufficient:

| Pair | `units_agree` | Required meaning |
|---|---:|---|
| `Hz` / `s^-1` | `True` | equivalent frequency spellings |
| `N.m^-2` / `kg.m^-1.s^-2` | `True` | equivalent pressure spellings |
| `J` / `N.m` | `False` | energy remains distinct from torque |
| `Hz` / `Bq` | `False` | frequency remains distinct from activity |
| `Gy` / `Sv` | `False` | absorbed dose remains distinct from equivalent dose |

The same public-API capture reports installed
`imas_standard_names.__version__ == "0.8.0rc67"` and reads the definition through
`get_grammar_context()`:

```text
The flux-surface location at which the safety-factor magnitude attains its minimum.
```

The lock scan searched for the complete previous rc66 commit
`6dd6eae9585f4244fe1ae164604d1de278eb82d0` and found **0 references** in
`uv.lock`.

## Watchdog determinism

The exact regression
`TestIdleExhaustionWatchdog::test_younger_replica_keeps_its_age_window_after_older_finishes`
ran in ten separate, sequential pytest invocations. Every invocation reported
one pass and exit 0; the aggregate capture reports **10 passed, 0 failed**. This
avoids pytest's collection deduplication of repeated identical node IDs and
proves ten real executions rather than ten aliases for one collected item.

## Retained logs

All Python commands reused `/home/ITER/mcintos/Code/imas-codex/.venv` through
`UV_PROJECT_ENVIRONMENT`, set `PYTHONPATH` to this worktree, and used
`uv run --no-sync`; no worktree-local environment or dependency sync was
created. Full command output is retained under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T095158804912-finalverify/logs/`:

| Log | SHA-256 |
|---|---|
| `standard-names-full.log` | `f031dadc7a946e417d0bca00765d7d2d1d661c5001b6101422d0bf840b462e2b` |
| `units.log` | `8ef71340be5f069f02393720e098573679dc07220a025d3b4d4cc685b079184b` |
| `watchdog-repeat-10.log` | `30880aec766e76f4b4ffaa0f56f2289088e5726eb83abd0ec2b3c5cdf800974a` |
| `six-focused-suites.log` | `d25ce91d1c4fecf8b315adac5eab857e3d3f89e7f85090652ee5b7f6d30c4329` |
| `semantic-assertions.log` | `6290a9fe6f82dd396f041a8456b4f9f1c1fa9f4545869e490437e0709000c4e3` |
| `lock-rc66-scan.log` | `484bfd0aa391403beacaba6f4db17556902811176920a70f96a4e8d9413903eb` |
| `ruff-check.log` | `7e8e995beb6b942867c80817910cf3a80d4b9cb10974bc60ce55e6e21f00e688` |
| `ruff-format-check.log` | `06739e9ff7db051cd52a7653d62d2186cbfffa9be617067eb5c90eae10a9c9b5` |
| `head-sha.log` | `336ff2f8b26d69bfc6fa4baf030747b74d87a69465e70266cefc96ba119e9e9e` |

Two non-failing warnings remain visible in the pytest logs: the already-known
`RepairAuthorityArtifact.schema` field-shadow warning and pytest's unknown
`cache_dir` configuration warning. Neither produced a failure or error, and no
out-of-fence source or configuration file was changed by this verification
node.
