# Would the catalog export silently drop or wrongly emit a name?

provisional: true

Scope: the four export test files named in this node's measure, read against
`imas_codex/standard_names/export.py` and `imas_codex/standard_names/catalog_release.py`.
Worktree `audit-s14-20260921T042900/n-the-export-path-would-not-silently-drop-or-misemit`,
base `0994e0313131052f9a0e87f1b26a75e1cae1ec9e`.

## Headline

**No silent drop and no wrong emission.** Every exit from the export population is
attributable to exactly one named ledger reason, and the arithmetic closes. Of the
ten red results, **nine are stale test fixtures** and **one is a real gate defect**
that fails *closed* — it refuses a cut it should pass, which is the safe direction.

## The reproduction

```
.venv/bin/python -m pytest -p no:cacheprovider \
  tests/standard_names/test_export_exclusion_ledger.py \
  tests/standard_names/test_export_domain_resolution.py \
  tests/standard_names/test_export_deprecation.py \
  tests/standard_names/test_catalog_status_reconcile.py
```

Log: `~/.config/reckon/crew/runs/export-review-042900/export-tests.log`

```
7 failed, 16 passed, 1 warning, 3 errors in 20.59s
```

| # | Test | Symptom |
|---|---|---|
| 1 | `test_export_exclusion_ledger::test_export_ledger_closes_over_fixture_population` | `assert 0 == 1` exported |
| 2 | `test_export_exclusion_ledger::test_export_emits_generic_source_bindings_and_preserves_accounting` | `FileNotFoundError: .../equilibrium.yml` |
| 3 | `test_export_exclusion_ledger::test_export_validates_cross_links_against_full_catalog` | `set() != {'electron_density','ion_density'}` |
| 4 | `test_export_exclusion_ledger::test_manifest_sources_reconcile_emitted_excluded_and_non_nameable` | `report.all_gates_passed` is False |
| 5 | `test_export_domain_resolution::test_accepted_name_with_source_domain_never_resolves_to_the_bucket` | `[] != ['etendue_of_soft_xray_detector']` |
| 6 | `test_export_domain_resolution::test_name_with_no_resolvable_domain_is_reported_not_emitted` | reason `no_producing_source`, expected `missing_physics_domain` |
| 7 | `test_catalog_status_reconcile::test_export_reader_withholds_valid_but_exhausted_names` | `[] != ['accepted_valid']` |
| E1–E3 | `test_export_deprecation::TestExportOmitsSupersessionLineage` (3 setup errors) | `FileNotFoundError: .../general.yml` |

## The single-variable experiment that attributes nine of them

Every failing fixture hand-builds candidate rows and mocks `_fetch_export_population`.
The producer-evidence predicate `_has_producing_source` (`export.py:789`) reads three
projection keys — `_has_derived_producer`, `_has_non_derived_producer`,
`_has_live_child` — that those fixtures predate and therefore omit. `dict.get`
returns `None` for an absent key, so the predicate **fails closed** and the
candidate is excluded under reason `no_producing_source` (`export.py:863`).

To attribute rather than assume, a session plugin replaced *only* that predicate
with `lambda candidate: True` and the same four files were re-run:

```
1 failed, 25 passed, 1 warning in 12.76s
```

Log: `~/.config/reckon/crew/runs/export-review-042900/export-tests-producer-neutralised.log`

**Nine of ten red results are one cause.** Tests 2, 3, 5 and E1–E3 are downstream
consequences of an empty emission (no YAML file written, no cross-link set built);
tests 1 and 7 assert the pre-predicate exclusion counts; test 6 asserts a reason
that the earlier producer branch now masks.

### Are these stale expectations or a real defect?

**Stale expectations.** The predicate is correct on the live path and the fixtures
are the thing that is out of date:

- The live population query projects all three keys explicitly
  (`export.py:697–702`), computed from `PRODUCED_NAME` edges and from a live
  `HAS_PARENT` child. A live candidate therefore always carries `True` or `False`,
  never absent — so the fail-closed `get` is a fixture-only path.
- Withholding a name no source produces is the intended contract, landed with its
  own tests in `tests/standard_names/test_export_producer_predicate.py`
  (`test_export_refuses_name_with_no_producing_source`, `:149`), which pass.
- The exclusion is **named and counted**, not silent: the report carries
  `no_producing_source: 3` for the four-name fixture in test 1, and
  `exported 0 + invalid_validation_status 1 + no_producing_source 3 = 4 = N`.

The one thing worth recording as a follow-on rather than a defect: test 6's fixture
deliberately omits the `physics_domain` key, and `_classify_export_population`
guards the `missing_physics_domain` branch on `has_physics_domain` being present
(`export.py:838`), so a domainless name is reported under whichever *later* reason
fires first. The name is still excluded and still counted — the reason string is
less specific than the test expects, not absent.

## Does the exclusion ledger still close?

**Yes.** Every excluded identity carries exactly one `ExclusionRecord` with a
`reason`, and `_classify_export_population` (`export.py:796`) is structured as a
single `if/elif` chain with a terminal `if reason is None: eligible` — there is no
path that drops a candidate without appending a record. The closure assertion
itself is exercised and green: `test_export_refuses_when_ledger_does_not_close`
passes in both runs above. In the failing run the fixture arithmetic still closes:
0 emitted + 1 + 3 excluded = 4 candidates.
