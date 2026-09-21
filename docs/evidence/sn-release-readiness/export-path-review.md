# Would the catalog export silently drop or wrongly emit a name?

provisional: false

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

## The tenth failure has a second, independent cause

`test_manifest_sources_reconcile_emitted_excluded_and_non_nameable` is the one
red result that survives the producer-neutral run, so it is stale twice over.
Its second cause is the `manifest_generability` gate: it refuses a cut in which
any manifest source fails to reach a name, and the test's fixture manifest
deliberately contains three such sources (an exhausted identity, a `reviewed`
identity, and a `non_nameable_coordinate: time axis`).

**This is the deliberate contract, held by a newer suite, not a defect.** The
refusal was landed by `e0c2d5ef3` and `tests/standard_names/test_manifest_generability.py`
asserts it directly, on a fixture built from the same shapes:

```
test_a_cut_refuses_a_manifest_it_cannot_generate_rather_than_counting
    assert report.all_gates_passed is False
    assert [gate.gate for gate in failed] == ["manifest_generability"]
```

Both adjacent suites are green — `test_manifest_generability.py` and
`test_export_producer_predicate.py`: **10 passed, exit 0**
(`~/.config/reckon/crew/runs/export-review-042900/adjacent-export-suites.log`).
So the older assertion in `test_export_exclusion_ledger.py` contradicts the
newer one, and the newer one is the contract in force.

### The accounting was checked rather than assumed

`_MANIFEST_GENERABILITY_MECHANISMS` (`export.py:2405`) looked at first read as
though it omitted the `recorded_refusal` mechanism the classifier can return,
which would have put uncarried sources in the list but not in `mechanism_counts`
— a silent accounting hole. A direct probe refutes that: over a manifest with an
exhausted identity and a documented non-nameable coordinate,
`uncarried = 2`, `mechanism_counts = {'recorded_refusal': 1, 'attempt_budget_exhausted': 1}`,
`sum = 2`. The tuple does declare `recorded_refusal`; the misreading was mine.
The one genuine oddity is that `refusal_cause_not_recorded` is declared but
never returned — `_manifest_source_mechanism` returns `composition_not_scheduled`
for that case (`export.py:2438`). That is a dead declared member, cosmetic, and
recorded as a follow-on rather than a defect.

## Does the exclusion ledger close? Yes — and the guard was made to fire

A passing suite never shows a guard firing, so the closure guard was made to
fail on purpose. `_classify_export_population` was wrapped to swallow exactly one
of its own exclusion records, over the same four-name fixture population
(`~/.config/reckon/crew/runs/export-review-042900/ledger-closure-probe.log`):

```
POPULATION N        = 4
exported_count      = 1
exclusion reasons   = {'invalid_validation_status': 1, 'documentation_not_accepted': 1,
                       'unreviewed_name': 1}
emitted + excluded  = 4
unnamed-reason rows = 0
LEDGER CLOSES       = True

=== guard fired on a deliberately short ledger ===
all_gates_passed    = False
failing gates       = ['exclusion_accounting']
   issue: {'type': 'unattributed_identity', 'identities': ['docs_pending_name']}
   issue: {'type': 'exclusion_accounting_mismatch', 'accepted_population': 4,
           'emitted': 1, 'excluded': 2, 'accounted_total': 3, 'unaccounted': 1}
```

The guard refuses, and it **names the identity that went unaccounted** rather
than reporting a bare arithmetic mismatch — so a short ledger is recoverable,
not merely detected.

The same probe doubles as the load control for the staleness diagnosis: supplying
only the missing `_has_non_derived_producer` key to the fixture of failure 1
produces exactly the `exported_count == 1` and the three exclusion reasons that
the stale test asserts.

## Disposition per failure

| Test | Verdict | Cause |
|---|---|---|
| `ledger_closes_over_fixture_population` | stale test | fixture omits producer projection keys |
| `emits_generic_source_bindings_and_preserves_accounting` | stale test | same; downstream of an empty emission |
| `validates_cross_links_against_full_catalog` | stale test | same; empty cross-link set |
| `manifest_sources_reconcile_emitted_excluded_and_non_nameable` | stale test, twice | same, **plus** asserts the pre-`e0c2d5ef3` generability contract |
| `accepted_name_with_source_domain_never_resolves_to_the_bucket` | stale test | same |
| `name_with_no_resolvable_domain_is_reported_not_emitted` | stale test | same; reason masked by the earlier producer branch |
| `export_reader_withholds_valid_but_exhausted_names` | stale test | same; calls the classifier directly |
| `TestExportOmitsSupersessionLineage` ×3 (setup errors) | stale test | same; no domain YAML written |

**Zero of the ten are export defects.** No name is dropped without a named,
counted reason, and no name is emitted that should not be.

## One prior finding no longer reproduces

§11 records `test_non_quantity_token_collision_blocks_export_with_registry_citation`
as a live failure — "an export-exclusion gate that fails open is exactly the
class of defect this plan exists to catch". It was re-run rather than carried:
it **passes** in both runs above (1.40 s, the slowest case in the file). The
collision gate no longer fails open, and §11's second release-relevant failure
is closed by measurement.

## What a WEST cut should expect

The generability gate means a cut **refuses** unless every manifest source
reaches a name — an exhausted identity or a documented non-nameable coordinate
is enough to stop it. That is the designed behaviour and it is the same refusal
the export-gate-chain audit reported against the WEST cut. It is a scoping
input for the release, not a defect to repair here.

provisional: false
