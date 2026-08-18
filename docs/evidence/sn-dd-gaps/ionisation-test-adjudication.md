# Ionisation-potential unit test adjudication

## Verdict

The failing expectation was stale; the current unit-authority implementation is
behaving as designed. The test had grouped one exact, actively resolved base path
with one unresolved descendant and expected the legacy graph correction to own
both. Those paths now intentionally use different authorities:

- `edge_profiles/ggd/ion/state/ionisation_potential` has an exact active unit
  resolution from observed `e` to effective `eV`.
- `plasma_profiles/ggd/ion/state/ionisation_potential` has the same exact active
  resolution.
- `plasma_profiles/ggd/ion/state/ionisation_potential/values` has no exact active
  record and remains covered by the legacy wildcard correction.

The regression now asserts both halves of that boundary: the two exact base paths
preserve raw `e` in `resolve_dd_unit`, while the unresolved descendant continues
to return corrected `eV`.

## Code evidence

- `imas_codex/units/dd_unit_exceptions.py:86-110` looks up exact active unit
  authority and retires a legacy pair only when its observed and effective values
  match the legacy pair.
- `imas_codex/units/dd_unit_exceptions.py:147-188` skips a matching
  `correct_in_graph` rule when that retirement predicate succeeds; otherwise it
  returns the legacy correction.
- `imas_codex/units/__init__.py:157-194` applies a returned legacy correction, or
  normalizes the raw declaration when no legacy correction is returned.
- `imas_codex/standard_names/dd_resolutions.py:1234-1256` defines active authority
  as an exact path, DD version, and field lookup.
- `imas_codex/standard_names/dd_resolutions.py:1607-1643` applies the effective
  value from an exact active record when a consumer resolves the raw DD field.
- `imas_codex/standard_names/config/dd_resolutions.yaml:908-945` records the two
  active base-path unit resolutions with observed `e` and effective `eV`.

A direct probe returned:

```text
edge_profiles/ggd/ion/state/ionisation_potential                 legacy=None resolved=e
plasma_profiles/ggd/ion/state/ionisation_potential               legacy=None resolved=e
plasma_profiles/ggd/ion/state/ionisation_potential/values        legacy=eV   resolved=eV
```

## Authority record

The legacy-retirement record at
`docs/evidence/archive/sn-dd-gaps-landed.html:156-165` states that legacy DD unit
correction became inert for exactly the paths carrying effective active authority:
66 exact path-to-rule bindings across 21 legacy rows were retired, while 41 of 62
legacy rows were left untouched. The live plan repeats those headline counts at
`docs/sn-dd-gaps.html:34-40`.

The remaining ionisation-potential cohort is explicitly incomplete at
`docs/sn-dd-gaps.html:75-86`: the full governed set is 14 paths, but expanding it
still needs renewed authority. Thus the active base-path records retire the legacy
fallback exactly, while the not-yet-active `/values` descendant correctly remains
on the legacy correction.

## Verification

- Focused authority-boundary regression: 2 passed, 0 failed.
- Complete `tests/core/test_dd_unit_resolution.py`: 23 passed, 0 failed, one
  pre-existing pytest configuration warning.
- Ruff format: passed; Ruff check: passed.
- Durable logs:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260818T205404690392-ionisation-test-adjudication/logs/`.
