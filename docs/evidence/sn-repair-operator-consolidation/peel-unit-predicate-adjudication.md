# Normalization-peel unit predicate adjudication

## Decision

**The live every-child Cypher predicate is wrong.** The production docstring's
every-unit-bearing-child reading is the intended contract. A child whose
`unit` property is null supplies no unit authority and therefore cannot veto
repair of a parent whose recorded `name_unit_consistency_check` already proves
that its scalar unit `1` contradicts the parent's name.

This decision changes **exactly 1 of the 7 pinned behavioral-suite verdicts**.
The null-unit normalization-child case changes from refusal (`[]`) to admission
(`['particle_mass']`). The other 6 verdicts remain unchanged.

## Conflicting production evidence

### Live predicate: every child

`imas_codex/standard_names/graph_ops.py:4237-4242` currently says:

```cypher
MATCH (c)-[:HAS_PARENT]->(sn)
WITH sn, collect(c) AS kids
WHERE all(k IN kids
          WHERE k.unit = '1'
            AND any(t IN split(k.id, '_')
                    WHERE t IN ['normalized', 'normalised']))
```

Because `k.unit = '1'` is not true for a missing/null property, any null-unit
child makes `all(...)` fail. The predicate therefore treats every child as a
unit witness.

### Production docstring: every unit-bearing child

`imas_codex/standard_names/graph_ops.py:4210-4223` states:

> Before the seeder excluded normalization-peel children from unit inheritance,
> a derived parent whose only unit signal was a `normalized_*` child was stamped
> `'1'` [...]. Repair is scoped to parents where ALL THREE hold [...]:
> `origin='derived'` with unit `'1'` and no normalization marker of its own;
> **every unit-bearing child** is a dimensionless normalization variant (the
> only possible source of the inherited `'1'`); and a
> `name_unit_consistency_check` finding is on record.

The qualifier "unit-bearing" excludes a child with no `unit` property from the
child-unit predicate. It does not remove the independent finding guard: the
finding is what establishes that the parent's present dimensionless stamp is
wrong, while the caller remains responsible for validation re-stamping
(`graph_ops.py:4225-4226`).

## Decisive unit-authority rule

The shared production unit-selection path defines the authoritative child set.
`imas_codex/standard_names/graph_ops.py:3682` describes it as:

> Select children whose units authoritatively constrain a parent.

Its consumer then constructs the authoritative unit set at
`imas_codex/standard_names/graph_ops.py:3718-3724`:

```python
eligible_unit_children = _eligible_derived_parent_unit_children(
    parent_id, child_data
)
eligible_units = {
    str(child["unit"])
    for child in eligible_unit_children
    if child.get("unit")
}
```

The same rule is repeated by the shared helper at
`imas_codex/standard_names/graph_ops.py:4735-4741`:

```python
eligible = _eligible_derived_parent_unit_children(parent_id, child_data)
units = {str(child["unit"]) for child in eligible if child.get("unit")}
return next(iter(units)) if len(units) == 1 else None
```

The deciding clause is `if child.get("unit")`: a null-unit child is outside the
unit-authority set. This is consistent with the repository rule headed "Units
are DD-authoritative" at `imas_codex/standard_names/AGENTS.md:103-106`, which
states that unit is injected post-LLM, and with the prohibition on bulk source
re-derivation at `AGENTS.md:121-124`. A missing unit is not contrary unit
evidence.

Normalization-peel children are already explicitly excluded from parent-unit
inheritance at `graph_ops.py:3754-3762`: their dimensionless unit is correct for
the child and wrong for a dimensional parent. The repair does not infer a
replacement unit from the null child; it only removes a parent stamp already
contradicted by the recorded consistency finding.

## Concrete correction

Keep every existing parent guard and mutation unchanged. Replace only the child
cohort portion of the Cypher with an explicit unit-bearing projection:

```cypher
MATCH (c)-[:HAS_PARENT]->(sn)
WITH sn, [k IN collect(c) WHERE k.unit IS NOT NULL] AS unit_kids
WHERE all(k IN unit_kids
          WHERE k.unit = '1'
            AND any(t IN split(k.id, '_')
                    WHERE t IN ['normalized', 'normalised']))
```

This makes the predicate match its docstring and the shared unit-authority
rule. In Cypher, `all(...)` over an empty list is true, so a parent whose only
child has a null unit is admitted when—and only when—the existing parent guards
also hold: derived origin, parent scalar unit `1`, no parent normalization
marker, and a recorded `name_unit_consistency_check` finding. A unit-bearing
non-normalization child or a child bearing a non-dimensionless unit continues
to refuse the repair.

No production or test change is made in this investigation node. The correction
belongs in a separately authorized implementation node, followed by an update
to the one pinned test whose expected verdict changes.

## Seven-case verdict impact

The cases are the seven graph-marked tests in
`tests/standard_names/test_normalization_peel_unit_repair_graph.py:159-340`.

| Pinned case | Current verdict | Corrected verdict | Changes? |
|---|---|---|---:|
| Mixed six-parent cohort admits exactly `electric_current` and `particle_mass` | those 2 admitted | same 2 admitted | No |
| Parent has its own normalization marker | `[]` | `[]` | No |
| Parent lacks the recorded name/unit finding | `[]` | `[]` | No |
| Unit-bearing non-normalization child is present | `[]` | `[]` | No |
| Only child is `normalized_particle_mass` with `unit=None` | `[]` | `['particle_mass']` | **Yes** |
| Scalar-only candidate has no `HAS_UNIT` edge | `['particle_mass']` | `['particle_mass']` | No |
| Replay after a successful repair | `[]`, byte-identical graph | same | No |

**Quantitative result: 1 changed verdict, 6 unchanged verdicts, 7 cases total.**

## Scope statement

This node changes **0 production-code files and 0 test files**. Its sole output
is this adjudication artifact.
