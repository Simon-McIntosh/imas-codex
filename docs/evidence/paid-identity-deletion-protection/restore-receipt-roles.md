# Restore receipt reports roles per identity

## Outcome

The archive reconstruction receipt now names, for every identity it restores,
which relationship roles came back and which the registry cannot put back. The
applied receipt carries `identity_roles`, keyed by archived identity, with two
per-type tallies:

| Tally | Meaning |
|---|---|
| `reinstated` | roles the reconstruction created back, taken from the live per-type count the parity guard read |
| `unreinstatable` | roles the archive record held that the reconstruction registry has no route for |

The parity guard already read every registry role per identity and raised on any
per-type difference, then discarded the comparison. That comparison is now the
receipt's content rather than a local variable.

## The defect and its reproduction

The receipt carried only aggregate counts, so a restore that reinstated an
identity could not say which roles returned. Nothing recorded it, and a role the registry cannot route was silently absent from the receipt even though the archive held it.

Reproduced by swapping the module to the base revision and running the receipt
tests against it:

```
$ git show 9d7e57f85:imas_codex/standard_names/signed_manifest.py > imas_codex/standard_names/signed_manifest.py
$ uv run pytest -p no:cacheprovider tests/standard_names/test_archive_reconstruction.py -k receipt
E   KeyError: 'identity_roles'
E   KeyError: 'identity_roles'
FAILED tests/standard_names/test_archive_reconstruction.py::test_archive_reconstruction_receipt_records_reinstated_roles_per_identity
FAILED tests/standard_names/test_archive_reconstruction.py::test_archive_reconstruction_receipt_names_unreinstatable_archived_roles
2 failed, 11 deselected, 1 warning in 7.30s
```

The base revision carries no `identity_roles` label anywhere in the module, so
the key is absent and a caller reading it raises. The reproduction run used
`-k receipt`, so the two tests selected were the two receipt tests. The module was then read-only restored; the tree is clean and the marker reads `True` afterwards.

## How the archive's own role counts enter

A role outside the reconstruction registry cannot be expressed as a bounded
reconstruction edge — the loader refuses any edge whose relationship type or
direction is outside the registry, and `test_archive_reconstruction_refuses_relationship_outside_registry`
asserts that refusal, which still passes. So the archive record's own
per-identity role counts enter through a new optional `archive_roles` object in
the authority, which can name a role no edge could carry.

`EVIDENCED_BY` was the only one of the 35 incident roles in the prior-art census
with no registry route (34 roles are routable). That role now appears in the
receipt as unreinstatable rather than vanishing, and the archive still refuses to
carry it as an edge.

## Guarding the archived counts

A registry role the archive held in greater number than the reconstruction edges
cover is refused, not silently under-restored:

```
archive reconstruction cannot reinstate an archived role the edges do not cover:
HAS_UNIT holds 2 in the archive and 1 in the reconstruction edges
```

Covered by `test_archive_reconstruction_refuses_archived_role_beyond_reconstruction_edges`.

## Code and tests

| Commit | Change |
|---|---|
| `a0bb18ccb` | `signed_manifest.py`: `identity_roles` on both the applied and the preview/refused receipt, `_load_archive_role_counts`, `_archive_role_outcomes` |
| same | `test_archive_reconstruction.py`: three tests, authority helper gained `archive_roles` |

The receipt is built on the parity loop's own read: `_archive_edge_counts` per
identity, which the guard already computes, is captured into the receipt rather
than discarded.

## Gate

`tests/standard_names/test_archive_reconstruction.py` on `all_debug`:

```
13 passed, 1 warning in 5.20s
```

Ten pre-existing tests plus the three added; the refusal-outside-registry and
parity-mismatch tests still pass, so the new receipt does not weaken the two
refusals. The reproduction run above swaps in the base module; the restored tree
marker check returns `head markers: 2` for `identity_roles`.

## Acceptance

- receipt names per-identity per-role outcomes: **13-passed gate**, receipt
  carries `identity_roles` on both preview and applied;
- `EVIDENCED_BY` named as unreinstatable when the archive record carries it:
  `test_archive_reconstruction_receipt_names_unreinstatable_archived_roles`;
- the ten pre-existing archive reconstruction tests still pass: **yes** (13 =
  10 + 3);
- no reconstruction edge carries a registry-external role: **asserted** in
  `test_archive_reconstruction_receipt_names_unreinstatable_archived_roles`.

## Follow-on

The archive record's `archive_roles` object is populated by the caller, not by
this node; the typed archive adapter that reads the archive dump per identity is
with the typed adapter (etendue-restore sweep remains separate). This node
supplies the receipt surface and its guard.