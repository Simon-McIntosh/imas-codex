# The standard-name CLI diff, reviewed

provisional: true

Diff under review: `git diff 3f8c06fcef82b8a769ee5ec5d81ae83b4fe3c5c5..HEAD --
imas_codex/cli/sn.py`, **420 added lines / 10 removed** across four commits —
`050b4624f` (benchmark banner), `21e4739e9` (`sn restore compose`/`apply`),
`cca977cbb` (authority refusal at the command boundary), `ccc257763` (compose
seat probe). Reviewed at `923daf733d10dfd8591ebeabf688aeab79c36945`.

## HEADLINE — the restore path can reinstate a delete permission: **CONFIRMED**

`compose_archive_reconstruction_authority` copies the archived property map
wholesale and changes exactly one key:

```
imas_codex/cli/sn.py:8131    node_properties = dict(properties)
imas_codex/cli/sn.py:8132    node_properties["id"] = node_id
```

**Nothing filters, rewrites, defaults, or even reads `origin` anywhere in the
restore path.** The whole restore region is lines 8072–8420; `origin` does not
appear in it. That absence is measured with the instrument shown to see the
thing when it is present — the same grep finds 12 occurrences elsewhere in the
same file, including `sn.py:4740`, an existing option whose help text is
*"Name(s) to reset from catalog_edit to pipeline origin"*:

```
$ awk 'NR>=8072 && NR<=8420' imas_codex/cli/sn.py | grep -n 'origin'
NO MATCH in 8072-8420
$ grep -c 'origin' imas_codex/cli/sn.py
12
$ grep -n 'catalog_edit' imas_codex/cli/sn.py
1056:    # operation we treat it as a boolean (override any catalog_edit).
4740:    help="[export] Name(s) to reset from catalog_edit to pipeline origin (repeatable; or 'all')",
6398:        "cascade to rename descendants with origin='catalog_edit'. Without "
```

So the file already knows `catalog_edit` is a value that has to be reset, on a
different command, and the restore path does not.

### The reproduction

Composing one identity exactly as a pre-correction store dump holds it:

```
$ python -c 'compose_archive_reconstruction_authority({"properties": {"id": "power_due_to_fusion",
      "origin": "catalog_edit", "status": "draft", "name_stage": "accepted"}, "edges": [...]})'
composed node properties: {"id": "power_due_to_fusion", "name_stage": "accepted",
                           "origin": "catalog_edit", "status": "draft"}
origin survives compose: catalog_edit
compose parameters: ['record', 'identity', 'counterpart_renames']
```

The third line is the part that closes it: the function's entire surface is
`record`, `identity` and `counterpart_renames`, and the command exposes exactly
`--identity` and `--rename-counterpart` (`sn.py:8229-8245`). **There is no knob
that states an origin**, so an operator who wants the corrected value has to
know to hand-edit the extraction JSON before composing, and nothing in the
command, its help, or its output says so.

Downstream the value is written verbatim — the adapter's node statement is
`MERGE (node:StandardName {id: $id}) ON CREATE SET node = $properties` with a
`properties(node) = $properties` post-check, so the archived map is what lands
and a live node carrying anything different makes the apply refuse rather than
merge. The restored identity therefore comes back holding precisely the
property the archive held.

### Why this is the delete permission and not a stale label

`origin = 'catalog_edit'` is what made these identities eligible for the
automatic cleanup: a bulk write of that value across 2096 rows is what removed
93 identities on 2026-09-08. A restore composed straight from the dump returns
each identity **to the exact state that authorised its deletion**, and the
acceptance test in use — per-type incident edge counts against the archive —
passes such a restore without complaint, because edge counts say nothing about
node properties.

Two things make this hard to catch rather than obvious:

- **The adapter's own test fixture carries `"origin": "catalog_edit"`** on its
  archived node (`tests/standard_names/test_archive_reconstruction.py`), so the
  suite exercises this exact shape and asserts it round-trips. A green gate is
  evidence the value survives, not evidence it should.
- Every control on this property is on the write path of *other* commands and
  none is on this one.

### The minimal guard that closes it — stated, not implemented

**Make the restore state the origin it is reinstating, and refuse to infer
it.** Concretely, the smallest change that cannot be bypassed by forgetting:

1. `compose_archive_reconstruction_authority` takes a required `origin`
   argument, surfaced as a required `--origin` option on `sn restore compose`,
   and sets `node_properties["origin"]` from it rather than from the archive.
2. It refuses the value `catalog_edit` outright with a message naming the
   incident class, so the failure mode cannot be reached by passing the archived
   value through.
3. Where the archived map already carries an origin that differs from the
   stated one, the difference is echoed in the command's output, so the operator
   sees which value was dropped.

Required rather than optional, and required at *compose* rather than at apply,
because compose is where the property enters the signed payload: an apply-time
check would have to reopen a signed artifact to fix it. This is a guard on the
write path for a property whose controls are all on other write paths — the
point is that it becomes impossible to compose a restore without deciding.

**Not implemented here.** `imas_codex/cli/sn.py` is outside this node's write
scope; it is recorded under follow-ons.
