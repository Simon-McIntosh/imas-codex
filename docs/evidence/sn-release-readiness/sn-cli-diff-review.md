# The standard-name CLI diff, reviewed

provisional: false

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

## 2 · The compose-seat guard is bypassed by an absent key — **CONFIRMED**

This is the recurring shape the review was told to look for, and it is here
verbatim: a guard that short-circuits when a key is **absent**, so the
eligibility clause after it never runs and the seat is wrongly **admitted**.

```
imas_codex/cli/sn.py:136-142   served = [str(entry.get("id","")) for entry in json.loads(body).get("data", []) ...]
imas_codex/cli/sn.py:143-147   except (ValueError, AttributeError, TypeError): return True, model_label
imas_codex/cli/sn.py:149-150   if not served: return True, model_label
imas_codex/cli/sn.py:151-153   if model_label in served: return True, model_label
                               return False, f"serves ..., not {model_label}"
```

`.get("data", [])` defaults an absent key to empty, `if not served` then returns
healthy, and `model_label in served` — the entire point of the change — is never
evaluated. Driven directly, with the gate's own two cases as the positive
control that the instrument distinguishes a match from a mismatch:

```
the gate's own match case           -> (True, 'deepseek-v4-flash')
the gate's own mismatch case        -> (False, 'serves deepseek-v4.1-flash, not deepseek-v4-flash')
body is a JSON list, not an object  -> (True, 'deepseek-v4-flash')
no data key (proxy reshaped it)     -> (True, 'deepseek-v4-flash')
data entries are plain strings      -> (True, 'deepseek-v4-flash')
served id is prefix-qualified       -> (False, 'serves deepseek-ai/deepseek-v4-flash, not deepseek-v4-flash')
```

Rows three to five are three distinct bodies that all name a model the seat does
not match, and the guard admits all three. `tests/standard_names/test_sn_service_checks.py`
constructs `{"data": [{"id": ...}]}`, an unparsable body and an empty listing,
and none of these three — so **its own gate passes while the guard is bypassed
by an input the gate never constructs.**

**Failure scenario.** The endpoint is fronted by a proxy that returns
`{"models": [...]}`, or a server whose `data` entries are bare strings. The
probe reports healthy, `_require_local_compose_ready` admits the run, and
generation 404s per request — the precise failure `ccc257763` was written to
convert into a configuration message.

**The fail-open is deliberate and the commit says so** ("treated as usable
rather than inventing a mismatch... fails closed only on positive evidence"),
and as a policy that is defensible: blocking every compose run on a listing
format is worse than a late 404. **What is not defensible is the return
value.** All three bypass rows return `(True, model_label)`, and `model_label`
is the **configured** name, while the docstring at `sn.py:83-85` states
`healthy → detail is the served model's short name`. The probe reports a
configured value in a field documented as an observation — an unmeasured value
presented as a measurement. The minimal repair is to distinguish them: return
the served name only when it was actually read, and a distinct detail such as
`"listing unreadable"` otherwise, so a reader can tell a confirmed seat from an
unchecked one.

## 3 · The seat comparison normalises one side only — **CONFIRMED (mechanism), PLAUSIBLE (reach)**

`model_label = (cfg.get("model") or "").rsplit("/", 1)[-1]` at `sn.py:104`
strips the provider prefix from the configured seat; the served ids from
`/models` are compared **unstripped** (`sn.py:151`). A server launched with an
org-qualified or path-like served name therefore never matches:

```
served id is prefix-qualified -> (False, 'serves deepseek-ai/deepseek-v4-flash, not deepseek-v4-flash')
```

That is the opposite failure from finding 2 and the more disruptive one:
`_require_local_compose_ready` raises a `ClickException` (`sn.py:191-198`), so
**every compose run is blocked against a server that does in fact serve the
configured seat.** Marked PLAUSIBLE on reach rather than CONFIRMED because it
depends on how the endpoint is launched — `ccc257763`'s own measurement reports
the live server offering the bare name `deepseek-v4.1-flash`, which compares
like-for-like today. Apply `rsplit("/", 1)[-1]` to both sides, or compare on the
suffix.

The live-endpoint half of this question belongs to the concurrently dispatched
compose-seat node; recorded here because the asymmetry is visible in the diff
under review and is a property of the code rather than of the endpoint.

## 4 · A counterpart rekey is consumed by the first matching edge — **CONFIRMED**

`renames.pop(str(counterpart_id))` at `imas_codex/cli/sn.py:8165` removes the
rekey from the map as it applies it, so an identity holding two edges to the
same renamed `StandardName` counterpart gets one rekeyed and one not:

```
$ compose(..., counterpart_renames={"old_parent": "new_parent"})   # two edges to old_parent
  HAS_PARENT -> new_parent
  REFERENCES -> old_parent
```

The `pop` is load-bearing — the residue drives the "a rekey names no archived
counterpart" refusal at `sn.py:8192-8196` — so it cannot simply become a lookup
without moving that check. Two outcomes, and only one is loud:

- the archived id no longer exists live → the adapter's counterpart `MATCH`
  fails and the apply raises `archive relationship counterpart changed before
  reconstruction`. Recoverable.
- **the archived id does still exist live** — a tombstone, a placeholder, or a
  name since reused → the edge is created against the wrong node **with no
  refusal at any layer**, because the parity guard compares per-role counts and
  never endpoints, so the count is correct and the restore reports success.

**Failure scenario.** Restoring a member of the pedestal-top density family
whose parent was renamed, where the archived parent spelling still exists as a
superseded identity: the `HAS_PARENT` edge is rekeyed, a second edge to the same
parent is bound to the superseded node, and the receipt says the restore
succeeded.

## 5 · The census counts a role before direction is routed, blocking the restore — **CONFIRMED**

`census[relationship_type] = census.get(relationship_type, 0) + 1` at
`imas_codex/cli/sn.py:8146` runs **before** the `(type, direction)` lookup
decides whether the edge is routable. Most registry roles are registered in one
direction only, so an archived edge carrying such a role in the other direction
is counted into the census, omitted from the edges, and the two disagree by
construction:

```
$ compose({... HAS_UNIT/outgoing, HAS_LOCUS/incoming ...})
  census: {'x': {'HAS_LOCUS': 1, 'HAS_UNIT': 1}}
  routed: ['HAS_UNIT']
```

The loader then refuses the whole authority — *"cannot reinstate an archived
role the edges do not cover: HAS_LOCUS holds 1 in the archive and 0 in the
reconstruction edges"* — so compose emits, with no warning, an artifact that
`sn restore apply` cannot load, and the message names coverage rather than
direction so it reads as an extraction defect. Note the asymmetry: a role with
**no** registry entry passes through as counted-and-unreinstatable, while a role
with an entry in the wrong direction is fatal. Compose should classify the
unroutable direction the same way it classifies an unroutable role, and say so
at compose time.

## 6 · The exception the command boundary catches — **verified correct, no finding**

`cca977cbb` wraps `apply_signed_manifest` in a `try` and re-raises exactly one
exception class as a usage error (`sn.py:8331-8348`). This was checked against
the swallowed-exception shape and is not one: a single named class is caught,
the loader's own message is carried into the replacement, `from exc` preserves
the chain, nothing is retried, and no other exception is intercepted — a
programming fault inside the adapter still surfaces as a traceback. The refusal
becomes more legible and no refusal is lost.

## What was verified sound and is not a finding

- `--apply` requires `--manifest-sha256` from a preview (`sn.py:8355`) and a
  digest without `--apply` is a usage error (`:8357`); the adapter re-reads the
  closure inside the transaction and compares, so the authorisation cannot be
  replayed against a graph that moved.
- Both digests are recomputed from the file the command was given, which is not
  vacuous only because the loader independently requires the embedded
  `signature.sha256` to equal the canonical payload digest. Worth stating
  plainly: that signature is **tamper-evidence for transport, not
  authorisation** — `signed_payload_sha256` is importable and `compose`
  re-signs, so any operator who can run the apply can sign arbitrary content.
- The node write is `MERGE … ON CREATE SET node = $properties` with a
  `properties(node) = $properties` post-check, so a live node holding different
  properties makes the apply refuse rather than overwriting it.

## Findings index

| # | file:line | Verdict | Failure scenario |
|---|---|---|---|
| 1 | `cli/sn.py:8131` | **CONFIRMED** | restore returns an identity carrying `origin: catalog_edit`, the value that authorised its deletion; edge-count acceptance passes it |
| 2 | `cli/sn.py:143-150` | **CONFIRMED** | absent `data` key short-circuits the seat check; three body shapes admit a seat the server does not serve, and report the configured name as the served one |
| 3 | `cli/sn.py:104` vs `:151` | CONFIRMED mechanism, **PLAUSIBLE** reach | prefix-qualified served id blocks every compose run against a server that does serve the seat |
| 4 | `cli/sn.py:8165` | **CONFIRMED** | a second edge to a renamed counterpart binds to the archived id; if that node exists, a wrong-target edge with no refusal anywhere |
| 5 | `cli/sn.py:8146` | **CONFIRMED** | a registry role archived in the unregistered direction composes an authority the loader refuses, reported as a coverage failure |

## Method

Probes ran through the module directly at
`923daf733d10dfd8591ebeabf688aeab79c36945`, each with a positive control in the
same run so an absence is measured by an instrument shown to see the thing when
present: `/tmp/sncli-origin-probe.py`, `/tmp/sncli-probe-shapes.py`,
`/tmp/sncli-restore-shapes.py`. No live graph and no live endpoint was touched.
