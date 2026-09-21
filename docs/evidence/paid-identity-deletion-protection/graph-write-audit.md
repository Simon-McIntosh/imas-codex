# The two graph-writing paths, audited

Scope: the restore-receipt parity change at `da2b5619e` in
`imas_codex/standard_names/signed_manifest.py`, and `sn restore compose` /
`sn restore apply` at `21e4739e9` in `imas_codex/cli/sn.py`. Three questions per
path — can it write the graph when it should refuse, can its refusal be
bypassed, can its receipt disagree with what was written — plus the adjudication
the plan asked for on the receipt's two expectation sources.

**Verdicts.**

| Path | Verdict |
|---|---|
| `signed_manifest.py` restore-receipt parity (`da2b5619e`) | **SAFE-FOR-WEST** |
| `sn restore compose` / `apply` (`21e4739e9`) | **MUST-FIX-FIRST** |

Nothing below is inferred from reading alone. Every claim is a probe that could
have come out the other way; the six probes and their output are
`~/.cache/pidp-graph-write-probe.py` and `~/.cache/pidp-graph-write-probes.log`.
The two test files run green at the audited revision: 26 passed
(`~/.cache/pidp-graph-write-audit-baseline.log`).

## Path A — the restore-receipt parity change

### A1 · The recorded finding, adjudicated — and it is narrower than recorded

The plan recorded that "the receipt's expected falls back to the archive census
while the refusal's expected is built from the reconstruction closure alone, so
a restore reinstating fewer edges than the archive recorded will now DISPLAY the
shortfall and not RAISE it."

The two sources are real. `signed_manifest.py:7272` builds the receipt's
expectation as `archived.get(role, closure.get(role, 0))`, while the refusal at
`signed_manifest.py:7411-7421` builds its own from the closure alone. **But a
displayed-not-raised shortfall on a *routable* role is unreachable**, because
the loader already refuses that case one layer earlier:

```
$ probe_registry_shortfall_is_refused          # archive_roles HAS_UNIT: 2, closure carries 1
RAISED SignedManifestAuthorityError: archive reconstruction cannot reinstate an
archived role the edges do not cover: HAS_UNIT holds 2 in the archive and 1 in
the reconstruction edges
```

`signed_manifest.py:6944` gates that refusal on `role in
_ARCHIVE_EDGE_COUNTERPARTS`. So the shortfall the finding describes survives
only for a role *outside* the reconstruction registry, which today means
`EVIDENCED_BY` and nothing else:

```
$ probe_evidenced_by_shortfall                 # archive_roles EVIDENCED_BY: 2
outcome: applied refusals: []
EVIDENCED_BY parity: {'expected': 2, 'observed': 0}
```

**Which expectation should be authoritative.** The reconstruction closure, for
the refusal; the archive census, for the receipt — but the receipt must say
which column it is showing. The closure is the only expectation the transaction
can act on, because it is the only one the restore can satisfy: `EVIDENCED_BY`
is held inbound from `PromotionCandidate` and no reconstruction edge can carry
it, so making the census govern the refusal would make every archived identity
that ever carried one unrestorable, with no route to a green restore. That is
the wrong trade for a plan whose whole purpose is to get 67 identities back.

**What a shortfall must do.** On a routable role it must fail, and it already
does, at load. On a non-routable role it must be recorded, not raised — which is
what `identity_roles.unreinstatable` and the parity block already do. The real
gap the finding names is neither of those: it is that **one receipt presents two
differently-sourced numbers in one column called `expected`, and nothing in the
receipt says which is which.** The repair is a receipt-shape change, not a guard
change: carry the census and the closure as separate named fields, so a reader
can tell "the archive held two of these and the restore cannot put them back"
apart from "the restore intended two and produced two".

That distinction is not cosmetic, because the column is not monotone — see A2.

### A2 · The census can also sit *below* the closure, and then the receipt reads as an overshoot (NEW)

The loader's guard at `signed_manifest.py:6944` is one-sided: it refuses
`reconstructable < count` and says nothing about `count < reconstructable`. An
archive census that under-counts a role therefore loads, applies, and prints a
parity row that disagrees with a completely correct restore:

```
$ probe_overshoot                              # archive_roles HAS_UNIT: 0, closure carries 1
outcome: applied changed: 1
HAS_UNIT parity: {'expected': 0, 'observed': 1}
edges actually written: [('archived_temperature', 'HAS_UNIT', 'unit:eV', {'source': 'archive'})]
```

The restore did exactly what the closure authorised and exactly one edge was
written. The receipt says it expected none. Trigger: any extraction whose
`archive_roles` census is stale or partial relative to the `edges` list it ships
with — which is precisely the shape `sn restore compose` produces when its
census and its routing disagree (B3).

`file:line` — `imas_codex/standard_names/signed_manifest.py:6944` (one-sided
guard), `:7272` (the fallback that consumes it).

### A3 · A role outside the parity set reports `observed: 0` without the graph ever being read (NEW)

`_archive_role_parity` iterates `_ARCHIVE_PARITY_ROLES | set(archived)`
(`signed_manifest.py:7275`) but `observed` comes from a live read taken only over
`_ARCHIVE_PARITY_ROLES` (`:7415`). For any role the archive names that is
neither in the registry nor `EVIDENCED_BY`, `live.get(role, 0)` is structurally
zero — the graph is never asked. The loader accepts any non-empty string as a
role name (`:6936-6941`), so the input is unvalidated:

```
$ probe_unknown_role       # archive_roles ATTRIBUTED_TO: 5, graph holds 2 such edges
ATTRIBUTED_TO parity: {'expected': 5, 'observed': 0} live count in graph: 2
unreinstatable: {'ATTRIBUTED_TO': 5}
```

The receipt states, with the same authority as every other row, that the graph
holds zero of a role it holds two of, and reports five as unreinstatable. A
zero that was never measured is the failure mode this programme has already
recorded twice. **Receipt-level only — nothing is written or withheld because of
it** — but it is a receipt disagreeing with the graph, which is one of the three
questions this audit was asked.

`file:line` — `imas_codex/standard_names/signed_manifest.py:7275` against `:7415`.

### A4 · The refusal was *not* weakened — verified, with a control that could have failed

`da2b5619e` replaced `if live != expected` with
`if any(live.get(role, 0) != count for role, count in expected.items())`, which
reads like a narrowing: the new form iterates only the keys `expected` holds.
It is not, because `expected` is seeded as
`dict.fromkeys(_ARCHIVE_EDGE_COUNTERPARTS, 0)` at `signed_manifest.py:7411` —
every registry role, zeros included. The rewrite was forced by the widened live
read (which now carries an `EVIDENCED_BY` key the closure can never match), and
it is exactly equivalent over the registry.

Negative control — a live registry edge the closure does not carry must still
conflict, and does:

```
$ probe_refusal_not_weakened   # graph pre-holds one HAS_PARENT; the closure carries none
RAISED SignedManifestConflict: archive-versus-live relationship counts differ
```

Had the change weakened the guard, this probe would have applied silently.

### Path A answers

| Question | Answer |
|---|---|
| Can it write when it should refuse? | **No.** Registry parity is enforced unchanged (A4); node and counterpart writes are `MERGE … ON CREATE SET` with a `properties(node) = $properties` post-check (`:7354-7365`), so an existing node with different properties raises rather than being overwritten. |
| Can its refusal be bypassed? | **No.** The closure is re-read inside the transaction and compared to the authorised digest (`:7348-7351`) after the preview rolled back. |
| Can the receipt disagree with what was written? | **Yes, two ways, both display-only: A2 and A3.** Neither changes what is written or withheld. |

**Verdict: SAFE-FOR-WEST.** A2 and A3 are receipt-truthfulness defects on a path
whose write and refusal behaviour is sound; they should be fixed before the
receipt is used as an acceptance instrument for the 67-identity restore, but
they cannot corrupt the graph.

## Path B — `sn restore compose` and `sn restore apply`

### B1 · Compose writes the archived `origin` back verbatim, re-arming the deletion this plan exists to prevent (HIGH)

`compose_archive_reconstruction_authority` copies the extraction's `properties`
into the node payload unchanged apart from the id
(`imas_codex/cli/sn.py:8130-8132`), and the command exposes `--identity` and
`--rename-counterpart` and nothing else (`:8226-8246`). The adapter then writes
that payload as-is. The §4 NEXT card states the constraint this collides with:

> The dump predates the origin correction, so a naive restore reinstates the
> false `catalog_edit` value the repair was right to remove.

That value is not a label. It is the delete permission that made these
identities removable in the first place, so a restore composed straight from the
dump returns each identity to the graph in the exact state that allowed the loss
— and the plan's own acceptance is per-type edge counts, which such a restore
passes. The test fixture in `tests/standard_names/test_archive_reconstruction.py`
carries `"origin": "catalog_edit"` on its archived node, so the suite exercises
this shape and cannot see it.

The operator *can* edit `properties.origin` in the extraction file before
composing — compose reads it verbatim, so the correction is available. Nothing
in the command, its help text, or its output says so, and no guard requires the
restore to state an origin. **The fix belongs to compose: require the restore to
state the identity's corrected `origin` explicitly, or refuse an extraction
carrying a `catalog_edit` origin.** Restore and reclassification are one action
in this plan; the entry point currently makes them two, with the safe one
optional.

`file:line` — `imas_codex/cli/sn.py:8130-8132`, options at `:8229-8245`.

### B2 · A counterpart rekey is consumed by the first edge only, and the rest keep the archived id (NEW)

`renames.pop(str(counterpart_id))` at `imas_codex/cli/sn.py:8165` removes the
rekey from the map as it applies it. An identity holding two edges to the same
renamed `StandardName` counterpart gets one rekeyed and one not:

```
$ probe_rename_pop     # --rename-counterpart old_parent=new_parent, two edges to old_parent
HAS_PARENT -> new_parent
REFERENCES -> old_parent
```

The `pop` is load-bearing — the leftover map drives the "a rekey names no
archived counterpart" refusal at `:8192-8196` — so it cannot simply become a
lookup without moving that check. Two outcomes downstream, and only one of them
is loud:

- the archived id no longer exists live → the adapter's counterpart `MATCH`
  fails and the apply raises `archive relationship counterpart changed before
  reconstruction`. Recoverable.
- the archived id *does* exist live, as a tombstone, placeholder, or a
  since-reused name → **the edge is created to the wrong node, with no refusal
  at any layer.** The parity guard counts roles, not endpoints, so the count is
  correct and the restore reports success.

The second is a silent graph corruption on a plan whose restore set explicitly
includes renamed parents (`etendue_of_spectrometer_channel` and its ratified
spectral child, the nineteen-member pedestal family).

### B3 · The census counts by role before direction is routed, so a half-routable role blocks the whole restore

`census[relationship_type] += 1` at `imas_codex/cli/sn.py:8146` runs before the
`(type, direction)` lookup decides whether the edge is routable. Most registry
roles are registered in one direction only (`HAS_LOCUS` outgoing,
`ENTAILED_FROM_CHILD` incoming, every `GrammarToken` role outgoing). An archived
edge carrying such a role in the other direction is counted into the census,
omitted from the edges, and then refused at load by the very guard from A1:

```
$ probe_direction_census   # extraction holds HAS_UNIT/outgoing and HAS_LOCUS/incoming
composed archive_roles: {'archived_name': {'HAS_LOCUS': 1, 'HAS_UNIT': 1}}
composed edge roles: ['HAS_UNIT']
RAISED SignedManifestAuthorityError: archive reconstruction cannot reinstate an
archived role the edges do not cover: HAS_LOCUS holds 1 in the archive and 0 in
the reconstruction edges
```

The authority composes without warning and is unusable, and the message names
coverage rather than direction, so an operator reads it as an extraction defect.
Note the asymmetry this creates: a role with *no* registry entry (`EVIDENCED_BY`)
passes through as counted-and-unreinstatable, while a role with a registry entry
in the wrong direction is fatal. Compose should either route the reverse
direction or report it as unreinstatable, the same as any other role it cannot
carry — and it should say so at compose time, not at load time.

### B4 · The apply gate holds — verified

`--apply` without `--manifest-sha256` is a usage error (`:8355`), a digest
without `--apply` is a usage error (`:8357`), and the digest authorises exactly
one closure that the adapter re-reads inside the transaction. The integrity
check is not vacuous despite both digests being recomputed from the file: the
loader independently requires `data["signature"]["sha256"]` to equal the
canonical payload digest (`signed_manifest.py:6982-6986`), so a body edited
without re-signing is refused.

One thing to state plainly rather than leave implied: **the signature is
tamper-evidence, not authorisation.** `signed_payload_sha256` is importable and
`compose` re-signs, so any operator who can run `sn restore apply` can produce a
correctly-signed authority for arbitrary content. The property the envelope
provides is that the applied bytes are the composed bytes and that the closure
did not move between preview and apply — not that a second party approved them.

### Path B answers

| Question | Answer |
|---|---|
| Can it write when it should refuse? | **Yes — B1 and B2.** B1 writes a `catalog_edit` origin no guard questions; B2 can write an edge to the wrong counterpart with no refusal at any layer. |
| Can its refusal be bypassed? | **No.** The preview/digest/re-read chain is intact (B4). B3 is the opposite failure — a refusal that fires where it should not. |
| Can its receipt disagree with what was written? | **Indirectly.** Compose is the producer of the `archive_roles` census that A2 and A3 turn into a misleading receipt; B3 is the case where its census and its edge list disagree by construction. |

**Verdict: MUST-FIX-FIRST.** B1 before any identity is restored, because a
restore that reinstates the false origin satisfies the plan's acceptance while
recreating the exposure. B2 before any identity whose counterpart was renamed —
which is the etendue and pedestal work specifically.

## Defect index

| # | file:line | Triggering input | Effect |
|---|---|---|---|
| A2 | `signed_manifest.py:6944`, `:7272` | `archive_roles` under-counts a routable role the edges carry | applied receipt shows `expected 0 / observed 1` on a correct restore |
| A3 | `signed_manifest.py:7275` vs `:7415` | `archive_roles` names a role outside registry ∪ `EVIDENCED_BY` | `observed: 0` reported without reading the graph; live count was 2 |
| B1 | `imas_codex/cli/sn.py:8130-8132` | any dump extraction predating the origin correction | restores `origin: catalog_edit`, re-arming automatic deletion |
| B2 | `imas_codex/cli/sn.py:8165` | two edges to one `--rename-counterpart` target | second edge silently bound to the archived id |
| B3 | `imas_codex/cli/sn.py:8146` | a registry role archived in the unregistered direction | authority composes, then is refused at load as a coverage failure |

## Gate

Base `1fa39615146cc6c88a61e36f4829fde60f4d6f86`, `all_debug` partition.
`python -m pytest -p no:cacheprovider tests/standard_names/test_archive_reconstruction.py
tests/standard_names/test_archive_reconstruction_entry_point.py` — exit `0`,
**26 passed** in 6.62s, log `~/.cache/pidp-graph-write-audit-baseline.log`.
The six probes ran against the same revision through the suite's own fake graph;
no live graph was touched and no graph-writing peer had to be quiet.
