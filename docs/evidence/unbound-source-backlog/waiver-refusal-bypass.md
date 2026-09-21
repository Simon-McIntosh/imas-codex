# The waiver contradiction refusal can be evaded, and the evasion is latent

An independent audit of the landed contradiction refusal returned verdict SOUND
at 88, with five findings. One is a confirmed, demonstrated bypass of the guard
itself; three are real gaps around it; one is a wording nit. A follow-up
measurement establishes that the bypass cannot fire against the current graph,
which is why the repair is queued rather than rushed.

## The bypass

`_source_disposition` is handed the **terminal** identity of a source's successor
chain, while the cut publishes whatever identity is `accepted`. So a declared
waiver whose published identity is an *intermediate* of that chain classifies
`waived`, exactly as before the repair.

`fetch_manifest_source_release_rows` collects `direct_ids` and walks the chain,
then emits a row carrying only `standard_name_id` set to the terminal's
`target_id` — discarding both. Two triggering inputs were run through the real
resolver and the real decision function:

| input | published | row identity | disposition |
|---|---|---|---|
| waived source → accepted identity in the cut, which has a successor that is pending and not in the cut | the accepted one | the pending successor | `waived` |
| waived source with two direct targets, `produced_sn_id` naming the unexported one | the other one | the unexported one | `waived` |

The second passes the ambiguity guard because that guard raises only when
`produced_id` is **absent** from `direct_ids`. The probe carried a positive
control on the same run — feeding the chain terminal raises — so the two `waived`
results are the guard staying silent rather than a probe that measures nothing.

## The evasion cannot fire today

Every one of the eleven declared-waived paths produces no identity at all:

```
declared waived paths: 11
reachable today: 0 of 11      (direct PRODUCED_NAME targets = 0, successors = 0, for all eleven)
```

A bypass needs a source that produces a published identity off the chain
terminal. None of the eleven produces anything, so the refusal has nothing to be
wrong about yet — and, read the other way, the waiver declaration is currently
doing exactly what it claims: eleven paths, no names, all waived.

**This is a statement about today's graph, not about the code.** The moment a
declared-waived source acquires a producer — which is precisely the stale-waiver
situation the refusal exists to catch — the bypass becomes reachable. The repair
is correct and owed; it is not urgent, and it was not worth a metered lane
sitting at 92 % of its window.

## The three further gaps, none of which is the bypass

- **The pre-staging preflight's refusal branch has no test.** All four new tests
  call the decision function directly; none drives an export over a manifest row.
  Deleting the preflight block leaves every test green, so what the node's
  negative control proved is the decision function's refusal, not that site's.
- **A dry-run release rehearsal swallows the refusal.** The rehearsal wraps the
  export leg in a bare `except Exception` and writes the message into a field no
  CLI or renderer reads, then reports success. A contradicted cut therefore
  rehearses as *accounting could not be measured*, which is exactly where an
  operator looks before spending a real cut.
- **The staging tree is deleted before the refusal fires.** Every other refusal
  that can fire after the staging directory exists writes an export report first;
  this one raises instead. The stated invariant — no staging artifact — holds for
  what the cut *writes* and not for what it *removes*.

## What this record corrects

An earlier statement in this session, and the plan note that accompanied the
landing, said the waiver repair chain was closed. **It is not.** Both halves
exist and both are reachable by their own tests, but the contradiction half can
be evaded by an identity off the chain terminal. The chain is closed against the
cases the tests construct and open against one they do not.
