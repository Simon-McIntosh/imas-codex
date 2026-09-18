# Which verdict field wins, and how many identities carry a disagreeing pair

**Answer in one line.** `name_stage` is the verdict and `refine_stop_reason` is a
cause label, not a second verdict; **62** live identities carry a disagreeing
pair, split **20 legitimate history / 42 legacy stage writes with no recorded
cause**, and the two shapes need different treatment.

Measured live on 2026-09-18 against the login-node graph (DD-pinned graph client,
`GraphClient()`), over a population of **5,130** `StandardName` nodes, and
corrected the same day against a review of the first pass: which function performs
the clear at `graph_ops.py:10721`, how many read paths `refine_stop_reason` has,
and whether the 42-row arm is a live fault or legacy state. The **62** count and
the **42 / 11 / 7 / 2** breakdown are unchanged.

## The named predicate

`DISAGREEING_VERDICT_PAIR` — a row whose two name-axis fields cannot be read
together as one verdict for the identity:

```
MATCH (sn:StandardName)
WHERE (sn.name_stage = 'accepted'  AND sn.name_stage = 'accepted' AND sn.refine_stop_reason IS NOT NULL)
   OR (sn.name_stage = 'exhausted' AND sn.refine_stop_reason IS NULL)
RETURN count(sn) AS n
```

Why those two arms are the disagreement, and nothing else is:

- **`accepted` with a bad table