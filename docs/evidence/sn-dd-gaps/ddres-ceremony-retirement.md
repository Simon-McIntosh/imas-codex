# DD resolution graph-authority cutover

Before either packaged resource was removed, a credentialed read compared the
packaged active authority with the live graph on the exact behavior tuple
`(path, DD version, field, published value, effective value)`.

| Measure | Result |
|---|---:|
| Packaged records | 37 |
| Graph records | 37 |
| Exact matches | 37 |
| Missing from graph | 0 |
| Extra in graph | 0 |
| Records with exactly one evidence and version edge | 37 |

The complete output is recorded in
`/tmp/ddres-ceremony-retirement-equivalence.log` with exit status 0.

Runtime loading now reads the graph through one typed boundary. It refuses an
empty or unavailable snapshot, a missing or ambiguous `EVIDENCED_BY` edge, a
missing upstream reference or explicit `none-yet` marker, a bridge-direction
mismatch, a DD-version edge mismatch, an incomplete who/when/why trail, and any
duplicate active exact key. The CLI retains read-only `list` and `show`
inspection over those records. File mutation, candidate promotion, transition
receipts, evidence-token validation, revision sequencing, digest fencing, and
file locking no longer exist.

The packaged active and candidate YAML resources were deleted only after the
37-of-37 comparison passed. The graph was not mutated by this cutover.
