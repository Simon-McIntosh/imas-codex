# A test that reads the statement instead of the behaviour

## What was measured

The mint path issues two Cypher statements: a base join over DD paths, and an
immediate-family closure over the base ids. The unit tests for it drove a stub
graph, and that stub chose which rows to return by searching the statement it
was handed for two tokens, `PRODUCED_NAME` and `HAS_PARENT`.

That is a test of the string, not of the behaviour. The reproduction is exact:
inject an unrelated field into the base join whose *name* carries the token the
family branch keys on, and the stub serves the family rows to the base join.
The base join then parses rows that have no `ids` key and reports every input
path as unmatched, so the behaviour under test is replaced by the shape of the
text.

```
statement carries HAS_PARENT after the unrelated field landed: True
old matcher, base statement as written   : base
old matcher, base statement perturbed    : family
new matcher, base call params            : base
new matcher, family call params          : family
```

## What changed

The stub routes on the parameters the mint path declares — `base_ids` for the
family closure, `paths` for the base join — so no branch inspects the statement.
A new case drives the same mint twice, once with both statements rewritten to
carry an unrelated field, and requires an identical batch and unmatched list.
The previous dispatch fails that case; the reproduction above is its mechanism.

The second module in this node's scope,
`tests/standard_names/test_exact_reset_parent_cleanup_graph.py`, already asserts
behaviour rather than text: it creates a fixture graph and checks node existence
after a reset, with no statement matching at all (`grep -rn 'in cypher' -e
'startswith'` over the two modules returns the minting stub only, before this
change and none after). Its unrelated-field lever is the fixture's `extra_set`,
which appends a field to a created node's statement — a node created that way is
classified by the same assertions as its plain twin.

## Gate

`pytest -m graph tests/standard_names/test_minting.py tests/standard_names/test_exact_reset_parent_cleanup_graph.py`
— exit 0, 4 passed, 7 deselected, 1 warning (an unknown `cache_dir` option in
the shared environment's pytest config), 37.81s.
