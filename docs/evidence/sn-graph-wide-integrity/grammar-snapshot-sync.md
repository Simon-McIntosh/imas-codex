# Grammar snapshot synchronization

Recorded 2026-08-20 against the installed `imas-standard-names 0.8.0rc66`
package and the live `codex` graph.

## Outcome

The graph now contains exactly one `ISNGrammarVersion` node for
`0.8.0rc66`, and `0.8.0rc66` is the only version carrying `active=true`.
The snapshot contains 22 `GrammarSegment` rows and 956 `GrammarToken` rows,
covering all 22 segments and every token exposed through the installed
package's public `get_grammar_context()` vocabulary sections. There are zero
missing segments and zero per-segment token-count mismatches.

The native ISN graph specification supplied 781 tokens but exposed zero
`physical_base` tokens. The Codex synchronization wrapper now also mirrors the
public grammar context, adding the 175 physical bases while preserving aliases
already supplied by the native graph specification. Re-running the sync is
idempotent.

## Version selection

Grammar-token resolution now follows three explicit authorities:

1. an exact token snapshot for the installed runtime version;
2. otherwise, the graph's single active `ISNGrammarVersion` snapshot;
3. otherwise, the newest available token snapshot under PEP 440 ordering.

The regression orders the representative candidates as
`0.8.0rc66 > 0.8.0rc65 > 0.8.0rc9`. A mutation probe replacing PEP 440
ordering with a plain descending string sort produced
`[0.8.0rc9, 0.8.0rc66, 0.8.0rc65]` and failed the same expected ordering with
exit 1. A grammar-specific source grep found zero remaining uses of `max()` or
raw descending version ordering for `GrammarToken` or `ISNGrammarVersion`
resolution.

## Verification

- Live synchronization and package-to-graph census:
  `/tmp/sgwi-grammar-snapshot-sync-2.log` — exit 0; 1 target version node,
  1 active version, 22/22 segments, 956/956 tokens, 0 missing segments, 0
  count mismatches.
- Resolver and context coverage:
  `/tmp/sgwi-grammar-ordering-focused-3.log` — 8 passed, 0 failed.
- Raw-string mutation probe:
  `/tmp/sgwi-grammar-raw-sort-mutation.log` — expected failure, exit 1.
- Grammar-resolution grep:
  `/tmp/sgwi-grammar-resolution-specific-grep.log` — 0 matches.
- Full Standard Names suite:
  `/tmp/sgwi-standard-names-suite.log` — 6,533 passed, 8 skipped, 245
  deselected, 0 failed, exit 0 in 201.56 seconds.
