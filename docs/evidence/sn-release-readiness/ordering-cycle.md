# Spectrometer signal-to-noise ordering-cycle disposition

Measured against the production `codex` graph on 2026-08-23, from source
commit `afde1095d682d527c79b49ea8f41a87894988155`. This was a read-only
investigation: every graph statement used `MATCH`/`RETURN`, no model was
called, and no node, relationship, or property was changed.

## Verdict

**A reversed `HAS_PARENT` edge closes the cycle; `HAS_ERROR` does not
participate.** The graph contains exactly two directed relationships between
the identities. The logarithm form's edge to the base is the legitimate
operator peel and must be preserved. The base identity's edge back to the
logarithm form is grammatically impossible and is the specific edge to remove
through a governed graph repair.

This is a two-row defect, not a systemic graph pattern. The production graph
contains exactly **1** bidirectional `HAS_PARENT` pair: this pair. The count of
**other** identity pairs with the same shape is **0**.

The unit and documentation evidence changes the catalog disposition beyond
the immediate edge repair. Both identities are bound to the same source path,
both carry the source-authoritative unit `dB`, and both define the same
`10 log10` power ratio. The base identity is therefore already logarithmic;
the explicit `logarithm_of_...` form is a redundant conflation, not a second
physical quantity obtained by applying a logarithm to a linear ratio.

## Complete edge inventory

These are **every** relationship between the two `StandardName` nodes in
either direction:

| Direction | Type | Relationship properties | Disposition |
|---|---|---|---|
| `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` &rarr; `spectral_signal_to_noise_ratio_of_spectrometer_channel` | `HAS_PARENT` | `operator_kind='unary_prefix'`, `operator='logarithm'` | **Legitimate; preserve.** Removing the outer unary `logarithm` operator from `logarithm_of_X` yields `X`, exactly the recorded parent. |
| `spectral_signal_to_noise_ratio_of_spectrometer_channel` &rarr; `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` | `HAS_PARENT` | `operator_kind='qualifier'`, `operator='spectral'` | **Cycle-closing defect; remove through governed repair.** Peeling the `spectral` qualifier cannot introduce a new `logarithm_of_` prefix or return to the child. |

Explicit negative checks found **0** `HAS_ERROR` relationships in either
direction, **1** base-to-log `HAS_PARENT`, and **1** log-to-base `HAS_PARENT`.
There are no parallel or third edges between the pair.

The grammar distinction is decisive. `HAS_PARENT` is directed
`(child)-[:HAS_PARENT]->(parent)` and represents removal of one ISN grammar
layer. The first edge removes a unary prefix. The second claims to remove the
qualifier `spectral`, but its target still contains `spectral` and additionally
contains the unary prefix `logarithm`; it therefore cannot be the result of
that peel. Because it points back to the original child, it is the edge that
turns the otherwise valid derivation into a two-node ordering cycle.

## Physics and source binding

| Identity | Origin | Source path | Stored/source unit | What the live prose defines |
|---|---|---|---|---|
| `spectral_signal_to_noise_ratio_of_spectrometer_channel` | `derived` | `spectrometer_visible/channel/isotope_ratios/signal_to_noise` | `dB` / `dB` | A logarithmic spectral-quality measure, explicitly `10 log10(P_line / P_ref)`. |
| `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` | `pipeline` | `spectrometer_visible/channel/isotope_ratios/signal_to_noise` | `dB` / `dB` | The base-ten logarithmic ratio of the same integrated signal-band and line-free reference-band powers, again explicitly `10 log10(P_s / P_n)`. |

A linear signal-to-noise ratio would be dimensionless and could legitimately
serve as the argument of a separate logarithmic representation. That is not
the graph state here: the supposed base is in decibels and its own description
and documentation already define the logarithmic transform. Treating the
explicit logarithm form as another live catalog quantity would amount to
applying identity syntax for a transformation already carried by the base's
unit and physics definition.

## Recommended repair, not applied

1. Remove only
   `spectral_signal_to_noise_ratio_of_spectrometer_channel
   -[:HAS_PARENT {operator_kind:'qualifier', operator:'spectral'}]->
   logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel`
   using the repository's exact-manifest, governed graph-repair path. This is
   the relationship that closes the ordering cycle.
2. Preserve
   `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel
   -[:HAS_PARENT {operator_kind:'unary_prefix', operator:'logarithm'}]->
   spectral_signal_to_noise_ratio_of_spectrometer_channel`. It is the correct
   grammar edge for the spelling, even though the identity itself needs a
   catalog disposition.
3. Route the explicit logarithm identity through the sanctioned catalog
   lifecycle as redundant, superseding it in favor of the shorter
   `spectral_signal_to_noise_ratio_of_spectrometer_channel` identity and
   retaining the `dB` source binding there. Do not hand-edit or direct-accept
   either name. The two nodes have the same source, logarithmic unit, and
   defining equation, so preserving both as live catalog identities would
   encode one physical observable twice.
4. Re-run the release dry-run after the governed relationship repair. The
   catalog-lifecycle disposition should remain separately auditable; it must
   not be approximated by deleting the legitimate log-to-base peel edge.

No graph repair or catalog edit was performed by this investigation.

## Read-only mutation sentinels

| Population inspected | Before | After | Delta |
|---|---:|---:|---:|
| `StandardName` nodes | 4,395 | 4,395 | **0** |
| `HAS_PARENT` relationships | 61,921 | 61,921 | **0** |
| `HAS_ERROR` relationships | 31,281 | 31,281 | **0** |
| `HAS_UNIT` relationships | 49,410 | 49,410 | **0** |

The sentinels prove the read-only investigation left the inspected graph
populations unchanged. LLM calls: **0**. LLM spend: **USD 0**.
