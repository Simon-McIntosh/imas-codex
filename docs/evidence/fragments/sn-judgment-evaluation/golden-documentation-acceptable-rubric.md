# Golden B rubric — is a standard name's documentation acceptable for the catalog

A documentation is labelled **acceptable** only when every check below passes.
It is labelled **not acceptable** when any one fails, and the label records the
first check that fails, by its key.

1. `subject` — it documents the quantity the name denotes. The DD source path or
   paths it was produced from measure that quantity. A provenance mismatch fails,
   even when the prose is good, because the catalog would publish the wrong
   meaning beside the name (e.g. `electron_energy` documented from
   `summary/global_quantities/energy_b_field_pol`).
2. `physics` — the definitions, equations and statements are correct. The
   dimensions agree with the unit symbol. Signs and directions follow the
   catalog's single coordinate convention, which is never named in the text.
3. `scope` — it claims no more than the name. It does not narrow a general name
   to one diagnostic, machine or code, and it does not widen a specific name
   into a general one.
4. `house_style` — where a conventions registry entry applies (the row's
   `conventions` field), the documentation uses that entry's definition and
   notation. A notation that conflicts with the registry symbol fails. So does a
   forbidden spelling, or a COCOS number or other catalog-level convention
   stated in prose.
5. `links` — every `name:` link points at a name whose meaning matches its link
   text. A link to the name itself, presented as an example of something else,
   fails.
6. `complete` — nothing is truncated, left as a placeholder, unresolved, or
   self-contradictory. Every symbol in an equation is defined.

Judgement notes:
- Style preferences outside the registry are not failures: wording, length, or
  the order of paragraphs.
- A minor omission that leaves every statement true does not fail `complete`.
  A statement that is false fails `physics`.
- When the DD source is a generic leaf such as `.../value` or `.../data`, judge
  `subject` from the full path, not the leaf.
- A bracketed phrase with no link target, such as `[ion decay length]`,
  fails `links`. The published entry shows the reader a reference that leads
  nowhere. (A `name:` link whose target is outside the current batch is not a
  failure; links resolve against the whole graph.)
- The template check is the catalog's own `template_opening` rule in
  `imas_codex/standard_names/docs_gates.py`: the first non-empty line must
  match, case-sensitively from its start, at least one selected template's
  opening, with each placeholder matching any text. A leading "The" that the
  template does not carry fails it.
