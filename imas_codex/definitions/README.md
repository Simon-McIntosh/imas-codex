# IMAS Codex Definitions

This directory contains data definitions and configurations used by the IMAS Codex server.

## Structure

- `physics/` — physics-domain definitions and mappings
- `clusters/` — the terminology used to group IMAS paths into semantic clusters

## Physics domain guidance

- `domains.yaml` — classification guidance for the physics domains: a scope for
  each domain and each category, the quantity kinds a domain covers, and the
  exclusions that separate a domain from a named neighbour. It is not a LinkML
  schema, and it repeats none of the standard-name vocabulary.

  The domain values, their one-line descriptions and their categories are
  canonical in **imas-standard-names**
  (`imas_standard_names/grammar/vocabularies/physics_domains.yml`). This file is
  keyed by those values and adds only what that vocabulary lacks. A domain or a
  category added upstream without an entry here fails
  `tests/graph/test_physics_domain_guidance.py`, so the guidance cannot fall
  behind the vocabulary silently.

- `ids_domains.json` — version-independent mapping from each IMAS IDS to its
  default physics domain, covering all DD versions.

- `path_domain_overrides.json` — deterministic domain pins for DD path patterns
  whose semantic subject the LLM classifier mis-assigns against the IDS default.

## Usage

These files are read by `imas_codex/definitions/physics/__init__.py`:

```python
from imas_codex.definitions.physics import IDS_DOMAINS_FILE, load_domain_guidance

guidance = load_domain_guidance()  # {"categories": {...}, "domains": {...}}
```

`load_domain_guidance` returns the guidance keyed by the ISN domain and category
values. `_provide_physics_domains` in `imas_codex/llm/prompt_loader.py` composes
it with the ISN vocabulary and serves it to any prompt that declares the
`physics_domains` schema need, so the classifier prompt and the chat prompt read
the same text.

## Maintenance

When the ISN vocabulary gains a domain or a category, add its entry here and run
`tests/graph/test_physics_domain_guidance.py`; it fails until the file matches
the vocabulary exactly.