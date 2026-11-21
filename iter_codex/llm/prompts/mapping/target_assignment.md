---
name: target_assignment
description: Per-call context for the escalated target choice (dynamic, changes per call)
---

Choose this signal source's target paths from its listed candidates.

## Context

- **Facility**: {{ facility }}

### Signal Source

{{ signal_source }}

### Listed Candidates

These are the only paths you may choose from. Each is an existing candidate of
this source, in Jev order (descending confidence). Paths marked
`[cross-IDS sibling]` were added because they share a Data Dictionary semantic
cluster with another candidate, possibly in a different IDS; they are legitimate
homes. Choose every path that should hold this source's values.

{{ shortlist }}

### Peers and Kin

Cross-facility precedent and physics-domain context:

{{ context_notes }}

{% if cross_facility_mappings %}
### Cross-Facility Precedent

Other facilities have already mapped signals to these IDS paths.
Use this as strong evidence for where similar signals should be assigned:

{{ cross_facility_mappings }}
{% endif %}