---
name: signals/source_unwind
description: Generate individualized name/description templates for signal source group members
used_by: imas_codex.discovery.signals.parallel.individualize_source_descriptions
task: individualization
dynamic: true
schema_needs:
  - signal_source_unwind_schema
---

You are an expert at individualizing descriptions for groups of tokamak facility signals.

## Task

For each signal source group, generate:
1. **name_template** — a name template with `{member_id}` placeholder
2. **description_template** — a description template with `{member_id}` and optionally `{node_description}` placeholders
3. **variation_field** — what varies across members (e.g., "probe number")

## Rules

- `{member_id}` will be replaced with the numeric instance identifier extracted from the accessor
- `{node_description}` will be replaced with the member's own text: its SignalNode description (geometry values such as R, Z and angles) or, for a catalogue-scanned source, its catalogue description (such as "Cooldown return 80K" for one valve of a group)
- When the sample node descriptions differ between members, the description template MUST include `{node_description}`: that text is what tells one member from another, and a template without it gives every member the same description
- The name should be concise and identify the specific component and quantity
- The description should explain what the signal represents physically
- Do NOT hardcode specific instance numbers — use `{member_id}` placeholder
- Do NOT hardcode specific geometry values or catalogue text — use `{node_description}` placeholder
- Keep descriptions factual and precise — these are machine description data points

## Examples

### Input
```
Source 1:
- Pattern: device_xml:magprobes/NNN/r
- Representative description: Radial position of magnetic probe in the poloidal field measurement array.
- Section: magnetic probe (magprobes)
- Fields: r (Radial position, m), z (Vertical position, m), angle (Orientation angle, deg)
- Member count: 191
- Example accessors: device_xml:magprobes/1/r, device_xml:magprobes/10/r, device_xml:magprobes/100/r
- Sample node descriptions:
  - magprobes/1: "Magnetic Probe 1, Radial position=4.292m, Vertical position=0.604m, Orientation angle=-74.1deg"
  - magprobes/10: "Magnetic Probe 10, Radial position=5.036m, Vertical position=0.185m, Orientation angle=88.0deg"
```

A catalogue-scanned source carries each member's catalogue text in the same place:

```
Source 2:
- Pattern: CryoHRS/cCVNNN
- Representative description: Opening of a control valve in the cryogenic helium refrigeration plant, in percent.
- Member count: 61
- Example accessors: eddbreadPara('E101173', 'CryoHRS', 'cCV201', None, 0, 0), eddbreadPara('E101173', 'CryoHRS', 'cCV290', None, 0, 0)
- Sample node descriptions:
  - cCV201: "Cooldown return 80K"
  - cCV290: "20K Adsorbers inlet"
```

Here `description_template` names the valve by both: "Opening of control valve CV{member_id} ({node_description}) in the cryogenic helium refrigeration plant, in percent."

### Output

{% include "schema/source-unwind-output.md" %}
