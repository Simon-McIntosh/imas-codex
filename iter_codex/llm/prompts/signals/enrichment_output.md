---
name: signals/enrichment_output
description: Explicit output contract appended to the complete signal task
used_by: iter_codex.discovery.signals.enrichment.fit_enrichment_prompt
task: enrichment
dynamic: true
schema_needs: []
---
{{ baseline }}

## Required field types and exact values

Return a single JSON object with a `results` array. Each result corresponds to
one input signal, with its 1-based `signal_index`. Include every input once.
The following contract applies to every result, including the worked example:

- `physics_domain`: a string, exactly one of {{ domain_values | tojson }}.
- `diagnostic`: null, or a string exactly one of {{ diagnostic_values | tojson }}.
- `context_quality`: a string, exactly one of {{ quality_values | tojson }}.
- `keywords`: an array of strings, even when empty or containing only one term.
- `name`, `description`, `analysis_code`, `sign_convention`: strings.
- `confidence`: a number from 0 to 1.
- `signal_index`: the integer identifying the input signal.

## Worked response

Given one signal whose authoritative catalogue description is "Measured
temperature of the cooling circuit", unit K, and whose wiki context identifies
the cooling circuit temperature sensor, the complete valid response is:

{{ worked_example }}

The example illustrates the wrapper, field types and enums. Describe the actual
signals using their supplied context and the task instructions above.

## Final output schema

Return JSON only, without Markdown fences, explanation or a bare result list.
Use exactly this schema:

{{ output_schema | tojson }}
