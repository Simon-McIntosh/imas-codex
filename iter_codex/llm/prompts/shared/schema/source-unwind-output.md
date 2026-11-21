## Required Output Format (CRITICAL)

You MUST return valid JSON matching this EXACT structure. The response MUST be parseable JSON.

**Schema derived from Pydantic model:**

```json
{{ signal_source_unwind_schema_example }}
```

### Field Requirements

{{ signal_source_unwind_schema_fields }}

### Critical Rules

1. Return ONE result per input source group, in the same order
2. `source_index` is the 1-based index matching the input source order
3. `{member_id}` is replaced with the numeric instance identifier extracted from the accessor
4. `{node_description}` is replaced with the member's own text — its SignalNode description (geometry values) or its catalogue description — and must appear in the description template whenever the sample node descriptions differ between members
5. Do NOT hardcode specific instance numbers, geometry values or catalogue text — use the placeholders
6. Ensure valid JSON - no trailing commas, proper quoting
7. Do NOT include any text outside the JSON object