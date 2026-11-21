Return one JSON object, a `SignalMappingBatch`, and nothing outside it. Use these
exact field names. A field marked required must be present on every object.

### `SignalMappingBatch`

{{ signal_mapping_batch_fields }}

### Each `mappings` entry — `SignalMappingEntry`

{{ signal_mapping_entry_fields }}

### Each `unmapped` entry — `UnmappedSignal`

{{ unmapped_signal_fields }}

### Each `escalations` entry — `EscalationFlag`

{{ escalation_flag_fields }}

`target_id` is always the full IMAS field path, for example
`magnetics/b_field_pol_probe/field/data`, and never a placeholder.
