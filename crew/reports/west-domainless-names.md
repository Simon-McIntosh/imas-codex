# WEST frozen-batch physics-domain assignment

## Outcome

All six requested `StandardName` nodes began with `physics_domain = null` and now carry one non-null value from the 32-member `PhysicsDomain` enum. Four assignments were copied mechanically from the live DD node bound through `StandardNameSource`; two derived names had no direct DD binding and were classified through the configured `sn-classifier` seat using their stored name/documentation/unit and child DD context. No domain was selected by hand.

The applying command exited `0`. The scoped relationship writer also exited `0`, and the final Cypher read exited `0` with **6/6 non-null**, **6/6 enum-valid**, and **6/6 carrying exactly one matching `HAS_PHYSICS_DOMAIN` edge**.

## Per-name evidence

| Standard name | Before | Authority route | Bound source paths and live DD `physics_domain` | Assigned domain | Durable change |
|---|---|---|---|---|---|
| `ipb98y2_confinement_time` | `null` | Bound DD source | `summary/global_quantities/tau_energy_98/value` → `general` | `general` | `sn-change:4bb194ba-9ce3-4b24-8062-5c86af841739` |
| `intensity_at_spectral_line` | `null` | Bound DD source | `spectrometer_visible/channel/grating_spectrometer/processed_line/intensity` → `radiation_measurement_diagnostics` | `radiation_measurement_diagnostics` | `sn-change:79ddf947-880b-4e0c-baa4-520d8d6049c3` |
| `radiative_temperature_at_ece_channel` | `null` | Bound DD source | `ece/channel/t_radiation` → `electromagnetic_wave_diagnostics` | `electromagnetic_wave_diagnostics` | `sn-change:2ae20781-ad17-4413-8956-03d5c4dfaa9a` |
| `voltage_of_ion_cyclotron_heating_antenna_amplitude` | `null` | Bound DD source | `ic_antennas/antenna/module/voltage/amplitude` → `auxiliary_heating` | `auxiliary_heating` | `sn-change:5cc11d5b-de47-45cd-a5ab-cc7f178d6bbd` |
| `coolant_mass` | `null` | Configured `sn-classifier` seat; no direct DD binding | No direct binding. Child context: `balance_of_plant/power_plant/system/component/port/mass_flow`, `calorimetry/cooling_loop/mass_flow`, `calorimetry/group/component/mass_flow` | `plant_systems` | `sn-change:98e2542c-8932-4f37-869c-d0aa2f07b628` |
| `voltage_of_ion_cyclotron_heating_antenna` | `null` | Configured `sn-classifier` seat; no direct DD binding | No direct binding. Child context: `ic_antennas/antenna/module/voltage/amplitude` | `auxiliary_heating` | `sn-change:b11ebdb0-19ee-4c90-b0cb-157c70e40ccc` |

The classifier route resolved at runtime to `openrouter/openai/gpt-5.5`. The applying call cost `$0.01461`; it returned `plant_systems` for `coolant_mass` and `auxiliary_heating` for `voltage_of_ion_cyclotron_heating_antenna`. Both outputs were validated against `PhysicsDomain` before any write.

Representative stored meaning supplied to the classifier:

- `coolant_mass` is the derived parent of `ratio_of_coolant_mass_to_time`; its child DD evidence describes coolant/fluid mass flow in balance-of-plant and calorimetry systems.
- `voltage_of_ion_cyclotron_heating_antenna` is the parent antenna feed-terminal voltage quantity; its amplitude child is bound to the IC-antenna module voltage path.

## Applying command and exit status

The live command used the repository root environment and source-shadowing contract:

```bash
env -u VIRTUAL_ENV \
  UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH="$PWD" \
  uv run --no-sync python - <<'PY'
# 1. Read the four direct StandardNameSource -> IMASNode bindings.
# 2. Require exactly one non-null PhysicsDomain value per direct name.
# 3. Read the two unbound derived names plus their child DD context.
# 4. Call dd_domain_classifier._classify_batch with get_model("sn-classifier").
# 5. Require one enum-valid result for each unbound name.
# 6. Call standard_names.edit.reclassify_domain for all six computed values.
#    Each call records a StandardNameChange carrying its authority reason.
PY
assignment_status=$?
echo "ASSIGNMENT_EXIT_STATUS=$assignment_status"
exit "$assignment_status"
```

Observed status:

```text
ASSIGNMENT_EXIT_STATUS=0
```

The applying script failed closed on any of these conditions: a missing requested node, missing direct binding, more than one distinct live DD domain for a direct name, a value outside `PhysicsDomain`, incomplete classifier coverage, or a refused `reclassify_domain` operation.

After scalar assignment, the six relationship projections were materialized through `_write_standard_name_edges(..., expand_closure=False)`, the same scoped writer used by normal Standard Name persistence:

```text
EDGE_WRITE_EXIT_STATUS=0
edge_writer_targets=6
```

## Post-write Cypher verification

The verification derived `$valid` from `list(PhysicsDomain)` in the same process and passed it into this Cypher read; it did not restate an assumed allow-list:

```cypher
UNWIND $names AS wanted
MATCH (sn:StandardName {id: wanted})
OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
OPTIONAL MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
OPTIONAL MATCH (sn)-[:HAS_PHYSICS_DOMAIN]->(pd:PhysicsDomain)
OPTIONAL MATCH (sn)-[:HAS_INTERNAL_CHANGE]->
               (ch:StandardNameChange {operation: 'reclassify_domain'})
WITH wanted, sn,
     collect(DISTINCT CASE WHEN dd IS NULL THEN null
                           ELSE {path: dd.id, domain: dd.physics_domain} END)
       AS dd_bindings,
     collect(DISTINCT pd.id) AS domain_edges,
     collect(DISTINCT {id: ch.id, reason: ch.reason}) AS changes
RETURN wanted AS name,
       sn.physics_domain AS physics_domain,
       sn.physics_domain IS NOT NULL AS non_null,
       sn.physics_domain IN $valid AS enum_valid,
       dd_bindings,
       domain_edges,
       changes
ORDER BY name
```

The positive control in the same graph session reported `4,683` `StandardName` candidates, `4,683` with the schema identity key `id`, and `4,668` with a non-null domain. This proves the read was aimed at the live `StandardName.id`/`physics_domain` fields rather than an invented key.

| Standard name | Final domain | Non-null | Enum-valid | Domain edge |
|---|---|---:|---:|---|
| `coolant_mass` | `plant_systems` | yes | yes | `plant_systems` |
| `intensity_at_spectral_line` | `radiation_measurement_diagnostics` | yes | yes | `radiation_measurement_diagnostics` |
| `ipb98y2_confinement_time` | `general` | yes | yes | `general` |
| `radiative_temperature_at_ece_channel` | `electromagnetic_wave_diagnostics` | yes | yes | `electromagnetic_wave_diagnostics` |
| `voltage_of_ion_cyclotron_heating_antenna` | `auxiliary_heating` | yes | yes | `auxiliary_heating` |
| `voltage_of_ion_cyclotron_heating_antenna_amplitude` | `auxiliary_heating` | yes | yes | `auxiliary_heating` |

Observed status:

```text
FINAL_CYPHER_EXIT_STATUS=0
```

No requested name lacked both a direct DD binding and a classifier route. No guessed-domain exception was used.
