# Documentation recovery for `hot_neutral_temperature_at_plasma_boundary`

Measured 2026-09-14 through the bounded, exact-name Standard Names pipeline.
The live graph was reached with `GraphClient.from_profile()`, which resolved
the configured compute-node Bolt endpoint `bolt://98dci4-gpu-0002:7687`.
No compose pool was invoked, and the other excluded identity
`inner_hard_xray_peak_width` was not read or changed.

## Starting state

The initial graph read found one `StandardName` with this state:

| Field | Starting value |
| --- | --- |
| id | `hot_neutral_temperature_at_plasma_boundary` |
| description | `Translational kinetic temperature, expressed as energy per particle, of the energetic neutral-atom component in the plasma edge or scrape-off layer.` |
| documentation | `null` |
| name stage / score | `accepted` / `0.9875` |
| docs stage / score | `pending` / `null` |
| status | `draft` |
| validation | `valid`, `validated_at=2026-09-14T10:05:01.865000000+00:00` |
| unit | `eV` |
| name-axis reviews | 2, scores `0.975` and `1.000`; name aggregate `0.9875` |
| docs-axis reviews | 0 |
| source paths | 2 |

The property-key probe returned the `StandardName` key set before any field
absence was inferred. It included `id`, `description`, `documentation`,
`name_stage`, `docs_stage`, `reviewer_score_name`, `reviewer_score_docs`,
`validation_status`, `validated_at`, `unit`, `source_paths`, and the review
and lifecycle fields used below. The absence of documentation was therefore
a value read from a known property, not a zero-row or missing-property result.

The two producing DD bindings and their enriched descriptions were:

| DD path | Enriched description | Terse DD documentation | Unit |
| --- | --- | --- | --- |
| `spectrometer_visible/channel/polarization_spectroscopy/temperature_hot_neutrals` | `Temperature of the hot neutral population (T_hot) derived from a spectral line fit in polarization spectroscopy. Characterizes the energetic neutral component, distinguished from the cold bulk neutral population measured in the same channel.` | `Fit of hot neutrals temperature` | `eV` |
| `spectrometer_visible/channel/isotope_ratios/isotope/hot_neutrals_temperature` | `Temperature of hot neutral atoms for a given hydrogen isotope, derived from Doppler broadening analysis of visible spectral line emission. Characterizes the energetic neutral component contributing to the isotope ratio measurement.` | `Temperature of hot neutrals for this isotope` | `eV` |

Both enriched descriptions establish the measured quantity as a hot-neutral
population temperature derived from visible spectroscopy, and distinguish it
from the cold neutral component. Neither enriched description names a
geometric locus as either the last-closed-flux-surface boundary or the
scrape-off layer. The DD hierarchy confirms the two leaves are under
`spectrometer_visible/channel/polarization_spectroscopy` and
`spectrometer_visible/channel/isotope_ratios/isotope`; its structural
documentation likewise names spectroscopy and isotope sets but no spatial
boundary. Thus the DD evidence does not independently prove an SOL locus or
contradict the accepted `plasma_boundary` qualifier. The name review's
explicit expert steering to the plasma-boundary locus is the authority used
for the documentation wording below. The old description was ambiguous and
has now been replaced by wording that consistently states that locus.

## Scoped dry run and pipeline

The exact-name dry run was:

```text
imas-codex sn run --name hot_neutral_temperature_at_plasma_boundary \
  --docs-only --skip-global-maintenance --dry-run
```

It returned:

```text
Exact-name dry run: 1 existing name(s) eligible; no graph writes performed
```

The sanctioned live run was the same exact-name scope without `--dry-run`,
with `--cost-limit 12.00`. It entered only `generate_docs`, `review_docs`,
and `refine_docs`. The receipt reported:

```text
names_composed    0
names_enriched    1
names_reviewed    1
names_regenerated 0
cost_spent        0.126908
```

The `generate_docs` pool produced one candidate. Its documentation was:

```text
This quantity is the energy-equivalent translational temperature of the hot neutral-atom population evaluated at the plasma boundary. It characterizes random translational motion and excludes directed bulk flow and the cold neutral component.

For an isotropic translational distribution after removal of the population bulk velocity, the temperature is defined by

$$
T_{\mathrm{hot}} = \frac{2}{3}\left\langle E_{\mathrm{rand}}\right\rangle
$$

where $T_{\mathrm{hot}}$ is the hot-neutral energy-equivalent temperature, $E_{\mathrm{rand}}$ is the random translational kinetic energy of one neutral atom, and the angle brackets denote an average over the hot neutral population.

The quantity applies specifically to hot neutral atoms at the plasma boundary and represents one population-level temperature rather than a sum of particle temperatures. It is distinct from the [cold neutral temperature](name:cold_neutral_temperature), which characterizes the cold neutral component, and from the broader [neutral temperature](name:neutral_temperature), which does not specify the hot component or this boundary locus.
```

The generated documentation resolves the semantic defect: it names the
plasma boundary consistently in the first and final paragraphs, while
preserving the DD-supported hot-neutral population, random-motion, bulk-flow
exclusion, and `eV` energy-equivalent interpretation. It does not claim that
the two DD leaves are separate quantities.

## Documentation review evidence

The generated candidate received two documentation-axis reviews in one
quorum group, both with aggregate score `1.000`:

| Reviewer role | Model | Score | Cycle | Resolution |
| --- | --- | ---: | ---: | --- |
| primary | `openrouter/anthropic/claude-sonnet-5` | `1.000` | 0 | canonical review record |
| secondary | `openrouter/x-ai/grok-4.5` | `1.000` | 1 | `quorum_consensus` |

The primary review awarded `20/20` for description quality, documentation
quality, completeness, and physics accuracy. It specifically accepted the
energy-equivalent relation `T_hot = (2/3)<E_rand>`, the exclusion of bulk flow
and the cold component, the two targeted sibling links, and the deliberate
plasma-boundary qualifier. The secondary review independently awarded the
same four `20/20` dimensions and described the boundary wording as consistent
with the expert steering note. Neither reviewer raised a locus contradiction,
and `docs_review_quorum_shortfall` is `null`.

The live review write reported:

```text
Wrote 2 StandardNameReview nodes; cleared suggestions: non-distinct=0, unparseable=0
persist_reviewed_docs: hot_neutral_temperature_at_plasma_boundary → docs_stage=accepted (score=1.000, chain=0/3, resolution=quorum_consensus, shortfall=None)
```

## Final state and WEST admission

The post-run graph read returned:

| Field | Final value |
| --- | --- |
| id | `hot_neutral_temperature_at_plasma_boundary` |
| description | `Energy-equivalent translational temperature of hot neutral atoms at the plasma boundary, characterizing their random motion after removal of bulk flow.` |
| documentation | present; candidate shown above |
| name stage / score | `accepted` / `0.9875` |
| docs stage / score | `accepted` / `1.000` |
| status | `draft` |
| validation | `valid`, `validated_at` unchanged and non-null |
| unit | `eV` |
| docs-axis reviews | 2 new rows, both `1.000`; winning method `quorum_consensus` |
| source paths | the same 2 DD bindings; no source mutation |

The normal export admission predicate was checked for this identity rather
than inferred from `docs_stage`. All five relevant conditions were true:

```text
name_ok=true
validation_ok=true
name_quorum_ok=true
docs_quorum_ok=true
winning_docs_review=true
```

Accordingly, the WEST cut would now carry this identity. It was withheld
before the run solely because documentation had not been generated and
accepted; no name-stage, validation, or source-path repair was needed.

Campaign accounting was measured before and after over the same bounded
cost-row horizon (`llm_at >= 2026-09-10T04:00:00Z`):

| Measurement | Rows | Spend |
| --- | ---: | ---: |
| before | 1,712 | `$102.872694` |
| after | 1,715 | `$102.999602` |
| this node | 3 | `$0.126908` |

The node stayed below its `$12.00` cap and the campaign stayed below the
authorised `$250.00` ceiling. No quarantine was cleared, nothing was deleted,
and no compose request was made.

## Follow-ons outside this scope

The scoped run log also emitted pre-existing global maintenance warnings for
22 live names with no source and 7 live names fed by DD paths absent from the
current DD. Those populations were outside this exact-name docs scope and
were not changed here. Their owning graph-repair work must remain separate
from this successful documentation recovery.
