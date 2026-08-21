# Compose-to-writer coordinate mismatch

## Verdict

The defect is owned by **imas-codex**, not by imas-standard-names. Both requested
identities compose deterministically from Codex's structured `GrammarSegments`
as geometry coordinates and both strict-parse and losslessly round-trip under
ISN **0.8.0rc66**. Codex then projects the rc66 IR enum value `geometry` through
a stale comparison against `geometry_carrier`, misclassifying the base token
`coordinate` as a `physical_base` instead of a `geometric_base`.

The exact seam in commit `c81913747f78ead2c3bf57ab0ea0208543105d94` is
`imas_codex/standard_names/graph_ops.py:213-220`, specifically line 218:

```python
"geometric_base" if base_kind == "geometry_carrier" else "physical_base"
```

ISN rc66 emits `BaseKind.GEOMETRY.value == "geometry"`. Codex's compose model
already uses that public spelling (`imas_codex/standard_names/models.py:193-195`)
and routes it to `geometric_base` (`models.py:446-449`). The writer compatibility
projection is therefore the disagreeing component. The strict writer gate at
`graph_ops.py:5321-5343` is correct and must remain fail-closed.

## Deterministic reproduction

The reproduction made **0 provider calls**, spent **USD 0.00**, and used the
public rc66 parser plus pure Codex compose/projection functions. It did not run
the generation pool or any graph writer.

| Requested identity | Compose-side classification and output | Strict-parser verdict | Codex writer projection |
|---|---|---|---|
| `toroidal_coordinate_of_shatter_cone` | `base_kind=geometry`, `base_token=coordinate`, `projection_shape=coordinate`; emitted the requested identity exactly | **ACCEPT**; IR `base.kind=geometry`, `base.token=coordinate`; exact round-trip | `physical_base=coordinate`, `geometric_base=null` — **wrong** |
| `toroidal_coordinate_of_reflectometer_antenna` | `base_kind=geometry`, `base_token=coordinate`, `projection_shape=coordinate`; emitted the requested identity exactly | **ACCEPT**; IR `base.kind=geometry`, `base.token=coordinate`; exact round-trip | `physical_base=coordinate`, `geometric_base=null` — **wrong** |

This reproduces the classification disagreement for **2/2 identities** with no
model nondeterminism. The active production grammar query returned exactly one
active snapshot, `0.8.0rc66`, containing **956 GrammarToken nodes**.

The single-draw log contains two distinct observable consequences and they must
not be conflated:

- The shatter-cone candidate was the requested, strict-valid identity. After the
  stale projection, decomposition reported `physical_base:coordinate` as a
  token miss; the exact-claim batch then failed atomically.
- The reflectometer model emitted
  `toroidal_coordinate_of_diagnostic_component_center`, not the requested
  identity. The strict gate correctly rejected that different spelling. The
  requested `toroidal_coordinate_of_reflectometer_antenna` itself strict-parses,
  and the deterministic probe shows it reaches the same stale Codex projection
  when represented from its canonical compose fields.

Thus another provider draw is neither required nor justified to diagnose the
seam. The code repair belongs in Codex: compare the public IR base kind with
`geometry` (or its enum) and add both coordinate identities as regression cases.
No ISN grammar or vocabulary change is indicated by this evidence.

## Zero-mutation and accounting proof

Production counts were read immediately before and after the offline
classification within one short read-only client session:

| Node label | Before | After | Delta |
|---|---:|---:|---:|
| `StandardName` | 4,393 | 4,393 | **0** |
| `LLMCost` | 27,631 | 27,631 | **0** |

The counts are identical, the reproduction recorded `provider_calls=0`, and
attributable spend is **USD 0.00**. No production graph mutation was attempted.

## Evidence and verification

- Reproducer:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194828572595-composewriter/reproduce_mismatch.py`
- Reproduction log and assertions (exit 0):
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194828572595-composewriter/reproduction.log`
  — SHA-256 `958f2ed1a64bfaa922a7165a7abb7a1d73f3b7957b6d513255346efe7ead2c4d`
- Original single-draw log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T143734253079-freshdraw/compose-absent-once.log`
  — SHA-256 `c0d98a0379c096113c53ff2dc3e2f176e45ec68172083741137e85c55fbd5046`

Verification command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
  uv run --no-sync python \
  /home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T194828572595-composewriter/reproduce_mismatch.py
```

The script asserts the rc66 package and graph snapshot, both exact compose and
strict-parser round-trips, both incorrect writer projections, and identical
before/after production counts.
