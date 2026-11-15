# Move the configured model seats to their providers' newest model lines

## Goal

Each paid model seat declared in `pyproject.toml` names the newest model line
its provider offers at the seat's tier, and the code's fallback defaults agree
with the configured seats.

## Authority

The lead authorised moving the model seats to the latest models without a
benchmark on 2026-11-15. The earlier seat check held every seat on its current
line because seating a new line was a benchmark decision; that condition is
lifted.

## What is known

- `tests/config/test_model_seats.py` and its catalogue snapshot under
  `tests/config/data/` derive each seat's newest release within its line. All
  63 declared identifiers were already on their line's newest release.
- That check recorded newer lines than any seat uses: a newer Anthropic line
  than the seated Sonnet one, a newer OpenAI line, newer Google lines, and a
  `-pro` tier of every seated OpenAI line.
- `imas_codex/settings.py` holds fallback defaults; its `ids-mapping` default
  names an external model while the configured seat names the local one.
- `imas_codex/config/litellm_config.yaml` lists two proxy routes that no seat
  names.

## Rules

- Refresh the catalogue snapshot from the providers' own model listings first,
  and record the date in the snapshot as the existing test does.
- A seat moves to the newest generally available line of the **same provider**
  at the **same tier**. Tier means the provider's own price and capability
  class: a standard seat does not move to a `-pro`, premium or reasoning tier,
  and a fast or small seat does not move to a standard one. Where a provider
  has renamed its lines so that "same tier" is not evident from the listing,
  choose the line whose listed price per token is closest to the current
  seat's, state both prices, and say it was a judgement.
- Preview, beta, experimental and deprecated models are not seated.
- Seats routed to the locally served model do not change; that lane serves one
  model.
- The decisions seats that call a provider's own decisions endpoint move only
  if the catalogue lists a newer version of that same endpoint model.
- Change each seat's comments that name a switch-back model so they stay true.
- Make every fallback default in `imas_codex/settings.py` equal the configured
  seat, so code and configuration cannot disagree.
- Do not remove the two unseated proxy routes; list them in the evidence with
  whether the catalogue still offers them.
- Never print or commit an API key.

## Done when

- The evidence fragment lists every declared seat with: the model before, the
  model after, the provider's listed price per million input and output tokens
  before and after, and whether the move was same-line, new-line or unchanged.
- The seat tests pass against the refreshed snapshot with no added failure
  against the base revision, and a test asserts that each fallback default
  equals its configured seat.
- Under the opt-in live marker, each changed seat answers one request with
  non-empty content; the response model name is recorded for each.
- The projected change in cost is stated per seat as the ratio of new to old
  listed price.
- Negative control: pointing a seat at a model its route does not serve fails
  the live request and is reported; a fallback default that differs from its
  seat fails the new test.

## Scope

Write paths: `pyproject.toml`, `imas_codex/settings.py`, `imas_codex/config/`,
`tests/config/`, and the evidence fragment
`docs/evidence/fragments/iter-signal-context-and-enrichment/move-model-seats-to-newest-lines.html`.

Tests run from the worktree against the shared environment with `--no-sync`.
Commit locally with a conventional subject and a body. Do not push or merge.
