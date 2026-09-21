# Can the review surface run on the local lane?

provisional: true

Measured at `409809bff7538002648a5fd42a324f52a864b74b`. Every routing claim below
is a resolution taken through the real registry, not a reading of
`pyproject.toml`: `~/.cache/review-lane-probe.log`, produced by
`/tmp/review-lane-probe.py`.

## Answer, in one line

**No. Exactly one model in the entire configuration resolves to the local
endpoint, and every other review surface reaches a paid provider without the
operator asking for one.**

## The two facts to verify rather than trust — both hold

**The endpoint serves exactly one model.** Live `GET /v1/models`:

```
$ curl -s http://98dci4-gpu-0003:18802/v1/models
{"object":"list","data":[{"id":"deepseek-v4.1-flash","object":"model",
 "created":1789971776,"owned_by":"sglang","root":"deepseek-v4.1-flash",
 "parent":null,"max_model_len":512000}]}
```

One entry, `deepseek-v4.1-flash`. Anything else named in a request is a 404 at
generation time.

**`_MODEL_ENDPOINTS` is keyed on the full seat string, and the lookup is an
exact dict hit.** `imas_codex/settings.py:303` declares the registry,
`:334` and `:358` are the only two writers, and `:379` is the whole of the
reader — `return _MODEL_ENDPOINTS.get(model)`. Measured, with a positive control
so the registry is shown non-empty before any absence is reported:

```
_MODEL_ENDPOINTS holds 1 keys
  local/deepseek-v4.1-flash                    -> http://98dci4-gpu-0003:18802/v1

=== negative control: a seat-shaped name that is not a configured seat ===
  local/deepseek-v4.1-flash        -> http://98dci4-gpu-0003:18802/v1
  deepseek-v4.1-flash              -> None (falls to proxy)
  local/deepseek-v4-flash          -> None (falls to proxy)
```

The second control row is the sharp one: **the name the server actually serves,
without the `local/` prefix, resolves no endpoint.** The third is last sprint's
seat spelling — one character of drift in a seat string silently moves the call
onto a paid lane.

## The silent fallback, with its line

`imas_codex/discovery/base/llm.py:1947-1952`:

```python
if not api_base:
    endpoint = get_model_endpoint(model)
    if endpoint:
        api_base = endpoint["api_base"]
        ...
```

`get_model_endpoint` returning `None` is not an error and is not logged as one.
Control falls straight through to the `else` at `llm.py:1974`, which picks the
LiteLLM proxy or OpenRouter direct. **There is no branch anywhere in
`_build_kwargs` that refuses an unresolved model, and no caller is required to
ask for a paid provider for this to happen** — a model the registry does not
hold is routed to a paid lane by default, and the only trace is a `logger.debug`
line on the *other* branch (`llm.py:1968`) that simply does not appear.

This is the absence-is-permission shape: the check is `if endpoint:`, and the
absent case is admitted rather than refused.

## Per-surface verdict

| Surface | Model actually used | Endpoint resolved | Local end to end? |
|---|---|---|---|
| `--target names`, default quorum | `local/deepseek-v4.1-flash`, `openrouter/anthropic/claude-haiku-4.5`, `openrouter/openai/gpt-5.5` | local for 1 of 3; **none** for 2 of 3 | **No** — 1 of 3 cycles is free |
| `--target names`, `local-only` profile | `local/deepseek-v4.1-flash` | local | **Yes** |
| `--target names`, `default` profile | grok-4.5, gpt-5.6-luna, claude-sonnet-5 | **none for all 3** | **No** — fully paid |
| `--target docs` | claude-sonnet-5, grok-4.5, gpt-5.5 | **none for all 3** | **No** — fully paid, and see below |
| parent/description synthesis (`sn-parent-enrich`) | `local/deepseek-v4.1-flash` | local | **Yes** |
| `sn-docs` (documentation generation) | `openrouter/openai/gpt-5.6-luna` | none | **No** |
| `sn-refine` | `openrouter/openai/gpt-5.5` | none | **No** |
| `sn-escalation` | `openrouter/anthropic/claude-fable-5` | none | **No** |
| `sn-classifier` | `openrouter/openai/gpt-5.5` | none | **No** |
| `sn-prose-adjudicator` | `openrouter/openai/gpt-5.6-luna` | none | **No** |
| `sn-release-notes` | `openrouter/anthropic/claude-sonnet-4.6` | none | **No** |

**`[tool.imas-codex.sn-review.docs]` carries no `model-route` at all**
(measured: `model-route=None`), where `[sn-review.names]` carries
`model-route = "ambix-local"`. The registry's second pass
(`settings.py:339-370`) binds a `local/`-prefixed member only when the node it
sits in resolves an `api_base`. So **the docs axis could not reach the local
endpoint even if a local model were added to its list** — the omission is
structural, not a matter of which models are named.
