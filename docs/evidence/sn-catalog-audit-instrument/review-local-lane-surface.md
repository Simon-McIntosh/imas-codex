# Can the review surface run on the local lane?

provisional: false

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
| `--target names`, no flags | grok-4.5, gpt-5.6-luna, claude-sonnet-5 | **none for all 3** | **No** — fully paid (see the profile trap below) |
| `--target names`, `local-only` profile | `local/deepseek-v4.1-flash` | local | **Yes** |
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

## The profile trap: the mixed list in `[sn-review.names]` is not what runs

`[tool.imas-codex.sn-review.names].models` lists one local model and two paid
ones, and its comment describes cycle 0 as *"local DS4-flash — catches
grammar/format/ISN violations for free"*. **That list does not run.** The
accessor the pipeline calls resolves a *profile* first, and the hard-coded
default at `imas_codex/settings.py:1556` is `"default"` — a different section,
whose three members are all paid:

```
active profile with no flag and no env: 'default'

get_sn_review_names_models() — what --target names actually runs:
  openrouter/x-ai/grok-4.5                       -> PAID (proxy/OpenRouter)
  openrouter/openai/gpt-5.6-luna                 -> PAID (proxy/OpenRouter)
  openrouter/anthropic/claude-sonnet-5           -> PAID (proxy/OpenRouter)

the local-only profile, for comparison:
  local/deepseek-v4.1-flash                      -> http://98dci4-gpu-0003:18802/v1
```

So **reading `pyproject.toml` gives the wrong answer** about the name-review
axis, and gives it in the reassuring direction: the config file says one free
cycle, the accessor says none. The free lane exists and has to be asked for —
`--reviewer-profile local-only`, or `IMAS_CODEX_SN_REVIEW_PROFILE`.

Measured through the accessors, not the file: `~/.cache/review-lane-effective.log`.

## The embedding path is local and free — the one surface that is entirely on-lane

The comparators' embeddings do **not** reach a paid provider. `Encoder`
(`imas_codex/embeddings/encoder.py:73-89`) selects between a local
SentenceTransformer and a remote HTTP GPU server, and resolves here to the
latter:

```
backend=<EmbeddingBackend.REMOTE: 'remote'> model='Qwen/Qwen3-Embedding-0.6B'
remote_url='http://98dci4-gpu-0002:18765'
```

`imas_codex/embeddings/openrouter_embed.py` does contain a paid embedding
client, which is why this needed checking rather than assuming — but nothing
outside that module constructs it. The only symbol `encoder.py` imports from it
is the `EmbeddingResult` dataclass (`encoder.py:35`). `Encoder` is explicit
about the policy in its own docstring: *"No silent fallback: if the configured
backend is unavailable, an error is raised"* (`encoder.py:7,47`), and
`:89` raises `EmbeddingBackendError` on an unknown backend. **This is the one
part of the review surface that both runs locally and refuses rather than
falling back.**

### But the audit that uses it swallows its own failure

`imas_codex/standard_names/review/audits.py:224-269` wraps the whole embedding
preflight — the batch re-embed *and* the graph persistence — in
`try: … except Exception: logger.warning(...)`. A bare `Exception`, a warning,
and execution continues. So the `Encoder`'s refusal is converted back into a
silent pass one layer up: when the embedding server is unreachable, the audit
returns an `EmbeddingReport` whose `refreshed_count` is 0 while
`missing_count` and `stale_count` retain their pre-flight values, and the
near-duplicate detection downstream runs against stale or absent vectors. **A
review that cannot embed still reports.** This is the same shape as the routing
fallback: the failing case is admitted rather than refused, and the only trace
is a log line.

## Layer 1 audits and consolidation reach no model at all

Both are deterministic and need no lane:

- **Layer 1 audits** (`review/audits.py`) — the module docstring states
  *"lexical lint, link integrity checks, and near-duplicate detection — all
  runnable without LLM access"* (`:3-4`), and a grep of the file for `llm`,
  `completion`, `get_model`, `openrouter` and `litellm` returns only the
  embedding references above. **Local, with the swallow caveat.**
- **Consolidation** (`review/consolidation.py`) — *"Purely deterministic — no
  LLM calls, no graph queries"* (`:3`). The same grep returns three hits, all
  in prose or a field name (`:3`, `:59`, `:390`) and none a call site.
  **Runs anywhere, costs nothing.**

## Every silent fallback, with its line

| Site | Shape | Consequence |
|---|---|---|
| `imas_codex/discovery/base/llm.py:1947-1952` | `if endpoint:` — an unresolved model is admitted, not refused | any model not registered by its exact seat string is routed to a paid provider, with no error and no warning |
| `imas_codex/settings.py:379` | `_MODEL_ENDPOINTS.get(model)` returns `None` for a miss | the miss is indistinguishable from "this model is meant to be paid" |
| `imas_codex/settings.py:1556` | profile defaults to `"default"` | the paid quorum runs unless the free one is requested; the mixed list in the config file never executes |
| `imas_codex/standard_names/review/audits.py:266-269` | bare `except Exception` around the embedding preflight | an unreachable embedding server produces a clean-looking audit over stale vectors |
| `[tool.imas-codex.sn-review.docs]` (no `model-route`) | the registry's second pass needs an `api_base` on the node | the docs axis cannot bind a local model even if one is listed |

Nothing in this list requires the operator to ask for a paid provider. Four of
the five are absent-value defaults; the fifth is a missing configuration key.

## What it would take to run the review surface end to end on the local lane

Not implemented — outside this node's scope, recorded as follow-ons:

1. `--reviewer-profile local-only` covers the name axis today, and is the only
   surface that is already reachable.
2. The docs axis needs `model-route = "ambix-local"` on
   `[tool.imas-codex.sn-review.docs]` **and** a `local/` member, in that order —
   adding the member alone changes nothing.
3. `sn-refine`, `sn-classifier`, `sn-prose-adjudicator` and `sn-escalation` have
   no local seat at all. The endpoint serves one model, so all four would have to
   share `deepseek-v4.1-flash`.
4. The routing fallback should refuse rather than default: a model carrying a
   local-lane prefix (`local/`, `hosted_vllm/`, `ollama/`) that resolves no
   endpoint is a configuration error, not a request for the proxy. That single
   check would have caught the seat-spelling drift the negative control shows.
