This file governs the `imas_codex/llm/` subtree — the LLM call layer, prompt loading and
rendering, model routing, service tagging, and prompt caching.



## LLM Access

All LLM interaction flows through two canonical modules. Never call `litellm.completion()` directly — the shared functions handle prompt caching flags, cost tracking, retries with exponential backoff, and structured output parsing.

### Calling LLMs

Use `call_llm_structured()` / `acall_llm_structured()` from `imas_codex.discovery.base.llm`:

```python
from imas_codex.discovery.base.llm import call_llm_structured

result, cost, tokens = call_llm_structured(
    model=get_model("language"),
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_model=MyPydanticModel,
)
```

These functions automatically: apply `inject_cache_control()` to system messages, retry on API/parse errors with backoff, accumulate cost across retries, and parse structured output via Pydantic `response_format`.

### Rendering Prompts

Use `render_prompt()` from `imas_codex.llm.prompt_loader` — never construct paths to prompt files manually:

```python
from imas_codex.llm.prompt_loader import render_prompt

system_prompt = render_prompt("paths/scorer", {"facility": "tcv", "batch": batch_data})
```

For path access (e.g., in tests), import `PROMPTS_DIR` from the same module — never hardcode path segments like `"llm" / "prompts"`.

### Rules

- Model identifiers require the `openrouter/` prefix to preserve `cache_control` blocks
- Use `get_model(section)` from `imas_codex.settings` for model selection — never hardcode model names
- All models live in `pyproject.toml` under `[tool.imas-codex.<section>].model`. **Never hardcode `cc:*`, raw provider IDs, or any other model string in pipeline code or worker prompts.** If you think you need a new model variant, add a section to `pyproject.toml` and reference it via `get_model("section")`.
- Pydantic schema injection via `get_pydantic_schema_json()` — never hardcode JSON examples in prompts
- Each prompt declares `schema_needs` in frontmatter to load only required schema context

### Routing: Direct vs Proxy

`call_llm_structured()` chooses between two paths automatically. Understanding both is critical because they have different cost-tracking and caching behavior.

| Path | Trigger | Cost tracking | Prompt caching | Use for |
|------|---------|---------------|----------------|---------|
| **Direct (bypass)** | `supports_cache(model)` AND `OPENROUTER_API_KEY_IMAS_CODEX` set | ✅ `response_cost` populated | ✅ `cache_control` preserved | All cache-capable models on the codex billing account |
| **Proxy** | Otherwise (no IMAS_CODEX key, or non-caching model) | ❌ `response_cost = 0` | ❌ `cache_control` stripped | Air-gapped clusters, models without caching, dev environments |

Keep `OPENROUTER_API_KEY_IMAS_CODEX` set and use `openrouter/anthropic/<model>` (or unprefixed — `ensure_model_prefix()` normalizes) to get cost tracking + caching for free (empirically ~87% cheaper on a warm cache). Bypass logic in `imas_codex/discovery/base/llm.py`.

**Anti-pattern: do not invent `cc:*` model strings.** A `cc:opus` / `cc:sonnet` / `cc:haiku` proxy alias exists in `litellm_config.yaml` for billing isolation to a separate OpenRouter account, but it routes via the **proxy path** which silently breaks both `response_cost` and `cache_control`. If you need spend isolation, add a per-service env var (`OPENROUTER_API_KEY_<SERVICE>`) and let the direct path handle routing — it preserves cost + cache.

### Service Tagging

All LLM calls are tagged with a `service` parameter for spend visibility. The tag flows to:
- **X-Title** header → OpenRouter dashboard (shows as `imas-codex:<service>`)
- **Langfuse metadata** → trace analytics
- **Per-service API keys** → optional spend isolation via separate OpenRouter keys

**Service taxonomy:**

| Service Tag | Description | Call Sites |
|-------------|-------------|------------|
| `facility-discovery` | Facility path/wiki/signal/code discovery | `discovery/paths/*`, `discovery/wiki/*`, `discovery/signals/*`, `discovery/base/image.py`, `discovery/code/*`, `discovery/static/*` |
| `standard-names` | Standard name generation, review, enrichment | `standard_names/workers.py`, `benchmark.py`, `review/pipeline.py` |
| `data-dictionary` | DD enrichment, domain classification, and cluster labeling | `graph/dd_enrichment.py`, `dd_ids_enrichment.py`, `dd_identifier_enrichment.py`, `dd_workers.py`, `clusters/labeler.py` |
| `imas-mapping` | IMAS signal-to-path mapping | `ids/mapping.py`, `ids/metadata.py` |
| `untagged` | Default — surfaces missed call sites | Any call without explicit `service=` |

**Usage:**
```python
result, cost, tokens = call_llm_structured(
    model=model, messages=messages, response_model=MyModel,
    service="facility-discovery",  # Required — AST test enforces this
)
```

**Per-service API keys** (optional): Set `OPENROUTER_API_KEY_<SERVICE_UPPER>` to use a separate OpenRouter key for a service pipeline. Hyphens become underscores. Falls back to `OPENROUTER_API_KEY_IMAS_CODEX`.

```bash
# Example: isolate discovery spend to its own key
export OPENROUTER_API_KEY_FACILITY_DISCOVERY=sk-or-v1-...
```

### Prompt Structure and Caching

Static-first ordering maximises OpenRouter prompt-cache hit rates. **System prompt** (schema, enums, rules, output format) is static and shared — `inject_cache_control()` sets a `cache_control: {"type": "ephemeral"}` breakpoint at its end. **User prompt** is per-call dynamic. In Jinja templates, place `{% include %}` schema/rules blocks BEFORE dynamic variables to maximise the cacheable prefix.
