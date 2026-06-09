# Reviewer bake-off — summary

- gold set: 20 positives / 10 negatives  | efforts: ['low', 'high']  | total cost: $0.77

Ranked by discrimination = mean(positive) − mean(negative). Lower false-accept and false-reject are better.

| Model | effort | disc | mean_pos | mean_neg | false_accept | false_reject | $/run | done |
|-------|--------|-----:|---------:|---------:|-------------:|-------------:|------:|:----:|
| `openrouter/qwen/qwen3.7-max` | high | 0.53 | 1.0 | 0.47 | 0.7 | 0.0 | 0.028 | ✓ |
| `openrouter/anthropic/claude-opus-4.8` | low | 0.52 | 0.995 | 0.475 | 0.6 | 0.0 | 0.1282 | ✓ |
| `openrouter/anthropic/claude-opus-4.8` | high | 0.512 | 0.99 | 0.478 | 0.6 | 0.0 | 0.1336 | ✓ |
| `openrouter/qwen/qwen3.7-max` | low | 0.495 | 0.965 | 0.47 | 0.7 | 0.05 | 0.0377 | ✓ |
| `openrouter/minimax/minimax-m3` | high | 0.49 | 0.96 | 0.47 | 0.6 | 0.05 | 0.0077 | ✓ |
| `openrouter/moonshotai/kimi-k2.6` | high | 0.48 | 0.95 | 0.47 | 0.7 | 0.1 | 0.0814 | ✗ |
| `openrouter/xiaomi/mimo-v2.5-pro` | high | 0.445 | 0.885 | 0.44 | 0.6 | 0.2 | 0.0282 | ✓ |
| `openrouter/google/gemini-3.1-pro-preview` | high | 0.42 | 0.99 | 0.57 | 0.7 | 0.0 | 0.0762 | ✓ |
| `openrouter/google/gemini-3.1-pro-preview` | low | 0.4 | 1.0 | 0.6 | 0.8 | 0.0 | 0.0497 | ✓ |
| `openrouter/minimax/minimax-m3` | low | 0.397 | 0.948 | 0.55 | 0.7 | 0.0 | 0.0112 | ✓ |
| `openrouter/xiaomi/mimo-v2.5-pro` | low | 0.37 | 0.96 | 0.59 | 0.8 | 0.05 | 0.0245 | ✓ |
| `openrouter/google/gemini-3.5-flash` | low | 0.36 | 1.0 | 0.64 | 0.8 | 0.0 | 0.0272 | ✓ |
| `openrouter/google/gemini-3.5-flash` | high | 0.36 | 1.0 | 0.64 | 0.8 | 0.0 | 0.0767 | ✓ |
| `openrouter/openrouter/owl-alpha` | default | None | None | None | None | None | 0.0 | ✗ |
| `openrouter/moonshotai/kimi-k2.6` | low | None | 0.9 | None | None | 0.2 | 0.0596 | ✗ |

## Pairwise Spearman agreement (blind-pair selection)

Lower agreement between two ACCURATE models = more complementary blind pair (disagreement meaningfully triggers the breaker).

- `openrouter/openrouter/owl-alpha@default` vs `openrouter/xiaomi/mimo-v2.5-pro@low`: ρ = None
- `openrouter/openrouter/owl-alpha@default` vs `openrouter/qwen/qwen3.7-max@low`: ρ = None
- `openrouter/minimax/minimax-m3@low` vs `openrouter/openrouter/owl-alpha@default`: ρ = None
- `openrouter/minimax/minimax-m3@low` vs `openrouter/xiaomi/mimo-v2.5-pro@low`: ρ = 0.466
- `openrouter/minimax/minimax-m3@low` vs `openrouter/qwen/qwen3.7-max@low`: ρ = 0.682
- `openrouter/minimax/minimax-m3@low` vs `openrouter/moonshotai/kimi-k2.6@low`: ρ = None
- `openrouter/qwen/qwen3.7-max@low` vs `openrouter/xiaomi/mimo-v2.5-pro@low`: ρ = 0.836
- `openrouter/google/gemini-3.5-flash@low` vs `openrouter/openrouter/owl-alpha@default`: ρ = None
- `openrouter/google/gemini-3.5-flash@low` vs `openrouter/minimax/minimax-m3@low`: ρ = 0.559
- `openrouter/google/gemini-3.5-flash@low` vs `openrouter/xiaomi/mimo-v2.5-pro@low`: ρ = 0.75
- `openrouter/google/gemini-3.5-flash@low` vs `openrouter/qwen/qwen3.7-max@low`: ρ = 0.787
- `openrouter/google/gemini-3.5-flash@low` vs `openrouter/moonshotai/kimi-k2.6@low`: ρ = None
- `openrouter/moonshotai/kimi-k2.6@low` vs `openrouter/openrouter/owl-alpha@default`: ρ = None
- `openrouter/moonshotai/kimi-k2.6@low` vs `openrouter/xiaomi/mimo-v2.5-pro@low`: ρ = -0.167
- `openrouter/moonshotai/kimi-k2.6@low` vs `openrouter/qwen/qwen3.7-max@low`: ρ = -0.167
- `openrouter/google/gemini-3.1-pro-preview@low` vs `openrouter/openrouter/owl-alpha@default`: ρ = None
- `openrouter/google/gemini-3.1-pro-preview@low` vs `openrouter/minimax/minimax-m3@low`: ρ = 0.559
- `openrouter/google/gemini-3.1-pro-preview@low` vs `openrouter/xiaomi/mimo-v2.5-pro@low`: ρ = 0.75
- `openrouter/google/gemini-3.1-pro-preview@low` vs `openrouter/qwen/qwen3.7-max@low`: ρ = 0.787
- `openrouter/google/gemini-3.1-pro-preview@low` vs `openrouter/google/gemini-3.5-flash@low`: ρ = 1.0
- `openrouter/google/gemini-3.1-pro-preview@low` vs `openrouter/moonshotai/kimi-k2.6@low`: ρ = None
- `openrouter/anthropic/claude-opus-4.8@low` vs `openrouter/openrouter/owl-alpha@default`: ρ = None
- `openrouter/anthropic/claude-opus-4.8@low` vs `openrouter/minimax/minimax-m3@low`: ρ = 0.613
- `openrouter/anthropic/claude-opus-4.8@low` vs `openrouter/xiaomi/mimo-v2.5-pro@low`: ρ = 0.707
- `openrouter/anthropic/claude-opus-4.8@low` vs `openrouter/qwen/qwen3.7-max@low`: ρ = 0.881
- `openrouter/anthropic/claude-opus-4.8@low` vs `openrouter/google/gemini-3.5-flash@low`: ρ = 0.807
- `openrouter/anthropic/claude-opus-4.8@low` vs `openrouter/moonshotai/kimi-k2.6@low`: ρ = None
- `openrouter/anthropic/claude-opus-4.8@low` vs `openrouter/google/gemini-3.1-pro-preview@low`: ρ = 0.807
