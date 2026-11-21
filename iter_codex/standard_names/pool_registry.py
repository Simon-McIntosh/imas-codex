"""Dependency-light standard-name worker pool registry.

This module owns immutable pool identity and admission weights. It deliberately
imports no discovery, CLI, settings, or worker modules so command-line option
construction can inspect pool names in a fresh interpreter without triggering
the operational pool stack.
"""

from __future__ import annotations

# Default per-pool weights for soft-fairness admission control.
# Sum to 1.0 across the seven pools of the refine pipeline.
#
# Review is the throughput bottleneck: it uses multiple blind model calls plus
# optional escalation, while generation is normally cheaper. Favouring review
# pools keeps names and docs flowing through the pipeline at comparable rates.
# Embedding remains a separate discovery worker and is not a pool.
#
# These weights ration the dollar budget across paid pools. A pool whose model
# routes to a local or zero-cost endpoint bypasses spend fairness and is removed
# from the fairness denominator, so its configured weight is inert while free
# and becomes active only if that seat is moved to a paid model.
# ``enrich_parents`` generalizes placeholder parent descriptions from accepted
# children. It is paced modestly because its results feed the documentation
# axis after structural name acceptance.
POOL_WEIGHTS: dict[str, float] = {
    "generate_name": 0.12,
    "review_name": 0.24,
    "refine_name": 0.10,
    "generate_docs": 0.14,
    "review_docs": 0.24,
    "refine_docs": 0.08,
    "enrich_parents": 0.08,
}

POOL_NAMES: tuple[str, ...] = tuple(POOL_WEIGHTS)

__all__ = ["POOL_NAMES", "POOL_WEIGHTS"]
