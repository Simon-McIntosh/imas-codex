"""
Search engines for ITER Codex.

This package contains different search engine implementations for the ITER Codex system.
Each engine provides a specific search strategy (semantic, lexical, hybrid).
"""

from .base_engine import MockSearchEngine, SearchEngine, SearchEngineError
from .hybrid_engine import HybridSearchEngine
from .lexical_engine import LexicalSearchEngine
from .semantic_engine import SemanticSearchEngine

__all__ = [
    "SearchEngine",
    "SearchEngineError",
    "MockSearchEngine",
    "SemanticSearchEngine",
    "LexicalSearchEngine",
    "HybridSearchEngine",
]
