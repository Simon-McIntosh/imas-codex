"""Graph quality and compliance tests.

Only tests marked ``graph`` need a live Neo4j connection; the default run
excludes that marker. The package also holds guards that need no graph — a
graph-client double stands in for the server — and those run by default. Run
the live tests explicitly with:

    uv run pytest tests/graph/ -m graph -v

Connection resolves via:
    1. Environment variables (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    2. pyproject.toml [tool.imas-codex.graph] section
    3. Built-in defaults (bolt://localhost:7687, neo4j, imas-codex)
"""
