"""Domain-owned typed dispatch policy resources.

Each migration adds one uniquely named JSON resource containing exact callsite
policies.  An empty package means every production expression remains on the
legacy transition side and typed dispatch fails closed.
"""
