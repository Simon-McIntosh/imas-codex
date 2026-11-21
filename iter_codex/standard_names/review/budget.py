"""Budget management for LLM review batches.

Re-exports the shared :class:`~iter_codex.standard_names.budget.BudgetManager`
under its original name for backward compatibility.  All new code should
import from ``iter_codex.standard_names.budget`` directly.
"""

from iter_codex.standard_names.budget import (
    BudgetExceeded,  # noqa: F401
    BudgetLease,  # noqa: F401
    BudgetManager as ReviewBudgetManager,  # noqa: F401, E501
)

__all__ = ["ReviewBudgetManager", "BudgetExceeded", "BudgetLease"]
