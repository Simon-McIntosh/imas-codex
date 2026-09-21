"""A skipped attachment audit cannot be mistaken for a clean audit."""

from __future__ import annotations

import logging

import pytest

from imas_codex.standard_names.attachment_audit import AttachmentAuditResult
from imas_codex.standard_names.loop import (
    _log_attachment_audit_result,
    _run_attachment_audit_for_pool_run,
)
from imas_codex.standard_names.provenance_rebuild import _attachment_violation_rows


@pytest.mark.asyncio
async def test_skipped_and_clean_attachment_audits_remain_distinct(caplog) -> None:
    caplog.set_level(logging.INFO, logger="imas_codex.standard_names.loop")

    bypassed = await _run_attachment_audit_for_pool_run(
        skip_global_maintenance=True,
        run_id="bounded-run",
    )
    bypass_messages = [record.getMessage() for record in caplog.records]
    clean = AttachmentAuditResult()

    assert bypassed.audit_ran is False
    assert clean.audit_ran is True
    assert bypassed.as_dict()["audit_ran"] is False
    assert clean.as_dict()["audit_ran"] is True
    assert bypassed != clean
    assert any(
        "attachment-consistency audit skipped" in message
        and "skip_global_maintenance=True" in message
        for message in bypass_messages
    )
    assert not any("attachment(s) rejected" in message for message in bypass_messages)

    caplog.clear()
    _log_attachment_audit_result(clean)
    assert caplog.records == []
    assert _attachment_violation_rows(clean) == []

    with pytest.raises(ValueError, match="attachment consistency was not audited"):
        _attachment_violation_rows(bypassed)
