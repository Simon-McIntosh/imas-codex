"""Regression coverage for Layer 1 findings passed to name reviewers."""

from imas_codex.standard_names.review.audits import (
    AuditReport,
    DuplicateComponent,
    LinkFinding,
    LintFinding,
)
from imas_codex.standard_names.review.pipeline import _extract_audit_findings


def test_real_audit_findings_reach_the_batch_prompt() -> None:
    """Every real finding type is summarized for its affected review batch."""
    report = AuditReport(
        lint_findings=[
            LintFinding(
                name_id="plasma_current",
                finding_type="round_trip_failure",
                detail="name does not survive parser round-trip",
                severity="error",
            ),
            LintFinding(
                name_id="outside_batch",
                finding_type="convention_violation",
                detail="this finding belongs to another batch",
            ),
        ],
        link_findings=[
            LinkFinding(
                name_id="electron_temperature",
                finding_type="dead_link",
                target="missing_temperature",
                detail="link target is absent from the catalog",
            )
        ],
        duplicate_components=[
            DuplicateComponent(
                names=["ion_temperature", "toroidal_ion_temperature"],
                max_similarity=0.97,
                pairs=[
                    ("ion_temperature", "toroidal_ion_temperature", 0.97),
                ],
            )
        ],
    )

    summaries = _extract_audit_findings(
        report,
        {"plasma_current", "electron_temperature", "ion_temperature"},
    )

    assert summaries == [
        "[error:lint:round_trip_failure] plasma_current: "
        "name does not survive parser round-trip",
        "[warning:link:dead_link] electron_temperature -> missing_temperature: "
        "link target is absent from the catalog",
        "[warning:duplicate] ion_temperature, toroidal_ion_temperature: "
        "near-duplicate component (max similarity 0.9700)",
    ]
    assert all("outside_batch" not in summary for summary in summaries)


def test_audit_finding_summaries_keep_the_batch_verbosity_cap() -> None:
    """Large reports still pass no more than twenty findings to a reviewer."""
    report = AuditReport(
        lint_findings=[
            LintFinding(
                name_id=f"candidate_{index}",
                finding_type="convention_violation",
                detail=f"finding {index}",
            )
            for index in range(25)
        ]
    )

    summaries = _extract_audit_findings(
        report,
        {f"candidate_{index}" for index in range(25)},
    )

    assert len(summaries) == 20
    assert summaries[0].endswith("candidate_0: finding 0")
    assert summaries[-1].endswith("candidate_19: finding 19")
