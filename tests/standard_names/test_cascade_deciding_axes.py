"""The per-axis deciding switch: which axes Jev decides alone."""

import pytest

from imas_codex import settings
from imas_codex.standard_names.review import cascade


def _set_deciding(monkeypatch, value):
    monkeypatch.setattr(
        settings, "_get_section", lambda _section: {"cascade-deciding": value}
    )


@pytest.mark.parametrize(
    "value,name_decides,docs_decides",
    [
        (False, False, False),
        (True, True, True),
        ([], False, False),
        (["name"], True, False),
        (["docs"], False, True),
        (["docs", "name"], True, True),
    ],
)
def test_the_accepted_forms_switch_each_axis(
    monkeypatch, value, name_decides, docs_decides
):
    _set_deciding(monkeypatch, value)
    assert cascade.deciding_enabled("name") is name_decides
    assert cascade.deciding_enabled("docs") is docs_decides


@pytest.mark.parametrize("value", ["name", "names", 1, ["names"], ["name", 1], {}])
def test_an_unrecognised_form_is_refused(monkeypatch, value):
    _set_deciding(monkeypatch, value)
    with pytest.raises(ValueError, match="cascade-deciding"):
        cascade.deciding_enabled("name")


def test_an_absent_setting_shadows_both_axes(monkeypatch):
    monkeypatch.setattr(settings, "_get_section", lambda _section: {})
    assert cascade.deciding_enabled("name") is False
    assert cascade.deciding_enabled("docs") is False


def test_the_shipped_configuration_makes_both_axes_deciding():
    """The configuration this checkout ships has Jev decide both axes.

    The paid panel runs at the configured audit rate over decided items and
    decides nothing, so its verdicts stay a drift watch rather than a second
    decider.
    """
    from imas_codex.standard_names.review import panel_audit

    assert cascade.deciding_enabled("name") is True
    assert cascade.deciding_enabled("docs") is True
    assert panel_audit.panel_audit_rate() == 0.05


class _Lease:
    def charge_event(self, *_args):
        pass

    def release_unused(self):
        pass


class _Manager:
    def reserve(self, *_args, **_kwargs):
        return _Lease()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "axis,jev_decides",
    [("name", True), ("docs", False)],
)
async def test_names_deciding_leaves_documentation_with_the_incumbent(
    monkeypatch, axis, jev_decides
):
    """With ``cascade-deciding = ["name"]`` only the name axis is Jev's.

    The chat reviewer's verdict is what a docs item returns, byte for byte, while
    the name axis returns Jev's own decision. That reviewer runs a single seat,
    so the fallback receives the head of the configured chain.
    """
    from imas_codex.standard_names.review.checks import (
        DOCS_DIMENSIONS,
        NAME_DIMENSIONS,
    )

    reviewer_result = {"winning_score": 0.62, "resolution_method": "quorum_consensus"}
    reviewer_calls = []

    async def judgment(_item, _model):
        dimensions = NAME_DIMENSIONS if axis == "name" else DOCS_DIMENSIONS
        return (
            cascade.CascadeRoute(
                "accept",
                0.9,
                (),
                dict.fromkeys(dimensions, 18.0),
                selected_score=0.9,
                decision_score=0.92,
            ),
            {},
            0.0,
        )

    async def chat_reviewer(*, models):
        reviewer_calls.append(list(models))
        return reviewer_result

    async def store(*_args, **_kwargs):
        pass

    monkeypatch.setattr(settings, "get_model", lambda _seat: "judge-model")
    monkeypatch.setattr(
        "imas_codex.discovery.base.judgment.decisions_key_present", lambda: True
    )
    monkeypatch.setattr(
        cascade, "_name_judgment" if axis == "name" else "_docs_judgment", judgment
    )
    monkeypatch.setattr(cascade, "_store_route", store)
    _set_deciding(monkeypatch, ["name"])

    result = await cascade.adjudicate_review(
        axis,
        {"id": "temperature"},
        ["primary", "secondary"],
        chat_reviewer,
        mgr=_Manager(),
    )

    if jev_decides:
        assert result["resolution_method"] == "jev_cascade_accept"
        assert result["winning_score"] == 0.92
        assert reviewer_calls == [], "a decided axis never calls the chat reviewer"
    else:
        assert result is reviewer_result
        assert reviewer_calls == [["primary"]]
