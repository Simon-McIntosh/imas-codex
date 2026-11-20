"""Group enrichment composes member text only from delivered group descriptions."""

import pytest

from imas_codex.discovery.signals.enrichment import (
    GroupDescription,
    GroupDescriptionBatch,
    GroupStatement,
)
from imas_codex.discovery.signals.group_enrichment import (
    AWAITING_CONTEXT,
    DESCRIBED,
    FAILED,
    GroupOutcome,
    GroupPlan,
    GroupRequest,
    PilotMember,
    compose_member_description,
    compose_plan,
    context_lines,
    describe_group,
    describe_groups,
    summarise,
)


def member(**changes):
    values = {
        "id": "demo:signal:1",
        "accessor": "DEMO-UNIT:QAZ0001-TEMP",
        "status": "discovered",
        "context_state": "derived",
        "component": "demo:plant_component:DEMO-UNIT:QAZ0001",
        "control_node": "demo:control_node:DEMO-UNIT",
        "field_code": "TEMP",
        "quantity": "temperature",
        "quantity_state": "decided",
        "unit": "kelvin",
        "unit_state": "decided",
        "role": "measurement",
        "role_state": "decided",
    }
    return PilotMember(**{**values, **changes})


def plan(members=None):
    component = "demo:plant_component:DEMO-UNIT:QAZ0001"
    pattern = "demo:archive_field_pattern:invented"
    members = members or [member()]
    return GroupPlan(
        members=members,
        components={
            component: GroupRequest(
                group_id=component, kind="component", context=("a",)
            )
        },
        patterns={
            pattern: GroupRequest(
                group_id=pattern, kind="field_pattern", context=("b",)
            )
        },
        component_pattern={component: pattern},
        component_members={component: members},
        pattern_members={pattern: members},
    )


COMPONENT = "demo:plant_component:DEMO-UNIT:QAZ0001"
PATTERN = "demo:archive_field_pattern:invented"


def test_composition_states_component_field_and_decided_values():
    text = compose_member_description(
        "Cooling circuit valve assembly.",
        "Shared control fields.",
        "TEMP: cooling temperature.",
        member(),
    )
    assert text == (
        "Cooling circuit valve assembly. TEMP: cooling temperature. "
        "Quantity: temperature. Unit: kelvin. Role: measurement."
    )


@pytest.mark.parametrize(
    ("state", "value"),
    [("undecided", None), ("conflict", "temperature"), ("decided", "unlisted")],
)
def test_composition_omits_an_axis_without_a_decided_vocabulary_value(state, value):
    text = compose_member_description(
        "Component.", "Pattern.", "", member(quantity=value, quantity_state=state)
    )
    assert "Quantity:" not in text
    assert "Unit: kelvin. Role: measurement." in text


def test_component_failure_fails_its_members_without_a_description():
    composed = compose_plan(
        plan(),
        {
            COMPONENT: GroupOutcome(
                error="No statement was supported by a supplied context item"
            )
        },
        {PATTERN: GroupOutcome(text="Shared fields.", fields={"TEMP": "temperature"})},
    )
    assert composed.component_states[COMPONENT] == FAILED
    assert composed.described == []
    assert [item["id"] for item in composed.failed] == ["demo:signal:1"]
    assert composed.failed[0]["reason"] == "Component description was not delivered"
    assert all("description" not in item for item in composed.failed)


def test_pattern_failure_fails_its_members_because_composition_needs_both():
    composed = compose_plan(
        plan(),
        {COMPONENT: GroupOutcome(text="Component.")},
        {PATTERN: GroupOutcome(error="call failed")},
    )
    assert composed.pattern_states[PATTERN] == FAILED
    assert composed.described == []
    assert composed.failed[0]["reason"] == (
        "Field-pattern description was not delivered"
    )


def test_signal_without_linked_context_is_awaiting_and_never_described():
    absent = member(
        id="demo:signal:2",
        context_state="absent",
        component=None,
        quantity=None,
        quantity_state="undecided",
        unit=None,
        unit_state="undecided",
        role=None,
        role_state="undecided",
    )
    composed = compose_plan(
        plan([member(), absent]),
        {COMPONENT: GroupOutcome(text="Component.")},
        {PATTERN: GroupOutcome(text="Pattern.", fields={"TEMP": "temperature"})},
    )
    assert [item["id"] for item in composed.awaiting] == ["demo:signal:2"]
    assert composed.awaiting[0]["state"] == AWAITING_CONTEXT
    assert "description" not in composed.awaiting[0]


def test_component_without_a_field_pattern_cannot_compose():
    cohort = plan()
    cohort.component_pattern = {}
    composed = compose_plan(
        cohort,
        {COMPONENT: GroupOutcome(text="Component.")},
        {PATTERN: GroupOutcome(text="Pattern.")},
    )
    assert composed.failed[0]["reason"] == "Component has no field-pattern group"


def test_context_lines_attribute_a_row_to_the_field_it_names():
    lines = context_lines(
        function_category="Temperature transmitter",
        function_category_code="MP",
        plant_structure="Magnet cold test bench",
        rows=[
            {"row_field": "TEMP", "description": "Outlet temperature"},
            {"row_field": "PRES", "description": "Inlet pressure"},
            {"row_field": None, "signal_field": "FLOW", "description": "Mass flow"},
        ],
        document_titles=[("Invented specification", "functional_specification")],
    )
    assert lines == (
        "function category meaning: Temperature transmitter (MP)",
        "plant structure: Magnet cold test bench",
        "source field TEMP: Outlet temperature",
        "source field PRES: Inlet pressure",
        "source field FLOW: Mass flow",
        "source document: Invented specification (functional_specification)",
    )


class RecordingCall:
    def __init__(self, batch, cost=0.0, tokens=0, error=None):
        self.batch = batch
        self.cost = cost
        self.tokens = tokens
        self.error = error
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.batch, self.cost, self.tokens


@pytest.mark.asyncio
async def test_only_cited_statements_survive():
    batch = GroupDescriptionBatch(
        results=[
            GroupDescription(
                group_id=COMPONENT,
                statements=[
                    GroupStatement(
                        text="Cooling circuit valve assembly.",
                        context_items=["context-1"],
                    ),
                    GroupStatement(
                        text="Pressurised helium supply.", context_items=["missing"]
                    ),
                    GroupStatement(text="Uncited supply."),
                ],
            )
        ]
    )
    call = RecordingCall(batch, cost=0.25, tokens=7)
    outcome = await describe_group(
        GroupRequest(group_id=COMPONENT, kind="component", context=("Meaning.",)),
        model="local/demo",
        call=call,
    )
    assert outcome.text == "Cooling circuit valve assembly."
    assert outcome.drops == 2
    assert outcome.cost == pytest.approx(0.25)
    assert outcome.tokens == 7
    assert "[context-1] Meaning." in call.calls[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_all_claims_dropped_delivers_nothing_and_reports_failure():
    batch = GroupDescriptionBatch(
        results=[
            GroupDescription(
                group_id=COMPONENT,
                statements=[
                    GroupStatement(text="Invented.", context_items=["absent"]),
                ],
            )
        ]
    )
    outcome = await describe_group(
        GroupRequest(group_id=COMPONENT, kind="component", context=("Meaning.",)),
        model="local/demo",
        call=RecordingCall(batch),
    )
    assert outcome.text == ""
    assert not outcome.delivered
    assert outcome.error == "No statement was supported by a supplied context item"


@pytest.mark.asyncio
async def test_a_call_error_fails_only_its_group():
    good = GroupDescriptionBatch(
        results=[
            GroupDescription(
                group_id=PATTERN,
                statements=[
                    GroupStatement(text="Shared fields.", context_items=["context-1"])
                ],
                fields={"TEMP": "temperature"},
            )
        ]
    )
    requests = [
        GroupRequest(group_id=COMPONENT, kind="component", context=("Meaning.",)),
        GroupRequest(group_id=PATTERN, kind="field_pattern", context=("Fields.",)),
    ]

    class PerGroupCall(RecordingCall):
        async def __call__(self, **kwargs):
            if "Meaning." in kwargs["messages"][1]["content"]:
                raise TimeoutError("local endpoint did not answer")
            return self.batch, 0.0, 0

    outcomes = await describe_groups(
        requests, model="local/demo", call=PerGroupCall(good), max_workers=2
    )
    assert outcomes[COMPONENT].error == "TimeoutError: local endpoint did not answer"
    assert outcomes[PATTERN].text == "Shared fields."
    assert outcomes[PATTERN].fields == {"TEMP": "temperature"}


def test_summary_counts_sum_to_the_pilot_total_and_flag_no_invalid_value():
    absent = member(
        id="demo:signal:2",
        context_state="absent",
        component=None,
    )
    cohort = plan([member(), absent])
    composed = compose_plan(
        cohort,
        {COMPONENT: GroupOutcome(text="Component.", cost=0.5, tokens=10)},
        {
            PATTERN: GroupOutcome(
                text="Pattern.",
                fields={"TEMP": "temperature"},
                cost=0.5,
                tokens=10,
            )
        },
    )
    summary = summarise(cohort, composed)
    assert summary["signals_total"] == 2
    assert (
        summary["signals_described"]
        + summary["signals_awaiting_context"]
        + summary["signals_failed"]
        == 2
    )
    assert summary["components_total"] == 1
    assert summary["components_described"] == 1
    assert summary["patterns_total"] == 1
    assert summary["patterns_described"] == 1
    assert summary["calls"] == 2
    assert summary["cost_usd"] == pytest.approx(1.0)
    assert summary["vocabulary"]["quantity"]["temperature"] == 1
    assert summary["vocabulary"]["unit"]["kelvin"] == 1
    assert summary["vocabulary"]["role"]["measurement"] == 1
    assert summary["invalid_vocabulary_values"] == 0


def test_summary_counts_a_state_that_holds_a_non_member_value():
    off_vocabulary = member(quantity="opacity", quantity_state="decided")
    cohort = plan([off_vocabulary])
    composed = compose_plan(
        cohort,
        {COMPONENT: GroupOutcome(text="Component.")},
        {PATTERN: GroupOutcome(text="Pattern.")},
    )
    summary = summarise(cohort, composed)
    assert summary["invalid_vocabulary_values"] == 1


class TwoCallReview:
    """Return the draft first and the review second, recording both requests."""

    def __init__(self, draft, reviewed, draft_cost=0.1, review_cost=0.2):
        self.draft, self.reviewed = draft, reviewed
        self.draft_cost, self.review_cost = draft_cost, review_cost
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return self.draft, self.draft_cost, 5
        return self.reviewed, self.review_cost, 7


@pytest.mark.asyncio
async def test_review_pass_removes_a_statement_its_check_refuses():
    from imas_codex.discovery.signals.enrichment import (
        DescriptionReview,
        ReviewedDescription,
        ReviewedStatement,
    )

    draft = GroupDescriptionBatch(
        results=[
            GroupDescription(
                group_id=COMPONENT,
                statements=[
                    GroupStatement(
                        text="Cooling circuit valve assembly.", context_items=["context-1"]
                    ),
                    GroupStatement(
                        text="He supply at 4 K.", context_items=["context-1"]
                    ),
                ],
            )
        ]
    )
    reviewed = DescriptionReview(
        results=[
            ReviewedDescription(
                signal_index=1,
                statements=[
                    ReviewedStatement(
                        draft_index=1,
                        text="Cooling circuit valve assembly.",
                        context_items=["context-1"],
                    )
                ],
            )
        ]
    )
    call = TwoCallReview(draft, reviewed)
    outcome = await describe_group(
        GroupRequest(group_id=COMPONENT, kind="component", context=("Meaning.",)),
        model="local/demo",
        call=call,
        review=True,
        review_thinking=True,
    )
    assert outcome.text == "Cooling circuit valve assembly."
    assert outcome.cost == pytest.approx(0.3)
    assert outcome.tokens == 12
    assert call.calls[1]["reasoning_effort"] == "high"
    assert call.calls[1]["temperature"] == 0


@pytest.mark.asyncio
async def test_review_failure_delivers_nothing_and_keeps_both_costs():
    draft = GroupDescriptionBatch(
        results=[
            GroupDescription(
                group_id=COMPONENT,
                statements=[
                    GroupStatement(text="Component.", context_items=["context-1"])
                ],
            )
        ]
    )

    class FailingReview(TwoCallReview):
        async def __call__(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return self.draft, self.draft_cost, 5
            raise ValueError("review response unparseable")

    outcome = await describe_group(
        GroupRequest(group_id=COMPONENT, kind="component", context=("Meaning.",)),
        model="local/demo",
        call=FailingReview(draft, None),
        review=True,
    )
    assert not outcome.delivered
    assert outcome.text == ""
    assert outcome.cost == pytest.approx(0.1)
    assert outcome.error is not None and outcome.error.startswith("Review failed:")

