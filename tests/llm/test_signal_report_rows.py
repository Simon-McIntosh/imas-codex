"""Signal search rows retain members while sharing family access instructions."""

from imas_codex.llm.search_formatters import format_signals_report


def _signal(
    name: str,
    category: str,
    *,
    checked: bool = False,
    role: str | None = None,
) -> dict:
    return {
        "id": f"facility:{category.lower()}/{name.lower().replace(' ', '_')}",
        "name": name,
        "description": f"Measured {name}",
        "data_source_name": "edas",
        "data_source_path": f"{category}/{name}",
        "unit": "V",
        "data_class": "time_series",
        "checked": checked,
        "example_shot": 101173 if checked else None,
        "signal_role": role,
        "accessor": f"read('{category}', '{name}')",
        "access_methods": [
            {
                "access_type": "local",
                "method_type": "edas",
                "imports_template": "import edas",
                "access_template": "data = db.{accessor}",
            }
        ],
    }


def test_family_members_share_one_row_and_one_access_template() -> None:
    signals = [
        _signal("EF6 Booster BPS Voltage", "MMSYS"),
        _signal("EF1 Booster BPS Voltage", "MMSYS", checked=True),
        _signal("CS4 Voltage", "DCSV"),
        _signal("EF2 Booster BPS Voltage", "MMSYS"),
        _signal("EF5 Booster BPS Voltage", "MMSYS"),
    ]
    report = format_signals_report(signals, [], {s["id"]: 0.02 for s in signals})

    assert report.count("### ") == 2
    assert "EF{1,2,5,6} Booster BPS Voltage (4 members)" in report
    assert "Rank: 1" in report
    assert "Unit: V" in report
    assert "Data class: time_series" in report
    assert "Check status: 1 checked, 3 unchecked" in report
    assert report.count("data = db.{accessor}") == 1
    assert "score: 0.02" not in report


def test_equal_name_templates_in_different_categories_remain_separate() -> None:
    signals = [
        _signal("EF1 Booster BPS Voltage", "MMSYS"),
        _signal("EF2 Booster BPS Voltage", "CONTROL"),
    ]

    report = format_signals_report(signals, [], {})

    assert report.count("### ") == 2
    assert "Category: MMSYS" in report
    assert "Category: CONTROL" in report


def test_single_member_row_prints_its_resolved_access_call() -> None:
    signal = _signal("CS4 Voltage", "DCSV", checked=True)

    report = format_signals_report([signal], [], {})

    assert "Data access (local/edas):" in report
    assert "data = db.read('DCSV', 'CS4 Voltage')" in report
    assert "{accessor}" not in report


def test_report_keeps_fused_rank_after_a_rerank() -> None:
    signal = _signal("EF2 Booster BPS Voltage", "MMSYS")

    report = format_signals_report([signal], [], {}, ranks={signal["id"]: 7})

    assert "Rank: 7" in report


def test_row_shows_the_stored_role_when_present() -> None:
    measured = _signal("EF2 Booster BPS Voltage", "MMSYS", role="measured")
    reference = _signal("CS4 Voltage", "DCSV", role="reference_or_setpoint")

    report = format_signals_report([measured, reference], [], {})

    measured_row = next(
        row for row in report.split("### ") if "EF2 Booster BPS Voltage" in row
    )
    reference_row = next(row for row in report.split("### ") if "CS4 Voltage" in row)
    assert "Role: measured" in measured_row
    assert "Role: reference_or_setpoint" in reference_row


def test_row_omits_the_role_when_the_signal_carries_none() -> None:
    signal = _signal("EF2 Booster BPS Voltage", "MMSYS")

    report = format_signals_report([signal], [], {})

    assert "Role:" not in report


def test_row_lists_every_distinct_role_in_a_family() -> None:
    signals = [
        _signal("EF6 Booster BPS Voltage", "MMSYS", role="measured"),
        _signal("EF5 Booster BPS Voltage", "MMSYS", role="reference_or_setpoint"),
    ]

    report = format_signals_report(signals, [], {})

    assert "Role: measured, reference_or_setpoint" in report
