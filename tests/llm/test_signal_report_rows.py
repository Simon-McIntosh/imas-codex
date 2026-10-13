"""Signal search rows retain members while sharing family access instructions."""

from imas_codex.llm.search_formatters import format_signals_report


def _signal(name: str, category: str, *, checked: bool = False) -> dict:
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


def test_report_keeps_fused_rank_after_a_rerank() -> None:
    signal = _signal("EF2 Booster BPS Voltage", "MMSYS")

    report = format_signals_report([signal], [], {}, ranks={signal["id"]: 7})

    assert "Rank: 7" in report
