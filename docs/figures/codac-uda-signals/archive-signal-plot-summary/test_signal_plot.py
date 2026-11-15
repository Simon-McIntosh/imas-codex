"""Adversarial export controls and sampling-aware signal summaries."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import signal_plot


def signal(time=(0.0, 1.0, 10.0), values=(0.0, 9.0, 3.0), **metadata):
    return SimpleNamespace(
        time=np.array(time),
        values=np.array(values),
        metadata={"time_base": "relative_s", "mode": "raw", "unit": "A", **metadata},
    )


def test_change_events_are_not_time_weighted_or_raw_samples():
    result = signal_plot.statistics(
        signal(semantics="zero_order_hold", boundary_values_synthesized=True)
    )
    assert result["statistics_kind"] == "unweighted_archived_on_change_values"
    assert result["count"] == 3
    assert result["time_span_s"] == 10
    assert result["sample_interval_s"] == 5
    assert (result["min"], result["max"], result["mean"]) == (0, 9, 4)
    assert result["mean"] != 8.1
    assert result["boundary_values_synthesized"] is True


@pytest.mark.parametrize(
    "semantics", ["archived_samples", "zero_order_hold", "unknown"]
)
def test_envelope_kind_precedes_sampling_semantics(semantics):
    result = signal_plot.statistics(
        signal(
            time=(0.0, 2.0),
            values=((4.0, 1.0, 7.0), (10.0, 8.0, 15.0)),
            mode="envelope",
            semantics=semantics,
        )
    )
    assert result["statistics_kind"] == "unweighted_envelope_bin_means"
    assert (result["min"], result["max"], result["mean"]) == (1, 15, 7)
    assert result["count"] == 2


def test_unknown_decimation_is_preserved():
    result = signal_plot.statistics(signal(decimated=None))
    assert result["statistics_kind"] == "unweighted_returned_samples"
    assert result["decimated"] is None


def test_empty_singleton_and_nonnumeric_summaries():
    empty = signal_plot.statistics(signal(time=(), values=()))
    assert empty["count"] == 0 and empty["mean"] is None
    singleton = signal_plot.statistics(signal(time=(1.0,), values=(3.0,)))
    assert singleton["sample_interval_s"] is None and singleton["mean"] == 3
    strings = signal_plot.statistics(signal(values=("on", "off", "on")))
    assert strings["statistics_kind"] == "nonnumeric_returned_values"
    assert strings["mean"] is None


@pytest.mark.parametrize("times", [[2.0, 1.0], [1.0, 1.0], [0.0, float("nan")]])
def test_invalid_timestamps_are_refused(times):
    with pytest.raises(ValueError, match="timestamps"):
        signal_plot.statistics(signal(time=times, values=(1.0, 2.0)))


def test_integer_epoch_nanoseconds_remain_distinct():
    times = np.array(
        [1_800_000_000_000_000_000, 1_800_000_000_000_000_001], dtype=np.uint64
    )
    result = signal_plot.statistics(
        signal(time=times, values=(1.0, 2.0), time_base="epoch_ns")
    )
    assert result["time_span_s"] == 1e-9
    np.testing.assert_equal(signal_plot._elapsed(times, int(times[0]), 1e9), [0, 1e-9])


@pytest.mark.parametrize(
    "case, names",
    [("clipped", ["outside canvas"]), ("overlap", ["first label", "second label"])],
)
def test_invalid_text_refuses_before_creating_file(tmp_path, case, names):
    figure = plt.figure(figsize=signal_plot.CANVAS)
    if case == "clipped":
        figure.text(1.1, 0.5, names[0])
    else:
        figure.text(0.5, 0.5, names[0])
        figure.text(0.5, 0.5, names[1])
    path = tmp_path / f"{case}.svg"
    try:
        with pytest.raises(ValueError) as caught:
            signal_plot.save_figure(figure, path)
        for name in names:
            assert name in str(caught.value)
        assert not path.exists()
        print(f"REFUSAL {case}: {caught.value}; file_written={path.exists()}")
    finally:
        plt.close(figure)


def test_direct_label_over_data_is_refused(tmp_path):
    figure, axis = plt.subplots(figsize=signal_plot.CANVAS)
    axis.plot([0, 1], [0, 1], label="trace")
    axis.text(0.5, 0.5, "on trace")
    try:
        with pytest.raises(
            ValueError, match="direct label overlaps data: on trace / trace"
        ):
            signal_plot.save_figure(figure, tmp_path / "data-label.svg")
        assert not (tmp_path / "data-label.svg").exists()
    finally:
        plt.close(figure)


def test_valid_export_passes_and_declares_slide_canvas(tmp_path):
    path = tmp_path / "valid.svg"
    signal_plot.plot([signal()], path, labels=["demonstration"])
    content = path.read_text()
    assert 'width="31.1cm" height="16cm"' in content
    assert "elapsed time [s]" in content
    assert "demonstration" in content


def test_overlay_uses_common_origin_and_held_steps(tmp_path, monkeypatch):
    observed = {}

    def inspect(figure, path):
        lines = figure.axes[0].lines
        observed["time"] = [line.get_xdata().tolist() for line in lines]
        observed["style"] = [line.get_drawstyle() for line in lines]

    monkeypatch.setattr(signal_plot, "save_figure", inspect)
    receipt = signal_plot.plot(
        [
            signal(time=(2.0, 3.0), values=(1.0, 2.0), semantics="zero_order_hold"),
            signal(time=(4.0, 5.0), values=(3.0, 4.0)),
        ],
        tmp_path / "overlay.svg",
        labels=["measurement", "reference"],
    )
    assert receipt["time_origin"] == 2
    assert observed["time"] == [[0, 1], [2, 3]]
    assert observed["style"] == ["steps-post", "default"]


def test_overlay_refuses_incompatible_units(tmp_path):
    with pytest.raises(ValueError, match="common unit"):
        signal_plot.plot(
            [signal(), signal(unit="V")],
            tmp_path / "invalid.svg",
            labels=["current", "voltage"],
        )
    assert not (tmp_path / "invalid.svg").exists()


def test_malformed_envelopes_are_not_summarised():
    with pytest.raises(ValueError, match="envelope"):
        signal_plot.statistics(signal(mode="envelope"))
    with pytest.raises(ValueError, match="between"):
        signal_plot.statistics(
            signal(time=(0.0,), values=((10.0, 1.0, 2.0),), mode="envelope")
        )
