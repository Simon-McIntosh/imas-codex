"""Adversarial export controls and sampling-aware signal summaries."""

import re
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.ticker import MaxNLocator

from iter_codex.plotting import save_figure, signals as signal_plot, style


def test_public_guard_and_packaged_style_are_importable():
    resource = files("iter_codex.plotting").joinpath("data-ink.mplstyle")
    assert resource.is_file()
    assert Path(str(resource)).resolve() == style.BASE_STYLE.resolve()
    assert save_figure is style.save_figure
    with plt.style.context(style.BASE_STYLE):
        assert plt.rcParams["axes.grid"] is False
        assert plt.rcParams["axes.spines.right"] is False
    print(f"MEASUREMENT_MODULE={signal_plot.__file__}")
    print(f"MEASUREMENT_STYLE_MODULE={style.__file__}")
    print(f"MEASUREMENT_CWD={Path.cwd().resolve()}")


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


def test_loader_pulse_seconds_time_base(tmp_path):
    loaded = signal(time_base="pulse_seconds")
    assert signal_plot.statistics(loaded)["time_span_s"] == 10
    receipt = signal_plot.plot([loaded], tmp_path / "pulse.svg", labels=["measurement"])
    assert receipt["time_base"] == "pulse_seconds"


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


def _declared_centimetres(content: str) -> tuple[float, float]:
    match = re.search(r'width="([\d.]+)cm" height="([\d.]+)cm"', content)
    assert match is not None, content[:300]
    return float(match.group(1)), float(match.group(2))


@pytest.mark.parametrize(
    "canvas",
    [style.CANVAS, (28.0, 7.2), (7.0, 9.0)],
    ids=["default", "wide", "narrow"],
)
def test_exported_aspect_ratio_matches_the_drawn_canvas(tmp_path, canvas):
    figure, axis = plt.subplots(figsize=canvas)
    axis.plot([0.0, 1.0], [0.0, 1.0], label="trace")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("signal [A]")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)
    path = tmp_path / f"aspect-{canvas[0]:g}x{canvas[1]:g}.svg"
    try:
        save_figure(figure, path)
    finally:
        plt.close(figure)
    width, height = _declared_centimetres(path.read_text())
    drawn = canvas[0] / canvas[1]
    assert abs(width / height - drawn) <= drawn / 1000


@pytest.mark.parametrize("count", [4, 6])
def test_legend_stays_on_one_row_up_to_six_traces(tmp_path, count):
    traces = [signal(values=(float(index), 2.0, 3.0)) for index in range(count)]
    labels = [f"trace {index}" for index in range(count)]
    receipt = signal_plot.plot(traces, tmp_path / f"legend-{count}.svg", labels=labels)
    assert receipt["legend_rows"] == 1


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


def test_several_signals_share_one_legend_and_axis(tmp_path):
    path = tmp_path / "several.svg"
    receipt = signal_plot.plot(
        [signal(), signal(values=(3.0, 2.0, 1.0)), signal(values=(4.0, 5.0, 6.0))],
        path,
        labels=["measurement", "comparison", "setpoint"],
    )
    assert len(receipt["summaries"]) == 3
    assert path.exists()


def test_subject_envelope_stands_above_neutral_traces(tmp_path, monkeypatch):
    observed = {}

    def inspect(figure, path):
        observed.update({line.get_label(): line for line in figure.axes[0].lines})

    monkeypatch.setattr(signal_plot, "save_figure", inspect)
    signal_plot.plot(
        [
            signal(
                time=(0.0, 2.0),
                values=((4.0, 1.0, 7.0), (10.0, 8.0, 15.0)),
                mode="envelope",
            ),
            signal(time=(0.0, 2.0), values=(2.0, 3.0), semantics="zero_order_hold"),
        ],
        tmp_path / "highlight.svg",
        labels=["envelope", "events"],
    )
    subject = observed.pop("envelope bin mean")
    assert subject.get_color() == style.SUBJECT
    assert style.SUBJECT != style.REFERENCE
    for line in observed.values():
        assert line.get_color() == style.REFERENCE
        assert line.get_zorder() < subject.get_zorder()


def test_malformed_envelopes_are_not_summarised():
    with pytest.raises(ValueError, match="envelope"):
        signal_plot.statistics(signal(mode="envelope"))
    with pytest.raises(ValueError, match="between"):
        signal_plot.statistics(
            signal(time=(0.0,), values=((10.0, 1.0, 2.0),), mode="envelope")
        )
