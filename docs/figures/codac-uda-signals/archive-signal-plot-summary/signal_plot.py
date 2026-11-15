"""Plot loaded scalar signals and report explicitly unweighted summaries.

Import this file with importlib.util.spec_from_file_location from figure scripts.
Inputs use the loader's time/values/metadata interface, without requiring a graph
connection. Archive selectors and queries are never copied to figure labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Annotation
from matplotlib.ticker import MaxNLocator

from scripts.codac_uda_prototype.loader import _save_figure

SUBJECT = "#555555"
REFERENCE = "#929292"
LINE_STYLES = ("-", "--", "-.", ":")
CANVAS = (14, 7.2)
BASE_STYLE = Path(__file__).with_name("data-ink.mplstyle")
STYLE = {"axes.formatter.useoffset": False, "svg.fonttype": "none"}


class LoadedSignal(Protocol):
    """The scalar array and metadata interface returned by archive loaders."""

    time: np.ndarray
    values: np.ndarray
    metadata: dict[str, Any]


def _arrays(signal: LoadedSignal):
    times, values = np.asarray(signal.time), np.asarray(signal.values)
    if times.ndim != 1 or values.ndim == 0 or len(times) != len(values):
        raise ValueError("time and value dimensions disagree")
    base = signal.metadata.get("time_base")
    if base not in {"epoch_ns", "pulse_seconds", "relative_s"}:
        raise ValueError("time_base must be epoch_ns, pulse_seconds or relative_s")
    if times.dtype.kind not in ("iu" if base == "epoch_ns" else "iuf"):
        raise ValueError("timestamp dtype disagrees with time base")
    if not np.all(np.isfinite(times)) or np.any(times[1:] <= times[:-1]):
        raise ValueError("timestamps must be finite and strictly increasing")
    envelope = signal.metadata.get("mode") == "envelope"
    if values.ndim != (2 if envelope else 1) or (envelope and values.shape[1] != 3):
        raise ValueError("select a scalar field, or mean/minimum/maximum envelope")
    return times, values, envelope, 1e9 if base == "epoch_ns" else 1.0


def _elapsed(times, origin, factor):
    # Subtract integer nanoseconds before converting to seconds.
    return np.array(
        [
            float((int(value) if times.dtype.kind in "iu" else value) - origin) / factor
            for value in times
        ]
    )


def statistics(signal: LoadedSignal) -> dict:
    """Describe returned points, never infer raw cadence or duration weighting.

    Envelope extrema describe the bounds; its mean averages returned bin means.
    Held boundary points count as returned points, and are identified separately.
    The median interval describes the returned timestamps, not a sampling rate.
    """
    times, values, envelope, factor = _arrays(signal)
    metadata = signal.metadata
    held = metadata.get("semantics") in {"zero_order_hold", "archived_on_change"}
    kind = (
        "unweighted_envelope_bin_means"
        if envelope
        else "unweighted_binned_values"
        if metadata.get("mode") == "binned"
        else "unweighted_archived_on_change_values"
        if held
        else "unweighted_returned_samples"
    )
    result = {
        "count": len(times),
        "time_span_s": float(times[-1] - times[0]) / factor if len(times) else 0.0,
        "sample_interval_s": (
            float(np.median(np.diff(times))) / factor if len(times) > 1 else None
        ),
        "interval_kind": "median_returned_timestamp_spacing",
        "statistics_kind": kind,
        "sampling_semantics": metadata.get("semantics", "unknown"),
        "decimated": metadata.get("decimated"),
        "boundary_values_synthesized": metadata.get(
            "boundary_values_synthesized", False
        ),
        "unit": metadata.get("unit", ""),
        "min": None,
        "max": None,
        "mean": None,
    }
    if values.dtype.kind not in "biuf":
        result["statistics_kind"] = "nonnumeric_returned_values"
        return result
    if not np.all(np.isfinite(values)):
        raise ValueError("summary requires finite numeric values")
    if len(times):
        lower, upper, means = (
            (values[:, 1], values[:, 2], values[:, 0])
            if envelope
            else (values, values, values)
        )
        if envelope and (np.any(lower > means) or np.any(means > upper)):
            raise ValueError("envelope mean must lie between minimum and maximum")
        result.update(
            min=float(lower.min()), max=float(upper.max()), mean=float(means.mean())
        )
    return result


def save_figure(figure, path: str | Path) -> None:
    """Use the existing text refusal guard for every SVG or raster export.

    Annotations are direct labels: their boxes must also avoid plotted lines.
    The text check runs at the raster export resolution before any file is opened.
    """
    path = Path(path)
    if path.suffix.lower() not in {".svg", ".png"}:
        raise ValueError("figure export must be SVG or PNG")
    figure.set_dpi(200)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        for label in axis.texts:
            if not label.get_visible() or not label.get_text().strip():
                continue
            box = label.get_window_extent(renderer)
            for line in axis.lines:
                if line.get_visible() and line.get_transform().transform_path(
                    line.get_path()
                ).intersects_bbox(box, filled=False):
                    raise ValueError(
                        f"direct label overlaps data: {label.get_text()} / {line.get_label()}"
                    )
            if isinstance(label, Annotation) and label.arrow_patch is not None:
                raise ValueError("direct labels must sit beside data without arrows")
    _save_figure(figure, path)
    if path.suffix.lower() == ".svg":
        # Preserve the slide width while allowing a taller explicitly sized canvas.
        content = path.read_text()
        height = 16 * figure.get_figheight() / CANVAS[1]
        content = content.replace('height="16cm"', f'height="{height:g}cm"', 1)
        path.write_text(
            "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
        )


def _scale(magnitude: float, unit: str):
    if unit not in {"A", "V", "W", "J", "Hz", "Pa", "H", "Ω", "s"} or not magnitude:
        return 1.0, unit or "unspecified unit"
    for exponent, prefix in (
        (9, "G"),
        (6, "M"),
        (3, "k"),
        (0, ""),
        (-3, "m"),
        (-6, "µ"),
        (-9, "n"),
    ):
        if magnitude >= 10.0 ** (exponent - 1):
            return 10.0**exponent, prefix + unit
    return 1.0, unit


def plot(signals: Sequence[LoadedSignal], path: str | Path, *, labels: Sequence[str]):
    """Overlay numeric signals with one shared, unre-based elapsed-time axis.

    Labels must be caller-supplied public aliases for internal data. The time origin
    is the earliest returned timestamp across inputs, retained in the receipt.
    Different physical units require separate figures rather than a misleading axis.
    """
    if not signals or len(labels) != len(signals):
        raise ValueError("provide signals and one alias per signal")
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError("signal aliases must be distinct and nonempty")
    summaries = [statistics(signal) for signal in signals]
    arrays = [_arrays(signal) for signal in signals]
    if any(
        not len(times) or values.dtype.kind not in "biuf"
        for times, values, _, _ in arrays
    ):
        raise ValueError("plot requires nonempty numeric signals")
    units = {signal.metadata.get("unit", "") for signal in signals}
    bases = {signal.metadata["time_base"] for signal in signals}
    if len(units) != 1 or len(bases) != 1:
        raise ValueError("overlay requires a common unit and time base")
    origin = min(int(a[0][0]) if a[3] == 1e9 else float(a[0][0]) for a in arrays)
    factor = arrays[0][3]
    elapsed = [_elapsed(a[0], origin, factor) for a in arrays]
    divisor, unit = _scale(max(float(np.abs(a[1]).max()) for a in arrays), units.pop())
    time_divisor, time_unit = _scale(max(float(t[-1]) for t in elapsed), "s")
    with plt.style.context(BASE_STYLE), plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=CANVAS, dpi=100)
        for index, (signal, (_, values, envelope, _), times, alias) in enumerate(
            zip(signals, arrays, elapsed, labels, strict=True)
        ):
            x, y = times / time_divisor, values / divisor
            style = (
                LINE_STYLES[index]
                if index < len(LINE_STYLES)
                else (0, (index + 2, 2, 1, 2))
            )
            colour = SUBJECT if index == 0 else REFERENCE
            if envelope:
                axis.plot(
                    x,
                    y[:, 1],
                    color=REFERENCE,
                    lw=2.6,
                    zorder=1,
                    label=f"{alias} minimum",
                )
                axis.plot(
                    x,
                    y[:, 2],
                    color=REFERENCE,
                    lw=2.6,
                    ls=":",
                    zorder=1,
                    label=f"{alias} maximum",
                )
                axis.plot(
                    x,
                    y[:, 0],
                    color=colour,
                    lw=3,
                    ls=style,
                    zorder=3,
                    label=f"{alias} bin mean",
                )
            else:
                held = signal.metadata.get("semantics") in {
                    "zero_order_hold",
                    "archived_on_change",
                }
                axis.plot(
                    x,
                    y,
                    color=colour,
                    lw=3 if index == 0 else 2.6,
                    ls=style,
                    drawstyle="steps-post" if held else "default",
                    zorder=3 if index == 0 else 2,
                    label=alias,
                )
        figure.legend(loc="upper center", ncol=min(3, len(axis.lines)))
        axis.set_xlabel(f"elapsed time [{time_unit}]")
        axis.set_ylabel(f"signal [{unit}]")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
        axis.ticklabel_format(style="plain", axis="both", useOffset=False)
        legend_rows = (len(axis.lines) + 2) // 3
        figure.subplots_adjust(
            left=0.13,
            right=0.97,
            bottom=0.14,
            top=min(0.86, 0.94 - 0.055 * legend_rows),
        )
        try:
            save_figure(figure, path)
        finally:
            plt.close(figure)
    return {
        "time_origin": origin,
        "time_base": signals[0].metadata["time_base"],
        "summaries": summaries,
        "path": str(Path(path).resolve()),
    }
