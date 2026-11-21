"""Load a bounded CODAC reply across an interpreter boundary."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iter_codex.plotting import save_figure as _save_figure


@dataclass(frozen=True)
class Signal:
    time: np.ndarray
    values: np.ndarray
    metadata: dict[str, Any]


class Loader:
    """Invoke an argv supplied by trusted deployment configuration.

    The command must run reader.py under the UDA module's interpreter and
    environment. Shell setup belongs in a deployment wrapper, never in requests.
    CODAC_UDA_READER_COMMAND is a JSON argv, with the reader path included.
    """

    def __init__(self, command: list[str] | None = None, *, timeout: float = 20):
        self.command = command or json.loads(os.environ["CODAC_UDA_READER_COMMAND"])
        if (
            not isinstance(self.command, list)
            or not self.command
            or not all(isinstance(item, str) and item for item in self.command)
        ):
            raise ValueError("reader command must be a nonempty JSON argument list")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def request(self, operation: str, **parameters) -> dict:
        process = subprocess.run(
            self.command,
            input=json.dumps({"operation": operation, **parameters}),
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        try:
            receipt = json.loads(process.stdout)
        except ValueError as exc:
            raise RuntimeError(
                f"reader returned invalid JSON (exit {process.returncode}): {process.stderr}"
            ) from exc
        if not receipt.get("ok") or process.returncode:
            raise RuntimeError(
                f"{receipt.get('error_type', 'ReaderError')}: {receipt.get('error', process.stderr)}"
            )
        return receipt["result"]

    def variables(self, pattern: str, **parameters) -> dict:
        return self.request("variables", pattern=pattern, **parameters)

    def pulses(self, pattern: str, **parameters) -> dict:
        return self.request("pulses", pattern=pattern, **parameters)

    def read(self, name: str, **parameters) -> Signal:
        result = self.request("read", name=name, **parameters)
        times = np.asarray(result.pop("time"), dtype=result["time_dtype"])
        values = np.asarray(result.pop("values"), dtype=result["value_dtype"])
        if times.ndim != 1 or values.ndim == 0 or len(times) != len(values):
            raise RuntimeError("reader receipt has inconsistent array dimensions")
        return Signal(times, values, result)


def statistics(signal: Signal) -> dict:
    """Report native sample statistics; envelope means are bin statistics."""
    times, values = signal.time, signal.values
    factor = 1e9 if signal.metadata["time_base"] == "epoch_ns" else 1
    result = {"count": len(times), "time_span_s": 0.0, "sample_interval_s": None}
    if len(times):
        result["time_span_s"] = float(times[-1] - times[0]) / factor
    if len(times) > 1:
        result["sample_interval_s"] = float(np.median(np.diff(times))) / factor
    if values.dtype.kind not in "biuf":
        result["statistics_kind"] = "nonnumeric"
        return result
    envelope = signal.metadata.get("mode") == "envelope"
    lower = values[:, 1] if envelope and len(values) else values
    upper = values[:, 2] if envelope and len(values) else values
    mean_values = values[:, 0] if envelope and len(values) else values
    result.update(
        min=float(np.nanmin(lower)) if values.size else None,
        max=float(np.nanmax(upper)) if values.size else None,
        mean=float(np.nanmean(mean_values)) if values.size else None,
        statistics_kind="unweighted_bin_means" if envelope else "archived_samples",
    )
    return result


def plot(signal: Signal, path: Path) -> dict:
    """Plot elapsed seconds with explicit envelope bounds, and print statistics.

    Live exports must remain below the internal cache directory.
    The figure uses neutral marks and no per-component palette.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    cache = (Path.home() / ".cache" / "iter-codex-codac-uda").resolve()
    path = path.expanduser().resolve()
    if not path.is_relative_to(cache):
        raise ValueError("signal plots must stay in the internal cache directory")
    if signal.values.dtype.kind not in "biuf" or not len(signal.time):
        raise ValueError("plot requires a nonempty numeric signal")
    if signal.values.ndim != 1 and signal.metadata.get("mode") != "envelope":
        raise ValueError("select one scalar record field for plotting")
    with plt.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.useoffset": False,
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(14, 7.2), dpi=100)
        factor = 1e9 if signal.metadata["time_base"] == "epoch_ns" else 1
        elapsed = (signal.time - signal.time[0]) / factor
        time_unit = "s"
        if elapsed[-1] >= 1000:
            elapsed = elapsed / 1000
            time_unit = "ks"
        values = signal.values
        unit = signal.metadata.get("unit") or "unspecified unit"
        magnitude = float(np.nanmax(np.abs(values)))
        if unit in {"A", "V", "W", "J", "Hz", "Pa"} and magnitude >= 1000:
            prefix, divisor = ("M", 1e6) if magnitude >= 1e6 else ("k", 1e3)
            values = values / divisor
            unit = prefix + unit
        if signal.metadata.get("mode") == "envelope":
            axis.plot(
                elapsed, values[:, 1], color="#929292", linewidth=2.6, label="minimum"
            )
            axis.plot(
                elapsed,
                values[:, 2],
                color="#929292",
                linestyle="--",
                linewidth=2.6,
                label="maximum",
            )
            axis.plot(
                elapsed,
                values[:, 0],
                color="#555555",
                linewidth=3,
                zorder=3,
                label="mean",
            )
            figure.legend(frameon=False, loc="upper center", ncol=3)
        else:
            axis.plot(elapsed, values, color="#555555", linewidth=1.2, zorder=3)
        axis.set_xlabel(f"elapsed time [{time_unit}]")
        axis.set_ylabel(f"signal [{unit}]")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
        axis.ticklabel_format(style="plain", axis="both", useOffset=False)
        figure.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)
        try:
            _save_figure(figure, path)
        finally:
            plt.close(figure)
    result = statistics(signal)
    print(json.dumps(result))
    return result
