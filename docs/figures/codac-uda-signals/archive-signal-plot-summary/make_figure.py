"""Render invented envelope bins beside an archived-on-change example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from iter_codex.plotting import plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path)
    options = parser.parse_args()
    metadata = {"time_base": "relative_s", "unit": "A", "semantics": "zero_order_hold"}
    envelope = SimpleNamespace(
        time=np.array([0.25, 0.75, 1.25, 1.75]),
        values=np.array(
            [
                [250.0, 100.0, 400.0],
                [650.0, 500.0, 800.0],
                [900.0, 800.0, 1000.0],
                [400.0, 250.0, 550.0],
            ]
        ),
        metadata={**metadata, "mode": "envelope", "decimated": True},
    )
    events = SimpleNamespace(
        time=np.array([0.0, 0.4, 1.1, 1.8, 2.0]),
        values=np.array([200.0, 600.0, 950.0, 300.0, 300.0]),
        metadata={**metadata, "mode": "raw", "decimated": None},
    )
    path = Path(__file__).with_name("synthetic-signal.svg")
    receipt = plot([envelope, events], path, labels=["envelope", "events"])
    if options.preview:
        plot([envelope, events], options.preview, labels=["envelope", "events"])
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
