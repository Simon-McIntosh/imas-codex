"""Figure scripts export through the shared checked save function."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from PIL import Image

from imas_codex.plotting import save_figure, style

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name: str):
    """Import a figure script from scripts/ without running its main()."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_script_given_a_label_beyond_the_canvas_writes_no_file(tmp_path, monkeypatch):
    """The study figure refuses to export a bar label wider than its canvas."""
    study = load_script("figures_jt60sa_mapping_study")
    out = tmp_path / "agreement.svg"
    monkeypatch.setattr(study, "OUT", out)
    wide = "a note that runs far past the end of the canvas " * 3
    monkeypatch.setattr(study, "ROWS", [("§8  PF and TF coils", 7, 6, 0, 12, 1, wide)])
    try:
        with pytest.raises(ValueError, match="clipped figure text"):
            study.render()
        print(f"REFUSAL {wide[:20]!r}; file_written={out.exists()}")
        assert not out.exists()
    finally:
        plt.close("all")


def test_animated_export_writes_a_gif_at_the_canvas_ratio(tmp_path):
    figure, axis = plt.subplots(figsize=style.CANVAS)
    axis.plot([0.0, 1.0], [0.0, 1.0], label="trace")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)
    animation = FuncAnimation(
        figure, lambda step: axis.set_title(f"frame {step}"), frames=2
    )
    path = tmp_path / "animation.gif"
    try:
        save_figure(figure, path, dpi=100, animation=animation, fps=5)
    finally:
        plt.close(figure)
    assert path.is_file()
    with Image.open(path) as written:
        pixels = written.size
        frames = getattr(written, "n_frames", 1)
    assert frames == 2
    assert pixels[0] / pixels[1] == pytest.approx(
        style.CANVAS[0] / style.CANVAS[1], rel=1e-3
    )
    print(f"GIF {pixels} frames={frames} bytes={path.stat().st_size}")


def test_declared_slide_size_keeps_the_ratio_of_any_canvas(tmp_path):
    canvas = (10.0, 14.0)
    figure, axis = plt.subplots(figsize=canvas)
    axis.plot([0.0, 1.0], [0.0, 1.0])
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.1, top=0.94)
    path = tmp_path / "tall.svg"
    try:
        save_figure(figure, path)
    finally:
        plt.close(figure)
    declared = re.search(
        r'width="([\d.]+)cm" height="([\d.]+)cm"', path.read_text()
    )
    assert declared is not None
    width, height = (float(value) for value in declared.groups())
    drawn = canvas[0] / canvas[1]
    assert abs(width / height - drawn) <= drawn / 1000
    print(f"DECLARED {width}x{height}cm drawn_ratio={drawn:.5f}")


def test_a_mark_covering_text_is_refused(tmp_path):
    figure = plt.figure(figsize=style.CANVAS)
    figure.text(0.5, 0.5, "under the mark")
    mark = Rectangle(
        (0.4, 0.4), 0.2, 0.2, transform=figure.transFigure, facecolor="none"
    )
    figure.add_artist(mark)
    path = tmp_path / "covered.svg"
    try:
        with pytest.raises(ValueError, match="figure mark covers text"):
            save_figure(figure, path, clear_of=[mark])
        assert not path.exists()
        # the same figure exports once the mark is not held clear of text
        save_figure(figure, path)
        assert path.is_file()
    finally:
        plt.close(figure)


def test_a_label_over_an_unlabelled_data_line_is_refused(tmp_path):
    """A trace the legend never names is still data a reader can confuse."""
    figure, axis = plt.subplots(figsize=style.CANVAS)
    axis.plot([0.0, 1.0], [0.5, 0.5])
    axis.text(0.5, 0.5, "7.4", ha="center", va="center")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)
    path = tmp_path / "unlabelled.svg"
    try:
        with pytest.raises(ValueError, match="direct label overlaps data"):
            save_figure(figure, path)
        assert not path.exists()
        # naming the line as decoration is the way to except it, and then it exports
        save_figure(figure, path, decoration=axis.lines)
        assert path.is_file()
    finally:
        plt.close(figure)
