"""Shared figure style, component roles and checked exports."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation, Text

SUBJECT = "#2878a5"
REFERENCE = "#929292"
LINE_STYLES = ("-", "--", "-.", ":")
CANVAS = (14, 7.2)
SLIDE_WIDTH_CM = 31.1
SLIDE_SIZE_CM = (SLIDE_WIDTH_CM, 16.0)
BASE_STYLE = Path(__file__).with_name("data-ink.mplstyle")
STYLE = {"axes.formatter.useoffset": False, "svg.fonttype": "none"}


def use_style(**overrides: Any) -> None:
    """Apply the packaged figure style, then the caller's rc overrides."""
    plt.style.use(BASE_STYLE)
    if overrides:
        plt.rcParams.update(overrides)


def title_strip(figure, share: float):
    """Reserve the band at the top of a slide figure for the slide's own title.

    A slide writes its title over the figure, so figure text that drifts into
    that band would be hidden under it. The returned mark is meant to be passed
    to :func:`save_figure` as ``clear_of``, which refuses text inside it.
    """
    strip = Rectangle(
        (0, 1 - share),
        1,
        share,
        transform=figure.transFigure,
        facecolor="none",
        edgecolor="none",
    )
    figure.add_artist(strip)
    return strip


def _declared_size(figure) -> tuple[float, float]:
    """The slide size the figure declares: the slide's own size for the default
    canvas, otherwise the slide width with the height its aspect ratio needs."""
    if tuple(figure.get_size_inches()) == CANVAS:
        return SLIDE_SIZE_CM
    width = SLIDE_WIDTH_CM
    return round(width, 2), round(
        width * figure.get_figheight() / figure.get_figwidth(), 2
    )


def _text_boxes(figure, renderer) -> list[tuple[Text, Any]]:
    boxes = []
    for artist in figure.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        boxes.append((artist, box))
    return boxes


def _validate_text(figure, renderer, clear_of: Iterable[Any]) -> None:
    """Refuse clipped or intersecting text before opening an export file.

    ``clear_of`` names marks, such as arrowheads, that must not cover any text.
    """
    canvas = figure.bbox
    boxes = _text_boxes(figure, renderer)
    for artist, box in boxes:
        if (
            box.x0 < canvas.x0
            or box.y0 < canvas.y0
            or box.x1 > canvas.x1
            or box.y1 > canvas.y1
        ):
            raise ValueError(f"clipped figure text: {artist.get_text()}")
    for index, (artist, box) in enumerate(boxes):
        for prior, other in boxes[:index]:
            if box.overlaps(other):
                raise ValueError(
                    f"overlapping figure text: {prior.get_text()} / {artist.get_text()}"
                )
    for mark in clear_of:
        if not mark.get_visible():
            continue
        extent = mark.get_window_extent(renderer)
        for artist, box in boxes:
            if extent.overlaps(box):
                raise ValueError(f"figure mark covers text: {artist.get_text()}")


def save_figure(
    figure,
    path: str | Path,
    *,
    dpi: int = 200,
    animation=None,
    fps: float = 15,
    clear_of: Iterable[Any] = (),
    decoration: Iterable[Any] = (),
) -> dict[str, Any]:
    """Refuse bad text layout and direct labels over data before exporting.

    All checks run at export resolution. A vector export declares the slide
    width and the height the canvas's own ratio needs. An animated export
    needs the ``animation`` drawing the frames and writes a GIF.

    ``clear_of`` names marks, such as arrowheads or a reserved title strip,
    that must not cover any text. ``decoration`` names line artists that are
    paint rather than data, such as a rule or a leader line: a label over one
    of those is allowed, and naming the artist is the only way to except it.
    The return value is a receipt of the file written.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".svg", ".png", ".gif"}:
        raise ValueError("figure export must be SVG, PNG or GIF")
    if (suffix == ".gif") != (animation is not None):
        raise ValueError("a GIF export needs an animation, and only a GIF takes one")
    figure.set_dpi(dpi)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    excepted = {id(mark) for mark in decoration}
    for axis in figure.axes:
        for label in axis.texts:
            if not label.get_visible() or not label.get_text().strip():
                continue
            box = label.get_window_extent(renderer)
            for line in axis.lines:
                if not line.get_visible() or id(line) in excepted:
                    continue
                if (
                    line.get_transform()
                    .transform_path(line.get_path())
                    .intersects_bbox(box, filled=False)
                ):
                    raise ValueError(
                        f"direct label overlaps data: {label.get_text()} / "
                        f"{line.get_label()}"
                    )
            if isinstance(label, Annotation) and label.arrow_patch is not None:
                raise ValueError("direct labels must sit beside data without arrows")
    _validate_text(figure, renderer, clear_of)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "path": str(path),
        "format": suffix.lstrip("."),
        "export_dpi": dpi,
        "pixel_size": (
            round(figure.get_figwidth() * dpi),
            round(figure.get_figheight() * dpi),
        ),
    }
    if animation is not None:
        from matplotlib.animation import PillowWriter

        animation.save(
            path,
            writer=PillowWriter(fps=fps),
            dpi=dpi,
            savefig_kwargs={"facecolor": "white"},
        )
    else:
        # a fixed salt keeps a redrawn vector file's element ids stable
        with plt.rc_context({"svg.hashsalt": "iter-codex"}):
            figure.savefig(path, dpi=dpi, facecolor="white", metadata={"Date": None})
    if suffix == ".svg":
        width, height = _declared_size(figure)
        content = re.sub(
            r'width="[^"]+pt" height="[^"]+pt"',
            f'width="{width:g}cm" height="{height:g}cm"',
            path.read_text(),
            count=1,
        )
        path.write_text(
            "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
        )
        receipt["declared_cm"] = (width, height)
    receipt["bytes"] = path.stat().st_size
    return receipt
