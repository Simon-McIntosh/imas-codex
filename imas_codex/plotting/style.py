"""Shared figure style, component roles and checked exports."""

import re
from pathlib import Path

from matplotlib.text import Annotation, Text

SUBJECT = "#2878a5"
REFERENCE = "#929292"
LINE_STYLES = ("-", "--", "-.", ":")
CANVAS = (14, 7.2)
SLIDE_WIDTH_CM = 31.1
BASE_STYLE = Path(__file__).with_name("data-ink.mplstyle")
STYLE = {"axes.formatter.useoffset": False, "svg.fonttype": "none"}


def _validate_text(figure, renderer):
    """Refuse clipped or intersecting text before opening an export file."""
    canvas = figure.bbox
    boxes = []
    for artist in figure.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        if (
            box.x0 < canvas.x0
            or box.y0 < canvas.y0
            or box.x1 > canvas.x1
            or box.y1 > canvas.y1
        ):
            raise ValueError(f"clipped figure text: {artist.get_text()}")
        for prior, other in boxes:
            if box.overlaps(other):
                raise ValueError(
                    f"overlapping figure text: {prior} / {artist.get_text()}"
                )
        boxes.append((artist.get_text(), box))


def save_figure(figure, path: str | Path) -> None:
    """Refuse bad text layout and direct labels over data before exporting.

    All checks run at raster-export resolution. SVG dimensions declare the
    slide width while retaining the explicitly sized figure's aspect ratio.
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
    _validate_text(figure, renderer)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    if path.suffix.lower() == ".svg":
        height = SLIDE_WIDTH_CM * figure.get_figheight() / figure.get_figwidth()
        content = re.sub(
            r'width="[^"]+pt" height="[^"]+pt"',
            f'width="{SLIDE_WIDTH_CM:g}cm" height="{height:.3g}cm"',
            path.read_text(),
            count=1,
        )
        path.write_text(
            "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
        )
