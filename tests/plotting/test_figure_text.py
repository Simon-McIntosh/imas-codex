"""Reader-facing properties of the figures: wrapping, contrast, paths, baselines."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
FIGURE_DIRS = (
    ROOT / "docs/figures/ai-stack",
    ROOT / "docs/figures/discovery-to-mapping",
    ROOT / "docs/figures/icp-knowledge-map",
    ROOT / "docs/figures/jt60sa-mapping-study",
    ROOT / "docs/figures/knowledge-graph-scale",
)
HOME_PATH = re.compile(r"/home/[A-Za-z]|/Users/[A-Za-z]")


def load_script(name: str):
    """Import a figure script from scripts/ without running its main()."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def relative_luminance(colour: str) -> float:
    channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(colour: str, other: str = "#ffffff") -> float:
    low, high = sorted((relative_luminance(colour), relative_luminance(other)))
    return (high + 0.05) / (low + 0.05)


def test_a_wrapped_json_line_breaks_between_words():
    """A wrapped line may end anywhere except inside a word."""
    mapping = load_script("figures_discovery_to_mapping")
    prose = (
        "Magnetic field measurement from the tangential (poloidal-field) magnetic "
        "probe TC-(5), in tesla. This probe is part of the SELENE diagnostic set "
        "and is used for real-time plasma control."
    )
    rows = mapping.styled_lines({"description": prose}, columns=52)
    body = ["".join(character for character, _ in row) for row in rows[1:-1]]
    rejoined = "".join(body)
    assert prose in rejoined, rejoined
    for line, following in zip(body, body[1:], strict=False):
        if not line or not following:
            continue
        # a break that splits a word leaves the two fragments adjacent in the
        # rejoined text with no space between them
        if line[-1].isalpha() and following[0].isalpha():
            assert line[-1] + following[0] not in prose, (
                f"line broke inside {line[-1] + following[0]!r}"
            )
    print("WRAP " + " | ".join(body))


def test_card_system_lines_share_one_baseline():
    """Every repository card's second line sits at the same height."""
    icp = load_script("figures_icp_knowledge_map")
    captured = {}

    def capture(fig, path, **options):
        captured["figure"] = fig
        return {"bytes": 0}

    icp.save_figure = capture
    try:
        icp.grid_figure(None)
        figure = captured["figure"]
        titles = {
            icp.GRID_NAMES.get(
                key, next(town.name for town in icp.TOWNS if town.key == key)
            )
            for key in icp.GRID_ORDER
        }
        systems = {
            text.get_text(): round(text.get_position()[1], 6)
            for text in figure.texts
            if text.get_text() not in titles
        }
        heights = sorted(set(systems.values()))
        assert len(heights) == 2, heights
        assert any("\n" in title for title in titles)
    finally:
        plt.close("all")
    print(f"CARD_LINES {len(systems)} texts at {heights}")


def test_every_segment_fill_carries_its_white_numeral():
    """A white count numeral needs 4.5:1 against the fill it sits on."""
    study = load_script("figures_jt60sa_mapping_study")
    fills = {
        "agreeing": study.AGREE,
        "chain-differ": study.CHAIN_DIFFERS,
        "disagreeing": study.DISAGREE,
        "unplaced": study.UNPLACED,
        "refused": study.REFUSED,
    }
    ratios = {name: contrast(colour) for name, colour in fills.items()}
    assert min(ratios.values()) >= 4.5, ratios
    print("CONTRAST " + ", ".join(f"{name}={ratio:.2f}" for name, ratio in ratios.items()))


def test_no_published_figure_names_a_machine_home_path():
    """A figure must not carry the absolute home path of the checkout."""
    files = 0
    offenders = []
    for directory in FIGURE_DIRS:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() in {".png", ".gif"}:
                continue
            files += 1
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if HOME_PATH.search(text):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], offenders
    print(f"PATHS {files} text files clean of absolute home paths")