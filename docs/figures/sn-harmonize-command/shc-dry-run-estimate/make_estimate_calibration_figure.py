"""The dry-run estimate read against what the apply then spent.

One panel: the estimate the dry run prints for the 17-member boundary-density
family, drawn beside the spend the second forced apply recorded, with the
30 % acceptance band the family's measurement is judged against. The estimate
is the subject and takes the colour; the recorded spend is the reference and
stays neutral.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent

INK = "#1b2733"
MUTED = "#8c8c8c"
COOL = "#2d6ca2"

#: The estimate the dry run printed for the family's 17 planned calls,
#: measured from 7 recent documentation runs covering 23 calls.
ESTIMATE = 7.04
#: What the second forced apply over the family actually spent.
APPLIED = 6.26
#: The acceptance band the family's measurement is judged against.
TOLERANCE = 0.30


def main() -> None:
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 5))

    y_est, y_act = 1.0, 0.0
    ax.barh(y_est, ESTIMATE, height=0.5, color=COOL, zorder=3)
    ax.barh(y_act, APPLIED, height=0.5, color=MUTED, zorder=3)

    lo, hi = APPLIED * (1 - TOLERANCE), APPLIED * (1 + TOLERANCE)
    ax.axvline(lo, color="#52514e", lw=1.2, ls=":", zorder=2)
    ax.axvline(hi, color="#52514e", lw=1.2, ls=":", zorder=2)
    ax.annotate(
        f"±30 % of applied (${lo:.2f}–${hi:.2f})",
        (hi, 1.55),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=20,
        color="#52514e",
        va="center",
    )

    ax.annotate(
        f"estimate  ${ESTIMATE:.2f}",
        (ESTIMATE, y_est),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=20,
        color=COOL,
        va="center",
        fontweight="bold",
    )
    ax.annotate(
        f"applied  ${APPLIED:.2f}",
        (APPLIED, y_act),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=20,
        color="#52514e",
        va="center",
    )

    ax.set_yticks([y_act, y_est])
    ax.set_yticklabels(["recorded apply", "dry-run estimate"])
    ax.set_xlabel("cost [USD]")
    ax.set_xlim(0, hi + 1.1)
    ax.set_ylim(-0.6, 1.85)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(OUT / "estimate-calibration.svg")
    print("wrote", OUT / "estimate-calibration.svg")


if __name__ == "__main__":
    main()
