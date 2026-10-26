"""Figure for the sensing report: felt-force error per imperfection, one bar per cumulative fix level.

    python -m blindtouch.eval.sensing_figure            # results/sensing.json -> results/figures/sensing.png (+ .pdf)
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blindtouch.paths import RESULTS
from blindtouch.eval.sensing import CONDS, LEVELS

RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#184f95")   # ordinal blue ramp: raw (light) -> all fixes (dark)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"clean": "clean", "friction": "joint\nfriction", "noise": "noise", "ripple": "torque\nripple",
         "latency": "delay", "model_error": "model\nerror", "all": "all five"}
LEVEL_LABEL = {"raw": "raw observer", "+dob": "+ contact-gated DOB", "+cog": "+ anticogging", "+int": "+ integral terms"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sensing")
    ap.add_argument("--out", default="sensing")
    args = ap.parse_args()
    rows = json.loads((RESULTS / f"{args.src}.json").read_text())
    conds = [c for c in CONDS if any(r["cond"] == c for r in rows)]

    def med(c, lv, phases):
        v = [np.mean([r[ph]["err_rms"] for ph in phases]) for r in rows if r["cond"] == c and r["level"] == lv]
        return float(np.median(v))

    panels = [("Felt force error in contact (press 3 N, 8 N, slide)", ("press 3 N", "press 8 N", "slide 3 N")),
              ("Phantom force in free space (hold, move, after contact)", ("free hold", "free move", "free after"))]
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True)
    x = np.arange(len(conds))
    w = 0.8 / len(LEVELS)
    for ax, (title, phases) in zip(axes, panels):
        for k, lv in enumerate(LEVELS):
            v = [med(c, lv, phases) for c in conds]
            ax.bar(x + (k - (len(LEVELS) - 1) / 2) * w, v, w * 0.9, color=RAMP[k], label=LEVEL_LABEL[lv], zorder=2)
        ax.set_title(title, loc="left", color=INK, fontsize=9)
        ax.set_ylabel("RMS error (N)", color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.axhline(2.5, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.text(-0.45, 2.55, "contact threshold 2.5 N", ha="left", va="bottom", color=INK2, fontsize=7)
    axes[1].set_xticks(x, [LABEL[c] for c in conds])
    axes[0].legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.02), fontsize=7, labelcolor=INK)
    fig.tight_layout(h_pad=2.2)
    out = RESULTS / "figures" / args.out
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
