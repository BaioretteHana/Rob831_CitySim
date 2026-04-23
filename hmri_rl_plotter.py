"""
hmri_rl_plotter.py 

Generates three figures from TensorBoard logs:

  Figure 1 — Distance subplots (one panel per segment, PPO vs DQN overlaid)
  Figure 2 — Success rate subplots (one panel per segment + segment_reached)
  Figure 3 — Learning curve: mean return per episode (PPO vs DQN vs Random)

Usage:
  python hmri_rl_plotter.py \
      --ppo_dir  ./tensorboard/hmri_ppo/ppo_1 \
      --dqn_dir  ./tensorboard/hmri_dqn/dqn_1 \
      --rand_dir ./tensorboard/hmri_random \
      --out_dir  ./plots \
      --smooth   30

Requirements:
  pip install tensorboard matplotlib scipy numpy
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



# CONSTANTS
SEGMENTS = ["highway", "merge", "roundabout", "intersection"]

SEGMENT_COLORS = {
    "highway":      "#5B8DB8",
    "merge":        "#E07B4A",
    "roundabout":   "#5BAD72",
    "intersection": "#9B6DBF",
}

ALGO_STYLE = {
    "PPO":    {"color": "#2166AC", "ls": "-",  "lw": 2.0},
    "DQN":    {"color": "#D6604D", "ls": "-",  "lw": 2.0},
    "Random": {"color": "#777777", "ls": "--", "lw": 1.6},
}

# add this dictionary at top of file (near constants)
DISTANCE_REF = {
    "highway":  290,
    "merge":    260,
    "roundabout": 575,
    "intersection": 100,
}

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    ":",
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "legend.fontsize":   9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.constrained_layout.use": True,
})


# TENSORBOARD LOADER
def load_tag(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a scalar tag from a TensorBoard log directory.
    Walks all subdirectories to find the event file containing the tag.
    Returns (steps, values) — both empty arrays if tag not found.
    """
    if not log_dir or not os.path.exists(log_dir):
        return np.array([]), np.array([])

    # Try the directory itself first, then walk subdirs
    candidates = [log_dir] + [
        str(p.parent)
        for p in sorted(Path(log_dir).rglob("*.tfevents.*"))
    ]
    seen = set()
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        try:
            ea = EventAccumulator(d, size_guidance={"scalars": 0})
            ea.Reload()
            if tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                steps  = np.array([e.step  for e in events], dtype=float)
                values = np.array([e.value for e in events], dtype=float)
                return steps, values
        except Exception:
            continue

    return np.array([]), np.array([])


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 2:
        return values
    w = min(window, len(values))
    return uniform_filter1d(values.astype(float), size=w)


def shade(ax, steps, raw, color, window, alpha=0.15):
    """Plot smoothed line + shaded rolling-std band. Returns smoothed values."""
    if len(steps) == 0:
        return
    sv  = smooth(raw, window)
    pad = window // 2
    std = np.array([raw[max(0, i-pad):i+pad+1].std() for i in range(len(raw))])
    ax.fill_between(steps, sv - std, sv + std, color=color, alpha=alpha, linewidth=0)
    return sv


def fmt_steps(ax):
    """Format x-axis as k-steps."""
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )
    ax.set_xlabel("Training Steps")


# DATA BUNDLE
def load_all(ppo_dir, dqn_dir, rand_dir):
    """
    Returns nested dict:  data[algo][tag] = (steps, values)
    Only loads the tags actually needed by the three figures.
    """
    needed_trained = (
        [f"distance/{s}"          for s in SEGMENTS] +
        [f"success/reached_{s}"   for s in SEGMENTS] +
        ["success/segment_reached",
         "episode/mean_return"]
    )
    needed_random = [
        "random/mean_return",
        "random/segment_reached",
    ]

    data = {}

    if ppo_dir:
        data["PPO"] = {t: load_tag(ppo_dir, t) for t in needed_trained}
        print(f"  PPO  loaded from {ppo_dir}")

    if dqn_dir:
        data["DQN"] = {t: load_tag(dqn_dir, t) for t in needed_trained}
        print(f"  DQN  loaded from {dqn_dir}")

    if rand_dir:
        data["Random"] = {t: load_tag(rand_dir, t) for t in needed_random}
        print(f"  Rand loaded from {rand_dir}")

    # Report missing tags
    for algo, tags in data.items():
        missing = [t for t, (s, v) in tags.items() if len(s) == 0]
        if missing:
            print(f"  [{algo}] tags not found: {missing}")

    return data


# FIGURE 1 — DISTANCE PER SEGMENT  (PPO vs DQN, one panel per segment)
def fig_distance(data, out_dir, smooth_w):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False)
    fig.suptitle("Distance Travelled per Segment  (PPO vs DQN)", fontsize=12, fontweight="bold")

    for col, seg in enumerate(SEGMENTS):
        ax  = axes[col]
        tag = f"distance/{seg}"
        ax.set_title(seg.capitalize(), color=SEGMENT_COLORS[seg])

        has_data = False
        for algo in ["PPO", "DQN"]:
            if algo not in data:
                continue
            steps, vals = data[algo].get(tag, (np.array([]), np.array([])))
            if len(steps) == 0:
                continue
            sv = shade(ax, steps, vals, ALGO_STYLE[algo]["color"], smooth_w)
            ax.plot(steps, sv,
                    color=ALGO_STYLE[algo]["color"],
                    ls=ALGO_STYLE[algo]["ls"],
                    lw=ALGO_STYLE[algo]["lw"],
                    label=algo)
            has_data = True

        if seg in DISTANCE_REF:
            ax.axhline(
                DISTANCE_REF[seg],
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label="reference" if col == 0 else None
        )

        fmt_steps(ax)
        ax.set_ylabel("Distance (m)" if col == 0 else "")
        if not has_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=9)

    # Shared legend
    handles = [Line2D([0], [0], color=ALGO_STYLE[a]["color"],
                      ls=ALGO_STYLE[a]["ls"], lw=ALGO_STYLE[a]["lw"], label=a)
               for a in ["PPO", "DQN"] if a in data]
    # add horizontal reference line to legend
    handles.append(
        Line2D([0], [0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Start to End Displacement for Each Environment Segment")
            )

    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    path = os.path.join(out_dir, "fig1_distance_per_segment.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")



# FIGURE 2 — SUCCESS RATES  (PPO vs DQN, one panel per segment + segment_reached)
def fig_success(data, out_dir, smooth_w):
    # 5 panels: one per segment binary flag + one for segment_reached (integer)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharey=False)
    fig.suptitle("Success Metrics  (PPO vs DQN)", fontsize=12, fontweight="bold")

    # First 4 panels: binary reached flags
    for col, seg in enumerate(SEGMENTS):
        ax  = axes[col]
        tag = f"success/reached_{seg}"
        ax.set_title(f"Reached\n{seg.capitalize()}", color=SEGMENT_COLORS[seg])

        has_data = False
        for algo in ["PPO", "DQN"]:
            if algo not in data:
                continue
            steps, vals = data[algo].get(tag, (np.array([]), np.array([])))
            if len(steps) == 0:
                continue
            sv = shade(ax, steps, vals, ALGO_STYLE[algo]["color"], smooth_w)
            ax.plot(steps, sv,
                    color=ALGO_STYLE[algo]["color"],
                    ls=ALGO_STYLE[algo]["ls"],
                    lw=ALGO_STYLE[algo]["lw"],
                    label=algo)
            has_data = True

        ax.set_ylim(-0.05, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        fmt_steps(ax)
        ax.set_ylabel("Rate" if col == 0 else "")
        if not has_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=9)

    # 5th panel: segment_reached integer (0–3) for PPO and DQN
    ax  = axes[4]
    tag = "success/segment_reached"
    ax.set_title("Furthest Segment\nReached (0–3)", color="#444444")

    has_data = False
    for algo in ["PPO", "DQN"]:
        if algo not in data:
            continue
        steps, vals = data[algo].get(tag, (np.array([]), np.array([])))
        if len(steps) == 0:
            continue
        sv = shade(ax, steps, vals, ALGO_STYLE[algo]["color"], smooth_w)
        ax.plot(steps, sv,
                color=ALGO_STYLE[algo]["color"],
                ls=ALGO_STYLE[algo]["ls"],
                lw=ALGO_STYLE[algo]["lw"],
                label=algo)
        has_data = True

    ax.set_ylim(-0.1, 3.3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Hwy", "Merge", "Rbt", "Int"], fontsize=7)
    fmt_steps(ax)
    if not has_data:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=9)

    # Shared legend
    handles = [Line2D([0], [0], color=ALGO_STYLE[a]["color"],
                      ls=ALGO_STYLE[a]["ls"], lw=ALGO_STYLE[a]["lw"], label=a)
               for a in ["PPO", "DQN"] if a in data]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    path = os.path.join(out_dir, "fig2_success_rates.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# FIGURE 3 — LEARNING CURVE  (PPO vs DQN vs Random, one plot)
def fig_learning_curve(data, out_dir, smooth_w):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_title("Mean Return per Episode vs Training Steps", fontsize=12, fontweight="bold")

    tag_map = {
        "PPO":    "episode/mean_return",
        "DQN":    "episode/mean_return",
        "Random": "random/mean_return",
    }

    any_plotted = False
    for algo in ["PPO", "DQN", "Random"]:
        if algo not in data:
            continue
        tag = tag_map[algo]
        steps, vals = data[algo].get(tag, (np.array([]), np.array([])))
        if len(steps) == 0:
            print(f"  [{algo}] no data for {tag}, skipping")
            continue

        sv = shade(ax, steps, vals, ALGO_STYLE[algo]["color"], smooth_w)
        ax.plot(steps, sv,
                color=ALGO_STYLE[algo]["color"],
                ls=ALGO_STYLE[algo]["ls"],
                lw=ALGO_STYLE[algo]["lw"],
                label=algo)
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No data found.\nCheck --ppo_dir / --dqn_dir / --rand_dir paths.",
                transform=ax.transAxes, ha="center", va="center",
                color="grey", fontsize=10)

    ax.set_ylabel("Episode Return (cumulative reward sum)")
    fmt_steps(ax)
    ax.legend(frameon=False)

    path = os.path.join(out_dir, "fig3_learning_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading TensorBoard logs...")
    data = load_all(args.ppo_dir, args.dqn_dir, args.rand_dir)

    if not data:
        print("ERROR: No log directories provided or found. Exiting.")
        return

    w = args.smooth
    print(f"\nGenerating figures (smooth window = {w} episodes)...")

    fig_distance(data,       args.out_dir, w)
    fig_success(data,        args.out_dir, w)
    fig_learning_curve(data, args.out_dir, w)

    print(f"\nDone. All figures saved to: {args.out_dir}/")


# ARGUMENT PARSER
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HMRI RL training results")

    parser.add_argument(
        "--ppo_dir",  type=str, default=None,
        help="PPO TensorBoard log dir, e.g. ./tensorboard/hmri_ppo/ppo_1"
    )
    parser.add_argument(
        "--dqn_dir",  type=str, default=None,
        help="DQN TensorBoard log dir, e.g. ./tensorboard/hmri_dqn/dqn_1"
    )
    parser.add_argument(
        "--rand_dir", type=str, default=None,
        help="Random agent TensorBoard log dir, e.g. ./tensorboard/hmri_random"
    )
    parser.add_argument(
        "--out_dir",  type=str, default="./plots",
        help="Output directory for saved figures (created if absent)"
    )
    parser.add_argument(
        "--smooth",   type=int, default=20,
        help="Moving-average window width in episodes for smoothing (default 20)"
    )

    args = parser.parse_args()
    main(args)