"""
Figures for the report. Every function returns the matplotlib figure and
also saves it as a PNG in ``figures/`` so the images ship with the report.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from .data_prep import CNY_2001

FIG_DIR = "figures"

# categorical slots (fixed order) + neutrals
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET, RED, GRAY = "#4a3aa7", "#e34948", "#8d8c86"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

QUAD_STYLE = {
    "Star": (BLUE, "o"),
    "Loss leader": (ORANGE, "D"),
    "Passenger": (GRAY, "."),
    "Dead weight": (VIOLET, "x"),
}

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "font.size": 10,
})


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    return fig


def daily_sales(lines):
    """Daily revenue and profit (two panels, one axis each) with CNY marked."""
    d = lines.groupby("date")[["price", "profit"]].sum() / 1e3
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for ax, col, color, label in [
        (axes[0], "price", BLUE, "Revenue (NT$ thousands)"),
        (axes[1], "profit", AQUA, "Profit (NT$ thousands)"),
    ]:
        ax.plot(d.index, d[col], color=color)
        ax.axvline(CNY_2001, color=INK2, ls="--", lw=1)
        ax.set_ylabel(label)
    axes[0].text(CNY_2001, axes[0].get_ylim()[1], " Chinese New Year",
                 va="top", color=INK2)
    axes[0].set_title("Daily sales, Nov 2000 - Feb 2001")
    fig.tight_layout()
    return _save(fig, "01_daily_sales")


def basket_baseline(baskets, expected, size_cap=25):
    """Basket-size distribution and the size baseline used by the halo."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sizes = baskets["size"].clip(upper=size_cap).value_counts().sort_index()
    axes[0].bar(sizes.index, sizes.values, color=BLUE, width=0.8)
    axes[0].set_xlabel(f"Distinct subclasses in basket ({size_cap} = {size_cap}+)")
    axes[0].set_ylabel("Baskets")
    axes[0].set_title("Most baskets are small")
    expected = pd.Series(expected).sort_index()
    axes[1].plot(expected.index, expected.values, color=AQUA, marker="o",
                 ms=4)
    axes[1].set_xlabel("Basket size")
    axes[1].set_ylabel("NT$")
    axes[1].set_title("Expected profit of the other items, $E_n$")
    fig.tight_layout()
    return _save(fig, "02_basket_baseline")


def quadrant_map(bap, label_top=4):
    """Own margin vs halo, one marker per subclass, coloured by quadrant."""
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for q, (color, marker) in QUAD_STYLE.items():
        d = bap[bap["quadrant"] == q]
        ax.scatter(d["own_margin"], d["halo"], c=color, marker=marker,
                   s=np.clip(d["n_baskets"] / 20, 10, 120), alpha=0.75,
                   linewidths=1 if marker == "x" else 0.5,
                   edgecolors="white" if marker not in ".x" else None,
                   label=f"{q} ({len(d)})")
    ax.axvline(0, color=INK2, lw=1)
    ax.axhline(bap.attrs["halo_median"], color=INK2, lw=1, ls="--")
    ax.set_xscale("symlog", linthresh=5)
    ax.set_yscale("symlog", linthresh=5)
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel("Own margin per basket (NT$, symlog)")
    ax.set_ylabel("Basket halo per basket (NT$, symlog)")
    ax.set_title("Which subclasses earn their place?")
    # label the loss leaders with the biggest halo
    lead = bap[bap["quadrant"] == "Loss leader"].nlargest(label_top, "halo")
    for k, (sc, r) in enumerate(lead.iterrows()):
        ax.annotate(sc, (r["own_margin"], r["halo"]), fontsize=8,
                    color=INK, xytext=(8, -12 - 11 * k),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              markerscale=1.2)
    fig.tight_layout()
    return _save(fig, "03_quadrant_map")


def bap_persistence(bap_train, bap_holdout):
    """Halo in training vs the holdout month, per subclass."""
    common = bap_train.index.intersection(bap_holdout.index)
    a, h = bap_train.loc[common, "halo"], bap_holdout.loc[common, "halo"]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(a, h, s=12, color=BLUE, alpha=0.5, linewidths=0)
    lim = np.nanpercentile(np.r_[a, h], [1, 99])
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Halo, training months (NT$)")
    ax.set_ylabel("Halo, holdout month (NT$)")
    ax.set_title("Does the halo persist out of time?")
    fig.tight_layout()
    return _save(fig, "04_halo_persistence")


def model_comparison(results):
    """NDCG@10 and profit per customer for each model and mode."""
    modes = [("all", "All items"), ("new", "Items new to the customer")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for col, (mode, title) in enumerate(modes):
        d = results[results["mode"] == mode].set_index("model")
        for row, (metric, label, color) in enumerate([
            ("ndcg@10", "NDCG@10", BLUE),
            ("profit_per_customer", "Feb profit captured per customer (NT$)",
             AQUA),
        ]):
            ax = axes[row, col]
            y = np.arange(len(d))
            ax.barh(y, d[metric], color=color, height=0.6)
            ax.set_yticks(y, d.index)
            ax.invert_yaxis()
            ax.set_xlabel(label)
            ax.grid(axis="y", visible=False)
            for yi, v in zip(y, d[metric]):
                ax.text(v, yi, f" {v:.3f}" if metric == "ndcg@10"
                        else f" {v:,.0f}", va="center", color=INK2,
                        fontsize=8)
            if row == 0:
                ax.set_title(title)
    fig.tight_layout()
    return _save(fig, "05_model_comparison")


def profit_frontier(sweeps, picks, name="06_profit_frontier"):
    """
    Accuracy-profit trade-off across lambda, BAP vs own-margin re-ranking.

    Top row: profit on the recommended items that were bought. Bottom
    row: profit of the whole baskets those items were bought in.

    Parameters
    ----------
    sweeps : dict
        ``{panel title: lambda_sweep DataFrame}``.
    picks : dict
        ``{panel title: chosen lambda}`` (circled on the BAP curve).
    """
    ys = [("profit_per_customer", "Item profit captured\nper customer (NT$)"),
          ("basket_profit_per_customer",
           "Basket profit captured\nper customer (NT$)")]
    fig, axes = plt.subplots(2, len(sweeps),
                             figsize=(5.6 * len(sweeps), 8), squeeze=False)
    for col, (title, sw) in enumerate(sweeps.items()):
        for row, (ycol, ylabel) in enumerate(ys):
            ax = axes[row, col]
            for sig, color, marker in [("BAP", ORANGE, "o"),
                                       ("Own margin", BLUE, "s")]:
                d = sw[sw["signal"] == sig].sort_values("lambda")
                ax.plot(d["ndcg@10"], d[ycol], color=color, marker=marker,
                        ms=5, label=sig)
                if sig == "BAP":
                    r = d.iloc[-1]
                    ax.annotate("λ=1", (r["ndcg@10"], r[ycol]), fontsize=8,
                                color=INK2, xytext=(4, -10),
                                textcoords="offset points")
            r0 = sw[sw["lambda"] == 0].iloc[0]
            ax.annotate("λ=0", (r0["ndcg@10"], r0[ycol]), fontsize=8,
                        color=INK2, xytext=(4, 4), textcoords="offset points")
            lam = picks.get(title)
            if lam is not None:
                r = sw[(sw["signal"] == "BAP") & (sw["lambda"] == lam)].iloc[0]
                ax.scatter(r["ndcg@10"], r[ycol], s=150, facecolors="none",
                           edgecolors=INK, linewidths=1.5, zorder=5,
                           label=f"chosen λ={lam:g}")
            ax.set_xlabel("NDCG@10")
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(title)
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def cny_check(sweeps):
    """Profit and NDCG across lambda, computed on two parts of January."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = [BLUE, ORANGE]
    for (title, sw), color in zip(sweeps.items(), colors):
        d = sw[sw["signal"] == "BAP"].sort_values("lambda")
        base = d.iloc[0]
        axes[0].plot(d["lambda"], d["ndcg@10"] / base["ndcg@10"],
                     color=color, marker="o", ms=4, label=title)
        axes[1].plot(d["lambda"],
                     d["profit_per_customer"] / base["profit_per_customer"],
                     color=color, marker="o", ms=4, label=title)
    axes[0].set_title("NDCG@10 relative to λ = 0")
    axes[1].set_title("Profit captured relative to λ = 0")
    for ax in axes:
        ax.set_xlabel("λ (weight on BAP)")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.legend()
    fig.tight_layout()
    return _save(fig, "07_cny_check")


def segment_overlap(overlap, title, name):
    """Jaccard overlap of top pair rules between segments (heatmap)."""
    fig, ax = plt.subplots(figsize=(1.2 * len(overlap) + 2.5,
                                    1.0 * len(overlap) + 1.5))
    im = ax.imshow(overlap.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(overlap)), overlap.columns, rotation=30,
                  ha="right")
    ax.set_yticks(range(len(overlap)), overlap.index)
    ax.grid(False)
    for i in range(len(overlap)):
        for j in range(len(overlap)):
            v = overlap.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else INK, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Jaccard overlap")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, name)
