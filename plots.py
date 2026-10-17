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


# --------------------------------------------------------------------
# explanatory diagrams (drawn in code so the notebook regenerates them)
# --------------------------------------------------------------------
LANE_FIM = ("#fdebe3", ORANGE)    # fill, edge
LANE_CF = ("#e3eefb", BLUE)
SHARED = ("#f2f2ef", GRAY)
FINAL = ("#ddf3ea", AQUA)


def _box(ax, x, y, text, style, w=2.3, h=1.15, size=9):
    """Rounded box centred on (x, y)."""
    from matplotlib.patches import FancyBboxPatch
    fill, edge = style
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fill, ec=edge, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK,
            linespacing=1.35)


def _arrow(ax, a, b, dashed=False, color=INK2, rad=0.0):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                ls="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=2, shrinkB=2))


def pipeline_diagram():
    """How the two halves of the project fit together, with profit linking them."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 6)
    ax.axis("off")
    top, mid, bot = 4.6, 3.0, 1.4
    _box(ax, 1.25, mid, "Purchase lines\n817,741 lines\nNov 2000 - Feb 2001", SHARED)
    _box(ax, 3.85, mid, "Baskets\none customer, one day\nsubclass level", SHARED)
    _box(ax, 6.65, top, "FP-Growth\nfrequent itemsets\n& association rules", LANE_FIM)
    _box(ax, 9.45, top, "Basket-adjusted profit\nown margin + halo\n(size-adjusted)", LANE_FIM)
    _box(ax, 12.25, top, "Quadrants, pull-through,\nbundles, delisting\nbreak-even", LANE_FIM)
    _box(ax, 6.65, bot, "Implicit ratings\nlog(1 + days bought)", LANE_CF)
    _box(ax, 9.45, bot, "Recommenders\npopularity, repeat-buy,\nitem-kNN, ALS, rules", LANE_CF)
    _box(ax, 12.25, bot, "Profit-aware re-ranking\n(1-λ)·relevance + λ·BAP", LANE_CF)
    _box(ax, 15.0, mid, "Action list\nkeep / drop /\nbundle / offers", FINAL, w=2.1)
    # main flow
    _arrow(ax, (2.4, mid), (2.7, mid))
    _arrow(ax, (5.0, mid + 0.3), (5.5, top - 0.2))
    _arrow(ax, (5.0, mid - 0.3), (5.5, bot + 0.2))
    for y in (top, bot):
        _arrow(ax, (7.8, y), (8.3, y))
        _arrow(ax, (10.6, y), (11.1, y))
    _arrow(ax, (13.4, top - 0.2), (13.95, mid + 0.3))
    _arrow(ax, (13.4, bot + 0.2), (13.95, mid - 0.3))
    # the links that make it one project
    _arrow(ax, (9.95, top - 0.6), (11.75, bot + 0.6), dashed=True, color=ORANGE)
    ax.text(11.05, mid + 0.12, "BAP as the\nvalue signal", fontsize=8,
            color=ORANGE, ha="left", va="center")
    _arrow(ax, (7.15, top - 0.6), (8.95, bot + 0.6), dashed=True, color=ORANGE)
    ax.text(8.25, mid + 0.12, "rules as a\nrecommender", fontsize=8,
            color=ORANGE, ha="left", va="center")
    ax.text(6.65, 5.55, "Frequent itemset mining", color=ORANGE,
            fontsize=10, fontweight="bold", ha="center")
    ax.text(6.65, 0.45, "Collaborative filtering", color=BLUE,
            fontsize=10, fontweight="bold", ha="center")
    ax.text(12.25, 0.45, "evaluated on actual February purchases",
            color=INK2, fontsize=8.5, ha="center", style="italic")
    ax.set_title("Project pipeline: profit runs through both halves", loc="left")
    fig.tight_layout()
    return _save(fig, "00a_pipeline")


def timeline_diagram():
    """Which months train, tune and test each part of the analysis."""
    import matplotlib.dates as mdates
    d = pd.Timestamp
    rows = [
        ("Store insights\n(FIM, BAP)", [(d("2000-11-01"), d("2001-02-01"), AQUA, "Nov - Jan")]),
        ("BAP persistence\ncheck", [(d("2000-11-01"), d("2001-02-01"), AQUA, "BAP"),
                                    (d("2001-02-01"), d("2001-03-01"), YELLOW, "BAP again")]),
        ("Validation\n(tune k, d, λ)", [(d("2000-11-01"), d("2001-01-01"), BLUE, "train Nov - Dec"),
                                       (d("2001-01-01"), d("2001-02-01"), ORANGE, "validate Jan")]),
        ("CNY check", [(d("2001-01-01"), d("2001-01-14"), ORANGE, "1-13 Jan"),
                       (d("2001-01-14"), d("2001-02-01"), RED, "14-31 Jan")]),
        ("Final test", [(d("2000-11-01"), d("2001-02-01"), BLUE, "train Nov - Jan"),
                        (d("2001-02-01"), d("2001-03-01"), ORANGE, "test Feb")]),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    for k, (label, bars) in enumerate(rows[::-1]):
        for start, end, color, text in bars:
            left = mdates.date2num(start)
            width = mdates.date2num(end) - left
            ax.barh(k, width, left=left, height=0.62, color=color,
                    edgecolor="white", linewidth=2)
            ax.text(left + width / 2, k, text, ha="center", va="center",
                    color="white", fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.grid(False)
    ax.axvline(mdates.date2num(CNY_2001), color=INK, ls="--", lw=1, zorder=0)
    ax.text(mdates.date2num(CNY_2001), len(rows) - 0.45, " CNY 24 Jan",
            fontsize=8.5, color=INK, va="bottom")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(mdates.date2num(d("2000-11-01")), mdates.date2num(d("2001-03-01")))
    ax.set_title("Time splits: nothing is tuned on the month it is tested on",
                 loc="left")
    fig.tight_layout()
    return _save(fig, "00b_timeline")


# --------------------------------------------------------------------
# figures for the extra analyses
# --------------------------------------------------------------------
def _binned_median(x, y, bins=10):
    q = pd.qcut(x, bins, duplicates="drop")
    g = pd.DataFrame({"x": x, "y": y}).groupby(q, observed=True).median()
    return g["x"], g["y"]


def size_bias(bap, summary):
    """Naive vs size-adjusted halo against average basket size."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    panels = [("naive_halo", "Naive halo (vs average basket)",
               summary.iloc[0]),
              ("halo", "Size-adjusted halo (vs same-size baskets)",
               summary.iloc[1])]
    for ax, (col, title, rho) in zip(axes, panels):
        lo, hi = np.nanpercentile(bap[col], [1, 99])
        ax.scatter(bap["avg_size"], bap[col].clip(lo, hi), s=10, color=BLUE,
                   alpha=0.35, linewidths=0)
        bx, by = _binned_median(bap["avg_size"], bap[col].clip(lo, hi))
        ax.plot(bx, by, color=INK, lw=2, label="binned median")
        ax.axhline(0, color=INK2, lw=1, ls="--")
        ax.set_title(f"{title}\nSpearman ρ with basket size = {rho:.2f}",
                     fontsize=11)
        ax.set_xlabel("Average size of the subclass's baskets")
        ax.set_ylabel("NT$ per basket")
        ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "03b_size_bias")


def pull_network(pull, bap, top_leaders=8, per_leader=3):
    """Bipartite map: loss leaders (left) and the profitable items they pull in (right)."""
    if pull.empty:
        return None
    lead_order = (pull.groupby("leader")["extra_profit"].sum()
                  .nlargest(top_leaders).index)
    p = (pull[pull["leader"].isin(lead_order)]
         .groupby("leader").head(per_leader))
    right = (p.groupby("consequent")["extra_profit"].sum()
             .sort_values(ascending=False).index)
    yl = {s: i for i, s in enumerate(lead_order[::-1])}
    yr = {s: i * (len(lead_order) - 1) / max(len(right) - 1, 1)
          for i, s in enumerate(right[::-1])}
    fig, ax = plt.subplots(figsize=(10, 0.55 * max(len(lead_order), len(right)) + 1.8))
    wmax = p["extra_profit"].max()
    for _, r in p.iterrows():
        ax.plot([0, 1], [yl[r["leader"]], yr[r["consequent"]]],
                color=BLUE, alpha=0.35,
                lw=0.8 + 5 * r["extra_profit"] / wmax, solid_capstyle="round")
    for s, y in yl.items():
        ax.scatter(0, y, s=90, marker="D", color=ORANGE, zorder=3,
                   edgecolors="white")
        b = bap.loc[s]
        ax.text(-0.04, y, f"{s}   own {b['own_margin']:+.1f}, halo {b['halo']:+.1f}",
                ha="right", va="center", fontsize=8.5, color=INK)
    for s, y in yr.items():
        q = bap.loc[s, "quadrant"]
        color = QUAD_STYLE.get(q, (GRAY, "o"))[0]
        ax.scatter(1, y, s=70, color=color, zorder=3, edgecolors="white")
        ax.text(1.04, y, f"{s}   {q.lower()}, margin {bap.loc[s, 'own_margin']:+.1f}",
                ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(-0.8, len(lead_order) - 0.2)
    ax.axis("off")
    ax.text(0, len(lead_order) - 0.35, "Loss leader", ha="center",
            fontweight="bold", color=ORANGE)
    ax.text(1, len(lead_order) - 0.35, "Profitable item pulled in", ha="center",
            fontweight="bold", color=BLUE)
    ax.set_title("Where the halo comes from (line width = extra profit per leader basket)",
                 loc="left")
    fig.tight_layout()
    return _save(fig, "04b_pull_through")


def delisting(curves, crossover):
    """Net three-month profit of two delisting rules vs the causal share of the halo."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    styles = {0: (VIOLET, "s"), 1: (ORANGE, "o")}
    for k, (rule, d) in enumerate(curves.groupby("rule", sort=False)):
        color, marker = styles[k % 2]
        ax.plot(d["causal_share"], d["net_profit"] / 1e3, color=color,
                marker=marker, ms=4, markevery=4,
                label=f"{rule} ({d['subclasses'].iloc[0]} subclasses)")
    ax.axhline(0, color=INK2, lw=1)
    if crossover is not None:
        ax.axvline(crossover, color=INK, lw=1, ls="--")
        ax.text(crossover, ax.get_ylim()[1], f"  break-even at {crossover:.0%}",
                va="top", fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of the halo actually caused by the item (unknown, so swept)")
    ax.set_ylabel("Net profit change, Nov-Jan\n(NT$ thousands)")
    ax.set_title("What would cutting loss-makers really save?", loc="left")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "04c_delisting")


MODEL_COLORS = [("Popularity", GRAY), ("Repeat-buy", VIOLET),
                ("Item-kNN", BLUE), ("Implicit ALS", ORANGE),
                ("FP-Growth rules", AQUA)]


def _model_color(name):
    return next((c for prefix, c in MODEL_COLORS if name.startswith(prefix)), INK2)


def activity(act):
    """Mean NDCG@10 by customer activity, per model and mode."""
    modes = [("all", "All items"), ("new", "Items new to the customer")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (mode, title) in zip(axes, modes):
        d = act[act["mode"] == mode]
        for name, g in d.groupby("model", sort=False):
            g = g.sort_values("activity")
            ax.plot(g["activity"].astype(str), g["ndcg@10"],
                    color=_model_color(name), marker="o", ms=5, label=name)
        n = d.drop_duplicates("activity").sort_values("activity")
        ax.set_xticks(range(len(n)),
                      [f"{a}\n(n={c:,})" for a, c in zip(n["activity"], n["customers"])])
        ax.set_xlabel("Training baskets per customer")
        ax.set_ylabel("NDCG@10")
        ax.set_title(title)
    # the left panel has every model (repeat-buy is skipped on the right)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, "05b_activity")
