"""
LOGISTICS WORKSTREAM - Step 4: Charts
=====================================
Nine explanatory figures. Each answers one question and carries its finding in
the title, so the chart can be read without the surrounding text.

Run: source venv/bin/activate && python notebooks/06_logistics_charts.py
Figures land in output/figures/.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import vizstyle as vz  # noqa: E402

vz.apply_style()
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

lo = pd.read_csv(CLEAN / "logistics_orders.csv", low_memory=False,
                 parse_dates=["order_purchase_timestamp",
                              "order_delivered_customer_date"])
it = pd.read_csv(CLEAN / "logistics_items.csv", low_memory=False)
d = lo[lo.is_analysable_delivery].copy()
di = it[it.order_id.isin(d.order_id)].copy()
print(f"plotting from {len(d):,} delivered orders / {len(di):,} line items")

saved = []


def save(fig, name):
    path = FIG / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(name)
    print(f"  saved {path.relative_to(ROOT)}")


# ===========================================================================
# FIG 1 - Where the time goes. One stacked bar: the whole journey, split by leg.
# Form: a part-to-whole of ONE total -> a single stacked bar, not a pie.
# ===========================================================================
# Payment approval is a rounding error (8 minutes at the median), so it is
# stated in the subtitle rather than given an unreadable sliver of bar.
rows = []
for label, mask in [("A typical on-time order", ~d.is_late),
                    ("A late order", d.is_late)]:
    g = d[mask]
    rows.append((label,
                 g.seller_handover_days.median(),
                 g.carrier_transit_days.median()))

fig, ax = plt.subplots(figsize=(11, 3.6))
seg_names = ["Seller handover", "Carrier transit"]
seg_cols = [vz.ORANGE, vz.BLUE]
for y, (label, handover, transit) in enumerate(rows):
    left = 0
    for val, c in zip([handover, transit], seg_cols):
        ax.barh([y], [val], left=left, height=0.52, color=c,
                edgecolor=vz.SURFACE, linewidth=2, zorder=3)
        ax.text(left + val / 2, y, f"{val:.1f} d", ha="center", va="center",
                color="white", fontweight="bold", fontsize=11, zorder=4)
        left += val
    ax.text(left + 0.5, y, f"{left:.1f} days total", va="center",
            fontsize=10.5, color=vz.INK, fontweight="bold")

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=vz.INK)
ax.invert_yaxis()
ax.set_xlim(0, 33)
ax.set_xlabel("Days from payment approval to the customer's door")
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=vz.GRID, linewidth=0.8)
ax.grid(axis="y", visible=False)
ax.tick_params(length=0)
# Legend is always present for >= 2 series; segments are also value-labelled.
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in seg_cols]
ax.legend(handles, seg_names, loc="center right", ncol=1, fontsize=10.5,
          bbox_to_anchor=(1.0, 0.55))
vz.title_block(
    ax, "When an order runs late, almost all the extra time is carrier transit",
    "Seller handover adds 1.3 extra days on a late order. The carrier leg adds 17.")
vz.source_note(fig, "n = 96,470 delivered orders. Medians within each group. "
                    "Payment approval omitted: 8 minutes at the median.")
save(fig, "01_journey_decomposition")


# ===========================================================================
# FIG 2 - THE headline chart. Satisfaction against delivery-vs-promise.
# Form: polarity around zero (early vs late) -> DIVERGING blue<->red,
# neutral gray at the on-time midpoint.
# ===========================================================================
r = d.dropna(subset=["review_score"]).copy()
bands = ["20+ d\nearly", "10-20 d\nearly", "5-10 d\nearly", "0-5 d\nearly",
         "1-3 d\nlate", "4-7 d\nlate", "8-15 d\nlate", "15+ d\nlate"]
r["band"] = pd.cut(r.delay_days, [-np.inf, -20, -10, -5, 0, 3, 7, 15, np.inf],
                   labels=bands)
g = r.groupby("band", observed=True).agg(
    avg=("review_score", "mean"),
    one_star=("review_score", lambda s: (s == 1).mean() * 100),
    n=("review_score", "size"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
# Diverging fill: cool for early, warm for late, gray for the near-zero band.
cols = [vz.DIV_COOL] * 3 + ["#8fa8bd"] + [vz.SERIES[1], "#e0654a", vz.CRITICAL, "#a82c2c"]

ax = axes[0]
ax.bar(range(len(g)), g.avg, color=cols, width=0.72,
       edgecolor=vz.SURFACE, linewidth=2)
for i, v in enumerate(g.avg):
    ax.text(i, v + 0.08, f"{v:.2f}", ha="center", color=vz.INK, fontsize=9.5,
            fontweight="bold")
ax.axvline(3.5, color=vz.AXIS, linewidth=1.2)
ax.text(3.45, 4.9, "promised date", ha="right", color=vz.MUTED, fontsize=9)
ax.set_xticks(range(len(g)))
ax.set_xticklabels(g.index, fontsize=8.8)
ax.set_ylabel("Average review score (stars)")
ax.set_ylim(0, 5.1)
vz.clean_axes(ax)
vz.title_block(ax, "One day late costs half a star.\nA week late costs two and a half.",
               "Average review score by delivery vs. promised date")

ax = axes[1]
ax.bar(range(len(g)), g.one_star, color=cols, width=0.72,
       edgecolor=vz.SURFACE, linewidth=2)
for i, v in enumerate(g.one_star):
    ax.text(i, v + 1.4, f"{v:.0f}%", ha="center", color=vz.INK, fontsize=9.5,
            fontweight="bold")
ax.axvline(3.5, color=vz.AXIS, linewidth=1.2)
ax.set_xticks(range(len(g)))
ax.set_xticklabels(g.index, fontsize=8.8)
ax.set_ylabel("Share of orders rated 1 star (%)")
ax.set_ylim(0, 80)
vz.clean_axes(ax)
vz.title_block(ax, "Past a week late, seven in ten customers\nleave the worst possible review",
               "Share of orders receiving 1 star")
fig.tight_layout(w_pad=3)
vz.source_note(fig, "n = 95,824 delivered orders carrying a review. "
                    "Bands are actual delivery date minus promised date.")
save(fig, "02_satisfaction_cliff")


# ===========================================================================
# FIG 3 - The promise is sandbagged. Distribution of days early/late.
# Form: one distribution with a meaningful zero -> histogram, diverging fill.
# ===========================================================================
fig, ax = plt.subplots(figsize=(11, 4.8))
vals = d.delay_days.clip(-45, 45)
bins = np.arange(-45, 46, 1.5)
n, edges, patches = ax.hist(vals, bins=bins, color=vz.DIV_COOL, linewidth=0)
for patch, lo_edge in zip(patches, edges[:-1]):
    patch.set_facecolor(vz.CRITICAL if lo_edge >= 0 else vz.DIV_COOL)
ax.axvline(0, color=vz.INK, linewidth=1.4)
ax.set_xlabel("Days early (left) / late (right) vs. the promised delivery date")
ax.set_ylabel("Orders")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
med = d.delay_days.median()
ax.annotate(f"median order arrives\n{abs(med):.0f} days EARLY",
            xy=(med, 3000), xytext=(-38, 5200), fontsize=10, color=vz.INK,
            arrowprops=dict(arrowstyle="->", color=vz.MUTED, lw=1.2))
ax.annotate(f"{d.is_late.mean() * 100:.1f}% arrive late,\n"
            "and this tail alone produces\n38% of all 1-star reviews",
            xy=(8, 800), xytext=(13, 5600), fontsize=10, color=vz.CRITICAL,
            ha="left", arrowprops=dict(arrowstyle="->", color=vz.CRITICAL, lw=1.2))
vz.clean_axes(ax)
vz.title_block(ax, "The delivery promise is padded by 12 days — and still misses 8% of the time",
               "Distribution of delivery date vs. promised date. Clipped at +/-45 days for display.")
vz.source_note(fig, "n = 96,470 delivered orders. Median promise 23.2 days; "
                    "median actual delivery 10.2 days.")
save(fig, "03_promise_vs_reality")


# ===========================================================================
# FIG 4 - Two panels: distance drives TIME, but weight drives FREIGHT.
# Form: two different measures -> two panels, never a dual axis.
# ===========================================================================
dist_order = ["0-50 km", "50-150", "150-400", "400-800", "800-1500", "1500+ km"]
d["dist_band"] = pd.cut(d.max_distance_km, [-1, 50, 150, 400, 800, 1500, 1e5],
                        labels=dist_order)
di["dist_band"] = pd.cut(di.distance_km, [-1, 50, 150, 400, 800, 1500, 1e5],
                         labels=dist_order)
wt_order = ["<0.5 kg", "0.5-1", "1-2", "2-5", "5-10", "10+ kg"]
di["wt_band"] = pd.cut(di.billable_weight_kg, [-.001, .5, 1, 2, 5, 10, 1e4],
                       labels=wt_order)

# Indexing every measure to the shortest band makes them unitless multiples,
# so three different quantities can honestly share one axis - and the gap
# between the lines IS the cross-subsidy.
g_d = di.groupby("dist_band", observed=True)
idx = pd.DataFrame({
    "Distance travelled": g_d.distance_km.median(),
    "Freight charged": g_d.freight_value.median(),
})
idx["Delivery time"] = d.groupby("dist_band", observed=True).total_delivery_days.median()
idx = idx / idx.iloc[0]
f_by_wt = di.groupby("wt_band", observed=True).freight_value.median()

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
# End labels are staggered vertically so the freight and time lines - which
# finish close together - do not collide.
for name, colr, voff in [("Distance travelled", vz.AQUA, 1.35),
                         ("Freight charged", vz.ORANGE, 0.72),
                         ("Delivery time", vz.BLUE, 1.30)]:
    ax.plot(range(len(idx)), idx[name].values, color=colr, linewidth=2,
            marker="o", markersize=8, markeredgecolor=vz.SURFACE,
            markeredgewidth=2, label=name)
    ax.text(len(idx) - 0.82, idx[name].iloc[-1] * voff,
            f"{idx[name].iloc[-1]:.0f}x", color=colr, fontweight="bold",
            ha="left", va="center", fontsize=11.5)
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
ax.set_yticklabels(["1x", "2x", "5x", "10x", "20x", "50x", "100x"])
ax.set_xticks(range(len(dist_order)))
ax.set_xticklabels(dist_order, fontsize=9)
ax.set_xlim(-0.35, len(idx) - 0.15)
ax.set_xlabel("Seller-to-customer distance")
ax.set_ylabel("Multiple of the 0–50 km band (log scale)")
ax.set_ylim(0.8, 260)
ax.legend(loc="upper left", fontsize=9.5)
vz.clean_axes(ax)
dist_mult = idx["Distance travelled"].iloc[-1]
frt_mult = idx["Freight charged"].iloc[-1]
vz.title_block(ax, f"A {dist_mult:.0f}x longer trip costs the customer {frt_mult:.0f}x more",
               "Every measure indexed to the 0–50 km band = 1x. Log scale.")

ax = axes[1]
ax.bar(range(len(f_by_wt)), f_by_wt.values, color=vz.ORANGE, width=0.7,
       edgecolor=vz.SURFACE, linewidth=2)
for i, v in enumerate(f_by_wt.values):
    ax.text(i, v + 0.7, f"R$ {v:.0f}", ha="center", color=vz.INK, fontsize=9.5,
            fontweight="bold")
ax.set_xticks(range(len(wt_order)))
ax.set_xticklabels(wt_order, fontsize=9)
ax.set_xlabel("Billable parcel weight")
ax.set_ylabel("Median freight charged (R$)")
ax.set_ylim(0, max(f_by_wt.values) * 1.2)
vz.clean_axes(ax)
vz.title_block(ax, "Because freight is priced on weight, not on the trip",
               "Freight vs weight r = 0.61; freight vs distance r = 0.39.")
fig.tight_layout(w_pad=3)
vz.source_note(fig, "Left: n = 110,189 line items (distance, freight) and 96,470 orders "
                    "(delivery time). Right: n = 110,189 line items. "
                    "Billable weight = max(actual, volumetric).")
save(fig, "04_distance_vs_weight")


# ===========================================================================
# FIG 5 - Route matrix. Region -> region median delivery days.
# Form: continuous magnitude on a 2-D grid -> heatmap, SEQUENTIAL one-hue ramp.
# ===========================================================================
cols = ["Southeast", "South", "Central-West", "Northeast", "North"]
# The North is dropped as a SELLER row: it originates 0.02% of items, so every
# cell would be suppressed anyway. That absence is itself the story, and it is
# stated in the source note rather than shown as an empty row.
rows_r = ["Southeast", "South", "Central-West", "Northeast"]
mat = (d.dropna(subset=["seller_region", "customer_region"])
       .pivot_table(index="seller_region", columns="customer_region",
                    values="total_delivery_days", aggfunc="median")
       .reindex(index=rows_r, columns=cols))
cnt = (d.dropna(subset=["seller_region", "customer_region"])
       .pivot_table(index="seller_region", columns="customer_region",
                    values="order_id", aggfunc="size")
       .reindex(index=rows_r, columns=cols))

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seqblue", vz.SEQ_BLUE)

# Suppress cells built on too few orders - a "33 day" median from 4 orders is
# noise, and colouring it the darkest blue would make it the loudest cell here.
MIN_N = 30
mat = mat.mask(cnt < MIN_N)

fig, ax = plt.subplots(figsize=(8.6, 5.6))
im = ax.imshow(mat.values, cmap=cmap, aspect="auto", vmin=6, vmax=24)
for i in range(len(rows_r)):
    for j in range(len(cols)):
        v, c = mat.values[i, j], cnt.values[i, j]
        if np.isnan(v):
            note = "too few\norders" if (not np.isnan(c) and c < MIN_N) else "–"
            ax.text(j, i, note, ha="center", va="center", fontsize=8.5,
                    color=vz.MUTED)
            continue
        # Ink flips to white on the dark end of the ramp so labels stay legible;
        # the count line uses the same flip rather than a muted gray.
        on_dark = v > 13
        ax.text(j, i - 0.04, f"{v:.0f}d", ha="center", va="center", fontsize=13,
                fontweight="bold", color="white" if on_dark else vz.INK)
        ax.text(j, i + 0.26, f"{int(c):,} orders", ha="center", va="center",
                fontsize=8.5, color="#e8eef6" if on_dark else vz.INK_2)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, fontsize=9.5)
ax.set_yticks(range(len(rows_r)))
ax.set_yticklabels(rows_r, fontsize=9.5)
ax.set_xlabel("Customer region  (where it ships TO)")
ax.set_ylabel("Seller region  (where it ships FROM)")
ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(rows_r), 1), minor=True)
ax.grid(which="minor", color=vz.SURFACE, linewidth=2)
ax.grid(which="major", visible=False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
cb.set_label("Median delivery days", color=vz.INK_2, fontsize=9.5)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelsize=9, labelcolor=vz.INK_2)
se_share = (di.seller_region == "Southeast").mean() * 100
vz.title_block(ax, "Shipping stays fast only inside the Southeast —\nand that is where everything ships from",
               "Median delivery days by seller region -> customer region")
vz.source_note(fig, f"n = 96,470 delivered orders. {se_share:.0f}% of all line items are sold by a "
                    f"Southeast seller; the North sells almost nothing, so it has no row. "
                    f"Cells under {MIN_N} orders are suppressed.")
save(fig, "05_route_matrix")


# ===========================================================================
# FIG 6 - The double penalty: slowest states also pay the most freight.
# Form: relationship between two continuous measures -> scatter.
# One colour (identity is carried by the direct labels, not by hue).
# ===========================================================================
st = (d.groupby("customer_state")
      .agg(orders=("order_id", "size"),
           days=("total_delivery_days", "median"),
           late=("is_late", "mean"),
           review=("review_score", "mean"),
           freight_pct=("freight_ratio", "median")))
st = st[st.orders >= 100]
fig, ax = plt.subplots(figsize=(11, 6.4))
sizes = np.sqrt(st.orders) * 3.6
ax.scatter(st.days, st.freight_pct * 100, s=sizes, color=vz.BLUE, alpha=0.55,
           edgecolor=vz.SURFACE, linewidth=2, zorder=3)
# Direct-label selectively: the extremes and the high-volume states only.
label_these = set(st.nlargest(7, "days").index) | set(st.nsmallest(4, "days").index) \
    | set(st.nlargest(5, "orders").index)
for s_, row in st.iterrows():
    if s_ in label_these:
        # Offset scales with the bubble so the label clears it, but is capped
        # so São Paulo's label does not float off on its own.
        off = min(np.sqrt(row.orders) * 0.16 + 8, 20)
        ax.annotate(s_, (row.days, row.freight_pct * 100),
                    xytext=(0, off), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=vz.INK)
ax.set_xlabel("Median delivery time (days)")
ax.set_ylabel("Median freight as a share of item price (%)")
ax.set_xlim(5, 29)
ax.set_ylim(16, 42)
ax.annotate("bubble size = order volume", xy=(0.985, 0.03),
            xycoords="axes fraction", ha="right", fontsize=9, color=vz.MUTED)
ax.annotate("Worst served:\nslow AND expensive",
            xy=(0.985, 0.60), xycoords="axes fraction", ha="right",
            fontsize=10.5, color=vz.CRITICAL, fontweight="bold",
            linespacing=1.35)
ax.annotate("São Paulo: fast and cheap\n(42% of all orders)",
            xy=(st.loc["SP", "days"], st.loc["SP", "freight_pct"] * 100),
            xytext=(9.2, 17.6), fontsize=9.5, color=vz.INK_2,
            arrowprops=dict(arrowstyle="->", color=vz.MUTED, lw=1.1))
vz.clean_axes(ax, grid_axis="both")
vz.title_block(ax, "The states that wait longest also pay the most to ship",
               "Each bubble is a state with 100+ orders. Up and to the right is worse on both counts.")
vz.source_note(fig, "n = 96,470 delivered orders across 25 states meeting the 100-order threshold.")
save(fig, "06_state_double_penalty")


# ===========================================================================
# FIG 7 - Did performance hold as volume grew? Two panels, shared x, no dual axis.
# ===========================================================================
mth = (d[d.purchase_month.between("2017-01", "2018-08")]
       .groupby("purchase_month")
       .agg(orders=("order_id", "size"),
            promise=("estimated_days", "median"),
            actual=("total_delivery_days", "median"),
            late=("is_late", "mean")))
mth["buffer"] = mth.promise - mth.actual
x = list(range(len(mth)))

fig, axes = plt.subplots(2, 1, figsize=(12.5, 8), sharex=True,
                         gridspec_kw={"height_ratios": [1.25, 1]})

# Panel A: both series are measured in DAYS, so they legitimately share one
# axis. The shaded gap between them IS the safety margin.
ax = axes[0]
ax.fill_between(x, mth.actual, mth.promise, color=vz.BLUE, alpha=0.10, zorder=1)
ax.plot(x, mth.promise, color=vz.BLUE, linewidth=2.2, marker="o", markersize=5.5,
        markeredgecolor=vz.SURFACE, markeredgewidth=1.5, zorder=3,
        label="Promised delivery window")
ax.plot(x, mth.actual, color=vz.ORANGE, linewidth=2.2, marker="o", markersize=5.5,
        markeredgecolor=vz.SURFACE, markeredgewidth=1.5, zorder=3,
        label="Actual delivery time")
ax.text(0.4, 41, "Promised", color=vz.BLUE, fontweight="bold", fontsize=10.5)
ax.text(0.4, 6.4, "Actual", color=vz.ORANGE, fontweight="bold", fontsize=10.5)
ax.annotate("safety margin:\n28 days", xy=(1.0, 24), xytext=(2.2, 30),
            fontsize=9.5, color=vz.INK_2,
            arrowprops=dict(arrowstyle="->", color=vz.MUTED, lw=1.1))
ax.annotate("margin cut to\n6 days", xy=(len(x) - 1.15, 10.2),
            xytext=(len(x) - 5.6, 27), fontsize=9.5, color=vz.INK_2,
            arrowprops=dict(arrowstyle="->", color=vz.MUTED, lw=1.1))
ax.set_ylabel("Days")
ax.set_ylim(0, 46)
ax.legend(loc="upper right", fontsize=9.5, ncol=2)
vz.clean_axes(ax)
vz.title_block(ax, "Delivery genuinely got faster — but the promise tightened faster still",
               "Median promised window vs. median actual delivery time, by month of purchase.")

# Panel B: the consequence.
ax = axes[1]
ax.plot(x, mth.late * 100, color=vz.CRITICAL, linewidth=2.2, marker="o",
        markersize=6, markeredgecolor=vz.SURFACE, markeredgewidth=1.6)
# The March peak is self-evident from the axis, so it gets no callout. The one
# annotation is the finding the chart exists to make.
last = len(x) - 1
ax.annotate("Aug 2018 was the FASTEST month on record (7.0 days)\n"
            "— and still 10% of its orders counted as 'late'",
            xy=(last, mth.late.iloc[-1] * 100), xytext=(9.4, 21.6),
            fontsize=10, color=vz.INK, fontweight="bold", linespacing=1.4,
            arrowprops=dict(arrowstyle="->", color=vz.MUTED, lw=1.2,
                            connectionstyle="arc3,rad=-0.18"))
ax.set_ylabel("Orders delivered late (%)")
ax.set_ylim(0, 26)
ax.set_xticks(x)
ax.set_xticklabels(mth.index, rotation=90, fontsize=8.5)
vz.clean_axes(ax)
vz.title_block(ax, "So 'lateness' now measures the promise, not the operation",
               "In 2018 the monthly late rate tracks the safety margin at r = -0.75.")
fig.tight_layout(h_pad=2.6)
vz.source_note(fig, "n = 96,203 delivered orders, Jan 2017 - Aug 2018. "
                    "2016 and Sep/Oct 2018 excluded as partial months. "
                    "Monthly order volume grew from 750 to ~6,400 over the period.")
save(fig, "07_promise_vs_operation")


# ===========================================================================
# FIG 8 - Blame concentration. Form: cumulative share -> Lorenz-style curve.
# ===========================================================================
pairs = di[["seller_id", "order_id"]].drop_duplicates()
ol = pairs.merge(d[["order_id", "is_late"]], on="order_id")
late_by_seller = ol[ol.is_late].seller_id.value_counts()
cum = (late_by_seller.cumsum() / late_by_seller.sum()).values
n_sellers = ol.seller_id.nunique()
share_sellers = np.arange(1, len(cum) + 1) / n_sellers * 100

fig, ax = plt.subplots(figsize=(9.6, 5.6))
ax.plot(share_sellers, cum * 100, color=vz.BLUE, linewidth=2.4)
ax.fill_between(share_sellers, 0, cum * 100, color=vz.BLUE, alpha=0.10)
ax.plot([0, 100], [0, 100], color=vz.AXIS, linewidth=1.4)
ax.text(66, 62, "if lateness were spread evenly", color=vz.MUTED, fontsize=9.5,
        rotation=31, ha="center")
for target, col in [(50, vz.SERIOUS), (80, vz.CRITICAL)]:
    idx = int(np.searchsorted(cum, target / 100))
    pct = share_sellers[idx]
    ax.plot([pct, pct], [0, target], color=col, linewidth=1.4)
    ax.plot([0, pct], [target, target], color=col, linewidth=1.4)
    ax.scatter([pct], [target], s=70, color=col, zorder=4,
               edgecolor=vz.SURFACE, linewidth=2)
    ax.annotate(f"{pct:.0f}% of sellers\ncause {target}% of late deliveries",
                xy=(pct, target), xytext=(pct + 7, target - 15),
                fontsize=10, color=col, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
ax.set_xlabel("Sellers, ranked by number of late deliveries (%)")
ax.set_ylabel("Cumulative share of all late deliveries (%)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
vz.clean_axes(ax, grid_axis="both")
vz.title_block(ax, "Lateness is not a marketplace-wide problem — it is a short list",
               "Cumulative share of late deliveries by seller, worst first.")
vz.source_note(fig, f"n = {int(ol.is_late.sum()):,} late deliveries across "
                    f"{n_sellers:,} sellers.")
save(fig, "08_seller_concentration")


# ===========================================================================
# FIG 9 - Seller scorecard: who to intervene on. Form: scatter, status colour
# reserved for the actual risk call (with labels, never colour alone).
# ===========================================================================
sc = di.groupby("seller_id").agg(
    orders=("order_id", "nunique"),
    sla_miss=("seller_missed_sla", "mean"),
    state=("seller_state", "first"))
sc2 = ol.groupby("seller_id").agg(late=("is_late", "mean"))
rev = (pairs.merge(d[["order_id", "review_score"]], on="order_id")
       .groupby("seller_id").review_score.mean())
sc = sc.join(sc2).join(rev.rename("review"))
big = sc[sc.orders >= 50].dropna(subset=["review"])

# The risk call: high SLA-miss AND high late rate. Colour is the CONCLUSION,
# so a reserved status colour is correct here - and the zone is also labelled.
risk = (big.sla_miss > 0.25) & (big.late > 0.10)
fig, ax = plt.subplots(figsize=(11, 6.4))
ax.scatter(big.loc[~risk, "sla_miss"] * 100, big.loc[~risk, "late"] * 100,
           s=np.sqrt(big.loc[~risk, "orders"]) * 7, color=vz.BLUE, alpha=0.42,
           edgecolor=vz.SURFACE, linewidth=1.6, zorder=3, label="Within tolerance")
ax.scatter(big.loc[risk, "sla_miss"] * 100, big.loc[risk, "late"] * 100,
           s=np.sqrt(big.loc[risk, "orders"]) * 7, color=vz.CRITICAL, alpha=0.75,
           edgecolor=vz.SURFACE, linewidth=1.6, zorder=4,
           label=f"Intervention list ({risk.sum()} sellers)")
ax.axvline(25, color=vz.AXIS, linewidth=1.2)
ax.axhline(10, color=vz.AXIS, linewidth=1.2)
ax.text(64, 10.6, "10% late", ha="right", fontsize=9, color=vz.MUTED)
ax.text(25.8, 33.5, "25% of parcels handed\nover past deadline",
        fontsize=9, color=vz.MUTED)
ax.set_xlabel("Share of parcels handed to the carrier AFTER the deadline (%)")
ax.set_ylabel("Share of that seller's orders that arrived late (%)")
ax.set_xlim(-3, 66)   # no seller sits past 60%, so the empty right is trimmed
ax.set_ylim(-1.5, 35)
ax.legend(loc="upper left", fontsize=9.5)
ax.annotate("bubble size = order volume", xy=(0.985, 0.035),
            xycoords="axes fraction", ha="right", fontsize=9, color=vz.MUTED)
vz.clean_axes(ax, grid_axis="both")
vz.title_block(ax, "Sellers who miss their handover deadline are the ones who deliver late",
               f"Each bubble is a seller with 50+ orders (n = {len(big)}). "
               "Correlation between the two axes: r = 0.49.")
vz.source_note(fig, "The 425 sellers shown carry 76% of all marketplace orders.")
save(fig, "09_seller_scorecard")

print(f"\n{len(saved)} figures written to output/figures/")
