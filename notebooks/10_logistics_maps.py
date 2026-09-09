"""
LOGISTICS WORKSTREAM - Step 6: Choropleth maps
==============================================
Two state-level choropleths of Brazil: delivery time and freight burden.

Boundary data: data/raw/geo/br_states.geojson - public Brazilian state
boundaries (27 features, IBGE-derived, `SIGLA` = the 2-letter state code that
matches customer_state). Downloaded once from
github.com/giuliano-macedo/geodata-br-states and committed to the repo so the
pipeline reproduces offline.

THE CHOROPLETH TRAP, AND HOW THIS HANDLES IT
--------------------------------------------
A choropleth encodes value as area, so it lies about importance: Amazonas is
the size of Western Europe and carries 145 orders, while São Paulo is visually
tiny and carries 40,494 - 42% of the marketplace. Colouring the map alone would
tell leadership the North IS the problem, when it is ~2% of volume.
So each map carries proportional circles sized by order volume on top of the
fill. Colour says "how bad", circle size says "how much it matters".

Run: source venv/bin/activate && python notebooks/10_logistics_maps.py
"""
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import vizstyle as vz  # noqa: E402

vz.apply_style()
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "output" / "figures"
GEO = ROOT / "data" / "raw" / "geo" / "br_states.geojson"

# Second sequential context takes the next categorical slot's hue (orange),
# built as its own single-hue light -> dark ramp.
SEQ_ORANGE = ["#fdeade", "#fbd8c3", "#f8c3a4", "#f5ad85", "#f29768", "#ee814d",
              "#e96c35", "#d55a28", "#b9491f", "#9b3a18", "#7d2c11"]

CMAP_BLUE = LinearSegmentedColormap.from_list("seq_blue", vz.SEQ_BLUE)
CMAP_ORANGE = LinearSegmentedColormap.from_list("seq_orange", SEQ_ORANGE)

lo = pd.read_csv(CLEAN / "logistics_orders.csv", low_memory=False)
d = lo[lo.is_analysable_delivery]

st = (d.groupby("customer_state")
      .agg(orders=("order_id", "size"),
           days=("total_delivery_days", "median"),
           late=("is_late", "mean"),
           freight_pct=("freight_ratio", "median"),
           review=("review_score", "mean")))
st["late"] *= 100
st["freight_pct"] *= 100
print(f"states with data: {len(st)}  |  orders: {st.orders.sum():,}")


# ---------------------------------------------------------------------------
def load_states(simplify_to=600):
    """
    Parse the geojson into {state_code: [ring arrays]}.

    Rings are decimated to at most `simplify_to` vertices. At the ~150 dpi this
    figure renders at, a 3,000-point coastline and a 600-point one are visually
    identical, and the decimation keeps the draw fast. The first and last
    vertices are always retained so rings stay closed.
    """
    gj = json.loads(GEO.read_text())
    out, kept, total = {}, 0, 0
    for feat in gj["features"]:
        code = feat["properties"]["SIGLA"]
        rings = []
        for poly in feat["geometry"]["coordinates"]:
            for ring in poly:
                arr = np.asarray(ring, dtype=float)
                total += len(arr)
                if len(arr) > simplify_to:
                    step = int(np.ceil(len(arr) / simplify_to))
                    arr = np.vstack([arr[::step], arr[-1]])
                kept += len(arr)
                if len(arr) >= 3:
                    rings.append(arr)
        out[code] = rings
    print(f"boundary vertices: {total:,} -> {kept:,} after decimation")
    return out


STATES = load_states()

# Centroid of each state's largest ring - where the volume circle and label go.
CENTROID = {}
for code, rings in STATES.items():
    big = max(rings, key=len)
    CENTROID[code] = (big[:, 0].mean(), big[:, 1].mean())
# The geometric mean of a ring is not always inside a concave state; nudge the
# few that land badly so their circle and label sit on the landmass.
CENTROID["PA"] = (-52.5, -4.5)
CENTROID["BA"] = (-41.7, -12.5)
CENTROID["MA"] = (-45.3, -5.2)
CENTROID["AM"] = (-64.5, -4.5)


def draw_panel(ax, values, cmap, vmin, vmax, cbar_label, title, subtitle,
               fmt="{:.0f}", label_states=()):
    patches, colors = [], []
    norm = Normalize(vmin=vmin, vmax=vmax)
    for code, rings in STATES.items():
        val = values.get(code, np.nan)
        for ring in rings:
            patches.append(MplPolygon(ring, closed=True))
            colors.append(cmap(norm(val)) if not np.isnan(val) else "#eceae4")
    # 0.6pt surface-coloured edge separates neighbours without drawing a border
    # around the data itself.
    pc = PatchCollection(patches, facecolor=colors, edgecolor=vz.SURFACE,
                         linewidths=0.6, zorder=2)
    ax.add_collection(pc)

    # Proportional circles = order volume. This is the correction for the
    # area-distortion trap described in the module docstring.
    for code, row in st.iterrows():
        if code not in CENTROID:
            continue
        x, y = CENTROID[code]
        ax.scatter(x, y, s=np.sqrt(row.orders) * 1.45, facecolor="none",
                   edgecolor=vz.INK, linewidth=1.1, alpha=.62, zorder=4)

    for code in label_states:
        if code in CENTROID and code in values:
            x, y = CENTROID[code]
            ax.annotate(f"{code}\n{fmt.format(values[code])}", (x, y),
                        xytext=(0, -1), textcoords="offset points",
                        ha="center", va="center", fontsize=8.2,
                        fontweight="bold", color=vz.INK, zorder=6,
                        linespacing=1.25,
                        bbox=dict(boxstyle="round,pad=0.18", fc=vz.SURFACE,
                                  ec="none", alpha=.72))

    ax.set_xlim(-75, -32)
    ax.set_ylim(-34, 7)
    ax.set_aspect(1.0)
    ax.axis("off")
    ax.set_title(title, pad=26, fontsize=13.5)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=vz.MUTED, va="bottom", ha="left")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax, shrink=.52, pad=.01, aspect=18)
    cb.set_label(cbar_label, color=vz.INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5, labelcolor=vz.INK_2)


LABEL = ["SP", "RJ", "MG", "RS", "BA", "MA", "AM", "PA", "CE", "AL", "RR"]

fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
draw_panel(axes[0], st.days.to_dict(), CMAP_BLUE, 6, 26,
           "Median delivery days",
           "Delivery time is a regional gradient, not a scatter",
           "Median days from purchase to doorstep, by customer state.",
           fmt="{:.0f}d", label_states=LABEL)
draw_panel(axes[1], st.freight_pct.to_dict(), CMAP_ORANGE, 15, 50,
           "Freight as % of item price",
           "…and the slowest states pay the most to ship",
           "Median freight as a share of what the customer paid for the item.",
           fmt="{:.0f}%", label_states=LABEL)

# One shared legend for the circles, since both panels use the same encoding.
for n, lab in [(40000, "40,000 orders"), (10000, "10,000"), (1000, "1,000")]:
    axes[0].scatter([], [], s=np.sqrt(n) * 1.45, facecolor="none",
                    edgecolor=vz.INK, linewidth=1.1, alpha=.62, label=lab)
leg = axes[0].legend(loc="lower left", fontsize=8.5, labelspacing=1.15,
                     borderpad=0.6, handletextpad=1.1, frameon=False,
                     title="Order volume")
leg.get_title().set_fontsize(8.5)
leg.get_title().set_color(vz.INK_2)

fig.tight_layout(w_pad=1.5)
vz.source_note(fig,
               f"n = 96,470 delivered orders; all {len(st)} states carry data. "
               "Circle size = order volume — São Paulo alone is 42% of the "
               "marketplace, so map AREA badly overstates how much of the problem "
               "the North represents. Read colour and circle together.")
out = FIG / "10_choropleth_delivery_freight.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"saved {out.relative_to(ROOT)}")
