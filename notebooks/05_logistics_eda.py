"""
LOGISTICS WORKSTREAM - Step 3: Detailed EDA
===========================================
Ten analyses on the cleaned logistics tables. Each one is written to answer a
specific question a logistics owner would actually be asked in a review.

Run: source venv/bin/activate && python notebooks/05_logistics_eda.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 175)

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"

lo = pd.read_csv(CLEAN / "logistics_orders.csv", low_memory=False, parse_dates=[
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"])
it = pd.read_csv(CLEAN / "logistics_items.csv")

# Every delivery-time analysis runs on the analysable universe only.
d = lo[lo.is_analysable_delivery].copy()
di = it[it.order_id.isin(d.order_id)].copy()


def header(n, title):
    print("\n" + "=" * 100)
    print(f"{n}. {title}")
    print("=" * 100)


print(f"analysable delivered orders : {len(d):,}")
print(f"their line items            : {len(di):,}")


# ---------------------------------------------------------------------------
header(1, "WHERE DOES THE TIME ACTUALLY GO? (decomposing the journey)")
# ---------------------------------------------------------------------------
legs = pd.DataFrame({
    "1_payment_approval": d.approval_hours / 24,
    "2_seller_handover": d.seller_handover_days,
    "3_carrier_transit": d.carrier_transit_days,
})
summary = legs.describe(percentiles=[.25, .5, .75, .9, .95]).T.round(2)
summary["share_of_median_total"] = (legs.median() / legs.median().sum() * 100).round(1)
print(summary[["mean", "50%", "75%", "90%", "95%", "max", "share_of_median_total"]])
print(f"\nmedian TOTAL delivery time : {d.total_delivery_days.median():.2f} days")
print(f"mean   TOTAL delivery time : {d.total_delivery_days.mean():.2f} days")
print("\nREAD: the carrier leg dominates. Seller handover is small at the median")
print("      but has a long tail - which is where late orders are made.")


# ---------------------------------------------------------------------------
header(2, "PROMISE vs REALITY - is the delivery estimate calibrated?")
# ---------------------------------------------------------------------------
print(f"median PROMISED delivery window : {d.estimated_days.median():.1f} days")
print(f"median ACTUAL   delivery time   : {d.total_delivery_days.median():.1f} days")
print(f"median BUFFER (promise - actual): {d.estimate_buffer_days.median():.1f} days")
print(f"\norders arriving EARLY : {(d.delay_days < 0).mean() * 100:.1f}%")
print(f"orders arriving LATE  : {d.is_late.mean() * 100:.1f}%")
print("\nhow early, among early arrivals:")
print(d.loc[d.delay_days < 0, "estimate_buffer_days"].describe(
    percentiles=[.25, .5, .75, .9]).round(1).to_string())
print("\nREAD: the platform pads its promise by ~12 days at the median. It is not")
print("      'accurate' - it is sandbagged, and still misses 6.8% of the time.")

# Is the padding uniform, or is it worst where service is worst?
buf = (d.groupby("customer_region")
       .agg(orders=("order_id", "size"),
            median_promise=("estimated_days", "median"),
            median_actual=("total_delivery_days", "median"),
            median_buffer=("estimate_buffer_days", "median"),
            late_rate=("is_late", "mean"))
       .assign(late_rate=lambda x: (x.late_rate * 100).round(2))
       .sort_values("median_actual", ascending=False).round(1))
print("\nestimate padding by region:")
print(buf.to_string())


# ---------------------------------------------------------------------------
header(3, "WHO CAUSES LATENESS - the seller or the carrier?")
# ---------------------------------------------------------------------------
# Compare each leg for on-time vs late orders. Whichever leg blows out is the
# one that owns the failure.
cmp_legs = d.groupby("is_late")[["approval_hours", "seller_handover_days",
                                 "carrier_transit_days", "total_delivery_days",
                                 "estimated_days", "max_distance_km"]].median().round(2)
cmp_legs.index = ["on time", "LATE"]
print("median value of each leg, on-time vs late orders:")
print(cmp_legs.to_string())

print("\nratio (late / on-time):")
print((cmp_legs.loc["LATE"] / cmp_legs.loc["on time"]).round(2).to_string())

# Attribute the excess: how much of the extra time sits in each leg?
excess = (cmp_legs.loc["LATE"] - cmp_legs.loc["on time"])
print("\nextra days in each leg for a late order vs an on-time one:")
print(pd.Series({
    "payment approval": excess["approval_hours"] / 24,
    "seller handover": excess["seller_handover_days"],
    "carrier transit": excess["carrier_transit_days"],
}).round(2).to_string())

# Seller SLA breach: the cleanest "seller's fault" signal we have.
sla_order = di.groupby("order_id")["seller_missed_sla"].any()
d["missed_sla"] = d.order_id.map(sla_order)
print(f"\norders where the seller missed the carrier-handover deadline: "
      f"{d.missed_sla.mean() * 100:.1f}%")
xt = pd.crosstab(d.missed_sla, d.is_late, normalize="index").mul(100).round(1)
xt.columns = ["arrived on time %", "arrived LATE %"]
xt.index = ["seller met SLA", "seller MISSED SLA"]
print(xt.to_string())
print("\nlate-order breakdown by responsibility:")
late = d[d.is_late]
print(f"  late orders where the seller ALSO missed its SLA : "
      f"{late.missed_sla.mean() * 100:.1f}%")
print(f"  late orders where the seller met its SLA (carrier/transit fault): "
      f"{(~late.missed_sla.fillna(False)).mean() * 100:.1f}%")
print("\nREAD: most lateness happens AFTER the parcel leaves the seller.")


# ---------------------------------------------------------------------------
header(4, "DISTANCE - what does shipping far actually cost in time and money?")
# ---------------------------------------------------------------------------
d["dist_band"] = pd.cut(d.max_distance_km,
                        [-1, 50, 150, 400, 800, 1500, 10000],
                        labels=["0-50 km", "50-150", "150-400", "400-800",
                                "800-1500", "1500+ km"])
dist_tbl = (d.groupby("dist_band", observed=True)
            .agg(orders=("order_id", "size"),
                 median_transit=("carrier_transit_days", "median"),
                 median_total=("total_delivery_days", "median"),
                 median_promise=("estimated_days", "median"),
                 late_rate=("is_late", "mean"),
                 median_freight=("total_freight", "median"),
                 avg_review=("review_score", "mean")))
dist_tbl["late_rate"] = (dist_tbl.late_rate * 100).round(2)
print(dist_tbl.round(2).to_string())
print("\ncorrelation, distance vs total delivery days: "
      f"{d[['max_distance_km', 'total_delivery_days']].corr().iloc[0, 1]:.3f}")
print("correlation, distance vs freight value       : "
      f"{di[['distance_km', 'freight_value']].corr().iloc[0, 1]:.3f}")
print("correlation, billable weight vs freight value: "
      f"{di[['billable_weight_kg', 'freight_value']].corr().iloc[0, 1]:.3f}")
print("\nREAD: freight tracks WEIGHT far more than DISTANCE. The marketplace is")
print("      effectively cross-subsidising long-haul deliveries.")


# ---------------------------------------------------------------------------
header(5, "THE ROUTE MATRIX - seller region -> customer region")
# ---------------------------------------------------------------------------
route = (di.dropna(subset=["seller_region", "customer_region"])
         .groupby(["seller_region", "customer_region"])
         .agg(items=("order_id", "size"),
              median_freight=("freight_value", "median"),
              median_dist=("distance_km", "median")))
route_orders = (d.dropna(subset=["seller_region", "customer_region"])
                .groupby(["seller_region", "customer_region"])
                .agg(median_days=("total_delivery_days", "median"),
                     late_rate=("is_late", "mean"),
                     avg_review=("review_score", "mean")))
route = route.join(route_orders)
route["late_rate"] = (route.late_rate * 100).round(1)
route["share_of_items"] = (route["items"] / route["items"].sum() * 100).round(2)
print(route.round(2).sort_values("items", ascending=False).head(20).to_string())

print("\nseller supply vs customer demand by region (the structural imbalance):")
supply = di.seller_region.value_counts(normalize=True).mul(100).round(1)
demand = di.customer_region.value_counts(normalize=True).mul(100).round(1)
imb = pd.DataFrame({"% of items SOLD from": supply, "% of items SHIPPED to": demand})
imb["gap"] = (imb.iloc[:, 0] - imb.iloc[:, 1]).round(1)
print(imb.to_string())
print("\nREAD: the Southeast sells far more than it buys; every other region is a")
print("      net importer, so its customers structurally wait longer.")


# ---------------------------------------------------------------------------
header(6, "STATE-LEVEL SCORECARD - where is service worst?")
# ---------------------------------------------------------------------------
st = (d.groupby("customer_state")
      .agg(orders=("order_id", "size"),
           median_days=("total_delivery_days", "median"),
           median_promise=("estimated_days", "median"),
           late_rate=("is_late", "mean"),
           median_dist=("max_distance_km", "median"),
           median_freight=("total_freight", "median"),
           avg_review=("review_score", "mean")))
st["late_rate"] = (st.late_rate * 100).round(2)
st["freight_pct_of_order"] = (d.groupby("customer_state").freight_ratio.median() * 100).round(1)
st = st.round(2)
print("worst 12 states by median delivery time:")
print(st.sort_values("median_days", ascending=False).head(12).to_string())
print("\nbest 8 states by median delivery time:")
print(st.sort_values("median_days").head(8).to_string())
print("\nworst 8 states by LATE RATE (min 200 orders):")
print(st[st.orders >= 200].sort_values("late_rate", ascending=False).head(8).to_string())


# ---------------------------------------------------------------------------
header(7, "DELIVERY PERFORMANCE -> CUSTOMER SATISFACTION (the payoff)")
# ---------------------------------------------------------------------------
r = d.dropna(subset=["review_score"])
print(f"orders with a review: {len(r):,}")
bins = [-np.inf, -20, -10, -5, 0, 3, 7, 15, np.inf]
labels = ["20+ d early", "10-20 d early", "5-10 d early", "0-5 d early",
          "1-3 d late", "4-7 d late", "8-15 d late", "15+ d late"]
r = r.assign(delay_band=pd.cut(r.delay_days, bins, labels=labels))
tbl = (r.groupby("delay_band", observed=True)
       .agg(orders=("order_id", "size"),
            avg_review=("review_score", "mean"),
            pct_1_star=("review_score", lambda s: (s == 1).mean() * 100),
            pct_5_star=("review_score", lambda s: (s == 5).mean() * 100)))
tbl["share_of_orders"] = (tbl.orders / tbl.orders.sum() * 100).round(1)
print(tbl.round(2).to_string())

print("\nheadline contrast:")
on_t, lt = r[~r.is_late], r[r.is_late]
print(f"  on-time orders : avg {on_t.review_score.mean():.2f} stars, "
      f"{(on_t.review_score == 1).mean() * 100:.1f}% one-star")
print(f"  late orders    : avg {lt.review_score.mean():.2f} stars, "
      f"{(lt.review_score == 1).mean() * 100:.1f}% one-star")
print(f"  -> a late delivery makes a one-star review "
      f"{(lt.review_score == 1).mean() / (on_t.review_score == 1).mean():.1f}x more likely")

# How much of the platform's 1-star problem is logistics?
tot1 = (r.review_score == 1).sum()
print(f"\ntotal 1-star reviews: {tot1:,}")
print(f"  from LATE orders   : {(lt.review_score == 1).sum():,} "
      f"({(lt.review_score == 1).sum() / tot1 * 100:.1f}% of all 1-stars, "
      f"from just {lt.shape[0] / len(r) * 100:.1f}% of orders)")

# Does being early actually help, or is on-time enough?
print("\nis 'very early' better than 'just on time'? (avg review by earliness)")
early = r[r.delay_days < 0]
print(early.groupby(pd.cut(early.estimate_buffer_days, [0, 3, 7, 14, 21, 200]),
                    observed=True).review_score.agg(["mean", "size"]).round(3).to_string())
print("\nREAD: satisfaction saturates around 4.3 stars. Being 20 days early buys")
print("      nothing extra - but being 1 day late costs a full star.")


# ---------------------------------------------------------------------------
header(8, "FREIGHT ECONOMICS - is shipping priced sanely?")
# ---------------------------------------------------------------------------
print("freight value distribution (per line item):")
print(di.freight_value.describe(percentiles=[.1, .25, .5, .75, .9, .99]).round(2).to_string())
print(f"\nfree shipping line items : {di.is_free_shipping.sum():,} "
      f"({di.is_free_shipping.mean() * 100:.2f}%)")
print(f"freight > product price  : {(di.freight_value > di.price).sum():,} "
      f"({(di.freight_value > di.price).mean() * 100:.2f}%)")

di["wt_band"] = pd.cut(di.billable_weight_kg, [-.001, .5, 1, 2, 5, 10, 1000],
                       labels=["<0.5 kg", "0.5-1", "1-2", "2-5", "5-10", "10+ kg"])
print("\nfreight by parcel weight band:")
print(di.groupby("wt_band", observed=True).agg(
    items=("order_id", "size"),
    median_freight=("freight_value", "median"),
    median_freight_per_kg=("freight_per_kg", "median"),
    median_dist_km=("distance_km", "median"),
).round(2).to_string())

di["dist_band"] = pd.cut(di.distance_km, [-1, 50, 150, 400, 800, 1500, 10000],
                         labels=["0-50 km", "50-150", "150-400", "400-800",
                                 "800-1500", "1500+ km"])
print("\nfreight by distance band (note how flat this is vs the weight table):")
print(di.groupby("dist_band", observed=True).agg(
    items=("order_id", "size"),
    median_freight=("freight_value", "median"),
    median_freight_per_km=("freight_per_km", "median"),
    median_weight_kg=("billable_weight_kg", "median"),
).round(3).to_string())

print("\nfreight as a share of product price, by customer region:")
print(di.groupby("customer_region").freight_ratio.median().mul(100).round(1)
      .sort_values(ascending=False).to_string())
print("\nREAD: a Northern customer pays a much larger share of the item price in")
print("      freight AND waits longest - a double penalty.")

print("\ncategories with the worst freight burden (min 500 items):")
cf = di.groupby("category").agg(items=("order_id", "size"),
                                median_freight=("freight_value", "median"),
                                median_ratio=("freight_ratio", "median"),
                                median_kg=("billable_weight_kg", "median"))
print(cf[cf["items"] >= 500].sort_values("median_ratio", ascending=False).head(10).round(2).to_string())


# ---------------------------------------------------------------------------
header(9, "IS LOGISTICS PERFORMANCE IMPROVING AS THE MARKETPLACE GROWS?")
# ---------------------------------------------------------------------------
mth = (d.groupby("purchase_month")
       .agg(orders=("order_id", "size"),
            median_days=("total_delivery_days", "median"),
            median_transit=("carrier_transit_days", "median"),
            median_handover=("seller_handover_days", "median"),
            median_promise=("estimated_days", "median"),
            late_rate=("is_late", "mean"),
            avg_review=("review_score", "mean")))
mth["late_rate"] = (mth.late_rate * 100).round(2)
print(mth.round(2).to_string())
print("\nREAD: watch for months where order volume jumps and late_rate jumps with")
print("      it - those are capacity failures, not seasonal noise.")


# ---------------------------------------------------------------------------
header(10, "SELLER LOGISTICS SCORECARD - who should leadership intervene on?")
# ---------------------------------------------------------------------------
sc = (di.groupby("seller_id")
      .agg(items=("order_id", "size"),
           orders=("order_id", "nunique"),
           revenue=("price", "sum"),
           sla_miss_rate=("seller_missed_sla", "mean"),
           median_handover_h=("seller_sla_hours", "median"),
           seller_state=("seller_state", "first")))
# One row per (seller, order) pair. A multi-seller order is attributed to EVERY
# seller in it - each of them had a hand in when that parcel finally arrived.
pairs = di[["seller_id", "order_id"]].drop_duplicates()
ord_lvl = pairs.merge(
    d[["order_id", "is_late", "total_delivery_days", "review_score"]],
    on="order_id", how="inner")
sc2 = (ord_lvl.groupby("seller_id")
       .agg(late_rate=("is_late", "mean"),
            median_days=("total_delivery_days", "median"),
            avg_review=("review_score", "mean")))
sc = sc.join(sc2)
sc["sla_miss_rate"] = (sc.sla_miss_rate * 100).round(1)
sc["late_rate"] = (sc.late_rate * 100).round(1)

print(f"total sellers: {len(sc):,}")
print(f"sellers with >= 50 orders: {(sc.orders >= 50).sum():,} "
      f"-> they carry {sc[sc.orders >= 50].orders.sum() / sc.orders.sum() * 100:.1f}% of all orders")

big = sc[sc.orders >= 50]
print("\nWORST high-volume sellers by late rate (>=50 orders):")
print(big.sort_values("late_rate", ascending=False).head(12)
      [["orders", "revenue", "seller_state", "sla_miss_rate", "late_rate",
        "median_days", "avg_review"]].round(2).to_string())

print("\nBEST high-volume sellers by late rate (>=50 orders):")
print(big.sort_values("late_rate").head(8)
      [["orders", "revenue", "seller_state", "sla_miss_rate", "late_rate",
        "median_days", "avg_review"]].round(2).to_string())

# Concentration of the problem: what share of late orders come from few sellers?
late_by_seller = ord_lvl[ord_lvl.is_late].seller_id.value_counts()
cum = late_by_seller.cumsum() / late_by_seller.sum()
print(f"\nlate orders are concentrated: the worst {(cum <= .5).sum()} sellers "
      f"({(cum <= .5).sum() / len(sc) * 100:.1f}% of sellers) account for "
      "50% of all late deliveries")
print(f"the worst {(cum <= .8).sum()} sellers "
      f"({(cum <= .8).sum() / len(sc) * 100:.1f}% of sellers) account for 80%")

print("\ncorrelation of seller SLA discipline with outcomes (>=50 order sellers):")
print(big[["sla_miss_rate", "late_rate", "median_days", "avg_review"]].corr().round(3).to_string())

sc.sort_values("orders", ascending=False).to_csv(ROOT / "output" / "seller_logistics_scorecard.csv")
print("\nwrote output/seller_logistics_scorecard.csv")

print("\n" + "=" * 100)
print("EDA complete.")
print("=" * 100)
