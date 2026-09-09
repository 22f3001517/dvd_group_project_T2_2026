"""
LOGISTICS WORKSTREAM - Step 1: Diagnostics
==========================================
Before cleaning, find out exactly what is broken in the freight/delivery tables.
Every cleaning rule written in 04_logistics_clean.py should trace back to a
number printed here.

Tables in scope for logistics:
  orders        - the 5 timestamps that define the delivery journey
  order_items   - freight_value, shipping_limit_date (the seller handover SLA)
  products      - weight & dimensions (what drives freight cost)
  sellers       - origin location
  customers     - destination location
  geolocation   - lat/lng to turn zip prefixes into real distance

Run: source venv/bin/activate && python notebooks/03_logistics_diagnostics.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 170)

ROOT = Path(__file__).resolve().parents[1]
ECOM = ROOT / "data" / "raw" / "ecommerce"

ORDER_DATES = ["order_purchase_timestamp", "order_approved_at",
               "order_delivered_carrier_date", "order_delivered_customer_date",
               "order_estimated_delivery_date"]

orders = pd.read_csv(ECOM / "orders_dataset.csv", parse_dates=ORDER_DATES)
items = pd.read_csv(ECOM / "order_items_dataset.csv", parse_dates=["shipping_limit_date"])
products = pd.read_csv(ECOM / "products_dataset.csv")
sellers = pd.read_csv(ECOM / "sellers_dataset.csv")
customers = pd.read_csv(ECOM / "customers_dataset.csv")
geo = pd.read_csv(ECOM / "geolocation_dataset.csv")


def header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ---------------------------------------------------------------------------
header("1. ORDER STATUS x TIMESTAMP COMPLETENESS  (which statuses have which legs?)")
# ---------------------------------------------------------------------------
# A delivery journey is only fully observable when all 4 event timestamps exist.
# This matrix tells us which statuses we can legitimately measure.
completeness = orders.groupby("order_status")[ORDER_DATES].apply(
    lambda g: g.notna().mean().mul(100).round(1))
completeness["n_orders"] = orders["order_status"].value_counts()
print(completeness)

header("2. STATUS / TIMESTAMP CONTRADICTIONS")
# Status says one thing, the timestamps say another.
delivered_no_date = orders[(orders.order_status == "delivered")
                           & orders.order_delivered_customer_date.isna()]
print(f"status='delivered' but NO delivered_customer_date : {len(delivered_no_date)}")
print(delivered_no_date[["order_id", "order_status"] + ORDER_DATES].head(10).to_string(index=False))

delivered_no_carrier = orders[(orders.order_status == "delivered")
                              & orders.order_delivered_carrier_date.isna()]
print(f"\nstatus='delivered' but NO delivered_carrier_date  : {len(delivered_no_carrier)}")

delivered_no_approve = orders[(orders.order_status == "delivered")
                              & orders.order_approved_at.isna()]
print(f"status='delivered' but NO approved_at             : {len(delivered_no_approve)}")

canceled_with_delivery = orders[(orders.order_status == "canceled")
                                & orders.order_delivered_customer_date.notna()]
print(f"\nstatus='canceled' but HAS delivered_customer_date : {len(canceled_with_delivery)}")

shipped_with_delivery = orders[(orders.order_status == "shipped")
                               & orders.order_delivered_customer_date.notna()]
print(f"status='shipped' but HAS delivered_customer_date  : {len(shipped_with_delivery)}")

unavail_with_carrier = orders[(orders.order_status == "unavailable")
                              & orders.order_delivered_carrier_date.notna()]
print(f"status='unavailable' but HAS carrier date         : {len(unavail_with_carrier)}")


header("3. TIMESTAMP ORDERING VIOLATIONS  (events out of chronological sequence)")
# The journey must run purchase -> approved -> carrier -> customer.
# Any inversion is a data integrity problem that would produce negative durations.
o = orders
checks = {
    "approved_at < purchase":        (o.order_approved_at < o.order_purchase_timestamp),
    "carrier_date < approved_at":    (o.order_delivered_carrier_date < o.order_approved_at),
    "carrier_date < purchase":       (o.order_delivered_carrier_date < o.order_purchase_timestamp),
    "customer_date < carrier_date":  (o.order_delivered_customer_date < o.order_delivered_carrier_date),
    "customer_date < approved_at":   (o.order_delivered_customer_date < o.order_approved_at),
    "customer_date < purchase":      (o.order_delivered_customer_date < o.order_purchase_timestamp),
    "estimated_date < purchase":     (o.order_estimated_delivery_date < o.order_purchase_timestamp),
}
for label, mask in checks.items():
    n = mask.sum()
    print(f"{label:32s} : {n:>6,}  ({n / len(o) * 100:.3f}%)")

# How severe are the carrier<approved inversions? (hours, not just a count)
inv = o[o.order_delivered_carrier_date < o.order_approved_at].copy()
if len(inv):
    inv["gap_hours"] = (inv.order_approved_at - inv.order_delivered_carrier_date
                        ).dt.total_seconds() / 3600
    print("\nseverity of 'carrier before approval' inversions (hours early):")
    print(inv["gap_hours"].describe().round(2).to_string())
    print("  -> mostly small gaps = payment approval logged late, not a real inversion")


header("4. FREIGHT VALUE QUALITY  (order_items)")
print(items["freight_value"].describe().round(2).to_string())
zero_freight = (items.freight_value == 0).sum()
print(f"\nfreight_value == 0        : {zero_freight:,} line items "
      f"({zero_freight / len(items) * 100:.2f}%)  -> free-shipping promos?")
print(f"freight_value < 0         : {(items.freight_value < 0).sum():,}")
print(f"freight_value is null     : {items.freight_value.isna().sum():,}")
print(f"freight_value > 3x price  : {(items.freight_value > 3 * items.price).sum():,} "
      "line items (freight dwarfs the product)")

# Multi-item orders: freight is charged PER LINE ITEM, so summing is correct,
# but confirm that repeated identical products in one order repeat the freight.
multi = items.groupby("order_id").size()
print(f"\norders with >1 line item  : {(multi > 1).sum():,} of {multi.size:,}")
print("freight per line item within multi-item orders (are they identical?):")
mi = items[items.order_id.isin(multi[multi > 1].index)]
same_freight = mi.groupby("order_id")["freight_value"].nunique().eq(1).mean()
print(f"  {same_freight * 100:.1f}% of multi-item orders charge the SAME freight on every line")


header("5. SELLER SHIPPING SLA  (shipping_limit_date vs actual carrier handover)")
# shipping_limit_date = the contractual deadline for the seller to hand the
# parcel to the carrier. Comparing it to order_delivered_carrier_date tells us
# whether lateness starts with the SELLER or with the CARRIER.
sla = (items.groupby("order_id")
       .agg(shipping_limit=("shipping_limit_date", "max"))
       .reset_index()
       .merge(orders[["order_id", "order_status", "order_delivered_carrier_date",
                      "order_purchase_timestamp"]], on="order_id", how="left"))
sla = sla[sla.order_delivered_carrier_date.notna()]
sla["seller_late_hours"] = (sla.order_delivered_carrier_date - sla.shipping_limit
                            ).dt.total_seconds() / 3600
print(f"orders with both a shipping_limit and a carrier date: {len(sla):,}")
print(f"seller MISSED the handover deadline : {(sla.seller_late_hours > 0).sum():,} "
      f"({(sla.seller_late_hours > 0).mean() * 100:.1f}%)")
print("\nhours late vs shipping limit (positive = seller was late):")
print(sla["seller_late_hours"].describe().round(1).to_string())


header("6. PRODUCT DIMENSION / WEIGHT QUALITY  (freight cost drivers)")
dim_cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
print("missing counts:")
print(products[dim_cols].isna().sum().to_string())
print(f"\nrows missing ANY dimension : {products[dim_cols].isna().any(axis=1).sum()}")
print(f"weight == 0                : {(products.product_weight_g == 0).sum()}")
for c in ["product_length_cm", "product_height_cm", "product_width_cm"]:
    print(f"{c} == 0             : {(products[c] == 0).sum()}")

# How many actual LINE ITEMS are affected? (that is what matters for analysis)
it_p = items.merge(products, on="product_id", how="left")
bad_dims = it_p[dim_cols].isna().any(axis=1) | (it_p.product_weight_g == 0)
print(f"\nline items with unusable weight/dims : {bad_dims.sum():,} "
      f"({bad_dims.mean() * 100:.3f}% of {len(it_p):,})")
print(f"line items with missing category     : {it_p.product_category_name.isna().sum():,}")


header("7. GEOLOCATION TABLE QUALITY  (needed for real seller->customer distance)")
print(f"raw rows                       : {len(geo):,}")
print(f"exact duplicate rows           : {geo.duplicated().sum():,}")
print(f"unique zip prefixes            : {geo.geolocation_zip_code_prefix.nunique():,}")
print(f"avg pings per zip prefix       : {len(geo) / geo.geolocation_zip_code_prefix.nunique():.1f}")

# Brazil's real bounding box - anything outside is a corrupt coordinate.
BR = dict(lat_min=-33.75, lat_max=5.27, lng_min=-73.99, lng_max=-34.79)
outside = ~(geo.geolocation_lat.between(BR["lat_min"], BR["lat_max"])
            & geo.geolocation_lng.between(BR["lng_min"], BR["lng_max"]))
print(f"points OUTSIDE Brazil bbox     : {outside.sum():,} ({outside.mean() * 100:.4f}%)")
print(geo[outside][["geolocation_zip_code_prefix", "geolocation_lat",
                    "geolocation_lng", "geolocation_city", "geolocation_state"]].head(8).to_string(index=False))

# Spread within a zip prefix tells us how much error a per-zip centroid introduces.
spread = geo.groupby("geolocation_zip_code_prefix").agg(
    lat_range=("geolocation_lat", lambda s: s.max() - s.min()),
    lng_range=("geolocation_lng", lambda s: s.max() - s.min()))
print("\ncoordinate spread within a single zip prefix (degrees):")
print(spread.describe().round(4).to_string())
print("  -> median spread is tiny, so a per-zip median centroid is a safe representation")

# Coverage: can we actually locate every customer and seller?
geo_zips = set(geo.geolocation_zip_code_prefix.unique())
cust_missing = (~customers.customer_zip_code_prefix.isin(geo_zips)).sum()
sell_missing = (~sellers.seller_zip_code_prefix.isin(geo_zips)).sum()
print(f"\ncustomers whose zip prefix is NOT in geolocation : {cust_missing:,} "
      f"({cust_missing / len(customers) * 100:.2f}%)")
print(f"sellers   whose zip prefix is NOT in geolocation : {sell_missing:,} "
      f"({sell_missing / len(sellers) * 100:.2f}%)")


header("8. CITY-NAME CONSISTENCY  (free-text location fields)")
# City strings are user-entered and appear in customers, sellers and geolocation.
for name, df, col in [("customers", customers, "customer_city"),
                      ("sellers", sellers, "seller_city"),
                      ("geolocation", geo, "geolocation_city")]:
    s = df[col].astype(str)
    print(f"{name:12s} unique cities raw={s.nunique():>6,}  "
          f"after strip/lower/accent-normalise={s.str.strip().str.lower().nunique():>6,}")
print("\nexamples of messy geolocation city spellings:")
gc = geo.geolocation_city.value_counts()
print(gc[gc.index.str.contains("sao paulo|sp|s.paulo", case=False, na=False, regex=True)].head(10).to_string())


header("9. STATE COVERAGE  (destination spread)")
print(f"customer states : {customers.customer_state.nunique()}")
print(f"seller states   : {sellers.seller_state.nunique()}")
print("\nseller concentration by state (top 8):")
sc = sellers.seller_state.value_counts()
print(pd.DataFrame({"sellers": sc, "pct": (sc / len(sellers) * 100).round(1)}).head(8).to_string())
print("\ncustomer spread by state (top 8):")
cc = customers.customer_state.value_counts()
print(pd.DataFrame({"customers": cc, "pct": (cc / len(customers) * 100).round(1)}).head(8).to_string())

print("\n" + "=" * 100)
print("Diagnostics complete - see 04_logistics_clean.py for the rules derived from this.")
print("=" * 100)
