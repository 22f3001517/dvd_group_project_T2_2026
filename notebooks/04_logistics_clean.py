"""
LOGISTICS WORKSTREAM - Step 2: Cleaning & Feature Engineering
=============================================================
Turns the raw freight/delivery tables into two analysis-ready datasets:

  data/clean/logistics_orders.csv  - ONE ROW PER ORDER.
        Use for delivery-time analysis (delivery time is an order-level event).

  data/clean/logistics_items.csv   - ONE ROW PER ORDER LINE ITEM.
        Use for freight analysis (freight is charged per line item, and
        weight/dimensions/seller only exist at item level).

Every cleaning rule below cites the diagnostic number it came from
(see output/03_logistics_diagnostics.txt).

Run: source venv/bin/activate && python notebooks/04_logistics_clean.py
"""
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 170)

ROOT = Path(__file__).resolve().parents[1]
ECOM = ROOT / "data" / "raw" / "ecommerce"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

ORDER_DATES = ["order_purchase_timestamp", "order_approved_at",
               "order_delivered_carrier_date", "order_delivered_customer_date",
               "order_estimated_delivery_date"]

# Brazil's five official macro-regions. Logistics performance is really a
# regional story (sellers cluster in the Southeast, customers do not), so we
# want this grouping available everywhere.
STATE_TO_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North",
    "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast",
    "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West",
    "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}

# Real bounding box of Brazil. Coordinates outside it are corrupt records
# (diagnostic 7 found 42 such points, e.g. one sitting in Spain).
BR_BBOX = dict(lat_min=-33.75, lat_max=5.27, lng_min=-73.99, lng_max=-34.79)

log = []


def step(msg):
    """Record every cleaning action so the report can show an audit trail."""
    print(msg)
    log.append(msg)


def strip_accents(s: pd.Series) -> pd.Series:
    """
    Normalise free-text city names: 'Santo Antônio' and 'santo antonio' are the
    same city. Diagnostic 8 showed both spellings coexisting in geolocation.
    """
    return (s.astype(str)
            .str.strip().str.lower()
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore").str.decode("utf-8")
            .str.replace(r"\s+", " ", regex=True))


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Vectorised over numpy arrays."""
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


print("=" * 100)
print("LOADING RAW TABLES")
print("=" * 100)
orders = pd.read_csv(ECOM / "orders_dataset.csv", parse_dates=ORDER_DATES)
items = pd.read_csv(ECOM / "order_items_dataset.csv", parse_dates=["shipping_limit_date"])
products = pd.read_csv(ECOM / "products_dataset.csv")
sellers = pd.read_csv(ECOM / "sellers_dataset.csv")
customers = pd.read_csv(ECOM / "customers_dataset.csv")
geo = pd.read_csv(ECOM / "geolocation_dataset.csv")
cat_tr = pd.read_csv(ECOM / "product_category_name_translation.csv")
reviews = pd.read_csv(ECOM / "order_reviews_dataset.csv",
                      parse_dates=["review_creation_date", "review_answer_timestamp"])
for n, d in [("orders", orders), ("items", items), ("products", products),
             ("sellers", sellers), ("customers", customers), ("geolocation", geo)]:
    print(f"  {n:14s} {d.shape}")


# ===========================================================================
print("\n" + "=" * 100)
print("A. CLEAN GEOLOCATION  ->  one trusted coordinate per zip prefix")
print("=" * 100)
# ===========================================================================
n0 = len(geo)

# A1. Duplicate rows are DELIBERATELY KEPT.
#     This table has no seller/customer/business identifier - it is purely a
#     zip-prefix -> coordinate lookup, so a repeated row is not a duplicated
#     entity. It is a repeated observation of an address at that coordinate,
#     i.e. FREQUENCY WEIGHT. Dropping the 261,831 repeats would make each
#     centroid the middle of the *distinct coordinates seen* rather than the
#     middle of *where addresses actually are*, biasing every prefix away from
#     its densest blocks. (Measured impact of dropping them: the median centroid
#     moves 33 m and the median line-item distance moves 52 m - small, but the
#     bias is systematic and there is no reason to accept it.)
step(f"A1. KEPT {geo.duplicated().sum():,} duplicate geolocation rows as frequency "
     "weight (no entity id in this table - repeats are address density)")

# A2. Drop coordinates physically outside Brazil (diagnostic 7: 42 rows).
inside = (geo.geolocation_lat.between(BR_BBOX["lat_min"], BR_BBOX["lat_max"])
          & geo.geolocation_lng.between(BR_BBOX["lng_min"], BR_BBOX["lng_max"]))
step(f"A2. dropped {(~inside).sum():,} coordinates outside Brazil's bounding box")
geo = geo[inside].copy()

# A3. A Brazilian CEP prefix cannot straddle a state border, so when a prefix
#     carries more than one state label the minority rows are wrong.
dominant_state = (geo.groupby("geolocation_zip_code_prefix")
                  .geolocation_state.transform(lambda s: s.mode().iat[0]))
bad_state = geo.geolocation_state != dominant_state
n_mixed = (geo.groupby("geolocation_zip_code_prefix")
           .geolocation_state.nunique() > 1).sum()
step(f"A3. dropped {bad_state.sum():,} rows whose state disagrees with the rest of "
     f"their zip prefix ({n_mixed} prefixes affected)")
geo = geo[~bad_state]

# A4. Within-prefix outliers. The bounding box only catches coordinates outside
#     the COUNTRY; it cannot catch a coordinate that is inside Brazil but in the
#     wrong part of it. Brazil has many repeated place names, and the geocoder
#     clearly resolved some to the wrong one - e.g. prefix 19274 ("primavera",
#     SP) has pings 2,113 km away, and prefix 35179 ("santana do paraiso", MG)
#     has pings sitting in Amapá. Ping distance from its own prefix median is
#     0.5 km at the median and 9.8 km at p99, so a 50 km cut is ~5x the 99th
#     percentile: comfortably outside any real prefix, and it removes 0.12% of
#     rows.
#     The reference point is the prefix's MODAL coordinate - the exact lat/lng
#     that appears most often - not its median. Two reasons: with the duplicate
#     rows retained (A1) the mode is the single densest real address location,
#     and unlike the median it is guaranteed to be a point that actually exists,
#     so a prefix split into two distant clusters keeps its larger cluster
#     instead of having every row cut for being far from an interpolated
#     midpoint that lies between them.
MAX_KM_FROM_PREFIX = 50
anchor = (geo.groupby(["geolocation_zip_code_prefix",
                       "geolocation_lat", "geolocation_lng"])
          .size().rename("n").reset_index()
          .sort_values("n", ascending=False)
          .drop_duplicates("geolocation_zip_code_prefix")
          .set_index("geolocation_zip_code_prefix"))
a_lat = geo.geolocation_zip_code_prefix.map(anchor.geolocation_lat)
a_lng = geo.geolocation_zip_code_prefix.map(anchor.geolocation_lng)
far = haversine_km(geo.geolocation_lat, geo.geolocation_lng,
                   a_lat, a_lng) > MAX_KM_FROM_PREFIX
step(f"A4. dropped {far.sum():,} pings more than {MAX_KM_FROM_PREFIX} km from their "
     "prefix's densest coordinate (geocoder resolved an ambiguous Brazilian place "
     "name to the wrong city)")
geo = geo[~far]

# A5. Collapse to ONE centroid per zip prefix using the MEDIAN over the
#     duplicate-weighted rows: the median is robust to whatever outliers
#     survived A4, and the retained duplicates let address density pull the
#     centroid toward where people actually are.
geo["geolocation_city"] = strip_accents(geo["geolocation_city"])
geo_zip = (geo.groupby("geolocation_zip_code_prefix")
           .agg(lat=("geolocation_lat", "median"),
                lng=("geolocation_lng", "median"),
                geo_city=("geolocation_city", lambda s: s.mode().iat[0]),
                geo_state=("geolocation_state", lambda s: s.mode().iat[0]),
                n_pings=("geolocation_lat", "size"))
           .reset_index())
step(f"A5. collapsed {len(geo):,} weighted pings -> {len(geo_zip):,} zip-prefix "
     f"centroids (median lat/lng); {n0 - len(geo):,} rows removed in total")


# ===========================================================================
print("\n" + "=" * 100)
print("B. CLEAN CUSTOMERS & SELLERS  ->  normalise names, attach coordinates")
print("=" * 100)
# ===========================================================================
customers["customer_city"] = strip_accents(customers["customer_city"])
sellers["seller_city"] = strip_accents(sellers["seller_city"])
step("B1. normalised city names (lowercase, trimmed, accents stripped)")

customers = customers.merge(
    geo_zip[["geolocation_zip_code_prefix", "lat", "lng"]]
    .rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                     "lat": "customer_lat", "lng": "customer_lng"}),
    on="customer_zip_code_prefix", how="left")
sellers = sellers.merge(
    geo_zip[["geolocation_zip_code_prefix", "lat", "lng"]]
    .rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix",
                     "lat": "seller_lat", "lng": "seller_lng"}),
    on="seller_zip_code_prefix", how="left")
step(f"B2. geocoded customers: {customers.customer_lat.notna().mean() * 100:.2f}% matched")
step(f"B3. geocoded sellers  : {sellers.seller_lat.notna().mean() * 100:.2f}% matched")

# B4. Unmatched zip prefixes fall back to their STATE centroid rather than being
#     dropped - losing 278 customers would silently bias the distance analysis
#     toward well-mapped (urban) areas.
state_centroid = (geo_zip.groupby("geo_state")[["lat", "lng"]].median()
                  .rename(columns={"lat": "s_lat", "lng": "s_lng"}))
for df, pre, statecol in [(customers, "customer", "customer_state"),
                          (sellers, "seller", "seller_state")]:
    m = df[f"{pre}_lat"].isna()
    fill = df.loc[m, statecol].map(state_centroid["s_lat"])
    df.loc[m, f"{pre}_lat"] = fill
    df.loc[m, f"{pre}_lng"] = df.loc[m, statecol].map(state_centroid["s_lng"])
    df[f"{pre}_geo_source"] = np.where(m, "state_centroid", "zip_centroid")
step(f"B4. filled {(customers.customer_geo_source == 'state_centroid').sum()} customers and "
     f"{(sellers.seller_geo_source == 'state_centroid').sum()} sellers with state centroids "
     "(flagged in *_geo_source)")

customers["customer_region"] = customers.customer_state.map(STATE_TO_REGION)
sellers["seller_region"] = sellers.seller_state.map(STATE_TO_REGION)
step("B5. mapped states to Brazil's 5 macro-regions")


# ===========================================================================
print("\n" + "=" * 100)
print("C. CLEAN PRODUCTS  ->  usable weight & dimensions (freight cost drivers)")
print("=" * 100)
# ===========================================================================
dim_cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]

# C1. Treat zero weight as missing - a parcel cannot weigh 0g
#     (diagnostic 6: 4 products, plus 2 with genuinely null dimensions).
zero_w = (products.product_weight_g == 0).sum()
products.loc[products.product_weight_g == 0, "product_weight_g"] = np.nan
step(f"C1. converted {zero_w} zero-gram weights to NaN (physically impossible)")

# C2. Impute missing dims from the product's own category median, since parcel
#     size is far more similar within a category than across the catalogue.
#     Falls back to the global median for products with no category either.
products["product_category_name"] = products["product_category_name"].fillna("unknown")
before = products[dim_cols].isna().sum().sum()
for c in dim_cols:
    products[c] = products[c].fillna(products.groupby("product_category_name")[c].transform("median"))
    products[c] = products[c].fillna(products[c].median())
step(f"C2. imputed {before} missing weight/dimension values (category median -> global median)")

# C3. Derived size measures. Volumetric ("dimensional") weight is what carriers
#     actually bill on for bulky-but-light parcels: L*W*H / 6000 kg.
products["product_volume_cm3"] = (products.product_length_cm
                                  * products.product_height_cm
                                  * products.product_width_cm)
products["product_weight_kg"] = products.product_weight_g / 1000
products["volumetric_weight_kg"] = products.product_volume_cm3 / 6000
products["billable_weight_kg"] = products[["product_weight_kg", "volumetric_weight_kg"]].max(axis=1)
step("C3. derived volume_cm3, weight_kg, volumetric_weight_kg, billable_weight_kg")

products = products.merge(cat_tr, on="product_category_name", how="left")
products["category"] = products.product_category_name_english.fillna(
    products.product_category_name)
step(f"C4. translated categories to English "
     f"({products.category.isna().sum()} still unnamed)")


# ===========================================================================
print("\n" + "=" * 100)
print("D. CLEAN ORDERS  ->  resolve status/timestamp contradictions")
print("=" * 100)
# ===========================================================================
orders["status_conflict"] = None

# D1. status='canceled' but a delivery date exists (diagnostic 2: 6 orders).
#     These are contradictory. We keep the row but flag it and exclude it from
#     delivery-performance stats, because we cannot tell which field is wrong.
m = (orders.order_status == "canceled") & orders.order_delivered_customer_date.notna()
orders.loc[m, "status_conflict"] = "canceled_but_delivered"
step(f"D1. flagged {m.sum()} orders as 'canceled_but_delivered'")

# D2. status='delivered' but no delivery date (diagnostic 2: 8 orders).
#     Delivery time is unmeasurable for these; flag and exclude.
m = (orders.order_status == "delivered") & orders.order_delivered_customer_date.isna()
orders.loc[m, "status_conflict"] = "delivered_no_date"
step(f"D2. flagged {m.sum()} orders as 'delivered_no_date'")

# D3. Impossible orderings that CANNOT be a logging lag - the parcel cannot
#     reach the carrier before the customer bought it, nor reach the customer
#     before the carrier had it. Null the offending timestamp so the affected
#     leg becomes NaN instead of producing a negative duration.
m = orders.order_delivered_carrier_date < orders.order_purchase_timestamp
step(f"D3a. nulled {m.sum()} carrier dates that precede the purchase (impossible)")
orders.loc[m, "order_delivered_carrier_date"] = pd.NaT

m = orders.order_delivered_customer_date < orders.order_delivered_carrier_date
step(f"D3b. nulled {m.sum()} carrier dates that postdate the delivery (impossible)")
orders.loc[m, "order_delivered_carrier_date"] = pd.NaT

# D4. carrier_date < approved_at (diagnostic 3: 1,359 orders, median only 17h).
#     This is NOT impossible - it means payment approval was WRITTEN to the log
#     after the parcel had already moved. The purchase->customer total is
#     unaffected, so we keep the row, flag it, and let the seller-handover leg
#     clip at zero rather than going negative.
m = orders.order_delivered_carrier_date < orders.order_approved_at
orders["approval_logged_late"] = m
step(f"D4. flagged {m.sum()} orders where approval was logged after carrier pickup "
     "(kept: total delivery time is unaffected)")


# ===========================================================================
print("\n" + "=" * 100)
print("E. FEATURE ENGINEERING  ->  break the journey into its four legs")
print("=" * 100)
# ===========================================================================
o = orders
# The four legs of the product-movement journey. Splitting total delivery time
# this way is the whole point of the logistics workstream: it lets us say WHO
# is responsible for a delay - the payment system, the seller, or the carrier.
o["approval_hours"] = (o.order_approved_at - o.order_purchase_timestamp).dt.total_seconds() / 3600
o["seller_handover_days"] = (o.order_delivered_carrier_date - o.order_approved_at).dt.total_seconds() / 86400
o["carrier_transit_days"] = (o.order_delivered_customer_date - o.order_delivered_carrier_date).dt.total_seconds() / 86400
o["total_delivery_days"] = (o.order_delivered_customer_date - o.order_purchase_timestamp).dt.total_seconds() / 86400

# Clip only the leg affected by the D4 logging lag; never clip the total.
o["seller_handover_days"] = o["seller_handover_days"].clip(lower=0)
step("E1. built 4 journey legs: approval_hours, seller_handover_days, "
     "carrier_transit_days, total_delivery_days")

# Promise vs reality.
o["estimated_days"] = (o.order_estimated_delivery_date - o.order_purchase_timestamp).dt.total_seconds() / 86400
o["delay_days"] = (o.order_delivered_customer_date - o.order_estimated_delivery_date).dt.total_seconds() / 86400
o["is_late"] = o["delay_days"] > 0
# Positive buffer = arrived earlier than promised. Large positive values mean
# the promise was padded, which is itself a finding.
o["estimate_buffer_days"] = -o["delay_days"]
step("E2. built estimated_days, delay_days, is_late, estimate_buffer_days")

o["purchase_date"] = o.order_purchase_timestamp.dt.date
o["purchase_month"] = o.order_purchase_timestamp.dt.to_period("M").astype(str)
o["purchase_dow"] = o.order_purchase_timestamp.dt.day_name()
o["purchase_hour"] = o.order_purchase_timestamp.dt.hour
step("E3. built calendar features (month, day-of-week, hour)")


# --- item-level: freight, weight, seller, distance -------------------------
it = (items
      .merge(products[["product_id", "category", "product_weight_kg",
                       "product_volume_cm3", "volumetric_weight_kg",
                       "billable_weight_kg"]], on="product_id", how="left")
      .merge(sellers, on="seller_id", how="left"))
it = it.merge(orders[["order_id", "customer_id", "order_status",
                      "order_delivered_carrier_date", "total_delivery_days",
                      "delay_days", "is_late", "purchase_month"]],
              on="order_id", how="left")
it = it.merge(customers[["customer_id", "customer_state", "customer_region",
                         "customer_city", "customer_lat", "customer_lng"]],
              on="customer_id", how="left")

# E4. Real shipping distance, seller doorstep -> customer doorstep.
it["distance_km"] = haversine_km(it.seller_lat, it.seller_lng,
                                 it.customer_lat, it.customer_lng)
step(f"E4. computed seller->customer haversine distance for "
     f"{it.distance_km.notna().mean() * 100:.2f}% of line items "
     f"(median {it.distance_km.median():.0f} km)")

# E5. Freight economics. freight_ratio is what the CUSTOMER feels (shipping as
#     a share of product price); freight_per_kg / per_km are what the
#     MARKETPLACE should watch (is this route/parcel priced sanely?).
it["freight_ratio"] = np.where(it.price > 0, it.freight_value / it.price, np.nan)
it["freight_per_kg"] = np.where(it.billable_weight_kg > 0,
                                it.freight_value / it.billable_weight_kg, np.nan)
it["freight_per_km"] = np.where(it.distance_km > 0,
                                it.freight_value / it.distance_km, np.nan)
it["is_free_shipping"] = it.freight_value == 0
step(f"E5. built freight_ratio, freight_per_kg, freight_per_km; "
     f"flagged {it.is_free_shipping.sum():,} free-shipping line items")

# E6. Route features - same-state shipping is the cheap, fast case.
it["same_state"] = it.seller_state == it.customer_state
it["seller_region"] = it.seller_state.map(STATE_TO_REGION)
it["route"] = it.seller_region.astype(str) + " -> " + it.customer_region.astype(str)
step(f"E6. built same_state ({it.same_state.mean() * 100:.1f}% of items ship "
     "within one state) and region->region route labels")

# E7. Seller SLA: did the seller hand the parcel over by its contractual
#     deadline? This is the cleanest available split of "seller's fault" vs
#     "carrier's fault" (diagnostic 5).
it["seller_sla_hours"] = (it.order_delivered_carrier_date - it.shipping_limit_date
                          ).dt.total_seconds() / 3600
it["seller_missed_sla"] = it.seller_sla_hours > 0
step(f"E7. built seller_sla_hours / seller_missed_sla "
     f"({it.seller_missed_sla.mean() * 100:.1f}% of line items late to carrier)")


# --- roll item features back up to order level ------------------------------
# An order can span several sellers. For delivery performance the binding
# constraint is the WORST case, so distance/SLA roll up with max.
item_roll = it.groupby("order_id").agg(
    n_items=("order_item_id", "count"),
    n_sellers=("seller_id", "nunique"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    total_weight_kg=("product_weight_kg", "sum"),
    total_billable_kg=("billable_weight_kg", "sum"),
    max_distance_km=("distance_km", "max"),
    any_missed_sla=("seller_missed_sla", "any"),
    worst_sla_hours=("seller_sla_hours", "max"),
    all_same_state=("same_state", "all"),
    main_category=("category", lambda s: s.mode().iat[0] if s.notna().any() else np.nan),
    main_seller_state=("seller_state", lambda s: s.mode().iat[0] if s.notna().any() else np.nan),
).reset_index()

# Last review per order, so logistics can be tied to satisfaction.
rev_last = (reviews.sort_values("review_answer_timestamp")
            .drop_duplicates("order_id", keep="last")[["order_id", "review_score"]])

lo = (orders
      .merge(item_roll, on="order_id", how="left")
      .merge(customers[["customer_id", "customer_state", "customer_region",
                        "customer_city", "customer_lat", "customer_lng"]],
             on="customer_id", how="left")
      .merge(rev_last, on="order_id", how="left"))
lo["freight_ratio"] = np.where(lo.total_price > 0, lo.total_freight / lo.total_price, np.nan)
lo["seller_region"] = lo.main_seller_state.map(STATE_TO_REGION)
step(f"E8. rolled item features up to order level -> {lo.shape}")


# ===========================================================================
print("\n" + "=" * 100)
print("F. DEFINE THE ANALYSIS UNIVERSE")
print("=" * 100)
# ===========================================================================
# Delivery-time questions are only answerable on orders that actually completed
# a delivery with a trustworthy timestamp. Everything else stays in the file but
# is excluded by this flag, so no analysis silently mixes the two.
lo["is_analysable_delivery"] = (
    (lo.order_status == "delivered")
    & lo.order_delivered_customer_date.notna()
    & lo.status_conflict.isna()
    & lo.total_delivery_days.between(0, 365)
)
step(f"F1. is_analysable_delivery = True for {lo.is_analysable_delivery.sum():,} orders "
     f"({lo.is_analysable_delivery.mean() * 100:.1f}% of all {len(lo):,})")

excluded = lo[~lo.is_analysable_delivery].order_status.value_counts()
print("\nexcluded orders by status:")
print(excluded.to_string())

# Outlier treatment: cap rather than drop. A 200-day delivery is a REAL event
# and dropping it would hide exactly the failures leadership needs to see, but
# it would also blow up any chart axis. We keep the true value and add a
# winsorised twin for plotting.
for col, lo_q, hi_q in [("total_delivery_days", 0, 0.995),
                        ("carrier_transit_days", 0, 0.995),
                        ("seller_handover_days", 0, 0.995),
                        ("max_distance_km", 0, 0.995)]:
    cap = lo[col].quantile(hi_q)
    lo[col + "_capped"] = lo[col].clip(upper=cap)
    step(f"F2. {col}: winsorised at p99.5 = {cap:.1f} for plotting "
         f"({(lo[col] > cap).sum():,} orders above cap, originals retained)")


# ===========================================================================
print("\n" + "=" * 100)
print("G. SAVE + VALIDATE")
print("=" * 100)
# ===========================================================================
order_out = CLEAN / "logistics_orders.csv"
item_out = CLEAN / "logistics_items.csv"
lo.to_csv(order_out, index=False)
it.to_csv(item_out, index=False)
print(f"wrote {order_out.relative_to(ROOT)}  {lo.shape}")
print(f"wrote {item_out.relative_to(ROOT)}   {it.shape}")

# Post-clean assertions - these must all hold or the cleaning is wrong.
a = lo[lo.is_analysable_delivery]
assert (a.total_delivery_days >= 0).all(), "negative total delivery time survived"
assert (a.carrier_transit_days.dropna() >= 0).all(), "negative transit time survived"
assert (a.seller_handover_days.dropna() >= 0).all(), "negative handover time survived"
assert a.order_id.is_unique, "duplicate order_id in order-level table"
assert it.distance_km.dropna().between(0, 6000).all(), "impossible distance survived"
print("\nall post-clean assertions passed.")

print("\nanalysable orders - key logistics features:")
print(a[["approval_hours", "seller_handover_days", "carrier_transit_days",
         "total_delivery_days", "estimated_days", "delay_days",
         "max_distance_km", "total_freight", "freight_ratio"]].describe().round(2).to_string())

with open(ROOT / "output" / "04_cleaning_log.txt", "w") as f:
    f.write("LOGISTICS CLEANING LOG\n" + "=" * 100 + "\n")
    f.write("\n".join(log))
print(f"\nwrote output/04_cleaning_log.txt ({len(log)} documented cleaning actions)")
