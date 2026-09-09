# dvd_group_project_T2_2026

A repository containing the codes used for data cleaning and preparing visualizations.

**Project:** A Visual Study of E-Commerce Orders, Delivery & Customer Satisfaction
(Olist Brazilian E-Commerce dataset, Sep 2016 – Oct 2018).

---

## Logistics workstream — freight & delivery

Covers the product-movement journey: payment approval → seller handover to the
carrier → transit → the customer's door, plus freight cost and the delivery
promise.

### Getting set up

The raw CSVs are **not** committed (too large, and we all have them). Unzip the
dataset so it looks like this:

```
data/raw/ecommerce/          orders_dataset.csv, order_items_dataset.csv,
                             products_dataset.csv, sellers_dataset.csv,
                             customers_dataset.csv, geolocation_dataset.csv,
                             order_reviews_dataset.csv, order_payments_dataset.csv,
                             product_category_name_translation.csv
data/raw/marketing_funnel/   marketing_qualified_leads_dataset.csv,
                             closed_deals_dataset.csv
```

Then:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Running the pipeline

Run in order — each step depends on the previous one.

```bash
python notebooks/03_logistics_diagnostics.py   # measure every defect first
python notebooks/04_logistics_clean.py         # apply + log the cleaning rules
python notebooks/05_logistics_eda.py           # ten analyses + seller scorecard
python notebooks/06_logistics_charts.py        # figures 01-09
python notebooks/10_logistics_maps.py          # figure 10 (choropleths)
```

| Script | What it does |
|---|---|
| `03_logistics_diagnostics.py` | Counts every data-quality defect **before** any rule is written, so each cleaning decision cites a number instead of an assumption. |
| `04_logistics_clean.py` | Applies the cleaning rules and writes `data/clean/logistics_orders.csv` (order level, for delivery-time analysis) and `data/clean/logistics_items.csv` (item level, for freight analysis). Every action is logged to `output/04_cleaning_log.txt`. |
| `05_logistics_eda.py` | Ten analyses: journey decomposition, promise vs reality, blame attribution, distance, route matrix, state scorecard, satisfaction, freight economics, time trend, seller scorecard. |
| `06_logistics_charts.py` | Figures 01–09. |
| `10_logistics_maps.py` | Figure 10 — state-level choropleths of delivery time and freight burden. |
| `src/vizstyle.py` | Shared palette and matplotlib defaults so every figure reads as one system. |

### Two output tables

- **`data/clean/logistics_orders.csv`** — one row per order (99,441 × 47).
  Use for delivery-time questions; delivery is an order-level event.
- **`data/clean/logistics_items.csv`** — one row per line item (112,650 × 40).
  Use for freight questions; freight is charged per line item, and
  weight/dimensions/seller only exist at item level.

Both files carry the flag **`is_analysable_delivery`** — `True` for the 96,470
orders (97.0%) with a complete, self-consistent delivery timeline. Filter on it
for any delivery-time analysis so no chart silently mixes a delivered order with
a cancelled one.

### Cleaning decisions worth knowing about

The full audit trail is in `output/04_cleaning_log.txt`. The judgment calls:

- **Duplicate geolocation rows are KEPT** (261,831 of them). That table has no
  seller/customer id — it is a zip-prefix → coordinate lookup, so a repeated row
  is address *density*, not a duplicated business. Dropping them biases every
  centroid away from the densest blocks.
- **2,718 pings dropped** for sitting >50 km from their zip prefix's *densest*
  coordinate. Brazil reuses place names heavily and the geocoder resolved some to
  the wrong city — prefix 19274 ("primavera", SP) had pings 2,113 km away. The
  Brazil bounding-box test cannot catch these, because they are inside Brazil.
- **1,193 orders where carrier pickup precedes payment approval are KEPT and
  flagged.** The median gap is 17 hours, so approval was *logged* late rather
  than the parcel moving early. Total delivery time is unaffected; only the
  handover leg is clipped at zero.
- **483 extreme deliveries (up to 209 days) are KEPT**, with a winsorised twin
  column for plotting. Dropping them would hide exactly the failures the analysis
  exists to surface.

### Note on the boundary file

`data/raw/geo/br_states.geojson` (27 Brazilian states, `SIGLA` = the 2-letter
state code) is committed so the maps reproduce offline. Sourced from the public
[geodata-br-states](https://github.com/giuliano-macedo/geodata-br-states)
repository. It is parsed as data only.
