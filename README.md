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
python notebooks/12_retention_padding_text.py  # retention, padding, text mining
```

| Script | What it does |
|---|---|
| `03_logistics_diagnostics.py` | Counts every data-quality defect **before** any rule is written, so each cleaning decision cites a number instead of an assumption. |
| `04_logistics_clean.py` | Applies the cleaning rules and writes `data/clean/logistics_orders.csv` (order level, for delivery-time analysis) and `data/clean/logistics_items.csv` (item level, for freight analysis). Every action is logged to `output/04_cleaning_log.txt`. |
| `05_logistics_eda.py` | Ten analyses: journey decomposition, promise vs reality, blame attribution, distance, route matrix, state scorecard, satisfaction, freight economics, time trend, seller scorecard. |
| `06_logistics_charts.py` | Figures 01–09. |
| `10_logistics_maps.py` | Figure 10 — state-level choropleths of delivery time and freight burden. |
| `12_retention_padding_text.py` | Three enquiries behind the sustainable-growth storyline — see below. |
| `src/vizstyle.py` | Shared palette and matplotlib defaults so every figure reads as one system. |

### Retention, padding and text mining (`12_...`)

Three tests, each built to survive the obvious objection:

- **Retention forensics.** A cohort test with a *fixed* observation window, so
  censoring cannot explain the result. Repeat purchase is 0.66% at 30 days and
  still only 3.29% with a full year of observable window. Those who do return
  come back fast (median 73 days), so it is not a slow repurchase cycle.
- **The padding test.** Holds *actual* delivery speed constant and varies only
  the promise length. Padding turns out to be expectation *insurance*, not
  expectation management — once an order is on time, promise length barely
  moves satisfaction (r = −0.066).
- **Text mining** over 40,977 Portuguese review comments: accent-stripped and
  tokenised, then tagged with keyword dictionaries for delivery, product,
  wrong-item, incomplete-order and praise themes.

**On the text dictionaries.** Every tag is validated against the delivery
timestamps rather than trusted — a delivery-complaint tag should fire far more
often on genuinely late orders, and a praise tag far less. The first version of
the `late / waiting` tag **failed** that check (0.94× lift) because it used the
bare word *prazo*, which is the neutral Portuguese for "deadline" — so
*"chegou no prazo"* ("arrived on time") was being counted as a complaint. The
committed version uses only unambiguous markers and passes at 2.98× lift. If you
extend these dictionaries, re-run the validation block at the bottom of the
script; a tag sitting near 1.0× lift is picking up noise.

Two caveats when quoting the text results: only 41.3% of reviews carry text, and
writing skews heavily to the unhappy (76.5% of 1-star reviewers write, vs 35.9%
of 5-star), so theme percentages describe *reviewers who wrote*, not customers.

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

---

## Products workstream — catalog, pricing & revenue mix

Covers the product catalog itself: category structure, pricing behaviour,
and how catalog composition connects to revenue and customer satisfaction.

### Running the pipeline

This workstream runs in Google Colab rather than locally.

1. Open `notebooks/products_dataset_eda.py` in Colab (or use the badge below).
2. Upload the e-commerce dataset when prompted (see the "Getting set up"
   section above for the expected file structure), or mount Google Drive if
   the CSVs are already stored there.
3. Run all cells top to bottom — cleaning, merging, EDA, and figures are all
   produced inline in the notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-org>/dvd_group_project_T2_2026/blob/main/notebooks/products_eda.ipynb)

| Notebook | What it does |
|---|---|
| `products_dataset_eda.py` | Diagnoses data-quality defects, applies and logs the cleaning rules, merges the products table with `order_items`/`orders`/`product_category_name_translation`, and runs the full EDA: category revenue concentration, price distribution by category, catalog size vs. revenue, weight-freight relationship and review score by category. Produces figures 01–06 inline. |

### Cleaning decisions worth knowing about

The judgment calls, made inline in the notebook:

- **610 rows (1.85%) dropped** — missing category, name-length,
  description-length, and photo-qty together on the same rows. These are
  incomplete listings, not isolated gaps, so imputing them would fabricate
  content that was never captured.
- **2 rows (0.01%) median-imputed** for missing weight/dimensions —
  negligible share, no material risk.
- **13 categories had no match** in the standard translation table. 2
  (`pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos`) were manually
  translated; the remaining 11 fell back to their original Portuguese name
  rather than being dropped, since the category itself is still valid, just
  untranslated.
- **0 orphaned products** — every product in the cleaned catalog matched at
  least one order after merging with `order_items`/`orders`.

### Output

The notebook produces a cleaned, merged products table (32,341 products,
98.1% of the raw catalog) flagged `is_analysable_product` — True for products
with a complete, English-labelled category. Filter on it for any
category-level analysis so no chart silently mixes an incomplete listing in.

---
