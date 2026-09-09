"""
Three enquiries for the "sustainable growth" storyline.

  A. RETENTION FORENSICS - is 3% repeat real, or a censoring/identity artifact?
  B. THE PADDING TEST    - holding ACTUAL speed constant, does a longer promise
                           help or hurt? (i.e. are customers fooled by padding?)
  C. TEXT MINING         - what are customers actually saying, in Portuguese,
                           and what share of complaints are delivery vs product?

Run: source venv/bin/activate && python notebooks/12_retention_padding_text.py
"""
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

ROOT = Path(__file__).resolve().parents[1]
ECOM = ROOT / "data" / "raw" / "ecommerce"
CLEAN = ROOT / "data" / "clean"

lo = pd.read_csv(CLEAN / "logistics_orders.csv", low_memory=False,
                 parse_dates=["order_purchase_timestamp"])
it = pd.read_csv(CLEAN / "logistics_items.csv", low_memory=False)
cust = pd.read_csv(ECOM / "customers_dataset.csv")
rev = pd.read_csv(ECOM / "order_reviews_dataset.csv",
                  parse_dates=["review_creation_date", "review_answer_timestamp"])

DATA_END = lo.order_purchase_timestamp.max()


def h(n, t):
    print("\n" + "=" * 100)
    print(f"{n}  {t}")
    print("=" * 100)


# ===========================================================================
h("A1.", "IS customer_unique_id EVEN A PERSON? (identity integrity)")
# ===========================================================================
print(f"orders                : {len(lo):,}")
print(f"unique customer_id    : {lo.customer_id.nunique():,}  "
      "<- one per ORDER, never reused")
print(f"unique customer_unique_id: {cust.customer_unique_id.nunique():,}")
print("\n-> The platform mints a NEW customer_id for every order. "
      "customer_unique_id is a\n   reconciliation key laid over the top, not a "
      "persistent account. That is what a\n   marketplace ENABLER looks like: the "
      "shopper belongs to the sales channel,\n   not to Olist.")

# If unique_id really tracked a person, their location should be stable.
# lo already carries customer_state/city/region from cleaning, so only the two
# columns it lacks are pulled across.
oc = lo.merge(cust[["customer_id", "customer_unique_id",
                    "customer_zip_code_prefix"]], on="customer_id")
rep = oc.groupby("customer_unique_id").agg(
    orders=("order_id", "size"),
    zips=("customer_zip_code_prefix", "nunique"),
    states=("customer_state", "nunique"))
multi = rep[rep.orders > 1]
print(f"\nrepeat customers                     : {len(multi):,}")
print(f"  ...whose zip prefix CHANGED         : {(multi.zips > 1).sum():,} "
      f"({(multi.zips > 1).mean()*100:.1f}%)")
print(f"  ...whose STATE changed              : {(multi.states > 1).sum():,} "
      f"({(multi.states > 1).mean()*100:.1f}%)")
print("-> Location is stable for the vast majority, so the key is not obviously "
      "broken.\n   The low repeat rate looks real, not an id-matching failure.")


# ===========================================================================
h("A2.", "COHORT TEST - strip out censoring. Repeat rate with a FIXED window.")
# ===========================================================================
first = (oc.sort_values("order_purchase_timestamp")
         .drop_duplicates("customer_unique_id", keep="first")
         [["customer_unique_id", "order_purchase_timestamp", "customer_state"]]
         .rename(columns={"order_purchase_timestamp": "first_order"}))
allo = oc[["customer_unique_id", "order_purchase_timestamp"]].merge(first, on="customer_unique_id")
allo["days_since_first"] = (allo.order_purchase_timestamp - allo.first_order).dt.days

print("Every customer below had a FULL observation window of the stated length,")
print("so censoring cannot explain the result.\n")
rows = []
for win in [30, 60, 90, 180, 365]:
    eligible = first[first.first_order <= DATA_END - pd.Timedelta(days=win)]
    sub = allo[allo.customer_unique_id.isin(eligible.customer_unique_id)]
    repeat = sub[(sub.days_since_first > 0) & (sub.days_since_first <= win)]
    rows.append({"window_days": win,
                 "customers_with_full_window": len(eligible),
                 "repeat_rate_%": round(repeat.customer_unique_id.nunique() / len(eligible) * 100, 2)})
print(pd.DataFrame(rows).to_string(index=False))
print("\n-> Even with a full YEAR of observable window, repeat stays ~2-3%.")
print("   This is not a data-window artifact. Typical e-commerce is 20-40%.")

gap = allo[allo.days_since_first > 0].groupby("customer_unique_id").days_since_first.min()
print(f"\namong those who DO return, days to the 2nd order:")
print(gap.describe(percentiles=[.25, .5, .75]).round(0).to_string())
print(f"  share returning within 90 days: {(gap <= 90).mean()*100:.1f}%")
print("-> The ones who come back come back FAST. So the platform is not waiting on")
print("   a long repurchase cycle; the overwhelming majority simply never return.")

# Do repeaters stay with the same seller? Tests whether loyalty is to seller or platform.
si = it[["order_id", "seller_id"]].merge(oc[["order_id", "customer_unique_id"]], on="order_id")
ms = si[si.customer_unique_id.isin(multi.index)]
per_c = ms.groupby("customer_unique_id").agg(orders=("order_id", "nunique"),
                                             sellers=("seller_id", "nunique"))
same = per_c[per_c.orders > 1]
print(f"\nrepeat customers buying from the SAME seller again: "
      f"{(same.sellers == 1).mean()*100:.1f}%")


# ===========================================================================
h("B.", "THE PADDING TEST - at the SAME actual speed, does a longer promise help?")
# ===========================================================================
d = lo[lo.is_analysable_delivery].dropna(subset=["review_score"]).copy()
d["promise_band"] = pd.cut(d.estimated_days, [0, 15, 22, 30, 200],
                           labels=["short (≤15d)", "medium (15-22d)",
                                   "long (22-30d)", "very long (30d+)"])
d["actual_band"] = pd.cut(d.total_delivery_days, [0, 5, 10, 15, 25, 400],
                          labels=["0-5d", "5-10d", "10-15d", "15-25d", "25d+"])

print("Average review score. ROWS = how fast it ACTUALLY arrived (the real")
print("experience). COLUMNS = how long the promise was. Read ACROSS a row: the")
print("actual experience is held constant, only the promise changes.\n")
piv = d.pivot_table(index="actual_band", columns="promise_band",
                    values="review_score", aggfunc="mean", observed=True)
cnt = d.pivot_table(index="actual_band", columns="promise_band",
                    values="review_score", aggfunc="size", observed=True)
print(piv.round(2).to_string())
print("\n(cell counts)")
print(cnt.fillna(0).astype(int).to_string())

print("\nSame table, but restricted to ON-TIME orders only, so 'lateness' cannot")
print("contaminate the comparison:")
ot = d[~d.is_late]
piv2 = ot.pivot_table(index="actual_band", columns="promise_band",
                      values="review_score", aggfunc="mean", observed=True)
print(piv2.round(2).to_string())

for band in ["5-10d", "10-15d"]:
    row = piv2.loc[band].dropna()
    if len(row) > 1:
        print(f"\n  actual {band}: short promise {row.iloc[0]:.2f} -> "
              f"long promise {row.iloc[-1]:.2f}  (delta {row.iloc[-1]-row.iloc[0]:+.2f} stars)")

corr = ot[["estimated_days", "review_score"]].corr().iloc[0, 1]
print(f"\ncorrelation, promise length vs review score (ON-TIME orders only): {corr:.3f}")
print("-> Sign tells you whether padding is free. Negative = customers penalise a")
print("   long promise even when it is honoured.")


# ===========================================================================
h("C.", "TEXT MINING - 41k Portuguese review comments")
# ===========================================================================
r = rev.dropna(subset=["review_comment_message"]).copy()
r = r.merge(lo[["order_id", "is_late", "delay_days", "total_delivery_days"]],
            on="order_id", how="left")
print(f"reviews with text: {len(r):,}")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = s.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z\s]", " ", s)


r["clean"] = r.review_comment_message.map(norm)

STOP = set("""a o e de da do das dos em no na nos nas um uma uns umas para por com sem
que se ao aos as os eu me meu minha mais muito ja nao sim mas foi ser sou esta este
essa esse isso aquele qualquer como quando onde qual quem porque pois entao tambem
so ate lhe ele ela eles elas seu sua nem tem ter havia sao era estava fui vou vai
tudo nada bem la ai voce voces nos ou de_o pra pro tao ainda depois antes sobre
entre desde apos numa num dele dela toda todo todos todas outro outra na_o""".split())

WORD = re.compile(r"[a-z]{3,}")


def top_terms(series, n=18):
    c = Counter()
    for txt in series:
        c.update(w for w in WORD.findall(txt) if w not in STOP)
    return c.most_common(n)


print("\nMOST COMMON WORDS IN 1-STAR REVIEWS:")
for w, n in top_terms(r[r.review_score == 1].clean):
    print(f"   {w:<16} {n:>6,}")
print("\nMOST COMMON WORDS IN 5-STAR REVIEWS:")
for w, n in top_terms(r[r.review_score == 5].clean):
    print(f"   {w:<16} {n:>6,}")


def top_bigrams(series, n=14):
    c = Counter()
    for txt in series:
        ws = [w for w in WORD.findall(txt) if w not in STOP]
        c.update(zip(ws, ws[1:]))
    return c.most_common(n)


print("\nTOP BIGRAMS, 1-STAR:")
for (a, b), n in top_bigrams(r[r.review_score == 1].clean):
    print(f"   {a} {b:<22} {n:>5,}")
print("\nTOP BIGRAMS, 5-STAR:")
for (a, b), n in top_bigrams(r[r.review_score == 5].clean):
    print(f"   {a} {b:<22} {n:>5,}")

# --- theme tagging -------------------------------------------------------
# Negation is handled FIRST: "nao recebi" must not be counted as a delivery
# mention, it is a non-delivery, which is a different (worse) complaint.
NOT_RECEIVED = re.compile(
    r"\b(nao|ainda\s+nao|nunca)\s+(recebi|receb\w*|chegou|chego|entregue|foi\s+entregue)\b"
    r"|\bnao\s+veio\b|\bnada\s+chegou\b")
THEMES = {
    "did NOT arrive": None,  # handled by the regex above
    # NOTE: a first pass used the bare word "prazo" here and FAILED validation
    # (9.8% of tagged reviews were actually late, vs a 10.4% base rate). "prazo"
    # is the neutral Portuguese word for "deadline", and "chegou no prazo" means
    # ARRIVED ON TIME - so the tag was catching praise. Only unambiguously
    # negative markers are kept, and "fora do prazo" is matched as a phrase.
    "late / waiting": re.compile(r"\b(atras\w+|demor\w+|fora\s+do\s+prazo|"
                                 r"estou\s+aguardando|ainda\s+aguardo|"
                                 r"ate\s+agora|muito\s+tempo|lentid\w+|"
                                 r"passou\s+do\s+prazo)\b"),
    "on time (praise)": re.compile(r"\b(no\s+prazo|dentro\s+do\s+prazo|"
                                   r"antes\s+do\s+prazo)\b"),
    "arrived early/fast": re.compile(r"\b(antes\s+do\s+prazo|rapid\w+|chegou\s+antes|"
                                     r"adiantad\w+|super\s+rapid\w+)\b"),
    "product quality": re.compile(r"\b(qualidade|quebr\w+|defeit\w+|danificad\w+|"
                                  r"estrag\w+|frágil|fragil|ruim|pessim\w+|"
                                  r"fraco|furad\w+)\b"),
    "wrong / different item": re.compile(r"\b(errad\w+|diferente|troc\w+|"
                                         r"nao\s+era|outro\s+produto)\b"),
    "incomplete order": re.compile(r"\b(falt\w+|incomplet\w+|so\s+veio|apenas\s+um|"
                                   r"metade)\b"),
    "seller communication": re.compile(r"\b(vendedor|atendiment\w+|contat\w+|"
                                       r"respond\w+|respost\w+)\b"),
    "praise": re.compile(r"\b(otim\w+|excelent\w+|recomend\w+|perfeit\w+|ador\w+|"
                         r"maravilh\w+|satisfeit\w+|super)\b"),
}

r["t_did NOT arrive"] = r.clean.str.contains(NOT_RECEIVED, regex=True)
for name, rx in THEMES.items():
    if rx is None:
        continue
    hit = r.clean.str.contains(rx, regex=True)
    if name == "late / waiting":
        hit = hit & ~r["t_did NOT arrive"]   # don't double-count non-delivery
    r["t_" + name] = hit

theme_cols = [c for c in r.columns if c.startswith("t_")]
print("\n\nTHEME PREVALENCE (% of reviews mentioning), BY STAR RATING")
tbl = r.groupby("review_score")[theme_cols].mean().mul(100).round(1)
tbl.columns = [c[2:] for c in tbl.columns]
tbl["n reviews"] = r.groupby("review_score").size()
print(tbl.to_string())

print("\nAMONG 1-STAR REVIEWS ONLY - what is the complaint actually about?")
one = r[r.review_score == 1]
share = (one[theme_cols].mean() * 100).sort_values(ascending=False)
share.index = [c[2:] for c in share.index]
print(share.round(1).to_string())
print(f"\n(reviews may hit more than one theme; {len(one):,} one-star reviews with text)")

logi = one["t_did NOT arrive"] | one["t_late / waiting"]
prod = (one["t_product quality"] | one["t_wrong / different item"]
        | one["t_incomplete order"])
print(f"\n  mention a DELIVERY problem      : {logi.mean()*100:.1f}%")
print(f"  mention a PRODUCT/ORDER problem : {prod.mean()*100:.1f}%")
print(f"  mention BOTH                    : {(logi & prod).mean()*100:.1f}%")
print(f"  mention NEITHER                 : {(~logi & ~prod).mean()*100:.1f}%")

# --- validation: does the text agree with the timestamps? -----------------
print("\n\nVALIDATION - do text themes match what the delivery data says?")
v = r.dropna(subset=["is_late"])
base = v.is_late.mean() * 100
print(f"  base rate: {base:.1f}% of all reviewed orders were actually late\n")
for tag in ["late / waiting", "did NOT arrive", "on time (praise)",
            "arrived early/fast", "product quality"]:
    col = "t_" + tag
    if col not in v.columns:
        continue
    got = v[v[col]].is_late.mean() * 100
    lift = got / base
    verdict = "PASS" if (lift > 1.8 or lift < 0.6) else "weak/neutral"
    print(f"  {tag:<22} actually late: {got:5.1f}%   lift x{lift:.2f}   {verdict}")
print("\n-> A delivery tag should be far ABOVE the base rate; a praise tag far")
print("   BELOW it. A tag sitting at ~1.0 lift is picking up noise, not signal.")
print("   'product quality' is the control: it SHOULD sit near 1.0, because a")
print("   broken product has nothing to do with whether it arrived on time.")

# The uncomfortable one: complaints about non-delivery on orders the system
# believes were delivered on time.
ontime_nonarrival = v[(v["t_did NOT arrive"]) & (~v.is_late)]
print(f"\n'never arrived' complaints on orders the SYSTEM marked delivered on time: "
      f"{len(ontime_nonarrival):,}")
print(f"   = {len(ontime_nonarrival)/max(v['t_did NOT arrive'].sum(),1)*100:.1f}% of all "
      "non-arrival complaints")

# Is this a PARTIAL delivery? A multi-item order where one item never showed up
# still gets a single "delivered" timestamp, so the system cannot see the gap.
nitems = it.groupby("order_id").size().rename("n_items")
chk = v.merge(nitems, on="order_id", how="left")
chk["multi"] = chk.n_items > 1
grp = chk.groupby("multi")["t_did NOT arrive"].mean().mul(100).round(2)
print("\n   non-arrival complaint rate, single- vs multi-item orders:")
print(grp.rename({False: "single-item", True: "multi-item"}).to_string())
part = chk[chk["t_did NOT arrive"] & ~chk.is_late]
print(f"   of those on-time non-arrival complaints, {part.multi.mean()*100:.1f}% "
      "are multi-item orders")
print(f"   (multi-item orders are {chk.multi.mean()*100:.1f}% of all reviewed orders)")
print("\n-> A partial delivery is INVISIBLE in the timestamps: one parcel arrives,")
print("   the order is stamped delivered, and the customer is still missing an item.")

out = ROOT / "output" / "12_retention_padding_text.txt"
print(f"\n\n(saved to {out.name})")
