"""Generate a messy raw transactional log (data/raw_sales.csv)."""
import numpy as np
import pandas as pd
from config import RAW_CSV

rng = np.random.default_rng(42)

N_STORES, N_SKUS, N_DAYS = 200, 30, 365
start = pd.Timestamp("2024-01-01")
skus = [f"SKU-{i:03d}" for i in range(1, N_SKUS + 1)]
base_price = {s: float(rng.uniform(20, 400)) for s in skus}

rows = []
oid = 100000
for st in range(1, N_STORES + 1):
    store_id = f"ST{st:04d}"
    rate = rng.uniform(0.04, 0.25)             # orders per day
    churner = rng.random() < 0.22
    stop_day = int(rng.integers(230, 340)) if churner else N_DAYS
    pref = rng.choice(skus, size=rng.integers(4, 12), replace=False)
    for d in range(N_DAYS):
        if d >= stop_day:
            break
        # churners slow down before leaving
        r = rate * (0.35 if churner and d > stop_day - 50 else 1.0)
        dow_boost = 1.3 if (start + pd.Timedelta(days=d)).dayofweek < 5 else 0.7
        if rng.random() < min(r * dow_boost, 0.95):
            for sku in rng.choice(pref, size=rng.integers(1, 4), replace=False):
                oid += 1
                qty = int(max(1, rng.poisson(8 * (1 + 0.3 * (int(sku[-3:]) % 5)))))
                rows.append((f"ORD{oid}", start + pd.Timedelta(days=d), store_id,
                             sku, qty, round(base_price[sku] * rng.uniform(0.95, 1.05), 2)))

df = pd.DataFrame(rows, columns=["order_id", "order_date", "store_id", "sku", "quantity", "unit_price"])
n = len(df)

# ---- Corrupt the data on purpose ----
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
df["quantity"] = df["quantity"].astype(object)
df["unit_price"] = df["unit_price"].astype(object)

idx = rng.permutation(n)
k = int(n * 0.02)
df.loc[idx[:k], "quantity"] = -df.loc[idx[:k], "quantity"].astype(int)            # negative qty
df.loc[idx[k:2*k], "order_date"] = ""                                              # missing date
df.loc[idx[2*k:3*k], "order_date"] = "not-a-date"                                  # junk date
df.loc[idx[3*k:4*k], "quantity"] = "abc"                                           # non numeric qty
df.loc[idx[4*k:5*k], "unit_price"] = 0                                             # bad price
alt = idx[5*k:15*k]                                                                # mixed date format
df.loc[alt, "order_date"] = pd.to_datetime(df.loc[alt, "order_date"]).dt.strftime("%d/%m/%Y")
ws = idx[15*k:25*k]                                                                # messy SKU codes
df.loc[ws, "sku"] = ["  " + s.lower() + " " for s in df.loc[ws, "sku"]]
dups = df.sample(frac=0.01, random_state=1)                                        # duplicate order ids
df = pd.concat([df, dups]).sample(frac=1, random_state=3).reset_index(drop=True)

df.to_csv(RAW_CSV, index=False)
print(f"Wrote {len(df):,} raw rows to {RAW_CSV}")
