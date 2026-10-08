# 📦 NexusIQ

**An end-to-end data and ML platform for a B2B retail/distribution business: from messy raw transactions to a CXO-ready dashboard.**

NexusIQ ingests a deliberately corrupted sales log, cleans it with a **Bronze → Silver → Gold medallion pipeline** in DuckDB, trains a **LightGBM demand forecaster** and an **XGBoost churn classifier**, and serves everything through a **Streamlit** app with executive KPIs, predictions, and a safe SQL playground.

---

## ✨ Features

| Area | What it does |
|---|---|
| **Data engineering** | Medallion architecture in DuckDB with audit columns, validation rules, and a quarantine table for rejected rows |
| **Data quality** | Every rejected row is kept with a `rejection_reason`, and the dashboard shows the clean-data percentage |
| **Demand forecasting** | LightGBM predicts units per store and SKU, with a 14-day forecast view |
| **Churn prediction** | XGBoost scores every store and assigns a Low / Medium / High risk band |
| **Executive dashboard** | Revenue, orders, active stores, data-quality KPIs, top SKUs, and date filtering |
| **SQL Analytics Engine** | Read-only SQL playground over the Silver and Gold tables |

---

## 🏗️ Architecture

```
generate_data.py          pipeline.py                     train.py                app.py
┌──────────────┐   ┌─────────────────────────────┐   ┌───────────────────┐   ┌────────────────┐
│ Messy raw CSV│──▶│ BRONZE  raw, all-varchar    │   │ LightGBM (demand) │   │ Executive      │
│ (~200 stores,│   │   ▼                         │──▶│ XGBoost  (churn)  │──▶│ Predictive     │
│  30 SKUs,    │   │ SILVER  clean + quarantine  │   │ → models/*.joblib │   │ SQL Engine     │
│  1 year)     │   │   ▼                         │   │ → gold_churn_     │   └────────────────┘
└──────────────┘   │ GOLD    KPIs + ML features  │   │   scores table    │        Streamlit
                   └─────────────────────────────┘   └───────────────────┘
                              DuckDB (data/nexusiq.duckdb)
```

### Medallion layers

**Bronze** (`bronze_sales`): the raw CSV loaded as-is (all columns as text) plus `_ingested_at` and `_batch_id` audit columns.

**Silver** (`silver_sales`, `silver_quarantine`): trimmed, typed, and validated. Rows that fail any rule go to quarantine with a reason:

| Rejection reason | Meaning |
|---|---|
| `missing_order_id` | Order ID empty |
| `unparseable_or_missing_date` | Date blank or not in `YYYY-MM-DD` / `DD/MM/YYYY` |
| `non_numeric_quantity` | Quantity can't be cast to an integer |
| `non_positive_quantity` | Quantity ≤ 0 |
| `invalid_price` | Price missing or ≤ 0 |
| `missing_store` / `missing_sku` | Required key empty |
| `duplicate_order_id` | Repeated order ID (first occurrence is kept) |

Cleansing also normalizes store and SKU codes (trim and uppercase), so values like `"  sku-007 "` are fixed rather than rejected.

**Gold**: analytics-ready tables and views:

| Object | Purpose |
|---|---|
| `gold_daily_kpis` (view) | Daily revenue, orders, units, active stores |
| `gold_demand_features` | Units per date, store, and SKU, with calendar features |
| `gold_churn_features` | Store features as of *(latest date − 45 days)*, labelled by whether the store ordered afterwards |
| `gold_churn_scoring` | Store features as of the latest date, used for live scoring |
| `gold_churn_scores` | Final churn probability and risk band per store (written by `train.py`) |

---

## 🤖 Machine learning

### Demand forecasting (LightGBM)
- **Target:** units ordered for a store and SKU on a given day
- **Features:** store, SKU (categorical), day of week, day of month, month
- **Validation:** time-based split (last 30 days held out), compared against a mean-baseline MAE
- The final model is refit on all data before being saved

### Churn prediction (XGBoost)
- **Definition:** a store is *churned* if it places no order in the final 45 days of data (`CHURN_WINDOW_DAYS` in `config.py`)
- **Features:** order count, lifetime spend, average basket size, days since last order
- **Leakage-safe:** features are computed as of a cutoff date and the label comes only from activity *after* it
- **Validation:** stratified hold-out with ROC-AUC
- **Risk bands:** Low (≤ 40%), Medium (40–70%), High (> 70%)

Metrics (MAE, AUC) are printed when you run `train.py`.

---

## 🖥️ The app

1. **Executive Overview**: revenue, orders, active stores (last 30 days), data-quality score, daily revenue with a 7-day average, top 10 SKUs, and a breakdown of why rows were quarantined.
2. **Predictive Analytics**: pick a store, SKU, and date for a demand prediction, see a 14-day forecast, then explore the churn risk matrix and a sortable table of at-risk stores.
3. **SQL Analytics Engine**: query the `silver_*` and `gold_*` tables directly.

### SQL playground security
- DuckDB connection is **read-only** with `enable_external_access` turned off (no file or network access)
- Only a **single** `SELECT` / `WITH` / `DESCRIBE` / `SHOW` statement is accepted

---

## 🚀 Getting started

**Requirements:** Python 3.10+

```bash
git clone https://github.com/<your-username>/nexusiq.git
cd nexusiq
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py   # creates data/raw_sales.csv
python pipeline.py        # Bronze → Silver → Gold in data/nexusiq.duckdb
python train.py           # trains models, writes churn scores
streamlit run app.py      # opens http://localhost:8501
```

Or run everything at once:

```bash
bash run_all.sh
```

> The app also builds the data and models automatically on first launch if they're missing, which is what makes one-click cloud deployment work.

### Docker

```bash
docker build -t nexusiq .
docker run -p 8501:8501 nexusiq
```

---

## ☁️ Deployment

**Streamlit Community Cloud** (recommended):
1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repo, branch `main`, main file `app.py`, then **Deploy**

The first load generates the data and trains the models (about 1–2 minutes). If the build fails on LightGBM, add a `packages.txt` containing `libgomp1`.

---

## 📁 Project structure

```
nexusiq/
├── app.py              # Streamlit app (3 pages)
├── config.py           # Paths and churn window
├── generate_data.py    # Synthetic messy data generator (seeded, reproducible)
├── pipeline.py         # Medallion pipeline (Bronze → Silver → Gold)
├── train.py            # Model training and churn scoring
├── run_all.sh          # One-shot local run
├── requirements.txt
├── Dockerfile
├── data/               # generated (git-ignored)
└── models/             # generated (git-ignored)
```

---

## 🧪 About the synthetic data

`generate_data.py` simulates 200 stores ordering from 30 SKUs over 365 days, with weekday/weekend seasonality, store-specific preferred SKUs, and about 22% of stores that slow down and then stop ordering (the churners). It then injects realistic defects:

- negative, non-numeric, and zero-price values (about 2% each)
- missing and junk dates (about 2% each)
- mixed date formats (`DD/MM/YYYY`) in about 10% of rows
- messy SKU codes (whitespace and lowercase) in about 10% of rows
- duplicated order IDs (1%)

The generator is seeded, so results are reproducible.

---

## ⚠️ Limitations and next steps

- The demand model learns from days when a store *did* order a SKU, so it predicts expected quantity **given an order**, not the probability of ordering. A two-stage model (order probability × quantity) would be the natural next step.
- Demand features are calendar-only; adding lag and rolling-average features would likely improve accuracy.
- Churn labels depend on a single cutoff; multiple rolling cutoffs would give more training data and a more robust model.
- The pipeline fully rebuilds on each run; incremental loading would suit larger datasets.
- Add automated tests for the validation rules.

---

## 🛠️ Tech stack

Python · DuckDB · pandas · LightGBM · XGBoost · scikit-learn · Streamlit · Plotly · Docker

---

## 📄 License

MIT, or choose your own: add a `LICENSE` file to the repo.
