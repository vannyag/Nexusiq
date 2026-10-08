
import re
import duckdb
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from config import DB_PATH, MODEL_DIR

st.set_page_config(page_title="NexusIQ", page_icon="📦", layout="wide")


@st.cache_resource
def get_con():
    # read-only + no external file access: safe for the SQL playground
    return duckdb.connect(str(DB_PATH), read_only=True, config={"enable_external_access": False})


@st.cache_resource
def get_models():
    return joblib.load(MODEL_DIR / "demand_lgbm.joblib"), joblib.load(MODEL_DIR / "churn_xgb.joblib")


def q(sql, params=None):
    return get_con().execute(sql, params or []).df()


import subprocess, sys

if not DB_PATH.exists() or not (MODEL_DIR / "churn_xgb.joblib").exists():
    with st.spinner("First run: building data and models (1-2 min)..."):
        for script in ("generate_data.py", "pipeline.py", "train.py"):
            subprocess.run([sys.executable, script], check=True)
    st.rerun()

page = st.sidebar.radio("NexusIQ", ["Executive Overview", "Predictive Analytics", "SQL Analytics Engine"])
st.sidebar.caption("Bronze → Silver → Gold → ML → CXO")

# ------------------------------------------------------------------ Executive
if page == "Executive Overview":
    st.title("📊 Executive Overview")
    kpi = q("SELECT * FROM gold_daily_kpis")
    kpi["order_date"] = pd.to_datetime(kpi["order_date"])
    lo, hi = kpi.order_date.min().date(), kpi.order_date.max().date()
    rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    if len(rng) == 2:
        kpi = kpi[(kpi.order_date.dt.date >= rng[0]) & (kpi.order_date.dt.date <= rng[1])]

    quarantined = q("SELECT count(*) n FROM silver_quarantine").n[0]
    clean = q("SELECT count(*) n FROM silver_sales").n[0]
    active = q("SELECT count(DISTINCT store_id) n FROM silver_sales "
               "WHERE order_date > (SELECT max(order_date) FROM silver_sales) - INTERVAL 30 DAY").n[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"₹{kpi.revenue.sum():,.0f}")
    c2.metric("Orders", f"{int(kpi.orders.sum()):,}")
    c3.metric("Active stores (last 30d)", f"{active:,}")
    c4.metric("Data quality", f"{clean / (clean + quarantined):.1%} clean",
              f"{quarantined:,} rows quarantined", delta_color="off")

    kpi["revenue_7d_avg"] = kpi.revenue.rolling(7, min_periods=1).mean()
    fig = px.line(kpi, x="order_date", y=["revenue", "revenue_7d_avg"], title="Daily revenue")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    top = q("SELECT sku, sum(revenue) revenue FROM silver_sales GROUP BY sku ORDER BY revenue DESC LIMIT 10")
    left.plotly_chart(px.bar(top, x="sku", y="revenue", title="Top 10 SKUs by revenue"), use_container_width=True)
    qr = q("SELECT rejection_reason, count(*) AS row_count FROM silver_quarantine GROUP BY 1 ORDER BY 2 DESC")
    right.plotly_chart(px.bar(qr, x="row_count", y="rejection_reason", orientation="h",
                              title="Why rows were quarantined"), use_container_width=True)

# ------------------------------------------------------------------ Predictive
elif page == "Predictive Analytics":
    st.title("🔮 Predictive Analytics")
    demand, churn = get_models()
    stores, skus = demand["stores"], demand["skus"]

    st.subheader("Demand forecast (LightGBM)")
    c1, c2, c3 = st.columns(3)
    store = c1.selectbox("Store", stores)
    sku = c2.selectbox("SKU", skus)
    last = pd.to_datetime(q("SELECT max(order_date) d FROM silver_sales").d[0])
    day = c3.date_input("Forecast date", (last + pd.Timedelta(days=1)).date())

    def make_X(dates):
        d = pd.to_datetime(pd.Series(dates))
        return pd.DataFrame({
            "store_code": stores.index(store), "sku_code": skus.index(sku),
            "day_of_week": d.dt.dayofweek, "day_of_month": d.dt.day, "month": d.dt.month,
        })[demand["features"]]

    pred = float(demand["model"].predict(make_X([day]))[0])
    st.metric(f"Expected units on {day} (on a day this store orders this SKU)", f"{max(pred, 0):.1f}")

    nxt = pd.date_range(last + pd.Timedelta(days=1), periods=14)
    fc = pd.DataFrame({"date": nxt, "predicted_units": demand["model"].predict(make_X(nxt)).clip(0)})
    st.plotly_chart(px.bar(fc, x="date", y="predicted_units", title="Next 14 days"), use_container_width=True)

    st.divider()
    st.subheader("Churn risk matrix (XGBoost)")
    sc = q("SELECT * FROM gold_churn_scores")
    colors = {"High": "red", "Medium": "orange", "Low": "green"}
    m1, m2, m3 = st.columns(3)
    for col, band in zip((m1, m2, m3), ("High", "Medium", "Low")):
        col.metric(f"{band} risk stores", int((sc.risk_band == band).sum()))

    st.plotly_chart(px.scatter(sc, x="days_since_last_order", y="lifetime_spend", color="risk_band",
                               color_discrete_map=colors, hover_data=["store_id", "churn_probability"],
                               title="Days since last order vs lifetime spend"), use_container_width=True)

    show = sc.sort_values("churn_probability", ascending=False)[
        ["store_id", "churn_probability", "risk_band", "order_count", "lifetime_spend",
         "avg_basket_size", "days_since_last_order"]].reset_index(drop=True)
    only_high = st.checkbox("Show only High risk", value=True)
    if only_high:
        show = show[show.risk_band == "High"]
    styled = show.style.map(lambda v: "background-color:#ffb3b3" if v == "High" else "", subset=["risk_band"]) \
                       .format({"churn_probability": "{:.1%}", "lifetime_spend": "{:,.0f}", "avg_basket_size": "{:,.0f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ SQL
else:
    st.title("🧮 SQL Analytics Engine")
    tables = q("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'silver%' "
               "OR table_name LIKE 'gold%' ORDER BY 1").table_name.tolist()
    st.caption("Queryable tables: " + ", ".join(tables))
    default = "SELECT store_id, sum(revenue) AS revenue\nFROM silver_sales\nGROUP BY 1\nORDER BY 2 DESC\nLIMIT 10"
    sql = st.text_area("Your SQL (read-only)", default, height=160)
    if st.button("Run query", type="primary"):
        stmt = sql.strip().rstrip(";")
        if not re.match(r"(?is)^(select|with|describe|show)\b", stmt) or ";" in stmt:
            st.error("Only a single SELECT / WITH / DESCRIBE / SHOW statement is allowed.")
        else:
            try:
                res = q(stmt)
                st.success(f"{len(res):,} rows")
                st.dataframe(res, use_container_width=True)
            except Exception as e:  # show SQL errors to the user
                st.error(str(e))
