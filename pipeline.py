"""Medallion pipeline in DuckDB: Bronze -> Silver (+Quarantine) -> Gold."""
import uuid
import duckdb
from config import RAW_CSV, DB_PATH, CHURN_WINDOW_DAYS


def run():
    con = duckdb.connect(str(DB_PATH))
    batch_id = uuid.uuid4().hex[:12]

    # ---------------- BRONZE: raw, untouched + audit columns ----------------
    con.execute("DROP TABLE IF EXISTS bronze_sales")
    con.execute(f"""
        CREATE TABLE bronze_sales AS
        SELECT *, now() AS _ingested_at, '{batch_id}' AS _batch_id
        FROM read_csv('{RAW_CSV.as_posix()}', header=true, all_varchar=true)
    """)

    # ---------------- SILVER: cleanse + validate ----------------
    con.execute("DROP TABLE IF EXISTS silver_stage")
    con.execute("""
        CREATE TABLE silver_stage AS
        SELECT
            NULLIF(trim(order_id), '')                         AS order_id,
            COALESCE(try_strptime(trim(order_date), '%Y-%m-%d'),
                     try_strptime(trim(order_date), '%d/%m/%Y'))::DATE AS order_date,
            upper(trim(store_id))                              AS store_id,
            upper(trim(sku))                                   AS sku,
            TRY_CAST(quantity AS INTEGER)                      AS quantity,
            TRY_CAST(unit_price AS DOUBLE)                     AS unit_price,
            _ingested_at, _batch_id,
            row_number() OVER (PARTITION BY NULLIF(trim(order_id), '') ORDER BY _ingested_at) AS rn
        FROM bronze_sales
    """)

    reason = """
        CASE
          WHEN order_id IS NULL                       THEN 'missing_order_id'
          WHEN order_date IS NULL                     THEN 'unparseable_or_missing_date'
          WHEN quantity IS NULL                       THEN 'non_numeric_quantity'
          WHEN quantity <= 0                          THEN 'non_positive_quantity'
          WHEN unit_price IS NULL OR unit_price <= 0  THEN 'invalid_price'
          WHEN store_id IS NULL OR store_id = ''      THEN 'missing_store'
          WHEN sku IS NULL OR sku = ''                THEN 'missing_sku'
          WHEN rn > 1                                 THEN 'duplicate_order_id'
        END
    """
    con.execute("DROP TABLE IF EXISTS silver_sales")
    con.execute("DROP TABLE IF EXISTS silver_quarantine")
    con.execute(f"""
        CREATE TABLE silver_sales AS
        SELECT order_id, order_date, store_id, sku, quantity, unit_price,
               quantity * unit_price AS revenue, _ingested_at, _batch_id
        FROM silver_stage WHERE ({reason}) IS NULL
    """)
    con.execute(f"""
        CREATE TABLE silver_quarantine AS
        SELECT *, {reason} AS rejection_reason
        FROM silver_stage WHERE ({reason}) IS NOT NULL
    """)
    con.execute("DROP TABLE silver_stage")

    # ---------------- GOLD: KPIs ----------------
    con.execute("""
        CREATE OR REPLACE VIEW gold_daily_kpis AS
        SELECT order_date,
               sum(revenue)                AS revenue,
               count(DISTINCT order_id)    AS orders,
               sum(quantity)               AS units,
               count(DISTINCT store_id)    AS active_stores
        FROM silver_sales GROUP BY order_date ORDER BY order_date
    """)

    # ---------------- GOLD: demand features ----------------
    con.execute("DROP TABLE IF EXISTS gold_demand_features")
    con.execute("""
        CREATE TABLE gold_demand_features AS
        SELECT order_date, store_id, sku,
               sum(quantity)                       AS units,
               dayofweek(order_date)               AS day_of_week,
               day(order_date)                     AS day_of_month,
               month(order_date)                   AS month
        FROM silver_sales GROUP BY order_date, store_id, sku
    """)

    # ---------------- GOLD: churn features ----------------
    # Training set: features as of a cutoff, label = no order in the window after cutoff.
    # Scoring set : features as of the latest date (no label) -> used for live risk.
    max_d = con.execute("SELECT max(order_date) FROM silver_sales").fetchone()[0]
    for name, offset, labelled in (("gold_churn_features", CHURN_WINDOW_DAYS, True),
                                   ("gold_churn_scoring", 0, False)):
        label_sql = (", CASE WHEN EXISTS (SELECT 1 FROM silver_sales s2 WHERE s2.store_id = f.store_id "
                     "AND s2.order_date > f.asof) THEN 0 ELSE 1 END AS churned") if labelled else ""
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"""
            CREATE TABLE {name} AS
            WITH a AS (SELECT CAST(DATE '{max_d}' - INTERVAL {offset} DAY AS DATE) AS asof),
            f AS (
              SELECT s.store_id,
                     a.asof,
                     count(DISTINCT s.order_id)                  AS order_count,
                     sum(s.revenue)                              AS lifetime_spend,
                     sum(s.revenue) / count(DISTINCT s.order_id) AS avg_basket_size,
                     date_diff('day', max(s.order_date), a.asof) AS days_since_last_order
              FROM silver_sales s, a
              WHERE s.order_date <= a.asof
              GROUP BY s.store_id, a.asof
            )
            SELECT f.*{label_sql}
            FROM f
        """)

    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ["bronze_sales", "silver_sales", "silver_quarantine",
                        "gold_demand_features", "gold_churn_features", "gold_churn_scoring"]}
    print("Pipeline complete:", counts)
    print(con.execute("SELECT rejection_reason, count(*) n FROM silver_quarantine GROUP BY 1 ORDER BY 2 DESC").df())
    con.close()


if __name__ == "__main__":
    run()
