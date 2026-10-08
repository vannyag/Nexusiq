"""Train LightGBM demand forecaster and XGBoost churn classifier from the Gold layer."""
import duckdb
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from config import DB_PATH, MODEL_DIR

DEMAND_FEATURES = ["store_code", "sku_code", "day_of_week", "day_of_month", "month"]
CHURN_FEATURES = ["order_count", "lifetime_spend", "avg_basket_size", "days_since_last_order"]


def train_demand(con):
    df = con.execute("SELECT * FROM gold_demand_features").df()
    stores, skus = sorted(df.store_id.unique()), sorted(df.sku.unique())
    df["store_code"] = df.store_id.map({s: i for i, s in enumerate(stores)})
    df["sku_code"] = df.sku.map({s: i for i, s in enumerate(skus)})

    cutoff = df.order_date.max() - pd.Timedelta(days=30)       # time-based split
    train, test = df[df.order_date <= cutoff], df[df.order_date > cutoff]

    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                              random_state=42, verbose=-1)
    model.fit(train[DEMAND_FEATURES], train.units,
              categorical_feature=["store_code", "sku_code"])
    pred = model.predict(test[DEMAND_FEATURES])
    base = np.full(len(test), train.units.mean())
    print(f"[Demand] MAE model={mean_absolute_error(test.units, pred):.3f} "
          f"vs mean-baseline={mean_absolute_error(test.units, base):.3f}")

    model.fit(df[DEMAND_FEATURES], df.units, categorical_feature=["store_code", "sku_code"])  # refit on all
    joblib.dump({"model": model, "stores": stores, "skus": skus, "features": DEMAND_FEATURES},
                MODEL_DIR / "demand_lgbm.joblib")


def train_churn(con):
    df = con.execute("SELECT * FROM gold_churn_features").df()
    X, y = df[CHURN_FEATURES], df.churned
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08,
                              subsample=0.9, eval_metric="logloss", random_state=42)
    model.fit(Xtr, ytr)
    print(f"[Churn] AUC={roc_auc_score(yte, model.predict_proba(Xte)[:, 1]):.3f} "
          f"(churn rate={y.mean():.1%}, n={len(df)})")
    model.fit(X, y)
    joblib.dump({"model": model, "features": CHURN_FEATURES}, MODEL_DIR / "churn_xgb.joblib")

    # Score every store as of the latest date and persist for the dashboard
    sc = con.execute("SELECT * FROM gold_churn_scoring").df()
    sc["churn_probability"] = model.predict_proba(sc[CHURN_FEATURES])[:, 1]
    sc["risk_band"] = pd.cut(sc.churn_probability, [-0.01, 0.4, 0.7, 1.01], labels=["Low", "Medium", "High"]).astype(str)
    con.execute("DROP TABLE IF EXISTS gold_churn_scores")
    con.execute("CREATE TABLE gold_churn_scores AS SELECT * FROM sc")
    print(sc.risk_band.value_counts().to_dict())


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    train_demand(con)
    train_churn(con)
    con.close()
