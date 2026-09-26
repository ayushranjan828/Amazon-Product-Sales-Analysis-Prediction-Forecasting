from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Dataset" / "amazon_enriched.csv"
MODEL_PATH = ROOT / "Notebook" / "rf_units_sold_model.joblib"


# ---------------- Data (same cleaning as Notebook2) ----------------
@lru_cache
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["about_product", "review_content", "user_id", "img_link", "product_link"])

    df["discounted_price"] = df["discounted_price"].str.replace(r"[₹,]", "", regex=True).astype(float)
    df["actual_price"] = df["actual_price"].str.replace(r"[₹,]", "", regex=True).astype(float)
    df["discount_percentage"] = df["discount_percentage"].str.rstrip("%").astype(float) / 100
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = pd.to_numeric(df["rating_count"].str.replace(",", "", regex=False), errors="coerce")
    for c in ["launch_date", "first_sale_date", "last_sale_date"]:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")

    df[["cat_l1", "cat_l2", "cat_l3"]] = df["category"].str.split("|", expand=True).iloc[:, :3]
    df = df.dropna(subset=["rating", "rating_count", "cat_l3"])

    df["active_days"] = (df.last_sale_date - df.first_sale_date).dt.days.clip(lower=1)
    df["units_per_day"] = df.total_units_sold / df.active_days
    df["product_age_days"] = (df.last_sale_date.max() - df.launch_date).dt.days
    return df


@lru_cache
def products() -> pd.DataFrame:
    return load_data().drop_duplicates("product_id").copy()


@lru_cache
def load_model() -> dict:
    return joblib.load(MODEL_PATH)


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts."""
    df = df.round(2).astype(object)
    return df.where(pd.notna(df), None).to_dict(orient="records")


# ---------------- Overview / EDA ----------------
def summary() -> dict:
    p = products()
    return {
        "products": int(len(p)),
        "brands": int(p.brand.nunique()),
        "categories": int(p.main_category.nunique()),
        "total_units": int(p.total_units_sold.sum()),
        "net_revenue": float(p.net_revenue_inr.sum()),
        "total_profit": float(p.total_profit_inr.sum()),
        "avg_rating": round(float(p.rating.mean()), 2),
        "avg_return_rate": round(float(p.return_rate_pct.mean()), 2),
    }


def group_perf(key: str, top: int | None = None) -> list[dict]:
    t = (products().groupby(key)
         .agg(products=("product_id", "count"),
              units=("total_units_sold", "sum"),
              revenue=("net_revenue_inr", "sum"),
              profit=("total_profit_inr", "sum"),
              avg_rating=("rating", "mean"),
              avg_delivery=("avg_delivery_days", "mean"),
              avg_return=("return_rate_pct", "mean"))
         .sort_values("profit", ascending=False)
         .reset_index()
         .rename(columns={key: "name"}))
    t["name"] = t["name"].astype(str)
    return _records(t.head(top) if top else t)


def top_products(by: str = "total_profit_inr", n: int = 10) -> list[dict]:
    cols = ["product_name", "brand", "main_category", "selling_price_inr",
            "total_orders", "net_revenue_inr", "total_profit_inr"]
    t = products()[cols].sort_values(by, ascending=False).head(n).copy()
    t["product_name"] = t.product_name.str[:60]
    return _records(t)


def distributions() -> dict:
    p = products()
    rating = p.rating.value_counts().sort_index()
    disc, edges = np.histogram(p.discount_percentage * 100, bins=10, range=(0, 100))
    return {
        "rating": {"labels": [str(x) for x in rating.index], "values": rating.tolist()},
        "discount": {"labels": [f"{int(edges[i])}-{int(edges[i + 1])}%" for i in range(len(disc))],
                     "values": disc.tolist()},
    }


# ---------------- Prediction ----------------
def options() -> dict:
    p = products()
    return {
        "warehouse_region": sorted(p.warehouse_region.dropna().unique().tolist()),
        "fulfilment_type": sorted(p.fulfilment_type.dropna().unique().tolist()),
        "demand_tier": ["Low", "Medium", "High"],
        "main_category": sorted(p.main_category.dropna().unique().tolist()),
        "brand": sorted(p.brand.dropna().unique().tolist()),
    }


def predict_units(payload: dict) -> dict:
    bundle = load_model()
    X = pd.DataFrame([payload])[bundle["features"]]
    units = max(float(bundle["model"].predict(X)[0]), 0.0)
    return {
        "predicted_units": round(units),
        "estimated_revenue": round(units * payload["selling_price_inr"], 2),
        "estimated_profit": round(units * (payload["selling_price_inr"] - payload["cost_price_inr"]), 2),
        "model_cv_mae": round(float(bundle["cv_mae"]), 2),
        "model_cv_r2": round(float(bundle["cv_r2"]), 3),
    }


# ---------------- Forecasting ----------------
@lru_cache
def monthly_units() -> pd.Series:
    # Spread each product's total units evenly across its active months
    records = []
    for _, r in load_data().dropna(subset=["first_sale_date", "last_sale_date"]).iterrows():
        start = r.first_sale_date.to_period("M").to_timestamp()
        end = r.last_sale_date.to_period("M").to_timestamp()
        if end < start:
            continue
        months = pd.date_range(start, end, freq="MS")
        for m in months:
            records.append((m, r.total_units_sold / len(months)))
    m = pd.DataFrame(records, columns=["month", "units"]).groupby("month").sum().sort_index()
    return m["units"].asfreq("MS")


@lru_cache
def forecast(periods: int = 6) -> dict:
    series = monthly_units()
    fit = ETSModel(series, error="add", trend="add", damped_trend=True).fit(disp=False)
    start = series.index[-1] + pd.offsets.MonthBegin(1)
    end = series.index[-1] + pd.offsets.MonthBegin(periods)
    fr = fit.get_prediction(start=start, end=end).summary_frame(alpha=0.05).clip(lower=0)
    return {
        "history": {"labels": series.index.strftime("%Y-%m").tolist(),
                    "values": series.round(0).tolist()},
        "forecast": {"labels": fr.index.strftime("%Y-%m").tolist(),
                     "mean": fr["mean"].round(0).tolist(),
                     "lower": fr["pi_lower"].round(0).tolist(),
                     "upper": fr["pi_upper"].round(0).tolist()},
    }


# ---------------- Business insights ----------------
def discount_bands() -> list[dict]:
    p = products()
    bins = [0, .10, .20, .30, .40, .50, .60, .70, 1.0]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70%+"]
    p["discount_band"] = pd.cut(p.discount_percentage, bins=bins, labels=labels, include_lowest=True)
    band = (p.groupby(["main_category", "discount_band"], observed=True)
            .agg(n=("product_id", "count"),
                 profit_per_product=("total_profit_inr", "median"),
                 margin=("profit_margin_pct", "median"))
            .reset_index().query("n >= 5"))
    best = (band.sort_values("profit_per_product", ascending=False)
            .groupby("main_category").head(1)
            .rename(columns={"discount_band": "optimal_band"}))
    best["optimal_band"] = best.optimal_band.astype(str)
    return _records(best)


def high_returns(n: int = 15) -> dict:
    p = products()
    p90 = p.groupby("main_category").return_rate_pct.transform(lambda s: s.quantile(.90))
    p["cat_ret_mean"] = p.groupby("main_category").return_rate_pct.transform("mean")
    p["return_loss_inr"] = p.gross_revenue_inr * p.return_rate_pct / 100
    flagged = p[p.return_rate_pct >= p90].sort_values("return_loss_inr", ascending=False)
    t = flagged[["product_name", "brand", "main_category", "return_rate_pct",
                 "cat_ret_mean", "rating", "fulfilment_type", "return_loss_inr"]].head(n).copy()
    t["product_name"] = t.product_name.str[:50]
    return {"flagged": int(len(flagged)), "total": int(len(p)),
            "revenue_lost": float(flagged.return_loss_inr.sum()), "rows": _records(t)}


def restock(horizon: int = 3, lead_time: int = 1, safety: float = 0.2, n: int = 20) -> dict:
    p = products()
    fc = forecast(6)["forecast"]["mean"]
    growth = np.mean(fc[:horizon]) / monthly_units().iloc[-1]

    p["monthly_demand"] = p.units_per_day * 30 * growth
    p["horizon_demand"] = p.monthly_demand * horizon
    p["safety_stock"] = p.monthly_demand * lead_time * safety
    p["reorder_point"] = p.monthly_demand * lead_time + p.safety_stock
    p["months_of_cover"] = p.stock_quantity / p.monthly_demand.replace(0, np.nan)
    p["restock_qty"] = (p.horizon_demand + p.safety_stock - p.stock_quantity).clip(lower=0).round()

    p["restock_action"] = np.select(
        [p.stock_quantity <= p.reorder_point, p.stock_quantity < p.horizon_demand, p.months_of_cover > 12],
        ["URGENT restock", "Restock", "Overstock"], default="Healthy")

    need = (p[p.restock_action.isin(["URGENT restock", "Restock"])]
            .sort_values(["restock_action", "months_of_cover"], ascending=[False, True]))
    t = need[["product_name", "main_category", "warehouse_region", "stock_quantity",
              "monthly_demand", "months_of_cover", "restock_qty", "restock_action"]].head(n).copy()
    t["product_name"] = t.product_name.str[:50]
    return {
        "growth_factor": round(float(growth), 3),
        "status_counts": {k: int(v) for k, v in p.restock_action.value_counts().items()},
        "by_warehouse": {k: float(v) for k, v in
                         p.groupby("warehouse_region").restock_qty.sum().sort_values(ascending=False).items()},
        "rows": _records(t),
    }
