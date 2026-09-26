from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from . import analytics as a          # uvicorn Backend.main:app (from project root)
except ImportError:
    import analytics as a                 # uvicorn main:app / python main.py (from Backend folder)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data, model and forecast once so first requests are fast
    a.load_data(); a.load_model(); a.forecast(6)
    yield


app = FastAPI(title="Amazon Sales Analysis API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ProductIn(BaseModel):
    selling_price_inr: float = Field(499.0, gt=0)
    mrp_inr: float = Field(1299.0, gt=0)
    cost_price_inr: float = Field(320.0, ge=0)
    discount_percentage: float = Field(0.62, ge=0, le=1)
    rating: float = Field(4.3, ge=0, le=5)
    rating_count: int = Field(15000, ge=0)
    profit_margin_pct: float = 22.5
    stock_quantity: int = Field(500, ge=0)
    avg_delivery_days: float = Field(3, ge=0)
    return_rate_pct: float = Field(4.5, ge=0, le=100)
    is_seasonal_product: bool = False
    product_age_days: int = Field(365, ge=0)
    warehouse_region: str = "North"
    fulfilment_type: str = "FBA"
    demand_tier: Literal["Low", "Medium", "High"] = "High"
    main_category: str = "Electronics"
    brand: str = "boAt"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary")
def summary():
    return a.summary()


@app.get("/api/eda")
def eda():
    return {
        "categories": a.group_perf("main_category"),
        "brands": a.group_perf("brand", top=10),
        "regions": a.group_perf("warehouse_region"),
        "fulfilment": a.group_perf("fulfilment_type"),
        "seasonal": a.group_perf("is_seasonal_product"),
        "distributions": a.distributions(),
        "top_profit": a.top_products("total_profit_inr"),
        "top_revenue": a.top_products("net_revenue_inr"),
    }


@app.get("/api/options")
def options():
    return a.options()


@app.post("/api/predict")
def predict(product: ProductIn):
    return a.predict_units(product.model_dump())


@app.get("/api/forecast")
def forecast(months: int = Query(6, ge=1, le=24)):
    return a.forecast(months)


@app.get("/api/insights")
def insights(horizon: int = Query(3, ge=1, le=6)):
    return {
        "discount_bands": a.discount_bands(),
        "high_returns": a.high_returns(),
        "restock": a.restock(horizon=horizon),
    }


# Serve the frontend from the same server (must be mounted last)
app.mount("/", StaticFiles(directory=a.ROOT / "Frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
