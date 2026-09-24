"""
FastAPI Backend Application & Web Server for Ujjain Onion Price Prediction System.
Serves REST API endpoints and static Vanilla JavaScript frontend.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_and_preprocess
from src.predict import predict_future, load_model_bundle

app = FastAPI(
    title="Ujjain Onion Price Prediction System API",
    description="End-to-end ML time-series forecasting API for Ujjain Mandi, M.P.",
    version="2.0.0"
)

# Enable CORS for potential external client calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directory setup
static_dir = os.path.join(PROJECT_ROOT, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def read_root():
    """Serve the Vanilla JavaScript frontend index.html."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        content={"message": "Ujjain Onion Price Prediction API is running. Frontend index.html not found."},
        status_code=200
    )


@app.get("/api/health")
def health_check():
    """Health check endpoint returning system status and dataset freshness."""
    try:
        df = load_and_preprocess(save_processed=False)
        latest_date = str(pd.to_datetime(df['date']).max().date())
        total_rows = len(df)
        bundle = load_model_bundle()
        model_name = bundle.get('model_name', 'Random Forest')
        return {
            "status": "healthy",
            "market": "Ujjain APMC, Madhya Pradesh",
            "commodity": "Onion",
            "latest_dataset_date": latest_date,
            "total_daily_records": total_rows,
            "active_model": model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict")
def get_prediction(horizon: int = Query(default=1, description="Forecast horizon in days (1, 7, or 14)")):
    """
    Generate future wholesale price predictions for Ujjain Mandi.
    Supports 1, 7, or 14 day horizons.
    """
    if horizon not in [1, 7, 14]:
        raise HTTPException(status_code=400, detail="Invalid horizon. Must be 1, 7, or 14 days.")
    try:
        results = predict_future(horizon_days=horizon)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/history")
def get_history(limit: int = Query(default=60, ge=10, le=365)):
    """Retrieve recent historical daily price observations."""
    try:
        df = load_and_preprocess(save_processed=False)
        df = df.sort_values('date').tail(limit).reset_index(drop=True)
        records = []
        for _, row in df.iterrows():
            records.append({
                "date": str(pd.to_datetime(row['date']).date()),
                "modal_price": float(row['modal_price']),
                "min_price": float(row['min_price']) if 'min_price' in row and pd.notna(row['min_price']) else float(row['modal_price']),
                "max_price": float(row['max_price']) if 'max_price' in row and pd.notna(row['max_price']) else float(row['modal_price']),
                "arrivals_tonnes": float(row['arrivals_tonnes']) if 'arrivals_tonnes' in row and pd.notna(row['arrivals_tonnes']) else 0.0
            })
        return {
            "count": len(records),
            "records": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
def get_metrics():
    """Retrieve model performance metrics and benchmark comparison."""
    try:
        bundle = load_model_bundle()
        comp_path = os.path.join(PROJECT_ROOT, 'models', 'model_comparison.csv')
        benchmark_table = []
        if os.path.exists(comp_path):
            comp_df = pd.read_csv(comp_path)
            benchmark_table = comp_df.to_dict(orient='records')
            
        return {
            "selected_model": bundle.get('model_name', 'Random Forest'),
            "test_metrics": bundle.get('test_metrics', {}),
            "train_date_range": bundle.get('train_date_range'),
            "test_date_range": bundle.get('test_date_range'),
            "benchmark_comparison": benchmark_table
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
