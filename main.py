from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import pandas as pd
import numpy as np
from typing import List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datasight")

# Initialize the API engine
app = FastAPI(title="DataSight API Engine")

# --- CORS SECURITY CONFIGURATION ---
# Add every frontend origin that needs to call this API.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",   # Vite dev server default port
    "https://yourfrontend.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- CONSTANTS ---
MIN_ROWS_FOR_MODEL = 3          # below this, a regression isn't meaningful
MIN_ROWS_FOR_SEASONALITY = 14   # need ~2 weeks before trusting day-of-week patterns


# --- SCHEMAS ---
class DataPoint(BaseModel):
    date: str
    users: int
    revenue: float

    @field_validator("users")
    @classmethod
    def users_non_negative(cls, v):
        if v < 0:
            raise ValueError("users must be >= 0")
        return v

    @field_validator("revenue")
    @classmethod
    def revenue_non_negative(cls, v):
        if v < 0:
            raise ValueError("revenue must be >= 0")
        return v


class AnalysisResponse(BaseModel):
    message: str
    total_rows_ingested: int
    computed_total_revenue: float
    computed_average_users: float
    predicted_next_day_revenue: float
    prediction_lower_bound: float
    prediction_upper_bound: float
    model_r2: float
    model_confidence: str
    features_used: List[str]


# --- ROUTES ---
@app.get("/")
def read_root():
    return {"status": "System Online", "message": "DataSight Backend is active."}


def _build_features(df: pd.DataFrame):
    """Engineer features, adapting to how much data is actually available."""
    df = df.copy()
    df["day_index"] = range(len(df))
    feature_cols = ["day_index", "users"]

    # Only trust day-of-week seasonality once there's enough history
    if len(df) >= MIN_ROWS_FOR_SEASONALITY:
        dow = df["date"].dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        feature_cols += ["dow_sin", "dow_cos"]

    # Rolling average smooths out single-day noise
    df["rolling_avg_3"] = df["revenue"].rolling(window=3, min_periods=1).mean()
    feature_cols.append("rolling_avg_3")

    return df, feature_cols


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_data(payload: List[DataPoint]):
    # 0. Guard against empty / insufficient payloads
    if not payload:
        raise HTTPException(status_code=400, detail="Payload cannot be empty.")
    if len(payload) < MIN_ROWS_FOR_MODEL:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_ROWS_FOR_MODEL} data points are required to build a model."
        )

    # 1. Convert JSON payload to DataFrame (pydantic's own serializer, not vars())
    try:
        df = pd.DataFrame([p.model_dump() for p in payload])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse payload: {str(e)}")

    if df[["users", "revenue"]].isnull().any().any():
        raise HTTPException(status_code=400, detail="Payload contains null users/revenue values.")

    # 2. Parse and sort by date; drop duplicate dates (keep the latest entry)
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        raise HTTPException(status_code=400, detail="One or more 'date' values could not be parsed.")

    df = df.drop_duplicates(subset="date", keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < MIN_ROWS_FOR_MODEL:
        raise HTTPException(
            status_code=400,
            detail="Not enough unique dated rows remain after removing duplicates to build a model."
        )

    # 3. Historical metrics
    total_revenue = float(df["revenue"].sum())
    avg_users = float(df["users"].mean())

    # --- MACHINE LEARNING ENGINE ---
    # 4. Feature engineering
    df, feature_cols = _build_features(df)
    X = df[feature_cols].values
    y = df["revenue"].values

    # 5. Train a scaled Ridge regression (more stable than plain OLS on small/noisy data)
    try:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        pipeline.fit(X, y)
        in_sample_preds = pipeline.predict(X)
        residuals = y - in_sample_preds
        residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0
        r2 = float(pipeline.score(X, y))
    except Exception as e:
        logger.exception("Model fitting failed")
        raise HTTPException(status_code=500, detail=f"Model fitting failed: {str(e)}")

    # 6. Build the feature row for "tomorrow" and predict
    next_row = {
        "day_index": len(df),
        "users": avg_users,
        "rolling_avg_3": float(df["revenue"].tail(3).mean()),
    }
    if "dow_sin" in feature_cols:
        next_date = df["date"].iloc[-1] + pd.Timedelta(days=1)
        next_dow = next_date.dayofweek
        next_row["dow_sin"] = np.sin(2 * np.pi * next_dow / 7)
        next_row["dow_cos"] = np.cos(2 * np.pi * next_dow / 7)

    try:
        X_next = np.array([[next_row[c] for c in feature_cols]])
        predicted_revenue = float(pipeline.predict(X_next)[0])
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Revenue can't be negative — clamp the point estimate and the interval
    predicted_revenue = max(predicted_revenue, 0.0)
    lower_bound = max(predicted_revenue - 1.96 * residual_std, 0.0)
    upper_bound = predicted_revenue + 1.96 * residual_std

    # 7. Plain-English confidence label from R²
    if r2 >= 0.7:
        confidence = "high"
    elif r2 >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "message": "Data successfully processed and modeled.",
        "total_rows_ingested": len(df),
        "computed_total_revenue": total_revenue,
        "computed_average_users": avg_users,
        "predicted_next_day_revenue": round(predicted_revenue, 2),
        "prediction_lower_bound": round(lower_bound, 2),
        "prediction_upper_bound": round(upper_bound, 2),
        "model_r2": round(r2, 3),
        "model_confidence": confidence,
        "features_used": feature_cols,
    }