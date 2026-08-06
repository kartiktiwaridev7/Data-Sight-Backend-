import io
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Large-dataset guardrails (new)
MAX_ROWS_PER_JSON_REQUEST = 300_000   # /analyze (JSON body) hard ceiling
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB cap for /analyze/upload
MAX_ROWS_FOR_TRAINING = 50_000        # cap on rows actually fit into the regression


# --- SCHEMAS ---
class DataPoint(BaseModel):
    date: str
    users: Optional[float] = None
    revenue: Optional[float] = None

    @field_validator("users", "revenue", mode="before")
    @classmethod
    def _coerce_numeric(cls, v):
        """
        Real-world CSVs (especially big, hand-exported ones) are messy:
        blank cells, "$1,200", stray whitespace, "N/A", etc. Previously
        `users`/`revenue` were plain `int`/`float` fields, so ONE bad cell
        anywhere in the payload raised a Pydantic ValidationError for the
        WHOLE list and FastAPI returned a 422 for the entire request. That
        is almost certainly why the dashboard failed on larger datasets
        while small hand-typed ones worked (the terminal log even shows
        "POST /analyze 422 Unprocessable Content"): bigger files simply
        have a higher chance of containing one messy row.

        Now we try to salvage the value here. If it truly can't be parsed
        we return None, and the row is dropped later during cleaning
        instead of failing the entire batch.
        """
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip().replace(",", "").replace("$", "")
            if v == "" or v.lower() in ("nan", "null", "none", "n/a", "-"):
                return None
            try:
                return float(v)
            except ValueError:
                return None
        return v

    @field_validator("users", "revenue")
    @classmethod
    def _drop_negative(cls, v):
        # Negative users/revenue is bad data, not a reason to fail the
        # whole upload -- treat it as missing instead of raising.
        if v is not None and v < 0:
            return None
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


def _build_features(df: pd.DataFrame, anchor_date=None):
    """
    Engineer features, adapting to how much data is actually available.

    `day_index` used to be `range(len(df))` -- a plain row counter. That
    breaks the moment there's a gap in the dates (a skipped weekend, a
    missed logging day, etc.), because two rows that are 3 calendar days
    apart end up only 1 index apart. The regression then reads that gap as
    a steep single-day jump in revenue and learns an inflated growth slope,
    which makes "tomorrow" predictions overshoot.

    Fix: `day_index` is now the actual number of calendar days since an
    anchor date, so a 3-day gap contributes a day_index difference of 3,
    not 1, and the learned slope reflects real elapsed time.
    """
    df = df.copy()
    if anchor_date is None:
        anchor_date = df["date"].min()
    df["day_index"] = (df["date"] - anchor_date).dt.days
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


def _clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Shared cleaning path for BOTH ingestion routes (JSON list and CSV
    upload). Coerces types and drops unusable rows instead of throwing
    the whole request away, which is the key change that lets large,
    real-world (i.e. imperfect) datasets get through at all.
    """
    starting_rows = len(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["users"] = pd.to_numeric(df["users"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    df = df.dropna(subset=["date", "users", "revenue"])
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    dropped_rows = starting_rows - len(df)
    return df, dropped_rows


def _analyze_dataframe(df: pd.DataFrame) -> dict:
    """
    Everything downstream of "I have a raw date/users/revenue dataframe" --
    cleaning, metrics, feature engineering, model fit, and prediction.
    Both /analyze and /analyze/upload funnel into this so behaviour stays
    identical no matter how the data arrived.
    """
    df, dropped_rows = _clean_dataframe(df)

    if len(df) < MIN_ROWS_FOR_MODEL:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(df)} usable row(s) remained after cleaning "
                f"({dropped_rows} row(s) were dropped for bad/missing date, "
                f"users, or revenue values). At least {MIN_ROWS_FOR_MODEL} "
                "valid rows are required to build a model."
            ),
        )

    # 3. Historical metrics -- always computed on the FULL cleaned dataset,
    # never on a sample, so totals/averages stay exact no matter how large
    # the input is.
    total_revenue = float(df["revenue"].sum())
    avg_users = float(df["users"].mean())

    # --- MACHINE LEARNING ENGINE ---
    # For very large datasets, fit the model on an evenly-spaced sample
    # instead of every single row. This keeps request latency bounded
    # (Ridge/StandardScaler are fast, but there's no reason to fit on
    # millions of rows when a representative subset gives the same fit).
    # Aggregates above and the "next row" features below still use the
    # FULL dataset, so the prediction remains as accurate as possible.
    model_source_df = df
    if len(df) > MAX_ROWS_FOR_TRAINING:
        step = max(len(df) // MAX_ROWS_FOR_TRAINING, 1)
        model_source_df = df.iloc[::step].reset_index(drop=True)

    # 4. Feature engineering. The anchor date is fixed to the FULL
    # dataset's earliest date (not the sampled subset's), so day_index
    # means the same thing whether or not sampling kicked in above, and
    # the next-day prediction row lines up with what the model was
    # trained on.
    anchor_date = df["date"].min()
    model_df, feature_cols = _build_features(model_source_df, anchor_date=anchor_date)
    X = model_df[feature_cols].values
    y = model_df["revenue"].values

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

    # 6. Build the feature row for "tomorrow" and predict.
    # rolling_avg_3/date use the FULL dataframe (df), not the (possibly
    # sampled) model_df, so the extrapolation point stays accurate.
    # "Tomorrow" is always the calendar day after the last observed date,
    # even if that date itself followed a gap -- day_index is measured
    # from the same anchor_date the model was trained against, so the
    # step from the last training row to this one is the true number of
    # elapsed days, not an artificial "+1 row".
    next_date = df["date"].iloc[-1] + pd.Timedelta(days=1)
    next_row = {
        "day_index": (next_date - anchor_date).days,
        "users": avg_users,
        "rolling_avg_3": float(df["revenue"].tail(3).mean()),
    }
    if "dow_sin" in feature_cols:
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

    message = "Data successfully processed and modeled."
    if dropped_rows:
        message += f" ({dropped_rows} row(s) were skipped for invalid/missing values.)"
    if len(df) > MAX_ROWS_FOR_TRAINING:
        message += (
            f" Model was trained on a {len(model_df)}-row sample of the "
            f"{len(df)}-row dataset for speed; totals/averages use all rows."
        )

    return {
        "message": message,
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


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_data(payload: List[DataPoint]):
    # 0. Guard against empty / insufficient / oversized payloads
    if not payload:
        raise HTTPException(status_code=400, detail="Payload cannot be empty.")
    if len(payload) < MIN_ROWS_FOR_MODEL:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_ROWS_FOR_MODEL} data points are required to build a model."
        )
    if len(payload) > MAX_ROWS_PER_JSON_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Payload has {len(payload)} rows, which exceeds the "
                f"{MAX_ROWS_PER_JSON_REQUEST}-row limit for JSON requests. "
                "For bigger files, POST the raw CSV to /analyze/upload instead "
                "-- it skips JSON serialization entirely and handles much "
                "larger datasets."
            ),
        )

    # 1. Convert to a DataFrame. Pulling each field into its own list is
    # noticeably faster than building a dict per row (model_dump()) once
    # you're dealing with tens/hundreds of thousands of rows.
    try:
        df = pd.DataFrame({
            "date": [p.date for p in payload],
            "users": [p.users for p in payload],
            "revenue": [p.revenue for p in payload],
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse payload: {str(e)}")

    return _analyze_dataframe(df)


@app.post("/analyze/upload", response_model=AnalysisResponse)
async def analyze_uploaded_csv(file: UploadFile = File(...)):
    """
    New endpoint, purpose-built for large datasets.

    Sending a big JSON array of objects (as /analyze does) means the browser
    has to build + stringify one JS object per row and the backend has to
    validate one Pydantic model per row. That overhead scales badly. Letting
    the frontend upload the raw CSV file instead -- and parsing it directly
    with pandas -- is dramatically cheaper for large files and is the
    recommended path once a dataset gets past a few tens of thousands of rows.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
        )
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    df.columns = df.columns.str.strip().str.lower()
    required_cols = {"date", "users", "revenue"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required column(s): {', '.join(sorted(missing_cols))}"
        )

    df = df[["date", "users", "revenue"]]

    return _analyze_dataframe(df)