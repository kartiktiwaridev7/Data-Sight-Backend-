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
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",   # Vite dev server default port
    "https://datasight.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONSTANTS ---
MIN_ROWS_FOR_MODEL = 3          # below this, a regression isn't meaningful
MIN_ROWS_FOR_SEASONALITY = 14   # need ~2 weeks before trusting day-of-week patterns

MAX_ROWS_PER_JSON_REQUEST = 300_000   # /analyze (JSON body) hard ceiling
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB cap for /analyze/upload
MAX_ROWS_FOR_TRAINING = 50_000        # cap on rows actually fit into the regression
MAX_CHART_POINTS = 500                # cap on points sent back for the frontend chart


# --- SCHEMAS ---
class DataPoint(BaseModel):
    date: str
    users: Optional[float] = None
    revenue: Optional[float] = None

    @field_validator("users", "revenue", mode="before")
    @classmethod
    def _coerce_numeric(cls, v):
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
        if v is not None and v < 0:
            return None
        return v


class HistoricalPoint(BaseModel):
    date: str
    revenue: float
    users: float


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
    historical_series: List[HistoricalPoint]


# --- ROUTES ---
@app.get("/")
def read_root():
    return {"status": "System Online", "message": "DataSight Backend is active."}


def _build_features(df: pd.DataFrame, anchor_date=None):
    """
    Engineer features, adapting to how much data is actually available.
    `day_index` is the actual number of calendar days since an anchor date
    (not a plain row counter), so gaps in the dates don't get read as a
    steep single-day revenue jump.
    """
    df = df.copy()
    if anchor_date is None:
        anchor_date = df["date"].min()
    df["day_index"] = (df["date"] - anchor_date).dt.days
    feature_cols = ["day_index", "users"]

    if len(df) >= MIN_ROWS_FOR_SEASONALITY:
        dow = df["date"].dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        feature_cols += ["dow_sin", "dow_cos"]

    df["rolling_avg_3"] = df["revenue"].rolling(window=3, min_periods=1).mean()
    feature_cols.append("rolling_avg_3")

    return df, feature_cols


def _clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Shared cleaning path for BOTH ingestion routes (JSON list and CSV/JSON
    upload). Coerces types, strips currency formatting like "$1,200", and
    drops unusable rows instead of throwing the whole request away.
    """
    starting_rows = len(df)
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ("users", "revenue"):
        # NOTE: don't check `df[col].dtype == object` here -- pandas 2.x
        # gives string columns `object` dtype, but pandas 3.x defaults to a
        # separate `str` dtype, so that check silently stops matching and
        # currency formatting like "$1,200" never gets stripped, quietly
        # dropping every such row later. `is_numeric_dtype` is dtype-version
        # agnostic: if it's not already numeric, run it through the same
        # string cleanup regardless of which "string" dtype pandas used.
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
            )
            df[col] = df[col].replace(
                {"": None, "nan": None, "None": None, "N/A": None, "n/a": None, "-": None}
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Negative users/revenue is bad data, not a reason to fail the whole
    # upload -- treat it as missing instead of raising.
    df.loc[df["users"] < 0, "users"] = np.nan
    df.loc[df["revenue"] < 0, "revenue"] = np.nan

    df = df.dropna(subset=["date", "users", "revenue"])
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    dropped_rows = starting_rows - len(df)
    return df, dropped_rows


def _analyze_dataframe(df: pd.DataFrame) -> dict:
    """
    Everything downstream of "I have a raw date/users/revenue dataframe" --
    cleaning, metrics, feature engineering, model fit, and prediction.
    BOTH /analyze and /analyze/upload funnel into this, so the response
    shape (and the actual forecasting logic) is always identical no matter
    how the data arrived.
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

    total_revenue = float(df["revenue"].sum())
    avg_users = float(df["users"].mean())

    model_source_df = df
    if len(df) > MAX_ROWS_FOR_TRAINING:
        step = max(len(df) // MAX_ROWS_FOR_TRAINING, 1)
        model_source_df = df.iloc[::step].reset_index(drop=True)

    anchor_date = df["date"].min()
    model_df, feature_cols = _build_features(model_source_df, anchor_date=anchor_date)
    X = model_df[feature_cols].values
    y = model_df["revenue"].values

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

    predicted_revenue = max(predicted_revenue, 0.0)
    lower_bound = max(predicted_revenue - 1.96 * residual_std, 0.0)
    upper_bound = predicted_revenue + 1.96 * residual_std

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

    # Build the series the frontend actually charts. This is the SAME
    # cleaned/aggregated dataframe the model was built from -- not a
    # re-parse of the raw upload -- so a file like a per-country monthly
    # sales log (no literal "date"/"revenue" columns, many rows per period)
    # charts correctly instead of plotting blank/undefined points.
    chart_df = df
    if len(chart_df) > MAX_CHART_POINTS:
        step = max(len(chart_df) // MAX_CHART_POINTS, 1)
        chart_df = chart_df.iloc[::step]
    historical_series = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "revenue": round(float(row.revenue), 2),
            "users": round(float(row.users), 2),
        }
        for row in chart_df.itertuples()
    ]

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
        "historical_series": historical_series,
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_data(payload: List[DataPoint]):
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
                "For bigger files, POST the raw CSV to /analyze/upload instead."
            ),
        )

    try:
        df = pd.DataFrame({
            "date": [p.date for p in payload],
            "users": [p.users for p in payload],
            "revenue": [p.revenue for p in payload],
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse payload: {str(e)}")

    return _analyze_dataframe(df)


def _find_column(df: pd.DataFrame, cols_lower: dict, candidates: List[str]) -> Optional[str]:
    """Return the first column whose lowercased name contains any of the
    given substrings, checked in priority order (earlier candidates win)."""
    for key in candidates:
        match = next((c for c in df.columns if key in cols_lower[c]), None)
        if match:
            return match
    return None


def _resolve_schema(df: pd.DataFrame) -> Tuple[pd.Series, Optional[str], Optional[str]]:
    """
    Figure out how to get a date, a users-like column, and a revenue-like
    column out of an arbitrary uploaded file.

    Real-world exports rarely use the literal words "date"/"users"/
    "revenue" -- they use things like `year` + `month`, `gross_revenue_usd`,
    `estimated_active_players`, etc. This resolves those cases instead of
    requiring an exact schema match.
    """
    cols_lower = {c: c.lower() for c in df.columns}

    # ---- date: literal column, else synthesize from year(+month, +day) ----
    date_col = _find_column(df, cols_lower, ["date"])
    if date_col:
        date_series = pd.to_datetime(df[date_col], errors="coerce")
    else:
        year_col = next((c for c in df.columns if cols_lower[c] in ("year", "yr", "fiscal_year")), None)
        month_col = next((c for c in df.columns if cols_lower[c] in ("month", "mon", "mo")), None)
        day_col = next((c for c in df.columns if cols_lower[c] in ("day", "dom")), None)
        if year_col and month_col:
            date_series = pd.to_datetime(
                {
                    "year": pd.to_numeric(df[year_col], errors="coerce"),
                    "month": pd.to_numeric(df[month_col], errors="coerce"),
                    "day": pd.to_numeric(df[day_col], errors="coerce") if day_col else 1,
                },
                errors="coerce",
            )
        elif year_col:
            # Year-only granularity: anchor every row to Jan 1st of that year.
            date_series = pd.to_datetime(
                pd.to_numeric(df[year_col], errors="coerce"), format="%Y", errors="coerce"
            )
        else:
            date_series = pd.Series([pd.NaT] * len(df))

    # ---- revenue: prefer an unambiguous "total/gross" revenue field over
    # narrower sub-revenue streams (dlc, add-on, etc.) ----
    revenue_col = _find_column(
        df, cols_lower, ["gross_revenue", "total_revenue", "net_revenue", "revenue", "gross_sales", "sales_usd"]
    )

    # ---- users: prefer an explicit active/unique-user metric over raw
    # transaction counts ----
    users_col = _find_column(
        df,
        cols_lower,
        ["active_player", "active_user", "dau", "mau", "unique_user", "users", "player", "customer"],
    )

    return date_series, revenue_col, users_col


@app.post("/analyze/upload", response_model=AnalysisResponse)
async def analyze_uploaded_csv(file: UploadFile = File(...)):
    """
    Delegates to the same `_analyze_dataframe` pipeline as /analyze, so the
    response shape is always identical no matter how the data arrived.

    Column handling has two tiers:
      1. Exact schema match (columns literally named date/users/revenue,
         one row per period) -- used as-is.
      2. Smart resolution -- date synthesized from year/month/day if no
         date column exists, revenue/users matched against a priority list
         of common real-world names, and (because files matched this way
         are almost always broken out by category -- country, platform,
         etc. -- with several rows per period) the rows are grouped and
         summed per date before modeling, instead of the exact-match path's
         "drop duplicate dates" behaviour, which would silently discard
         nearly the whole file.
    """
    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is {len(contents) / (1024 * 1024):.1f}MB, which exceeds "
                f"the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
            ),
        )

    filename = (file.filename or "").lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Please upload a .csv or .json file.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file has no rows.")

    df.columns = [str(c).strip() for c in df.columns]
    cols_lower = {c: c.lower() for c in df.columns}

    # Tier 1: exact date/users/revenue columns present -- use the original,
    # stricter path (no forced aggregation).
    exact = {}
    for target in ("date", "users", "revenue"):
        match = next((c for c in df.columns if cols_lower[c] == target), None)
        if match:
            exact[target] = match

    if len(exact) == 3:
        clean_df = df.rename(columns={v: k for k, v in exact.items()})[["date", "users", "revenue"]]
        if len(clean_df) > MAX_ROWS_PER_JSON_REQUEST:
            clean_df = clean_df.iloc[:MAX_ROWS_PER_JSON_REQUEST].copy()
        return _analyze_dataframe(clean_df)

    # Tier 2: smart resolution + aggregation for real-world exports that
    # don't use the exact column names, and/or have multiple rows per period.
    date_series, revenue_col, users_col = _resolve_schema(df)

    missing = []
    if date_series.isna().all():
        missing.append("date (or year/month)")
    if revenue_col is None:
        missing.append("revenue")
    if users_col is None:
        missing.append("users")

    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Couldn't find a usable column for: {', '.join(missing)}. "
                "The file needs a date (or year/month) column, a revenue-like "
                "column (e.g. 'revenue', 'gross_revenue_usd'), and a user/player "
                "count column (e.g. 'users', 'active_players', 'customers'). "
                f"Columns found in this file: {', '.join(df.columns)}"
            ),
        )

    work = pd.DataFrame({
        "date": date_series,
        "users": pd.to_numeric(df[users_col], errors="coerce"),
        "revenue": pd.to_numeric(df[revenue_col], errors="coerce"),
    }).dropna(subset=["date", "users", "revenue"])

    if work.empty:
        raise HTTPException(
            status_code=400,
            detail="No rows had usable values in all of the date, users, and revenue columns after cleaning.",
        )

    # Collapse multiple rows per period (e.g. one row per country per month)
    # into a single global total per date -- the model expects one row per
    # time period, not one row per (period, category) combination.
    agg_df = work.groupby("date", as_index=False).agg(users=("users", "sum"), revenue=("revenue", "sum"))
    agg_df = agg_df.sort_values("date").reset_index(drop=True)

    if len(agg_df) > MAX_ROWS_PER_JSON_REQUEST:
        agg_df = agg_df.iloc[:MAX_ROWS_PER_JSON_REQUEST].copy()

    result = _analyze_dataframe(agg_df)
    result["message"] += (
        f" (Detected columns: date from '{exact.get('date', 'year/month')}', "
        f"revenue from '{revenue_col}', users from '{users_col}'; "
        f"{len(df)} input rows were grouped into {len(agg_df)} time periods.)"
    )
    return result