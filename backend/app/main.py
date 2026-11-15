from __future__ import annotations

import io
import uuid
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

app = FastAPI(title="DataSentry AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS: dict[str, pd.DataFrame] = {}
REPORTS: dict[str, dict[str, Any]] = {}
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 200


class ModelRequest(BaseModel):
    target: str


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if pd.isna(value):
        return None
    return value


def profile_dataframe(df: pd.DataFrame, dataset_id: str, filename: str) -> dict[str, Any]:
    rows, cols = df.shape
    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    column_details = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    outlier_total = 0
    constant_cols = []
    id_like_cols = []

    for col in df.columns:
        series = df[col]
        missing_count = int(missing[col])
        unique_count = int(series.nunique(dropna=True))
        dtype = str(series.dtype)
        detail: dict[str, Any] = {
            "name": str(col),
            "dtype": dtype,
            "missing": missing_count,
            "missing_pct": round((missing_count / rows) * 100, 2) if rows else 0,
            "unique": unique_count,
            "unique_pct": round((unique_count / rows) * 100, 2) if rows else 0,
            "sample": [json_safe(x) for x in series.dropna().head(3).tolist()],
        }
        if unique_count <= 1:
            constant_cols.append(str(col))
        if rows >= 5 and unique_count / max(rows, 1) > 0.95:
            id_like_cols.append(str(col))
        if col in numeric_cols:
            clean = pd.to_numeric(series, errors="coerce").dropna()
            if len(clean) >= 4:
                q1, q3 = clean.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    count = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
                else:
                    count = 0
                detail["outliers_iqr"] = count
                outlier_total += count
                detail["mean"] = json_safe(clean.mean())
                detail["median"] = json_safe(clean.median())
        column_details.append(detail)

    recommendations = []
    if int(missing.sum()) > 0:
        recommendations.append({
            "severity": "high",
            "title": "Handle missing values",
            "detail": "Choose an imputation strategy or remove rows/columns only after checking why values are missing."
        })
    if duplicate_rows:
        recommendations.append({
            "severity": "medium",
            "title": "Review duplicate rows",
            "detail": f"{duplicate_rows:,} duplicate rows were found. Confirm whether they are accidental before removing them."
        })
    if constant_cols:
        recommendations.append({
            "severity": "low",
            "title": "Review constant columns",
            "detail": "These columns contain one distinct non-null value: " + ", ".join(constant_cols[:8])
        })
    if id_like_cols:
        recommendations.append({
            "severity": "medium",
            "title": "Check identifier-like columns",
            "detail": "High-cardinality columns may be IDs. Exclude them from model features unless they carry meaningful signal: " + ", ".join(id_like_cols[:8])
        })
    if outlier_total:
        recommendations.append({
            "severity": "medium",
            "title": "Inspect numeric outliers",
            "detail": f"The IQR heuristic flagged {outlier_total:,} numeric values. These may be valid rare observations, not necessarily errors."
        })
    if not recommendations:
        recommendations.append({
            "severity": "low",
            "title": "No major basic quality issues detected",
            "detail": "Continue with domain-specific validation, leakage checks, and feature suitability review."
        })

    numeric_summary = []
    for col in numeric_cols[:12]:
        clean = pd.to_numeric(df[col], errors="coerce").dropna()
        numeric_summary.append({
            "name": str(col),
            "mean": json_safe(clean.mean()) if len(clean) else None,
            "median": json_safe(clean.median()) if len(clean) else None,
            "min": json_safe(clean.min()) if len(clean) else None,
            "max": json_safe(clean.max()) if len(clean) else None,
        })

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "rows": rows,
        "columns": cols,
        "memory_kb": round(float(df.memory_usage(deep=True).sum()) / 1024, 2),
        "duplicate_rows": duplicate_rows,
        "missing_cells": int(missing.sum()),
        "missing_pct": round(float(missing.sum()) / max(rows * cols, 1) * 100, 2),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": int(cols - len(numeric_cols)),
        "column_details": column_details,
        "numeric_summary": numeric_summary,
        "recommendations": recommendations,
        "constant_columns": constant_cols,
        "id_like_columns": id_like_cols,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "DataSentry AI API"}


@app.get("/api/sample")
def sample():
    df = pd.DataFrame({
        "CustomerID": [f"C{i:04d}" for i in range(1, 13)],
        "Age": [22, 35, 41, 29, 52, 46, 31, 27, 63, 38, 44, 25],
        "MonthlySpend": [25.5, 80.0, 65.0, 32.0, 120.0, 95.0, 45.0, 29.0, 150.0, 70.0, 88.0, 35.0],
        "SupportCalls": [0, 2, 1, 0, 4, 3, 1, 0, 5, 2, 2, 0],
        "Plan": ["Basic", "Pro", "Pro", "Basic", "Pro", "Pro", "Basic", "Basic", "Pro", "Pro", "Basic", "Basic"],
        "Churn": ["No", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No"],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return {"filename": "customer_churn_sample.csv", "csv": csv_bytes.decode("utf-8")}


@app.post("/api/audit")
async def audit(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File is too large. Maximum size is 15 MB.")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV has no data rows.")
    if len(df) > MAX_ROWS or len(df.columns) > MAX_COLUMNS:
        raise HTTPException(status_code=400, detail=f"Limit is {MAX_ROWS:,} rows and {MAX_COLUMNS} columns.")
    if len(df.columns) < 1:
        raise HTTPException(status_code=400, detail="The CSV must contain at least one column.")
    df.columns = [str(c).strip() or f"column_{i+1}" for i, c in enumerate(df.columns)]
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df
    report = profile_dataframe(df, dataset_id, filename)
    REPORTS[dataset_id] = report
    return report


@app.post("/api/model/{dataset_id}")
def train_models(dataset_id: str, request: ModelRequest):
    df = DATASETS.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload it again.")
    target = request.target
    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column was not found.")
    if df[target].nunique(dropna=True) < 2:
        raise HTTPException(status_code=400, detail="Target needs at least two distinct non-null values.")

    working = df.dropna(subset=[target]).copy()
    if len(working) < 10:
        raise HTTPException(status_code=400, detail="At least 10 rows with a non-missing target are required.")
    y = working[target]
    X = working.drop(columns=[target])
    if X.shape[1] == 0:
        raise HTTPException(status_code=400, detail="At least one feature column besides the target is required.")

    # Remove columns that are entirely missing; remaining missing values are imputed in the pipeline.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No usable feature columns remain.")
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    is_classification = (y.dtype == "object" or str(y.dtype) == "bool" or y.nunique() <= 12)
    if is_classification:
        y = y.astype(str)
        counts = y.value_counts()
        stratify = y if counts.min() >= 2 else None
    else:
        stratify = None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Could not split the data: {exc}") from exc

    models = []
    candidates = (
        [("Logistic Regression", LogisticRegression(max_iter=1000), "classification"),
         ("Random Forest", RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced"), "classification")]
        if is_classification else
        [("Ridge Regression", Ridge(), "regression"),
         ("Random Forest", RandomForestRegressor(n_estimators=120, random_state=42), "regression")]
    )
    for name, estimator, kind in candidates:
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        try:
            pipe.fit(X_train, y_train)
            predictions = pipe.predict(X_test)
            if kind == "classification":
                score = accuracy_score(y_test, predictions)
                secondary = f1_score(y_test, predictions, average="weighted", zero_division=0)
                models.append({"model": name, "accuracy": round(float(score), 4), "f1_weighted": round(float(secondary), 4)})
            else:
                score = r2_score(y_test, predictions) if len(y_test) > 1 else 0.0
                secondary = mean_absolute_error(y_test, predictions)
                models.append({"model": name, "r2": round(float(score), 4), "mae": round(float(secondary), 4)})
        except Exception as exc:
            models.append({"model": name, "error": str(exc)})

    if not any("error" not in m for m in models):
        raise HTTPException(status_code=400, detail="The baseline models could not train on this dataset. Check data types and target values.")
    return {
        "target": target,
        "task": "classification" if is_classification else "regression",
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "results": models,
        "notice": "Single holdout evaluation for exploration only; use cross-validation and leakage checks before drawing conclusions.",
    }


@app.get("/api/report/{dataset_id}")
def get_report(dataset_id: str):
    report = REPORTS.get(dataset_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found. Upload the dataset again.")
    return report
