"""
train_model.py
==============
Trains a selling-price prediction model on the Ecommerce (Flipkart) dataset.

Usage:
    python train_model.py --data ../Zomato-data-.csv
    python train_model.py --data path/to/your/dataset.csv

Outputs:
    models/price_model.pkl   – trained Random Forest pipeline
    models/label_encoders.pkl – fitted category/brand mappings
    models/model_metrics.json – MAE, RMSE, R² on test set
"""

import argparse
import json
import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# ─────────────────────────── helpers ────────────────────────────────────────

def parse_price(series: pd.Series) -> pd.Series:
    """Strip commas/₹ and convert to numeric."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "").str.replace("₹", "").str.strip(),
        errors="coerce"
    )


def parse_discount(series: pd.Series) -> pd.Series:
    """Extract numeric percentage from strings like '69% off'."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )


def load_and_clean(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)

    # ── standardise column names ──────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop useless columns if present
    drop_cols = ["unnamed:_0", "_id", "images", "pid", "product_details", "url",
                 "crawled_at", "description", "description2"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print(f"   Shape after drop: {df.shape}")

    # ── numeric cleaning ──────────────────────────────────────────────────
    df["actual_price"]  = parse_price(df["actual_price"])
    df["selling_price"] = parse_price(df["selling_price"])
    df["discount_pct"]  = parse_discount(df.get("discount", pd.Series(dtype=str)))

    # Derived feature: savings amount
    df["savings"] = df["actual_price"] - df["selling_price"]

    # Drop rows with missing target
    before = len(df)
    df.dropna(subset=["selling_price"], inplace=True)
    print(f"   Dropped {before - len(df)} rows with missing selling_price")

    # Cap extreme outliers (99th percentile)
    cap = df["selling_price"].quantile(0.99)
    df = df[df["selling_price"] <= cap]
    print(f"   After outlier cap (99th pct ≤ {cap:.0f}): {len(df)} rows")

    return df


def encode_categoricals(df: pd.DataFrame):
    """Label-encode low-cardinality string columns; return df + encoder dict."""
    cat_cols = ["brand", "category", "sub_category", "seller"]
    encoders = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def build_features(df: pd.DataFrame):
    feature_cols = [
        "actual_price", "discount_pct", "savings",
        "average_rating",
        "brand", "category", "sub_category",
        "out_of_stock",
    ]
    # only keep features that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    df["out_of_stock"] = df["out_of_stock"].astype(int) if "out_of_stock" in df.columns else 0

    X = df[feature_cols].copy()
    y = df["selling_price"].copy()
    return X, y, feature_cols


# ─────────────────────────── training ───────────────────────────────────────

def train(path: str):
    df = load_and_clean(path)
    df, encoders = encode_categoricals(df)
    X, y, feature_cols = build_features(df)

    print(f"\n[2/5] Feature matrix: {X.shape}  |  target: selling_price")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Model pipeline ────────────────────────────────────────────────────
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    print("\n[3/5] Training Random Forest pipeline …")
    model.fit(X_train, y_train)

    # ── Metrics ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n[4/5] Test-set metrics:")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   R²   : {r2:.4f}")

    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Feature importances plot ──────────────────────────────────────────
    rf_step = model.named_steps["rf"]
    importances = pd.Series(rf_step.feature_importances_, index=feature_cols)
    importances.sort_values().plot(kind="barh", figsize=(8, 5), color="#3b82d4")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "feature_importances.png"), dpi=120)
    plt.close()

    # ── Actual vs Predicted scatter ───────────────────────────────────────
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test[:2000], y_pred[:2000], alpha=0.3, s=10, color="#3b82d4")
    lim = max(y_test.max(), y_pred.max())
    plt.plot([0, lim], [0, lim], "r--", linewidth=1)
    plt.xlabel("Actual Price (₹)")
    plt.ylabel("Predicted Price (₹)")
    plt.title("Actual vs Predicted Selling Price")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "actual_vs_predicted.png"), dpi=120)
    plt.close()

    # ── Save artefacts ────────────────────────────────────────────────────
    print("\n[5/5] Saving model artefacts …")
    with open(os.path.join(MODELS_DIR, "price_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    # Store encoder categories for UI dropdowns
    encoder_classes = {col: list(le.classes_) for col, le in encoders.items()}
    with open(os.path.join(MODELS_DIR, "encoder_classes.json"), "w") as f:
        json.dump(encoder_classes, f, indent=2)

    print("\n✅ Training complete. Artefacts saved to models/")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


# ─────────────────────────── entry point ────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ecommerce price prediction model")
    parser.add_argument("--data", default="../Zomato-data-.csv",
                        help="Path to the CSV dataset")
    args = parser.parse_args()
    train(args.data)
