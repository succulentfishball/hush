"""
ByteLife — Model Training
==========================
Trains two models from bytelife_clean.csv:

  1. INCOME MODEL — Gradient Boosted Quantile Regression
     Predicts P10, P25, P50, P75, P90 of net monthly income.
     Uses log_income as the training target (right-skew correction),
     then exponentiates predictions back to £ for display.

  2. LIFE SATISFACTION MODEL — Gradient Boosted Ordinal Classifier
     Predicts P(satisfaction = 1), P(satisfaction = 2), ..., P(satisfaction = 7).
     Outputs a full probability vector over the 7-point scale.

Both models use scikit-learn's GradientBoostingRegressor /
GradientBoostingClassifier (no GPU required — these are small enough
datasets that CPU training takes minutes). If you want to switch to
GPU-accelerated XGBoost later, the swap is one line.

Usage:
    python train_bytelife.py --data bytelife_clean.csv

Outputs:
    model_income.pkl          — pickled income model (list of 5 quantile models)
    model_satisfaction.pkl    — pickled satisfaction model
    evaluation_report.txt     — printed metrics + predictions on held-out test set
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    classification_report, log_loss
)


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

# Quantiles to predict for income
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_LABELS = ["P10", "P25", "P50", "P75", "P90"]

# Life satisfaction scale in the data (1–7)
SATISFACTION_CLASSES = [1, 2, 3, 4, 5, 6, 7]

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# GBM hyperparameters — conservative defaults, fast to train
GBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------------
# 2. LOAD & VALIDATE
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # The occupation dummies may have been saved as string "True"/"False"
    for col in ["occ_2.0", "occ_3.0"]:
        if col in df.columns:
            df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0})
            # In case they're already numeric booleans
            df[col] = df[col].astype(int)

    # ever_married and sex_female should also be int
    for col in ["ever_married", "sex_female"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Drop any unnamed index column that pandas adds on re-read
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Validate targets exist
    for target in ["log_income", "sclfsato"]:
        if target not in df.columns:
            sys.exit(f"ERROR: Target column '{target}' not found in {path}")

    # Check for age_dv and warn if missing
    if "age_dv" not in df.columns:
        print("⚠  WARNING: age_dv not found in the dataset.")
        print("   Age is a strong predictor — consider re-running extraction to add it.")
        print("   Proceeding without it.\n")

    print(f"Loaded {len(df):,} individuals from {path}")
    print(f"Columns: {list(df.columns)}\n")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Dynamically determine input features from whatever columns are present.
    Excludes known target/ID columns.
    """
    exclude = {"fimnnet_dv", "log_income", "sclfsato", "pidp", "Unnamed: 0"}
    features = [c for c in df.columns if c not in exclude]
    print(f"Using {len(features)} input features: {features}\n")
    return features


# ---------------------------------------------------------------------------
# 3. INCOME MODEL — Quantile Regression
# ---------------------------------------------------------------------------

def train_income_model(X_train, y_train, X_test, y_test, raw_income_test):
    """
    Train one GBM per quantile. Target is log_income.
    Evaluation is done in original £ scale (exponentiate predictions).
    """
    print("=" * 60)
    print("  INCOME MODEL — Quantile Regression (log scale)")
    print("=" * 60)

    models = {}
    predictions = {}

    for q, label in zip(QUANTILES, QUANTILE_LABELS):
        print(f"  Training {label} (quantile={q}) ...", end=" ")
        model = GradientBoostingRegressor(
            loss="quantile", quantile=q, **GBM_PARAMS
        )
        model.fit(X_train, y_train)
        models[label] = model

        # Predict on test set, convert back to £
        pred_log = model.predict(X_test)
        pred_pounds = np.expm1(pred_log)  # inverse of log1p
        predictions[label] = pred_pounds
        print(f"done  (median predicted: £{np.median(pred_pounds):,.0f})")

    # --- Evaluation ---
    print(f"\n  {'':>6} {'Median Pred £':>14} {'MAE £':>10} {'RMSE £':>10}")
    print(f"  {'':>6} {'─'*14:>14} {'─'*10:>10} {'─'*10:>10}")

    # Actual median for context
    actual_median = np.median(raw_income_test)
    print(f"  {'Actual':>6} {'£{:,.0f}'.format(actual_median):>14}")

    for label in QUANTILE_LABELS:
        pred = predictions[label]
        mae  = mean_absolute_error(raw_income_test, pred)
        rmse = np.sqrt(mean_squared_error(raw_income_test, pred))
        print(f"  {label:>6} {'£{:,.0f}'.format(np.median(pred)):>14} "
              f"{'£{:,.0f}'.format(mae):>10} {'£{:,.0f}'.format(rmse):>10}")

    # --- Quantile coverage check ---
    # For a well-calibrated model, ~80% of actual values should fall between P10 and P90
    coverage = np.mean(
        (raw_income_test >= predictions["P10"]) &
        (raw_income_test <= predictions["P90"])
    )
    print(f"\n  P10–P90 coverage: {coverage*100:.1f}%  (target: ~80%)")

    # --- Feature importance (from median model) ---
    print(f"\n  Feature importance (P50 model):")
    importances = models["P50"].feature_importances_
    for feat, imp in sorted(zip(X_train.columns, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 100)
        print(f"    {feat:>20}  {imp:.3f}  {bar}")

    return models


# ---------------------------------------------------------------------------
# 4. LIFE SATISFACTION MODEL — Multiclass Classification
# ---------------------------------------------------------------------------

def train_satisfaction_model(X_train, y_train, X_test, y_test):
    """
    Train a single multiclass GBM classifier.
    Output: full probability vector P(sat=1) … P(sat=7).
    """
    print("\n" + "=" * 60)
    print("  LIFE SATISFACTION MODEL — Multiclass Classification")
    print("=" * 60)

    # GBM classifier doesn't support quantile loss, use default (deviance/log loss)
    model = GradientBoostingClassifier(
        n_estimators=GBM_PARAMS["n_estimators"],
        max_depth=GBM_PARAMS["max_depth"],
        learning_rate=GBM_PARAMS["learning_rate"],
        subsample=GBM_PARAMS["subsample"],
        random_state=GBM_PARAMS["random_state"],
    )

    print("  Training ... ", end="")
    model.fit(X_train, y_train)
    print("done")

    # Predictions
    y_pred_class = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)   # shape: (n_samples, n_classes)
    classes = model.classes_                      # actual class labels present

    # --- Accuracy & classification report ---
    print(f"\n  Accuracy: {(y_pred_class == y_test).mean()*100:.1f}%")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred_class,
        target_names=[f"sat={c}" for c in classes],
        digits=3
    ))

    # --- Log loss (proper scoring rule for probability outputs) ---
    ll = log_loss(y_test, y_pred_proba, labels=classes)
    print(f"  Log loss: {ll:.3f}")

    # --- Mean predicted satisfaction vs actual ---
    # Expected value from probability vector
    expected_sat = (y_pred_proba * classes).sum(axis=1)
    print(f"\n  Mean predicted satisfaction: {expected_sat.mean():.2f}")
    print(f"  Mean actual satisfaction:    {y_test.mean():.2f}")
    print(f"  MAE (expected vs actual):    {mean_absolute_error(y_test, expected_sat):.2f}")

    # --- Feature importance ---
    print(f"\n  Feature importance:")
    importances = model.feature_importances_
    for feat, imp in sorted(zip(X_train.columns, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 100)
        print(f"    {feat:>20}  {imp:.3f}  {bar}")

    # --- Sample predictions (show probability vectors for 5 test individuals) ---
    print(f"\n  Sample predictions (first 5 test individuals):")
    print(f"  {'':>4}", end="")
    for c in classes:
        print(f"  sat={int(c)}", end="")
    print(f"  {'→ pred':>8} {'actual':>7}")

    for i in range(min(5, len(X_test))):
        print(f"  [{i}]", end="")
        for p in y_pred_proba[i]:
            print(f"  {p:5.2f} ", end="")
        print(f"  {int(y_pred_class[i]):>7} {int(y_test.iloc[i]):>7}")

    return model


# ---------------------------------------------------------------------------
# 5. SAVE MODELS
# ---------------------------------------------------------------------------

def save_models(income_models, satisfaction_model, output_dir="."):
    income_path = os.path.join(output_dir, "model_income.pkl")
    sat_path    = os.path.join(output_dir, "model_satisfaction.pkl")

    with open(income_path, "wb") as f:
        pickle.dump(income_models, f)
    with open(sat_path, "wb") as f:
        pickle.dump(satisfaction_model, f)

    print(f"\n  Saved: {income_path}")
    print(f"  Saved: {sat_path}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ByteLife model training")
    parser.add_argument("--data", required=True, help="Path to bytelife_clean.csv")
    parser.add_argument("--output_dir", default=".", help="Directory to save models")
    args = parser.parse_args()

    # Load
    df = load_data(args.data)

    # Features and targets
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y_income = df["log_income"]
    y_raw_income = df["fimnnet_dv"]          # keep for evaluation in £
    y_satisfaction = df["sclfsato"].astype(int)

    # Drop rows with any NaN in features or targets
    mask = X.notna().all(axis=1) & y_income.notna() & y_satisfaction.notna()
    print(f"Dropping {(~mask).sum()} rows with missing values → {mask.sum():,} usable rows\n")
    X = X[mask]
    y_income = y_income[mask]
    y_raw_income = y_raw_income[mask]
    y_satisfaction = y_satisfaction[mask]

    # Train/test split — same split for both models so test sets are comparable
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, X.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_income_train      = y_income.loc[idx_train]
    y_income_test       = y_income.loc[idx_test]
    y_raw_income_test   = y_raw_income.loc[idx_test]
    y_sat_train         = y_satisfaction.loc[idx_train]
    y_sat_test          = y_satisfaction.loc[idx_test]

    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # Train
    income_models     = train_income_model(X_train, y_income_train, X_test, y_income_test, y_raw_income_test)
    satisfaction_model = train_satisfaction_model(X_train, y_sat_train, X_test, y_sat_test)

    # Save
    save_models(income_models, satisfaction_model, args.output_dir)


if __name__ == "__main__":
    main()