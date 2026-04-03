"""
model_utils.py
--------------
Trains a Random Forest and an XGBoost regressor on the salary dataset,
evaluates both on a held-out test split, selects the better one as the
"best" model, and exposes a prediction helper that returns a point
estimate plus a confidence interval.

Key design decisions
--------------------
1.  Random Forest is kept in addition to XGBoost because its ensemble of
    independent trees gives us a natural prediction interval: we collect
    each tree's prediction for a new point and report the 10th–90th
    percentile range as an "80 % confidence interval".  XGBoost doesn't
    expose this directly.

2.  When XGBoost beats RF on RMSE we still use RF's interval as the
    uncertainty estimate, combining XGBoost's point accuracy with RF's
    distributional insight.  This is a common pragmatic pattern in
    production forecasting pipelines.

3.  We use log-transformed salaries during training (then exponentiate
    predictions back to USD) because salary distributions are right-skewed.
    Log-space RMSE is equivalent to relative percentage error, which is a
    more natural metric for salary data.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from data_utils import generate_dataset, engineer_features, FEATURE_NAMES

# Path where trained models are cached so we don't re-train on every page load
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".model_cache")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _metrics(y_true, y_pred, label: str) -> dict:
    """
    Returns a dict of evaluation metrics for one model.
    All dollar metrics are in original USD (we exponentiate log predictions
    before computing so numbers are human-readable).
    """
    return {
        "Model":       label,
        "RMSE ($)":    int(_rmse(y_true, y_pred)),
        "MAE ($)":     int(mean_absolute_error(y_true, y_pred)),
        "R² Score":    round(r2_score(y_true, y_pred), 4),
        "MAPE (%)":    round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2),
    }


def _cache_path(name: str) -> str:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return os.path.join(MODEL_CACHE_DIR, f"{name}.joblib")


# ── Training ───────────────────────────────────────────────────────────────────

def train_and_evaluate(force_retrain: bool = False):
    """
    Train RF + XGBoost, evaluate on a 20 % held-out test set, and return
    everything the Streamlit app needs.

    Parameters
    ----------
    force_retrain : bool
        If False (default) and cached models exist, skip training and load
        from disk.  Set to True to always retrain.

    Returns
    -------
    dict with keys:
        rf_model      : trained RandomForestRegressor
        xgb_model     : trained XGBRegressor
        best_model    : whichever has lower test RMSE
        best_name     : "Random Forest" or "XGBoost"
        metrics_df    : pd.DataFrame — one row per model, columns = metrics
        X_test        : test features (DataFrame)
        y_test        : test targets  (Series, original USD)
        rf_test_preds : RF predictions on test set (array, original USD)
        xgb_test_preds: XGB predictions on test set (array, original USD)
        df_raw        : full raw DataFrame (for EDA charts in the app)
        feature_names : list of feature column names
    """
    rf_path  = _cache_path("rf")
    xgb_path = _cache_path("xgb")
    meta_path = _cache_path("meta")

    if not force_retrain and all(os.path.exists(p) for p in [rf_path, xgb_path, meta_path]):
        # Load cached artefacts — keeps app start-up fast after first run
        rf_model  = joblib.load(rf_path)
        xgb_model = joblib.load(xgb_path)
        meta      = joblib.load(meta_path)
        return {**meta, "rf_model": rf_model, "xgb_model": xgb_model}

    # ── Generate and prepare data ─────────────────────────────────────────────
    df_raw      = generate_dataset(n_samples=4_000)
    df_feat     = engineer_features(df_raw)

    X = df_feat[FEATURE_NAMES]
    y = df_raw["salary_usd"]

    # Log-transform target: reduces skew and makes RMSE ≈ relative % error
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )
    y_test = np.expm1(y_test_log)   # keep original-scale test targets handy

    # ── Random Forest ─────────────────────────────────────────────────────────
    # n_estimators=300: more trees → narrower prediction intervals, small cost
    # max_features="sqrt": standard best-practice to decorrelate trees
    # min_samples_leaf=5: prevents overfitting on our medium-sized dataset
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf_model.fit(X_train, y_train_log)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    # learning_rate=0.05 + n_estimators=500: slow learning with many rounds
    # typically beats RF on tabular data once tuned
    # subsample / colsample_bytree add stochastic regularisation
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_model.fit(
            X_train, y_train_log,
            eval_set=[(X_test, y_test_log)],
            verbose=False,
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    rf_test_preds  = np.expm1(rf_model.predict(X_test))
    xgb_test_preds = np.expm1(xgb_model.predict(X_test))

    rf_metrics  = _metrics(y_test, rf_test_preds,  "Random Forest")
    xgb_metrics = _metrics(y_test, xgb_test_preds, "XGBoost")
    metrics_df  = pd.DataFrame([rf_metrics, xgb_metrics])

    best_name  = "Random Forest" if rf_metrics["RMSE ($)"] <= xgb_metrics["RMSE ($)"] else "XGBoost"
    best_model = rf_model if best_name == "Random Forest" else xgb_model

    # ── Feature importance DataFrames ─────────────────────────────────────────
    rf_importance = pd.DataFrame({
        "Feature":    FEATURE_NAMES,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    xgb_importance = pd.DataFrame({
        "Feature":    FEATURE_NAMES,
        "Importance": xgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # ── Cache everything ──────────────────────────────────────────────────────
    joblib.dump(rf_model,  rf_path)
    joblib.dump(xgb_model, xgb_path)

    meta = {
        "best_model":      best_model,
        "best_name":       best_name,
        "metrics_df":      metrics_df,
        "X_test":          X_test,
        "y_test":          y_test,
        "rf_test_preds":   rf_test_preds,
        "xgb_test_preds":  xgb_test_preds,
        "df_raw":          df_raw,
        "feature_names":   FEATURE_NAMES,
        "rf_importance":   rf_importance,
        "xgb_importance":  xgb_importance,
    }
    joblib.dump(meta, meta_path)

    return {**meta, "rf_model": rf_model, "xgb_model": xgb_model}


# ── Prediction with confidence ─────────────────────────────────────────────────

def predict_with_confidence(
    rf_model,
    xgb_model,
    best_name: str,
    X_input: pd.DataFrame,
    ci: float = 0.80,
) -> dict:
    """
    Return a point prediction from the best model plus a prediction interval
    derived from Random Forest individual-tree variance.

    Why RF for the interval even when XGBoost is the best model?
    RF's bagged trees are trained on independent bootstrap samples, so the
    spread of their individual predictions is a principled measure of
    epistemic uncertainty.  XGBoost builds additive corrections and doesn't
    offer the same decomposition without a separate quantile-regression pass.

    Parameters
    ----------
    ci : float
        Desired coverage, e.g. 0.80 gives a 10th–90th percentile interval.

    Returns
    -------
    dict with keys:
        point_estimate : float — USD salary (from best model)
        lower          : float — lower CI bound from RF
        upper          : float — upper CI bound from RF
        rf_estimate    : float — RF's own point estimate (for reference)
        xgb_estimate   : float — XGBoost's point estimate (for reference)
    """
    alpha = (1 - ci) / 2   # tail probability on each side

    # Collect each tree's log-scale prediction for the input row
    tree_preds_log = np.array(
        [tree.predict(X_input.values)[0] for tree in rf_model.estimators_]
    )

    lower_log = np.percentile(tree_preds_log, alpha * 100)
    upper_log = np.percentile(tree_preds_log, (1 - alpha) * 100)

    rf_point  = float(np.expm1(rf_model.predict(X_input)[0]))
    xgb_point = float(np.expm1(xgb_model.predict(X_input)[0]))
    lower     = float(np.expm1(lower_log))
    upper     = float(np.expm1(upper_log))

    point_estimate = rf_point if best_name == "Random Forest" else xgb_point

    return {
        "point_estimate": point_estimate,
        "lower":          lower,
        "upper":          upper,
        "rf_estimate":    rf_point,
        "xgb_estimate":   xgb_point,
    }
