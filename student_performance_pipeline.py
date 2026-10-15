# -*- coding: utf-8 -*-
"""
Student Performance - Exam Score Prediction Pipeline
=====================================================
Random seed : 42
Target      : Exam_Score (regression)
Evaluation  : MAE, RMSE, R2  (test set)
Leakage prevention: all preprocessing fitted on training split only,
                    applied to test split via pipeline transform.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42

# -- 1. Load dataset -----------------------------------------------------------
df = pd.read_csv("StudentPerformanceFactors.csv")
print(f"Dataset shape: {df.shape}")

# -- 2. Inspect missing values & duplicates ------------------------------------
print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0])

dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"  -> Dropped duplicates. New shape: {df.shape}")

# -- 3. Separate features and target -------------------------------------------
TARGET = "Exam_Score"
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\nFeature matrix : {X.shape}")
print(f"Target series  : {y.shape}")
print(f"Target range   : {y.min()} - {y.max()}  |  mean={y.mean():.2f}  std={y.std():.2f}")

# -- 4. Train / test split (BEFORE any fitting to prevent leakage) -------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# -- 5. Identify column types --------------------------------------------------
numerical_cols   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
# Note: explicit "str" silences Pandas 4 deprecation for StringDtype columns

print(f"\nNumerical features   ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# -- 6. Build preprocessing pipelines -----------------------------------------
#  Numerical: impute median (robust to outliers) -> standard scale
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

#  Categorical: impute most-frequent -> ordinal encode (good for tree models)
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# -- 7. Full model pipeline ----------------------------------------------------
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )),
])

# -- 8. Baseline model (predicts training-set mean for every sample) -----------
baseline = DummyRegressor(strategy="mean")

# -- 9. Fit both models on training data only ----------------------------------
print("\nTraining Random Forest ...")
model_pipeline.fit(X_train, y_train)
print("Training Baseline ...")
baseline.fit(X_train, y_train)

# -- 10. Evaluate on held-out test set ----------------------------------------
def evaluate(name, estimator, X_t, y_t):
    preds = estimator.predict(X_t)
    mae  = mean_absolute_error(y_t, preds)
    rmse = np.sqrt(mean_squared_error(y_t, preds))
    r2   = r2_score(y_t, preds)
    print(f"\n{'-'*40}")
    print(f"  {name}")
    print(f"{'-'*40}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")
    return {"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

print("\n\n===== TEST SET EVALUATION =====")
rf_metrics = evaluate("Random Forest Regressor",  model_pipeline, X_test, y_test)
bl_metrics = evaluate("Baseline (Mean Predictor)", baseline,       X_test, y_test)

results_df = pd.DataFrame([rf_metrics, bl_metrics])
print("\n\nSummary table:")
print(results_df.to_string(index=False))

# -- 11. Feature importance ----------------------------------------------------
rf_model   = model_pipeline.named_steps["regressor"]
feat_names = numerical_cols + categorical_cols
importances = pd.Series(rf_model.feature_importances_, index=feat_names)
print("\n\nTop-10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10).to_string())

# -- 12. Save artefacts --------------------------------------------------------
joblib.dump(model_pipeline, "rf_exam_score_pipeline.joblib")
joblib.dump(preprocessor,   "preprocessor.joblib")
print("\n\nSaved: rf_exam_score_pipeline.joblib")
print("Saved: preprocessor.joblib")

print("\nDone.")
