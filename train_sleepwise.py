"""SleepWise AI: standard supervised ML regression workflow.

Predicts the dataset's self-reported Quality of Sleep score (1-10).
This is an educational dataset exercise, not a medical assessment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
TARGET = "Quality of Sleep"
NUMERIC_FEATURES = [
    "Age", "Sleep Duration", "Physical Activity Level", "Stress Level",
    "Systolic BP", "Diastolic BP", "Heart Rate", "Daily Steps",
]
CATEGORICAL_FEATURES = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]


def clean_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load, validate and prepare the public Sleep Health and Lifestyle CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    required = {
        "Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
        "Stress Level", "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps",
        "Gender", "Occupation",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "The uploaded CSV does not have the expected columns: " + ", ".join(missing)
            + ". Please use the Sleep Health and Lifestyle Dataset CSV."
        )

    # In the source dataset, a blank Sleep Disorder entry represents no listed disorder.
    if "Sleep Disorder" not in df.columns:
        df["Sleep Disorder"] = "None"
    else:
        df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None").astype(str).str.strip()
        df.loc[df["Sleep Disorder"].isin(["", "nan", "NaN"]), "Sleep Disorder"] = "None"

    # Split values such as 126/83 into two numeric features.
    bp = df["Blood Pressure"].astype(str).str.extract(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
    df["Systolic BP"] = pd.to_numeric(bp[0], errors="coerce")
    df["Diastolic BP"] = pd.to_numeric(bp[1], errors="coerce")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).drop_duplicates().copy()

    # Use only known predictor columns. Person ID is deliberately excluded because it is an identifier.
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("object")

    X = df[feature_columns].copy()
    y = df[TARGET].astype(float).copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_models() -> dict[str, object]:
    """A mean baseline plus six commonly taught regression algorithms."""
    return {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            loss="huber", random_state=RANDOM_STATE
        ),
    }


def run_project(csv_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = clean_dataset(csv_path)

    print("\nDATASET SUMMARY")
    print(f"Rows after cleaning: {len(X)}")
    print(f"Input features: {X.shape[1]}")
    print("Target: Quality of Sleep (self-reported score)")
    print("Target distribution:")
    print(y.value_counts().sort_index().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}

    print("\nTRAINING AND COMPARING MODELS")
    for name, regressor in build_models().items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ])
        # Cross-validation is performed on training data only; the test set stays held out.
        cv_mae_scores = -cross_val_score(
            pipeline, X_train, y_train, cv=cv,
            scoring="neg_mean_absolute_error", n_jobs=1
        )
        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "CV MAE (mean)": float(cv_mae_scores.mean()),
            "CV MAE (std)": float(cv_mae_scores.std()),
            "Test MAE": float(mean_absolute_error(y_test, test_pred)),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "Test R2": float(r2_score(y_test, test_pred)),
        })
        fitted_models[name] = pipeline
        print(f"Completed: {name}")

    comparison = pd.DataFrame(results).sort_values("CV MAE (mean)").reset_index(drop=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON (lower MAE/RMSE is better; higher R2 is better)")
    print(comparison.round(4).to_string(index=False))

    # Select on cross-validation MAE, not on the held-out test set.
    best_name = str(comparison.loc[0, "Model"])
    best_model = fitted_models[best_name]
    print(f"\nSelected model by lowest 5-fold CV MAE: {best_name}")

    best_predictions = best_model.predict(X_test)
    predictions_df = pd.DataFrame({
        "Actual Quality of Sleep": y_test.to_numpy(),
        "Predicted Quality of Sleep": best_predictions,
        "Absolute Error": np.abs(y_test.to_numpy() - best_predictions),
    })
    predictions_df.to_csv(output_dir / "best_model_test_predictions.csv", index=False)

    joblib.dump(best_model, output_dir / "sleepwise_best_model.joblib")
    (output_dir / "selected_model.txt").write_text(
        f"Selected model: {best_name}\nSelection basis: lowest mean 5-fold cross-validation MAE on training data.\n",
        encoding="utf-8",
    )

    # Permutation importance works for all selected model types and preserves original input columns.
    perm = permutation_importance(
        best_model, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=1
    )
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance (MAE increase)": perm.importances_mean,
        "Importance Std": perm.importances_std,
    }).sort_values("Importance (MAE increase)", ascending=False)
    importance.to_csv(output_dir / "permutation_feature_importance.csv", index=False)

    # Plots saved to the output folder.
    plt.figure(figsize=(10, 5))
    plt.bar(comparison["Model"], comparison["CV MAE (mean)"])
    plt.ylabel("Mean 5-fold CV MAE (lower is better)")
    plt.title("SleepWise AI — Model Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 5))
    top_features = importance.head(10).sort_values("Importance (MAE increase)")
    plt.barh(top_features["Feature"], top_features["Importance (MAE increase)"])
    plt.xlabel("Increase in test MAE when feature is shuffled")
    plt.title(f"Permutation Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.show()
    plt.close()

    print("\nFINAL HELD-OUT TEST RESULTS")
    final_row = comparison.loc[comparison["Model"] == best_name].iloc[0]
    print(f"Model: {best_name}")
    print(f"MAE : {final_row['Test MAE']:.3f} score points")
    print(f"RMSE: {final_row['Test RMSE']:.3f} score points")
    print(f"R²  : {final_row['Test R2']:.3f}")

    # Demonstrate how the saved model predicts for one held-out record.
    one_example = X_test.iloc[[0]]
    example_prediction = float(best_model.predict(one_example)[0])
    print("\nEXAMPLE PREDICTION (one held-out record)")
    print(f"Actual score: {float(y_test.iloc[0]):.1f}/10")
    print(f"Predicted score: {example_prediction:.2f}/10")

    print(f"\nSaved model, metrics and plots in: {output_dir.resolve()}")
    print("Reminder: this model estimates a self-reported dataset score; it is not a medical diagnostic tool.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare SleepWise AI regression models.")
    parser.add_argument("--data", required=True, help="Path to Sleep Health and Lifestyle CSV")
    parser.add_argument("--output", default="sleepwise_outputs", help="Folder for models, metrics and plots")
    args = parser.parse_args()
    run_project(Path(args.data), Path(args.output))


if __name__ == "__main__":
    main()
