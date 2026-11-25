import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

from config import (
    DATABASE_PATH,
    FREIGHT_FEATURES,
    FREIGHT_MODEL_PATH,
    TARGET_COLUMN,
)
from data_loader import load_vendor_invoices


def evaluate_models(X_train, X_test, y_train, y_test):
    """Train baseline regressors and return their evaluation metrics."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append(
            {
                "Model": name,
                "MAE ($)": round(
                    mean_absolute_error(y_test, predictions), 2
                ),
                "RMSE ($)": round(
                    np.sqrt(mean_squared_error(y_test, predictions)), 2
                ),
                "R2 Score": round(
                    r2_score(y_test, predictions), 4
                ),
            }
        )

    return pd.DataFrame(results)


def tune_random_forest(X_train, y_train):
    """Tune a Random Forest regressor using 3-fold cross-validation."""
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    return search


def train_freight_model(save_path=FREIGHT_MODEL_PATH):
    """Train the freight prediction model and save it to disk."""
    vendor_df = load_vendor_invoices(DATABASE_PATH)

    clean_df = vendor_df.dropna(
        subset=FREIGHT_FEATURES + [TARGET_COLUMN]
    ).copy()

    X = clean_df[FREIGHT_FEATURES]
    y = clean_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.8,
        random_state=42,
    )

    print("=== Baseline Freight Models ===")
    print(evaluate_models(X_train, X_test, y_train, y_test))

    grid_search = tune_random_forest(X_train, y_train)
    tuned_rf = grid_search.best_estimator_

    tuned_predictions = tuned_rf.predict(X_test)

    print("\n=== Tuned Random Forest ===")
    print("Best Parameters:", grid_search.best_params_)
    print(
        f"R2 Score: {r2_score(y_test, tuned_predictions):.4f}"
    )

    # Preserve the notebook's final model choice.
    final_model = LinearRegression()
    final_model.fit(X_train, y_train)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, save_path)

    print(f"\nSaved freight model to: {save_path}")

    return final_model


def main():
    parser = argparse.ArgumentParser(
        description="Train the vendor freight cost prediction model."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(FREIGHT_MODEL_PATH),
        help="Path for the serialized model.",
    )
    args = parser.parse_args()

    train_freight_model(FREIGHT_MODEL_PATH.parent / Path(args.output).name)


if __name__ == "__main__":
    main()
