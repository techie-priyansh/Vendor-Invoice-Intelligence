from __future__ import annotations
"""
Train an invoice risk classifier using the invoice_flagging notebook workflow.

Default model: RandomForestClassifier trained on X1 (features that exclude the
invoice dollar values directly used in the rule-based target).

Usage:
    python train_risk_classifier.py --db-path data/inventory.db
    python train_risk_classifier.py --db-path data/inventory.db --output models/invoice_risk_classifier.pkl

Dependencies:
    pandas, numpy, scikit-learn, joblib
"""
from config import (
    DATABASE_PATH,
    RISK_CLASSIFIER_PATH,
    RISK_FEATURES,
    RISK_FEATURES_X,
    RISK_TARGET_COLUMN,
    RISK_DOLLAR_DIFF_THRESHOLD,
    RISK_RECEIVING_DELAY_THRESHOLD_DAYS,
)



import argparse
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split


# Features used in the notebook's X1 experiment.
# RISK_FEATURES = [
#     "invoice_quantity",
#     "Freight",
#     "total_brands",
#     "total_item_quantity",
#     "days_po_to_invoice",
# ]

# Original X features, retained only for a side-by-side experimental comparison.
# invoice_dollars and total_item_dollars are directly involved in the target rule,
# so this feature set may contain target leakage.
# X_FEATURES = [
#     "invoice_quantity",
#     "invoice_dollars",
#     "Freight",
#     "total_brands",
#     "total_item_quantity",
#     "days_po_to_invoice",
#     "total_item_dollars",
# ]

TARGET_COLUMN = RISK_TARGET_COLUMN


def load_invoice_data(db_path: Path) -> pd.DataFrame:
    """Load invoice records and PO-level purchase aggregates from SQLite."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Pass the correct path with --db-path."
        )

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(
                julianday(p.ReceivingDate) - julianday(p.PODate)
            ) AS avg_receiving_delay
        FROM purchases AS p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (
            julianday(vi.InvoiceDate) - julianday(vi.PODate)
        ) AS days_po_to_invoice,
        (
            julianday(vi.PayDate) - julianday(vi.InvoiceDate)
        ) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice AS vi
    LEFT JOIN purchase_agg AS pa
        ON vi.PONumber = pa.PONumber
    """

    with sqlite3.connect(db_path) as connection:
        df = pd.read_sql_query(query, connection)

    if df.empty:
        raise ValueError("The SQL query returned no invoice rows.")

    return df


def create_invoice_risk_label(row: pd.Series) -> int:
    """
    Label an invoice as flagged when:
      1) invoice dollars differ from purchase item dollars by more than $5, OR
      2) average receiving delay is greater than 10 days.
    """
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > RISK_DOLLAR_DIFF_THRESHOLD:
        return 1

    if row["avg_receiving_delay"] > RISK_RECEIVING_DELAY_THRESHOLD_DAYS:
        return 1

    return 0


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create the notebook's rule-based target and prepare model-ready rows."""
    df = df.copy()
    df[TARGET_COLUMN] = df.apply(create_invoice_risk_label, axis=1)

    required = RISK_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing from data: {missing}")

    # Model training requires numeric, finite feature values.
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError(
            "No rows remain after dropping missing values in X1 features/target."
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def evaluate_predictions(
    y_true: pd.Series,
    y_pred,
    model_name: str,
) -> dict[str, float]:
    """Print classification metrics and return them as a dictionary."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }

    print(f"\n=== {model_name} ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_true, y_pred))

    return metrics


def compare_x_and_x1(df: pd.DataFrame, random_state: int = 42) -> None:
    """
    Compare baseline Random Forests for X and X1 on the same split.
    This is an analysis-only comparison; X may leak target-rule information.
    """
    X = df[RISK_FEATURES_X]
    X1 = df[RISK_FEATURES]
    y = df[TARGET_COLUMN]

    indices = df.index
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    model_x = RandomForestClassifier(random_state=random_state)
    model_x1 = RandomForestClassifier(random_state=random_state)

    model_x.fit(X.loc[train_idx], y_train)
    model_x1.fit(X1.loc[train_idx], y_train)

    pred_x = model_x.predict(X.loc[test_idx])
    pred_x1 = model_x1.predict(X1.loc[test_idx])

    results = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "X": [
                accuracy_score(y_test, pred_x),
                precision_score(y_test, pred_x, zero_division=0),
                recall_score(y_test, pred_x, zero_division=0),
                f1_score(y_test, pred_x, zero_division=0),
            ],
            "X1": [
                accuracy_score(y_test, pred_x1),
                precision_score(y_test, pred_x1, zero_division=0),
                recall_score(y_test, pred_x1, zero_division=0),
                f1_score(y_test, pred_x1, zero_division=0),
            ],
        }
    )

    print("\n=== Baseline Random Forest: X vs X1 ===")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "\nNote: X includes invoice_dollars and total_item_dollars, "
        "which directly contribute to the rule-generated target."
    )


def train_best_x1_model(
    df: pd.DataFrame,
    random_state: int = 42,
    n_iter: int = 20,
    cv: int = 3,
):
    """Tune and train the X1 Random Forest using cross-validated F1."""
    X1 = df[RISK_FEATURES]
    y = df[TARGET_COLUMN]

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
    }

    rf = RandomForestClassifier(random_state=random_state)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=make_scorer(f1_score),
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
    )

    search.fit(X1_train, y1_train)

    print("\nBest cross-validation F1:", f"{search.best_score_:.4f}")
    print("Best hyperparameters:")
    print(search.best_params_)

    y1_pred = search.best_estimator_.predict(X1_test)
    evaluate_predictions(y1_test, y1_pred, "Tuned Random Forest - X1")

    importance = pd.DataFrame(
        {
            "feature": RISK_FEATURES,
            "importance": search.best_estimator_.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nX1 feature importances:")
    print(importance.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    return search.best_estimator_, search, X1_test, y1_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the invoice risk Random Forest classifier."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DATABASE_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=RISK_CLASSIFIER_PATH,
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the baseline X vs X1 Random Forest comparison.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of RandomizedSearchCV parameter combinations (default: 20).",
    )
    args = parser.parse_args()

    raw_df = load_invoice_data(args.db_path)
    df = prepare_data(raw_df)

    print(f"Rows available for modeling: {len(df)}")
    print("Target distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    if not args.skip_comparison:
        compare_x_and_x1(df)

    model, search, _, _ = train_best_x1_model(
        df,
        n_iter=args.n_iter,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": model,
        "feature_names": RISK_FEATURES,
        "target_name": TARGET_COLUMN,
        "target_rule": (
            "1 if abs(invoice_dollars - total_item_dollars) > 5 "
            "or avg_receiving_delay > 10; otherwise 0"
        ),
        "best_params": search.best_params_,
        "best_cv_f1": search.best_score_,
    }
    joblib.dump(model_bundle, args.output)

    print(f"\nSaved X1 model bundle to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
