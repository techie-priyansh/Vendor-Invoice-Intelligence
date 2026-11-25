import argparse

import joblib
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    DATABASE_PATH,
    FREIGHT_MODEL_PATH,
    RISK_CLASSIFIER_PATH,
    RISK_FEATURES,
    SCALER_PATH,
)
from data_loader import load_vendor_invoices


def build_risk_dataset(vendor_df, freight_model):
    """Create model features and the high-risk target."""
    df = vendor_df.copy()

    df["predicted_freight"] = freight_model.predict(
        df[["Quantity", "Dollars"]]
    )
    df["freight_diff"] = (
        df["Freight"] - df["predicted_freight"]
    )
    df["freight_per_unit"] = (
        df["Freight"] / df["Quantity"]
    )

    risk_threshold = df["freight_diff"].quantile(0.90)

    df["is_high_risk"] = (
        df["freight_diff"] >= risk_threshold
    ).astype(int)

    return df, risk_threshold


def run_hypothesis_tests(df):
    """Run Welch's t-tests for the notebook's exploratory analysis."""
    normal_group = df[df["is_high_risk"] == 0]
    risk_group = df[df["is_high_risk"] == 1]

    print("\n=== Statistical Significance ===")

    for column in ["Quantity", "freight_per_unit", "Dollars"]:
        statistic, p_value = stats.ttest_ind(
            normal_group[column],
            risk_group[column],
            equal_var=False,
        )

        significance = (
            "Statistically significant"
            if p_value < 0.05
            else "Not statistically significant"
        )

        print(
            f"{column}: "
            f"t-stat={statistic:.2f}, "
            f"p-value={p_value:.4e} "
            f"({significance})"
        )


def evaluate_baseline_classifiers(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Evaluate the baseline classifiers used in the notebook."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    print("\n=== Baseline Classification Models ===")

    for name, classifier in classifiers.items():
        if name == "Logistic Regression":
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test

        classifier.fit(X_train_model, y_train)
        predictions = classifier.predict(X_test_model)

        print(f"\n--- {name} ---")
        print(
            f"Precision: {precision_score(y_test, predictions):.4f}"
        )
        print(
            f"Recall:    {recall_score(y_test, predictions):.4f}"
        )
        print(
            f"F1-Score:  {f1_score(y_test, predictions):.4f}"
        )

    return scaler


def tune_random_forest(X_train, y_train):
    """Tune the Random Forest classifier used for production."""
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search


def train_risk_classifier(
    freight_model_path=FREIGHT_MODEL_PATH,
    classifier_path=RISK_CLASSIFIER_PATH,
):
    """Train and save the invoice risk classifier."""
    freight_model = joblib.load(freight_model_path)

    vendor_df = load_vendor_invoices(
        DATABASE_PATH,
        require_freight=True,
        require_positive_quantity=True,
    )

    vendor_df, risk_threshold = build_risk_dataset(
        vendor_df,
        freight_model,
    )

    print(
        f"High-Risk Overcharge Threshold: "
        f"${risk_threshold:.2f}"
    )

    print("\nClass Distribution:")
    print(
        vendor_df["is_high_risk"]
        .value_counts(normalize=True)
    )

    run_hypothesis_tests(vendor_df)

    X = vendor_df[RISK_FEATURES]
    y = vendor_df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Preserved from the notebook for exploratory/baseline use.
    scaler = evaluate_baseline_classifiers(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    grid_search = tune_random_forest(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)

    print("\n=== Tuned Random Forest Results ===")
    print("Best Hyperparameters:", grid_search.best_params_)
    print(
        f"Tuned F1-Score: "
        f"{f1_score(y_test, predictions):.4f}"
    )
    print(
        "\nClassification Report:\n",
        classification_report(y_test, predictions),
    )

    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, classifier_path)

    print(
        f"Saved risk classifier to: {classifier_path}"
    )

    return best_model


def main():
    parser = argparse.ArgumentParser(
        description="Train the vendor invoice risk classifier."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RISK_CLASSIFIER_PATH),
        help="Path for the serialized classifier.",
    )
    args = parser.parse_args()

    train_risk_classifier(
        classifier_path=RISK_CLASSIFIER_PATH.parent
        / Path(args.output).name
    )


if __name__ == "__main__":
    main()
