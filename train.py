"""
Train the baseline high-cost classifier.

Loads data/member_summary.csv, holds out 20% of members as a test set,
fits a preprocessing + random forest pipeline, prints test metrics and the
top features, and saves the fitted pipeline to model.joblib.

Leakage note: the first version of this model used every numeric column and
scored ROC-AUC 1.000 with perfect recall, because the label is defined from
plan-paid dollars and three of the money columns restate it. With the four
in-year money columns removed, the honest model scores ROC-AUC 0.892.

Run:   python train.py
Test:  pytest tests/test_leakage.py tests/test_model.py -v
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")

CATEGORICAL_FEATURES = ["age_bucket", "sex", "region"]

# Leaky columns removed (see FEATURE_AUDIT in features.py):
#   total_plan_paid_cents  - the label is defined from it (AUC 1.000 on its own)
#   total_allowed_cents    - plan-paid + out-of-pocket (AUC 0.999)
#   total_billed_cents     - moves with plan-paid (AUC 0.981)
#   total_oop_cents        - same in-year spending as the label
# Debatable columns kept: they are real signal (AUC 0.73-0.87 alone), and a
# production system would use the prior year's version of each.
NUMERIC_FEATURES = [
    "total_claims",
    "medical_claims",
    "pharmacy_claims",
    "specialty_drug_fills",
    "unique_diagnosis_codes",
    "inpatient_claims",
    "has_specialty_drug",
]
TARGET = "is_high_cost"

# evaluate.py and fairness.py import these so every script uses the same split.
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the member summary, generating the synthetic data first if needed."""
    if not DATA_PATH.exists():
        print("data/member_summary.csv not found — generating synthetic data first.")
        from generate_claims_data import generate

        generate()
    return pd.read_csv(DATA_PATH)


def build_pipeline() -> Pipeline:
    """Preprocessing and classifier in one object, so the API can pass raw rows in.

    - One-hot encode categories, ignoring unseen values instead of crashing.
    - Scale numeric columns (not needed for a forest, but keeps the pipeline
      ready for a different model).
    - class_weight="balanced": only ~6.7% of members are high-cost, so each
      positive counts ~14x during training. Without it the model can score
      93% accuracy by predicting nobody.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def main():
    df = load_data()
    print(f"High-cost threshold: ${HIGH_COST_THRESHOLD_USD:,} in plan-paid claims")
    print(f"Prevalence in this cohort: {100 * df[TARGET].mean():.1f}%")

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET].astype(int)

    # Stratify so the rare positive class has the same share in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Accuracy is misleading at 6.7% prevalence: judge the model by ROC-AUC and
    # by recall on the High Cost class. Recall here is at the default 0.5 cutoff;
    # evaluate.py picks the operating point actually used (0.05).
    print("\n=== Test Set Results ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "High Cost"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # If one feature dominates every other one combined, suspect leakage.
    feature_names = pipeline["preprocessor"].get_feature_names_out()
    importances = pipeline["classifier"].feature_importances_
    top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(5)
    print("\nTop 5 features:")
    for name, imp in top.items():
        print(f"  {name:35s} {imp:.3f}")

    joblib.dump(pipeline, "model.joblib")
    print("\nModel saved to model.joblib")


if __name__ == "__main__":
    main()