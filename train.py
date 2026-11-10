"""
WEEK 4 STARTER — train the baseline classifier.

Fill in the sections marked TODO. When you run this file, it should:
  1. Load data/member_summary.csv (generating it if missing)
  2. Split into training and test sets
  3. Fit a scikit-learn Pipeline that preprocesses features and trains a classifier
  4. Print evaluation metrics on the test set
  5. Save the trained pipeline to model.joblib

Run:
    python train.py

Check your work:
    pytest tests/test_leakage.py -v

--------------------------------------------------------------------------------
STEP 0 — before you change anything.

The NUMERIC_FEATURES list below is every numeric column that comes out of
data/member_summary.csv, straight off the shelf. Get the TODOs working with
that list first, run the script, and write down two numbers:

    ROC-AUC:  ______        recall on the "High Cost" class: ______

Then ask yourself whether you would believe those numbers if a colleague
showed them to you in a review, and only then run the leakage tests.
--------------------------------------------------------------------------------
"""

import joblib
import numpy as np
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

# Every numeric column in member_summary.csv, as generated. Some of these belong
# in a model and some do not — that is your call to make, not the dataset's.
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

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print("data/member_summary.csv not found — generating synthetic data first.")
        from generate_claims_data import generate

        generate()
    return pd.read_csv(DATA_PATH)


def build_pipeline() -> Pipeline:
    """
    TODO 2 — Build and return a scikit-learn Pipeline with two steps.

    Step 1 — a ColumnTransformer named "preprocessor" that:
        - One-hot encodes CATEGORICAL_FEATURES.
          Use handle_unknown="ignore" so unseen categories at predict time
          don't crash the API.
        - Standard-scales NUMERIC_FEATURES.

    Step 2 — a classifier named "classifier".
        A RandomForestClassifier is a reasonable starting point.
        The dataset is imbalanced (positives are well under 15%), so think about
        how you want to handle that. Hint: see the `class_weight` parameter.
        Pass random_state=RANDOM_STATE so your results are reproducible.

    Docs:
        https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
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

    # TODO 1: Split X and y into training and test sets.
    #   - TEST_SIZE of the data goes to the test set
    #   - Stratify on y so the rare positive class is represented in both splits
    #   - Use random_state=RANDOM_STATE for reproducibility
    # Replace the line below with your implementation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # TODO 3: Score the model on the test set.
    #   - Compute the predicted class labels (y_pred)
    #   - Compute the predicted probability of the positive class (y_proba)
    # Hint: pipeline.predict(...) and pipeline.predict_proba(...)[:, 1]
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Test Set Results ===")
    # TODO 4: Print the classification_report comparing y_test and y_pred.
    # Use target_names=["Normal", "High Cost"] so the output is readable.

    # TODO 5: Print the ROC-AUC score using y_test and y_proba.

    print(classification_report(y_test, y_pred, target_names=["Normal", "High Cost"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # TODO 6 (do this one last): print the 5 features the model leaned on most.
    #   feature_names = pipeline["preprocessor"].get_feature_names_out()
    #   importances = pipeline["classifier"].feature_importances_
    # Sort them together and print the top 5. If one feature dominates every
    # other feature combined, that is worth explaining before you move on.

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
