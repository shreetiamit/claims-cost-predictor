"""
Fairness audit of the care-management model by sex and age.

Scores every member at the Week 5 operating point (0.05), reports per-group
base rates, flag rates and error rates, applies the 80% (four-fifths) rule,
and compares against a model retrained without sex and age. Writes the
results to reports/fairness_report.md.

Run:   python fairness.py
Test:  pytest tests/test_fairness.py -v
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_PATH = Path("reports/fairness_report.md")

PROTECTED_COLUMNS = ["sex", "age_bucket"]
DECISION_THRESHOLD = 0.05  # care-management operating point from evaluate.py

# Written assessment, based on this cohort's numbers (seed 3).
ASSESSMENT = (
    "At my care-management threshold (0.05), sex passes the 80% rule (ratio 0.857): "
    "women are flagged 17.2% vs men 20.1%, in line with their base rates (5.8% vs 7.5%). "
    "Age fails (ratio 0.366): 18-29 year olds are flagged 9.9% vs 26.9% for 60-64. "
    "That gap reflects real differences in need rather than model bias: base rates run "
    "from 3.2% to 12.0%, an even wider ratio (0.26), and the model under- rather than "
    "over-states it. Being flagged means getting a nurse call, so the group worse off "
    "is the one whose high-cost members get missed: 60-64 is the only age group with a "
    "nonzero FNR (10.7%, 3 of 28 high-cost members), and it is the group with the most "
    "need. Removing sex and age barely changed the ratios (sex 0.857 to 0.877, age 0.366 "
    "to 0.406), because claim counts and specialty-drug use rise with age and carry the "
    "same information. Caveat: this audit scores all 2,000 members, including the 1,600 "
    "the model trained on, so TPRs near 1.0 are inflated. On the 400-member holdout the "
    "60-64 group has only about 6 high-cost members, too few to conclude that its FNR "
    "is really higher. Equal demographic parity here would mean flagging young members "
    "who are less likely to need care, so I would audit equal FNR on a larger holdout "
    "rather than force equal flag rates."
)


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Count true/false positives and negatives for 0/1 labels and predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def group_metrics(df: pd.DataFrame, group_col: str,
                  y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> pd.DataFrame:
    """Per-group rates, one row per group value.

    base_rate      share actually high-cost (reality, no model involved)
    positive_rate  share flagged (demographic parity)
    tpr / fnr      of the actually high-cost: share caught / share missed
    fpr            of the actually normal: share flagged anyway
    precision      of the flagged: share actually high-cost

    A rate with nothing to divide by is NaN, not 0: a group with no high-cost
    members has no miss rate, and 0% would wrongly claim the model caught them all.
    """
    def safe_divide(num, den):
        return num / den if den > 0 else np.nan

    rows = []
    for group, sub in df.groupby(group_col, sort=True):
        c = confusion_counts(sub[y_true_col], sub[y_pred_col])
        n = len(sub)
        actual_pos = c["tp"] + c["fn"]
        actual_neg = c["fp"] + c["tn"]
        flagged = c["tp"] + c["fp"]
        tpr = safe_divide(c["tp"], actual_pos)
        rows.append({
            "group": group,
            "n": n,
            "base_rate": safe_divide(actual_pos, n),
            "positive_rate": safe_divide(flagged, n),
            "tpr": tpr,
            "fpr": safe_divide(c["fp"], actual_neg),
            "fnr": 1 - tpr if not np.isnan(tpr) else np.nan,
            "precision": safe_divide(c["tp"], flagged),
        })

    columns = ["group", "n", "base_rate", "positive_rate", "tpr", "fpr", "fnr", "precision"]
    return pd.DataFrame(rows, columns=columns).sort_values("group").reset_index(drop=True)


def disparate_impact_ratio(rates: pd.Series) -> float:
    """Lowest group rate divided by the highest; 1.0 means every group is equal.

    Unknown (NaN) rates are skipped rather than treated as 0.
    """
    rates = pd.Series(rates, dtype=float).dropna()
    if rates.empty:
        return np.nan
    highest = rates.max()
    if highest == 0:
        return 1.0  # every group at zero: equal, if useless
    return float(rates.min() / highest)


def passes_80_percent_rule(ratio: float) -> bool:
    """EEOC four-fifths guideline: passes at 0.80 or above.

    The small tolerance keeps an exact 0.80 passing despite floating-point error.
    """
    if ratio is None or np.isnan(ratio):
        return False
    return bool(ratio >= 0.8 - 1e-9)


def blind_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Retrain without sex and age on the same training rows, and compare ratios.

    My prediction before running it: the age disparity would mostly survive,
    because older members have more claims and specialty drugs. Result: it did
    (age 0.366 with demographics, 0.406 without). The claim counts carry the
    same information the age column did.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from train import RANDOM_STATE, TEST_SIZE

    full_pipeline = joblib.load(MODEL_PATH)
    full_features = list(full_pipeline["preprocessor"].feature_names_in_)
    blind_features = [c for c in full_features if c not in PROTECTED_COLUMNS]

    cat_cols = [c for c in blind_features if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in blind_features if c not in cat_cols]

    # Same training rows train.py used: same size, seed and stratification.
    y = df["y_true"].astype(int)
    train_df, _ = train_test_split(
        df, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    blind = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ])),
        ("classifier", RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    blind.fit(train_df[blind_features], train_df["y_true"].astype(int))

    # Score with the blind model's own predictions, not the full model's.
    blind_proba = blind.predict_proba(df[blind_features])[:, 1]
    blind_scored = df.assign(y_pred=(blind_proba >= DECISION_THRESHOLD).astype(int))

    rows = []
    for col in PROTECTED_COLUMNS:
        with_demo = group_metrics(df, col).set_index("group")["positive_rate"]
        without_demo = group_metrics(blind_scored, col).set_index("group")["positive_rate"]
        rows.append({
            "protected_column": col,
            "ratio_with_demographics": disparate_impact_ratio(with_demo),
            "ratio_without_demographics": disparate_impact_ratio(without_demo),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    features = list(pipeline["preprocessor"].feature_names_in_)
    scored = df.assign(
        y_true=df["is_high_cost"].astype(int),
        y_proba=pipeline.predict_proba(df[features])[:, 1],
    )
    scored["y_pred"] = (scored["y_proba"] >= DECISION_THRESHOLD).astype(int)

    print(f"Auditing at threshold {DECISION_THRESHOLD} "
          f"(label: >${HIGH_COST_THRESHOLD_USD:,} plan-paid)")

    lines = [f"# Fairness audit\n", f"Decision threshold: {DECISION_THRESHOLD}\n"]
    for col in PROTECTED_COLUMNS:
        table = group_metrics(scored, col)
        ratio = disparate_impact_ratio(table.set_index("group")["positive_rate"])
        verdict = "PASS" if passes_80_percent_rule(ratio) else "FAIL"

        print(f"\n=== {col} ===")
        print(table.to_string(index=False))
        print(f"disparate impact ratio (positive rate): {ratio:.3f} → {verdict}")

        lines += [f"\n## {col}\n", table.to_markdown(index=False),
                  f"\n\nDisparate impact ratio: {ratio:.3f} ({verdict})\n"]

    print("\n=== Blind model comparison ===")
    comparison = blind_model_comparison(scored)
    print(comparison.to_string(index=False))
    lines += ["\n## Blind model comparison\n", comparison.to_markdown(index=False),
              f"\n\n## Assessment\n\n{ASSESSMENT}\n"]

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()