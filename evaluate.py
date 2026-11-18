"""
Evaluate the model on the held-out test set and choose an operating point.

ROC-AUC says how well the model ranks members; it is not a decision. This
script rebuilds train.py's 400-member holdout, sweeps decision thresholds,
puts a dollar value on each one, and plots ROC, precision-recall and
calibration curves to reports/evaluation_curves.png.

Chosen use case: care management at threshold 0.05 (see justification below).

Run:   python evaluate.py
Test:  pytest tests/test_evaluation.py -v
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import HIGH_COST_THRESHOLD_USD

DATA_PATH = Path("data/member_summary.csv")
MODEL_PATH = Path("model.joblib")
REPORT_DIR = Path("reports")

# Must match train.py, or the "holdout" would include members the model trained on.
TEST_SIZE = 0.2
RANDOM_STATE = 42

VALID_USE_CASES = ("care_management", "stop_loss", "pricing")

# Operating point. A nurse team calls flagged members early. False positives
# waste a call; false negatives mean a serious condition is found late.
CHOSEN_USE_CASE = "care_management"
CHOSEN_THRESHOLD = 0.05
THRESHOLD_JUSTIFICATION = (
    "Care management: a call costs 250 dollars and catching a high-cost member saves "
    "about 4,000, so a call breaks even if just 1 in 16 flagged members is high-cost. "
    "At 0.05 we flag 88 of 400, catch 24 of 27 (recall 0.89) at precision 0.27, the "
    "highest net value in the sweep (74,000 dollars). The trade-off I accept: about 3 in "
    "4 calls go to members who turn out fine, roughly 440 calls a year at full cohort "
    "size, which assumes the nurse team can absorb that volume. Missing a high-cost "
    "member (a late diagnosis) costs more than a wasted call."
)

# Rough care-management economics, in dollars. Crude on purpose: the point is
# to make the trade-off between missed members and wasted calls explicit.
OUTREACH_COST_PER_FLAGGED = 250
SAVINGS_PER_TRUE_POSITIVE = 4_000


def load_holdout() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Rebuild train.py's test split and score it.

    Feature columns come from the fitted pipeline, so this stays correct if
    the feature list in train.py changes.
    """
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    feature_cols = list(pipeline["preprocessor"].feature_names_in_)
    X = df[feature_cols]
    y = df["is_high_cost"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return X_test, y_test, y_proba


def threshold_sweep(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """One row per threshold: how many get flagged, and how many of them are right.

    A member is flagged when y_proba >= threshold. When nothing is flagged,
    precision is reported as 0.0 rather than 0/0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    rows = []
    for t in thresholds:
        flagged = y_proba >= t
        tp = int(((y_true == 1) & flagged).sum())
        fp = int(((y_true == 0) & flagged).sum())
        fn = int(((y_true == 1) & ~flagged).sum())
        n_flagged = tp + fp
        precision = tp / n_flagged if n_flagged > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({
            "threshold": round(float(t), 2),
            "n_flagged": n_flagged,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        })
    return pd.DataFrame(rows)


def expected_value(tp: int, fp: int, outreach_cost: float, savings_per_tp: float) -> float:
    """Net dollars: savings from members caught, minus the cost of every call made."""
    return tp * savings_per_tp - (tp + fp) * outreach_cost


def plot_curves(y_true, y_proba, out_path: Path = REPORT_DIR / "evaluation_curves.png") -> Path:
    """Save ROC, precision-recall and calibration curves side by side.

    The red dot marks CHOSEN_THRESHOLD. On the calibration panel, points above
    the diagonal mean the model under-predicts risk. Mine sits close to the
    diagonal, slightly above at the low end, which matters little for ranking
    members for outreach but would matter for stop-loss or pricing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax1.plot(fpr, tpr, label=f"model (AUC = {roc_auc_score(y_true, y_proba):.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="coin flip")
    ax1.set(title="ROC curve", xlabel="false positive rate", ylabel="true positive rate (recall)")

    # Precision-recall, with prevalence as the random-flagging baseline
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax2.plot(recall, precision, label=f"model (AP = {average_precision_score(y_true, y_proba):.3f})")
    ax2.axhline(y_true.mean(), linestyle="--", color="gray", label="prevalence")
    ax2.set(title="Precision-recall curve", xlabel="recall", ylabel="precision")

    # Chosen operating point on both curves
    if CHOSEN_THRESHOLD is not None:
        flagged = y_proba >= CHOSEN_THRESHOLD
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        pos = int(y_true.sum())
        neg = len(y_true) - pos
        point_recall = tp / pos if pos else 0.0
        point_precision = tp / (tp + fp) if (tp + fp) else 0.0
        label = f"threshold {CHOSEN_THRESHOLD}"
        ax1.scatter(fp / neg, point_recall, color="red", zorder=5, label=label)
        ax2.scatter(point_recall, point_precision, color="red", zorder=5, label=label)

    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    ax3.plot(prob_pred, prob_true, marker="o", label="model")
    ax3.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax3.set(title="Calibration", xlabel="predicted probability", ylabel="actual share high-cost")

    for ax in (ax1, ax2, ax3):
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    X_test, y_test, y_proba = load_holdout()
    print(f"Scored {len(y_test)} held-out members "
          f"({int(y_test.sum())} of them high-cost at ${HIGH_COST_THRESHOLD_USD:,})")

    sweep = threshold_sweep(y_test, y_proba)
    sweep = sweep.assign(
        net_dollars=[
            expected_value(row.tp, row.fp, OUTREACH_COST_PER_FLAGGED, SAVINGS_PER_TRUE_POSITIVE)
            for row in sweep.itertuples()
        ]
    )
    print("\n=== Threshold sweep ===")
    print(sweep.to_string(index=False))

    path = plot_curves(y_test, y_proba)
    print(f"\nCurves written to {path}")

    print(f"\n=== My operating point ===")
    print(f"Use case:      {CHOSEN_USE_CASE}")
    print(f"Threshold:     {CHOSEN_THRESHOLD}")
    print(f"Justification: {THRESHOLD_JUSTIFICATION}")


if __name__ == "__main__":
    main()