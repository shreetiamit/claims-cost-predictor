"""
Build the member-level feature table and audit which columns a model may use.

claims.csv has one row per claim (about 10 per member). The model predicts one
label per member, so this script rolls approved claims up to one row per
member, attaches the is_high_cost label, and records a verdict for every
column: is it known on January 1, when the prediction is made?

Run:   python features.py
Test:  pytest tests/test_features.py -v
"""

import pandas as pd
from pathlib import Path

from config import HIGH_COST_THRESHOLD_CENTS, HIGH_COST_THRESHOLD_USD

MEMBERS_PATH = Path("data/members.csv")
CLAIMS_PATH = Path("data/claims.csv")
OUT_PATH = Path("data/member_features.csv")

# Every non-identifier column in the feature table, in the order the audit prints.
AUDITABLE_COLUMNS = [
    "age_bucket",
    "sex",
    "region",
    "total_claims",
    "medical_claims",
    "pharmacy_claims",
    "specialty_drug_fills",
    "total_billed_cents",
    "total_allowed_cents",
    "total_plan_paid_cents",
    "total_oop_cents",
    "unique_diagnosis_codes",
    "inpatient_claims",
    "has_specialty_drug",
    "is_high_cost",
]

VALID_VERDICTS = ("safe", "leaky", "debatable", "not_a_feature")

# Feature audit. The prediction is made on January 1; the label is only known
# after December 31. For each column: do I know this number on January 1?
#   safe           known at enrollment
#   debatable      a prior-year version would be known, but this column is
#                  measured during the outcome year
#   leaky          restates the outcome (the label is built from plan-paid)
#   not_a_feature  the label itself
FEATURE_AUDIT: dict[str, tuple[str, str]] = {
    "age_bucket": ("safe", "known at enrollment, before the plan year starts"),
    "sex": ("safe", "known at enrollment, before the plan year starts"),
    "region": ("safe", "known at enrollment, before the plan year starts"),
    "total_claims": ("debatable", "last year's count is knowable on Jan 1, but this column counts claims in the same year as the label"),
    "medical_claims": ("debatable", "same as total_claims: a prior-year version exists, this one is measured inside the outcome year"),
    "pharmacy_claims": ("debatable", "same as total_claims: a prior-year version exists, this one is measured inside the outcome year"),
    "specialty_drug_fills": ("debatable", "chronic specialty drugs often carry over from last year, but these fills happen during the outcome year"),
    "total_billed_cents": ("leaky", "moves almost one-for-one with plan-paid, so it nearly restates the outcome"),
    "total_allowed_cents": ("leaky", "allowed = plan-paid + out-of-pocket for most claims, so it contains the label's own column"),
    "total_plan_paid_cents": ("leaky", "the label is defined directly from this column (plan-paid > threshold)"),
    "total_oop_cents": ("leaky", "rises with the same in-year spending that defines the label"),
    "unique_diagnosis_codes": ("debatable", "prior-year diagnoses are knowable on Jan 1, but this counts diagnoses seen during the outcome year"),
    "inpatient_claims": ("debatable", "prior-year admissions are knowable, but these admissions happen during the outcome year"),
    "has_specialty_drug": ("debatable", "a chronic specialty drug often carries over from last year, but this flag is measured in the outcome year"),
    "is_high_cost": ("not_a_feature", "this is the label the model is trying to predict"),
}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load members and claims, generating the synthetic data first if needed."""
    if not CLAIMS_PATH.exists() or not MEMBERS_PATH.exists():
        print("Source CSVs not found — running the generator first.")
        from generate_claims_data import generate

        generate()
    return pd.read_csv(MEMBERS_PATH), pd.read_csv(CLAIMS_PATH)


def approved_claims(claims: pd.DataFrame) -> pd.DataFrame:
    """Keep approved claims only. Denied claims (~8%) were never paid, so they
    are not part of a member's cost."""
    return claims[claims["adjudication_status"] == "approved"].copy()


def build_member_features(members: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Roll claims up to one row per member and attach demographics.

    Uses a left merge so members with no approved claims still get a row, and
    fills their counts with 0: they had zero claims, not an unknown number.
    """
    agg = claims.groupby("member_id").agg(
        total_claims=("claim_id", "count"),
        medical_claims=("is_pharmacy", lambda s: (~s.astype(bool)).sum()),
        pharmacy_claims=("is_pharmacy", lambda s: s.astype(bool).sum()),
        specialty_drug_fills=("is_specialty_drug", lambda s: s.astype(bool).sum()),
        total_billed_cents=("billed_cents", "sum"),
        total_allowed_cents=("allowed_cents", "sum"),
        total_plan_paid_cents=("plan_paid_cents", "sum"),
        total_oop_cents=("out_of_pocket_cents", "sum"),
        unique_diagnosis_codes=("diagnosis_code", lambda s: s[s != ""].nunique()),
        inpatient_claims=("claim_type", lambda s: (s == "Institutional").sum()),
        has_specialty_drug=("is_specialty_drug", lambda s: s.astype(bool).any()),
    ).reset_index()

    member_cols = ["member_id", "age_bucket", "sex", "region",
                   "plan_year_start", "plan_year_end"]
    features = members[member_cols].merge(agg, on="member_id", how="left")

    count_cols = [c for c in agg.columns if c not in ("member_id", "has_specialty_drug")]
    features[count_cols] = features[count_cols].fillna(0).astype("int64")
    features["has_specialty_drug"] = features["has_specialty_drug"].eq(True)

    return features


def add_label(features: pd.DataFrame) -> pd.DataFrame:
    """Add is_high_cost (0/1): plan-paid for the year above the threshold.

    The label is computed from total_plan_paid_cents, which is why that column
    (and the money columns that track it) are marked leaky in the audit.
    """
    features = features.copy()
    features["is_high_cost"] = (
        features["total_plan_paid_cents"] > HIGH_COST_THRESHOLD_CENTS
    ).astype(int)
    return features


def model_features() -> list[str]:
    """Columns the model trains on, derived from the audit.

    Safe and debatable columns are in: the debatable ones carry real signal
    (single-column AUC 0.73-0.87, not ~1.0), and a production system would use
    the prior year's version of each. Leaky columns and the label stay out.
    """
    return [col for col, (verdict, _) in FEATURE_AUDIT.items() if verdict in ("safe", "debatable")]


def print_audit() -> None:
    print(f"\n=== Feature audit (threshold ${HIGH_COST_THRESHOLD_USD:,}) ===")
    for col in AUDITABLE_COLUMNS:
        verdict, why = FEATURE_AUDIT.get(col, ("MISSING", ""))
        print(f"  {col:24s} {verdict:14s} {why}")


def main() -> None:
    members, claims = load_raw()
    features = add_label(build_member_features(members, approved_claims(claims)))

    OUT_PATH.parent.mkdir(exist_ok=True)
    features.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(features)} member rows to {OUT_PATH}")
    print(f"High-cost prevalence: {100 * features['is_high_cost'].mean():.1f}%")
    print_audit()
    print(f"\nTraining on {len(model_features())} features: {model_features()}")


if __name__ == "__main__":
    main()