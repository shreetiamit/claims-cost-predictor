# Claims Cost Predictor

Predicts which health-plan members are likely to become **high-cost** (more than $25,000 in plan-paid claims in a year) so a care-management team can reach them early.

**Live API:** https://claims-risk-api-kdy9.onrender.com/docs

The data is synthetic: 2,000 members and about 21,000 claims, generated from a fixed seed (`STUDENT_SEED=3`) so every result here is reproducible.

> The free Render plan sleeps when idle, so the first request can take 30–60 seconds.

## Results

| Step | Finding |
|---|---|
| Exploration | 6.7% of members are high-cost, so predicting "nobody" already scores 93.3% accuracy. Accuracy is the wrong metric. |
| Leakage | The first model scored ROC-AUC **1.000**. Three money columns restate the label (plan-paid alone scores 1.000). Removing the four in-year money columns gives an honest **0.892**. |
| Operating point | For care management (call cost 250 dollars, savings 4,000 per member caught), threshold **0.05** catches 24 of 27 high-cost members in the holdout and has the highest net value: 74,000 dollars vs 25,000 at the default 0.5. |
| Calibration | Close to the diagonal, slightly under-predicting at the low end. Fine for ranking members; would need work for pricing or reserving. |
| Fairness | Sex passes the 80% rule (0.857). Age fails (0.366), but the real high-cost rate differs even more by age (3.2% to 12.0%), so the gap tracks need. The 60–64 group is the only one with missed members. Removing sex and age barely changes this, because claim counts carry age information. |

## How it works

```
claims.csv (one row per claim)
   → features.py    approved claims only, rolled up to one row per member, feature audit
   → train.py       random forest (class_weight="balanced"), leaky columns excluded
   → evaluate.py    threshold sweep, dollar value, ROC / PR / calibration curves
   → fairness.py    per-group rates by sex and age, 80% rule, blind-model comparison
   → app/           FastAPI service: /health and /predict
```

Each column was audited with one question: *on January 1, when the prediction is made, do I already know this number?* See `FEATURE_AUDIT` in `features.py`.

## Run it locally

Requires Python 3.11.

```bash
git clone https://github.com/shreetiamit/claims-cost-predictor.git
cd claims-cost-predictor
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export STUDENT_SEED=3

python generate_claims_data.py   # synthetic members and claims -> data/
python features.py               # member feature table + audit
python train.py                  # trains and saves model.joblib
python evaluate.py               # threshold sweep + reports/evaluation_curves.png
python fairness.py               # reports/fairness_report.md
```

Exploratory analysis is in `notebooks/week-02-eda.ipynb`.

## Test

```bash
pytest
```

74 tests cover the data, the feature table, leakage, the model's quality gate, evaluation, fairness metrics, and the API (including a train/serve skew check). CI runs them on every push and pull request.

## Serve the API

```bash
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000/docs
```

**Endpoints**

- `GET /health` returns `{"status": "ok", "model_loaded": true}`
- `POST /predict` scores one member:

```bash
curl -X POST https://claims-risk-api-kdy9.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age_bucket": "60-64", "sex": "M", "region": "Midwest",
       "total_claims": 20, "medical_claims": 14, "pharmacy_claims": 6,
       "specialty_drug_fills": 2, "unique_diagnosis_codes": 8,
       "inpatient_claims": 3, "has_specialty_drug": true}'
```

Response: `{"is_high_cost": true, "probability": ..., "threshold_dollars": 25000}`

Invalid input returns **422**: unknown categories, negative counts, or counts that contradict each other (for example `medical_claims + pharmacy_claims` not equal to `total_claims`).

## Deploy

The service deploys to [Render](https://render.com) from `render.yaml`:

1. On Render, choose **New → Blueprint** and connect this repository.
2. Render reads `render.yaml`. The build regenerates the data and retrains the model with `STUDENT_SEED=3`, so the deployed model matches the one audited here. `model.joblib` is never committed.
3. Each merge to `main` triggers a redeploy.

## Limitations

- The claim-count features are measured in the same year as the label. A production version would use the **prior year's** counts, which this one-year dataset doesn't have.
- The test set has only 27 high-cost members, so recall and per-group error rates are noisy.
- The fairness audit scores all members, including training rows, which inflates the true-positive rates.
- The dollar values in `evaluate.py` are rough assumptions, not real program costs.

## Project layout

```
config.py                  seed and threshold settings
generate_claims_data.py    synthetic data generator
features.py                feature table and audit
train.py                   model training
evaluate.py                evaluation and operating point
fairness.py                fairness audit
app/                       FastAPI service (main.py, schemas.py)
notebooks/                 exploratory analysis
tests/                     pytest suite
render.yaml                Render deploy config
```
