# Customer Churn Prediction — Model Comparison, Explainability & Deployment

Predicts whether a telecom customer is likely to churn, using the IBM Telco
Customer Churn dataset (7,043 customers, 21 raw features). Built to compare
multiple algorithms with justified model selection, explain predictions with
SHAP, and deploy the result as a usable app — not just a notebook.

## Dataset

[IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
— 7,043 rows, 21 columns, target = `Churn` (Yes/No). Class balance: ~73% no-churn,
~27% churn (imbalanced, handled via `class_weight="balanced"` / `scale_pos_weight`
rather than resampling, to keep the pipeline simple and avoid synthetic data).

## Data preparation (`src/data_prep.py`)

- **Missing values:** `TotalCharges` is read as text because 11 brand-new
  customers (tenure = 0) have a blank value instead of `0.00`. This isn't
  random — it's a specific segment (zero-tenure customers) — so it's imputed
  with `0.0` rather than dropped or mean-imputed.
- **Outlier checks:** explicit assertions on `tenure` and `MonthlyCharges` for
  implausible values (none found).
- **Feature engineering:**
  - `TenureGroup` — bucketed tenure (0–6mo, 7–12mo, 1–2yr, 2–4yr, 4–6yr), since
    churn risk is non-linear in tenure.
  - `NumAddonServices` — count of subscribed add-ons (security, backup, device
    protection, tech support, streaming TV/movies), a proxy for how embedded a
    customer is in the product.
  - `AvgMonthlySpend` — `TotalCharges / tenure`, flags customers whose recent
    spend has shifted from their historical average.

## Model comparison (`src/train.py`)

Three models trained on an 80/20 stratified split, with a fixed random seed:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7395 | 0.5059 | 0.8021 | 0.6205 | 0.8447 |
| **Random Forest (selected)** | **0.7608** | **0.5339** | **0.7781** | **0.6333** | 0.8435 |
| XGBoost | 0.7480 | 0.5168 | 0.7807 | 0.6219 | 0.8376 |

**Why Random Forest won:** it has the best F1 (0.633) of the three, and a good
recall/precision balance. F1 — not accuracy — is the right selection metric here
because the classes are imbalanced (~73/27): a model that just predicts
"no churn" for everyone would still score ~73% accuracy while being useless.
Logistic Regression actually edges it slightly on ROC-AUC and recall, which
makes sense — it's a well-regularized linear model on a dataset where several
features (contract type, tenure) have a fairly linear relationship with churn,
so it isn't dominated by the more flexible tree ensembles here. Random Forest's
edge comes from capturing non-linear interactions (e.g., short tenure combined
with month-to-month contract compounding risk in a way a linear model can't
represent as one term) without XGBoost's tendency to overfit this specific
feature set at the tried hyperparameters. In a real production setting, the
"right" choice actually depends on the business cost of a false negative
(missed churner) vs. false positive (wasted retention offer) — if recall
mattered more than precision, Logistic Regression would be the better pick
despite the lower F1, which is exactly the kind of tradeoff worth raising in
an interview rather than presenting F1 as the only correct answer.

## Explainability

SHAP (`TreeExplainer`) on the Random Forest model — see `outputs/shap_summary.png`.
Top churn drivers: contract type (month-to-month customers churn far more than
1-2yr contracts), tenure, internet service type (fiber optic customers churn
more than DSL), and monthly charges.

## Deployment

`app.py` — a Streamlit app that takes a customer's account details as input
and returns a churn probability, prediction, and the SHAP-driven model
performance summary. Run with:

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/train.py
streamlit run app.py
```

## Repo structure

```
churn_prediction/
├── data/                  # raw + cleaned dataset
├── src/
│   ├── data_prep.py       # cleaning + feature engineering
│   └── train.py           # model comparison, SHAP, artifact export
├── outputs/                # metrics.json, plots, saved model
├── app.py                  # Streamlit deployment
├── requirements.txt
└── README.md
```
# Results:

<img width="830" height="911" alt="image" src="https://github.com/user-attachments/assets/d7139141-2f2c-4626-bc80-3506e7fb690a" />

<img width="702" height="373" alt="image" src="https://github.com/user-attachments/assets/5610bd98-e676-41f7-9843-7753c66947b9" />
