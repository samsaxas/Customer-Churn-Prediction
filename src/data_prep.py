"""
Data preparation & feature engineering for the Telco Customer Churn dataset.
Handles missing values, outliers, and encodes features for modeling.
"""
import pandas as pd
import numpy as np

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
RAW_PATH = os.path.join(_PROJECT_ROOT, "data", "Telco-Customer-Churn.csv")


def load_and_clean(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # --- Missing values ---
    # TotalCharges is read as object because a handful of brand-new customers
    # (tenure == 0) have a blank string instead of 0.00. Coerce to numeric and
    # impute those blanks with 0, since a customer with 0 months of tenure has
    # by definition paid nothing in total yet — this is not "missing at random",
    # it's a data-entry quirk for a specific customer segment.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    print(f"[data_prep] Imputed {n_missing} missing TotalCharges values (tenure=0 customers) with 0.0")

    # --- Outliers ---
    # Check tenure and MonthlyCharges for implausible values (e.g. negative,
    # or beyond physically sensible bounds). None found in this dataset, but
    # the check is kept explicit rather than assumed.
    assert (df["tenure"] >= 0).all(), "Negative tenure found"
    assert (df["MonthlyCharges"] > 0).all(), "Non-positive MonthlyCharges found"

    # --- Drop non-predictive identifier ---
    df = df.drop(columns=["customerID"])

    # --- Feature engineering ---
    # Tenure buckets: a customer's likelihood to churn is highly non-linear in
    # tenure (very new and very old customers behave differently), so bucketing
    # gives tree models an easier signal and gives us a clean chart to reason about.
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"],
    )

    # Count of subscribed add-on services — a proxy for "how embedded" a
    # customer is in the product ecosystem, which is a common churn driver.
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumAddonServices"] = (df[service_cols] == "Yes").sum(axis=1)

    # Average monthly spend implied by total charges vs. actual monthly charge
    # — flags customers whose spend has recently changed (upgraded/downgraded).
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # --- Target encoding ---
    df["Churn_Flag"] = (df["Churn"] == "Yes").astype(int)

    return df


if __name__ == "__main__":
    df = load_and_clean()
    print(df.shape)
    print(df["Churn_Flag"].value_counts(normalize=True).rename("churn_rate"))
    out_path = os.path.join(_PROJECT_ROOT, "data", "telco_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_prep] Saved cleaned dataset to {out_path}")
