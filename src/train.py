"""
Trains and compares Logistic Regression, Random Forest, and XGBoost on the
cleaned Telco churn dataset. Saves metrics, the best model, and a SHAP
explainability plot to outputs/.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier
import shap

from data_prep import load_and_clean, _PROJECT_ROOT
import os

OUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # One-hot encode categoricals (low cardinality throughout this dataset,
    # so no leakage/dimensionality risk); scale numerics for the linear model
    # (tree models don't need it but scaling doesn't hurt them either).
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", pd.get_dummies, cat_cols),  # placeholder, replaced below
        ]
    )


def main():
    df = load_and_clean()
    y = df["Churn_Flag"]
    X = df.drop(columns=["Churn", "Churn_Flag"])

    # One-hot encode up front (simpler and inspectable for SHAP later than a
    # ColumnTransformer black box).
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for Logistic Regression only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=42,
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
        }
        fitted[name] = model
        print(f"{name}: {results[name]}")

    # --- Model selection ---
    # Pick the best model by F1 (balances precision/recall, appropriate here
    # since churn classes are imbalanced ~73/27 and both false positives —
    # wasted retention offers — and false negatives — lost customers — cost
    # the business something, so we don't want to over-optimize one at the
    # expense of the other the way accuracy or precision-only would).
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = fitted[best_name]
    print(f"\nBest model by F1: {best_name}")

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    # --- Confusion matrix for the best model ---
    X_eval = X_test_scaled if best_name == "Logistic Regression" else X_test
    cm = confusion_matrix(y_test, best_model.predict(X_eval))
    plt.figure(figsize=(4, 4))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix — {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # --- Model comparison bar chart ---
    comp_df = pd.DataFrame(results).T
    comp_df[["accuracy", "precision", "recall", "f1"]].plot(kind="bar", figsize=(7, 4))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
    plt.close()

    # --- SHAP explainability (on the best tree-based model if applicable) ---
    if best_name in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        plt.figure()
        shap.summary_plot(sv, X_test, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[train] Saved SHAP summary plot.")

    # --- Persist artifacts for the Streamlit app ---
    joblib.dump(best_model, f"{OUT_DIR}/best_model.joblib")
    joblib.dump(scaler, f"{OUT_DIR}/scaler.joblib")
    joblib.dump(list(X_encoded.columns), f"{OUT_DIR}/feature_columns.joblib")
    with open(f"{OUT_DIR}/best_model_name.txt", "w") as f:
        f.write(best_name)

    print(f"\nAll artifacts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
