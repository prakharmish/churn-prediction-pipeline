# src/shap_explain.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from ingest import load_data
from features import engineer_features
from train import split_data

def generate_shap_plots(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)

    # Plot 1: Global feature importance
    plt.figure()
    shap.summary_plot(shap_vals, X_test, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", bbox_inches="tight")
    plt.close()
    print("Saved: reports/shap_summary.png")

    # Plot 2: Single customer explanation
    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=X_test.iloc[0],
            feature_names=X_test.columns.tolist()
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig("reports/shap_waterfall.png", bbox_inches="tight")
    plt.close()
    print("Saved: reports/shap_waterfall.png")

if __name__ == "__main__":
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Train best model
    best_params = {
        "n_estimators": 307,
        "max_depth": 3,
        "learning_rate": 0.021205794819626343,
        "subsample": 0.6205212622208144,
        "colsample_bytree": 0.955764735687341,
        "scale_pos_weight": 8.664403420711734,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": 42
    }
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    generate_shap_plots(model, X_test)
    print("\nDone! Check reports/ folder for charts.")