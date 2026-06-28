# src/evaluate.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlflow
import mlflow.xgboost
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from ingest import load_data
from features import engineer_features
from train import split_data

def run_experiment(X_train, X_test, y_train, y_test, params, run_name="xgb_run"):
    mlflow.set_experiment("telco-churn")

    with mlflow.start_run(run_name=run_name):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds_proba = model.predict_proba(X_test)[:, 1]
        preds       = model.predict(X_test)

        auc = roc_auc_score(y_test, preds_proba)
        f1  = f1_score(y_test, preds)

        # Log params and metrics
        mlflow.log_params(params)
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_f1",  f1)

        # Log model
        mlflow.xgboost.log_model(model, "model")

        print(f"\n--- {run_name} ---")
        print(f"AUC: {auc:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, preds, zero_division=0))

    return model, auc, f1

if __name__ == "__main__":
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Default XGBoost
    default_params = {
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": 42
    }
    run_experiment(X_train, X_test, y_train, y_test,
                   default_params, run_name="xgb_default")

    # Optuna tuned
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
    best_model, best_auc, best_f1 = run_experiment(
        X_train, X_test, y_train, y_test,
        best_params, run_name="xgb_optuna"
    )