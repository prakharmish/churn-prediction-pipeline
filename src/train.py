# src/train.py
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import xgboost as xgb
from ingest import load_data
from features import engineer_features
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report

def split_data(df, target="Churn", test_size=0.2, seed=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=seed)

def evaluate(model, X_test, y_test, model_name="Model"):
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds       = model.predict(X_test)
    auc  = roc_auc_score(y_test, preds_proba)
    f1   = f1_score(y_test, preds)
    print(f"\n--- {model_name} ---")
    print(f"AUC:  {auc:.4f}")
    print(f"F1:   {f1:.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    return {"model": model_name, "auc": auc, "f1": f1}

if __name__ == "__main__":
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Churn rate in test: {y_test.mean():.2%}")

    # Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline_results = evaluate(dummy, X_test, y_test, "Dummy Baseline")

    # XGBoost default
    xgb_model = xgb.XGBClassifier(
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_results = evaluate(xgb_model, X_test, y_test, "XGBoost Default")