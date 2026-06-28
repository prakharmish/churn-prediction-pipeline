# src/tune.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from ingest import load_data
from features import engineer_features
from train import split_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X_train, y_train):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

if __name__ == "__main__":
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train),
                   n_trials=50,
                   show_progress_bar=True)

    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")