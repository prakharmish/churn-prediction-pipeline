# api/main.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import shap
import pickle

from features import engineer_features
from ingest import load_data
from train import split_data

app = FastAPI(title="Churn Prediction API")

# Load model at startup
def get_model():
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
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, X_train.columns.tolist()

model, feature_names = get_model()
explainer = shap.TreeExplainer(model)

class Customer(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float
    charges_per_month: float
    is_long_tenure: int
    is_new_customer: int
    num_services: int
    high_risk: int

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    proba = model.predict_proba(df)[0][1]
    shap_vals = explainer.shap_values(df)[0]
    top_feature = feature_names[list(shap_vals).index(max(shap_vals, key=abs))]

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": int(proba > 0.5),
        "top_reason": top_feature,
        "shap_values": dict(zip(feature_names, [round(float(v), 4) for v in shap_vals]))
    }

@app.get("/health")
def health():
    return {"status": "ok"}