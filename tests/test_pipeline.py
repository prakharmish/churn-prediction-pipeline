# tests/test_pipeline.py
import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingest import load_data
from features import engineer_features
from train import split_data

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# --- ingest tests ---

def test_load_data_shape():
    df = load_data(DATA_PATH)
    assert df.shape[0] > 0, "Dataset should have rows"
    assert df.shape[1] == 20, "Should have 20 columns after dropping customerID"

def test_load_data_no_nulls():
    df = load_data(DATA_PATH)
    assert df.isnull().sum().sum() == 0, "No nulls should exist after loading"

def test_load_data_target_binary():
    df = load_data(DATA_PATH)
    assert set(df["Churn"].unique()) == {0, 1}, "Churn should be binary 0/1"

def test_load_data_churn_rate():
    df = load_data(DATA_PATH)
    churn_rate = df["Churn"].mean()
    assert 0.20 < churn_rate < 0.35, "Churn rate should be between 20-35%"

# --- feature engineering tests ---

def test_engineer_features_adds_columns():
    df = load_data(DATA_PATH)
    df_feat = engineer_features(df)
    new_cols = ["charges_per_month", "is_long_tenure", "is_new_customer", "num_services", "high_risk"]
    for col in new_cols:
        assert col in df_feat.columns, f"Missing engineered feature: {col}"

def test_engineer_features_no_nulls():
    df = load_data(DATA_PATH)
    df_feat = engineer_features(df)
    assert df_feat.isnull().sum().sum() == 0, "No nulls after feature engineering"

def test_engineer_features_binary_flags():
    df = load_data(DATA_PATH)
    df_feat = engineer_features(df)
    assert df_feat["is_long_tenure"].isin([0, 1]).all(), "is_long_tenure should be binary"
    assert df_feat["is_new_customer"].isin([0, 1]).all(), "is_new_customer should be binary"
    assert df_feat["high_risk"].isin([0, 1]).all(), "high_risk should be binary"

def test_engineer_features_charges_per_month_positive():
    df = load_data(DATA_PATH)
    df_feat = engineer_features(df)
    assert (df_feat["charges_per_month"] >= 0).all(), "charges_per_month should be non-negative"

# --- train/test split tests ---

def test_split_data_sizes():
    df = load_data(DATA_PATH)
    df_feat = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df_feat)
    total = len(X_train) + len(X_test)
    assert total == len(df_feat), "Train + test should equal total rows"

def test_split_data_stratified():
    df = load_data(DATA_PATH)
    df_feat = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df_feat)
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.01, "Churn rate should be similar in train and test"