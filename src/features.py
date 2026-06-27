# src/features.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Interaction features ---

    # How much does this customer pay per month on average?
    # Normalizes TotalCharges for tenure length
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Is this a long-term loyal customer?
    # EDA showed churners are concentrated in first 10 months
    df["is_long_tenure"] = (df["tenure"] > 24).astype(int)

    # Is this a new customer? High risk group from EDA
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    # How many services does this customer have?
    # More services = more locked in = less likely to churn
    df["num_services"] = (
        (df["PhoneService"] == "Yes").astype(int) +
        (df["InternetService"] != "No").astype(int) +
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int) +
        (df["OnlineBackup"] == "Yes").astype(int) +
        (df["OnlineSecurity"] == "Yes").astype(int)
    )

    # High monthly charges + month-to-month = highest risk combo
    df["high_risk"] = (
        (df["MonthlyCharges"] > 65) &
        (df["Contract"] == "Month-to-month")
    ).astype(int)

    # --- Encode categoricals ---
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    print(f"Features shape: {df.shape}")
    print(f"New features added: charges_per_month, is_long_tenure, is_new_customer, num_services, high_risk")
    return df

if __name__ == "__main__":
    from ingest import load_data
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = engineer_features(df)
    print(df.head())
    print(df.dtypes)