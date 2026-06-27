# src/ingest.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fix: TotalCharges has spaces instead of nulls
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Convert target to binary: Yes=1, No=0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Drop customer ID — it's just a label, model can't learn from it
    df = df.drop(columns=["customerID"])

    # Sanity checks
    assert df["Churn"].nunique() == 2, "Target should be binary"
    assert df.isnull().sum().sum() == 0, "Unexpected nulls found"

    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    print(f"No nulls found")
    return df

if __name__ == "__main__":
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(df.head())