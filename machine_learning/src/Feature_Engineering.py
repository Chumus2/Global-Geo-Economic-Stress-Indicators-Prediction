# ==================
# IMPORTS
# ==================
import pandas as pd

from utils import load_data

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def feature_engineering(input_path: str, output_path: str):
    # ==================
    # DATA PREPARATION
    # ==================
    merged = load_data(input_path)

    # ==================
    # DROPPING UNNECESSARY FEATURES
    # ==================
    # Remove identifiers and already encoded/derived features
    # to avoid model's accuracy.
    merged = merged.drop(columns=[
        "iso3", "country_name", "country_code", "region", "final_economic_stress_score",
        "inflation_score", "unemployment_score", "gdp_growth_score", "income_vulnerability_score"
    ])

    # ==================
    # ENCODING
    # ==================
    merged = pd.get_dummies(merged, columns=["income_group"], drop_first=True)
    # Standardize column names (removes spaces for ML compatibility).
    merged.columns = merged.columns.str.replace(" ", "_", regex=False)

    # ==================
    # FEATURE CREATION (INTERACTIONS)
    # ==================
    # Capture nonlinear and interactions effects between macro indicators

    # Inflation-unemployment interaction (economic stress coupling)
    merged["inflation_unemployment"] = merged["inflation"] * merged["unemployment"]
    # Inflation-growth imbalance
    merged["inflation_minus_growth"] = merged["inflation"] - merged["gdp_growth"]
    # Non-linear inflation effect
    merged["inflation_squared"] = merged["inflation"] ** 2

    # ==================
    # SAVING PROCESSED DATAFRAME
    # ==================
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    return merged


# ==================
# RUN PIPELINE
# ==================
if __name__ == "__main__":
    try:
        df = feature_engineering(
            input_path="../data/cleaned_data/cleaned_merged_data.csv",
            output_path="../data/processed_data/processed_merged_data.csv"
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise