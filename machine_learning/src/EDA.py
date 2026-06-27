# ==================
# IMPORTS
# ==================
import numpy as np
import pandas as pd

from utils import load_data, clip_outliers, fill_missing, drop_unfillable

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def eda():
    # ==================
    # DATA PREPARATION
    # ==================
    metadata = load_data("../data/raw_data/raw_country_metadata.csv")
    year_indicators = load_data("../data/raw_data/raw_country_year_indicators.csv")
    stress_score = load_data("../data/raw_data/raw_economic_stress_score.csv")


    # ==================
    # NULLS & DUPLICATES METADATA
    # ==================
    # lat/long don't vary by year, so a simple global median fill is fine here
    metadata["latitude"] = metadata["latitude"].fillna(metadata["latitude"].median())
    metadata["longitude"] = metadata["longitude"].fillna(metadata["longitude"].median())


    # ==================
    # NULLS & DUPLICATES YEAR INDICATORS
    # ==================
    # Target column - rows without it are useless for modeling, drop upfront
    year_indicators = year_indicators[year_indicators["economic_stress_score"].notna()]

    year_indicators = clip_outliers(year_indicators, target="inflation")

    # GOTCHA: .drop() takes a list, not multiple positional args
    numeric_year_cols = year_indicators.select_dtypes(include="number").columns.drop(["year", "economic_stress_score"])

    # Fill what we can per-year, drop what's still NaN after that
    for col in numeric_year_cols:
        year_indicators = fill_missing(year_indicators, target=col, feature="year")
        year_indicators = drop_unfillable(year_indicators, target=col)
        print(f"{col}: NaN left = {year_indicators[col].isna().sum()}")
        print("")


    # ==================
    # ANOMALIES YEAR INDICATORS
    # ==================
    cols_to_clip = [
        "gdp_growth", "inflation", "unemployment",
        "cereal_yield", "dietary_energy_supply_adequacy",
        "food_production_index"
    ]

    for col in cols_to_clip:
        # NOTE: this branch never triggers - gdp_per_capita/population/cereal_production_tonnes
        # aren't in cols_to_clip, so log1p is
        # never actually applied here. Likely leftover/dead logic.
        if col in ["gdp_per_capita", "population", "cereal_production_tonnes"]:
            year_indicators[col] = np.log1p(year_indicators[col])
        else:
            year_indicators = clip_outliers(year_indicators, target=col, lower_quantile=0.05, upper_quantile=0.95)


    # ==================
    # NULLS & DUPLICATES STRESS SCORE
    # ==================
    numeric_stress_cols = stress_score.select_dtypes(include="number").columns.drop("year")

    for col in numeric_stress_cols:
        stress_score = fill_missing(stress_score, target=col, feature="year")
        stress_score = drop_unfillable(stress_score, target=col)
        print(f"{col}: NaN left = {stress_score[col].isna().sum()}")
        print("")


    # ==================
    # MERGING
    # ==================
    # Only pull lat/long from metadata - rest of it's columns already
    # exist in year_indicators, would create duplicate x/y columns
    merged = year_indicators.merge(
        metadata[["country_code", "latitude", "longitude"]],
        on="country_code",
        how="left"
    )

    # Same idea: only pull the score columns unique to stress_score
    score_cols = [
        "country_code", "year",
        "inflation_score", "unemployment_score", "gdp_growth_score",
        "income_vulnerability_score", "food_pressure_score",
        "final_economic_stress_score"
    ]

    # Composite key (country_code + year) - stress_score has multiple
    # rows per country, so country_code alone would duplicate rows
    merged = merged.merge(
        stress_score[score_cols],
        on=["country_code", "year"],
        how="left"
    )

    # Sanity check: merge should never multiply rows if keys are unique on the right side
    assert merged.shape[0] == year_indicators.shape[0], "Row count mismatch after merge!"
    print(f"Final shape: {merged.shape}, NaNs: {merged.isna().sum().sum()}") # index=False avoids an 'Unnamed: 0' column


    # ==================
    # SAVING DATAFRAME
    # ==================
    Path("../data/cleaned_data").mkdir(parents=True, exist_ok=True)
    merged.to_csv("../data/cleaned_data/cleaned_merged_data.csv", index=False)


# ==================
# RUN PIPELINE
# ==================
if __name__ == "__main__":
    try:
        eda()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise