# ==================
# IMPORTS
# ==================
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Global Geo Economic Stress Prediction",
    description="",
    version="1.0.0"
)


# ==================
# MODEL PREPARATION
# ==================
model = joblib.load("../machine_learning/models/model.pkl")

# Exact feature order the model was trained on
FEATURE_ORDER = [
    "year", "gdp_growth", "inflation", "unemployment", "gdp_per_capita",
    "population", "food_production_index", "cereal_yield",
    "cereal_production_tonnes", "agricultural_land_pct",
    "dietary_energy_supply_adequacy", "data_completeness_score",
    "latitude", "longitude", "food_pressure_score",
    "income_group_Low_income", "income_group_Lower_middle_income",
    "income_group_Not_classified", "income_group_Upper_middle_income",
]


# ==================
# REQUEST SCHEMA
# ==================
class StressData(BaseModel):
    """
    Raw input fields - the same units/scale as the original dataset
    (e.g. inflation as a percentage like 5.2, not a normalized score).
    """
    year: int = Field(..., example=2020)
    gdp_growth: float = Field(..., example=3.5)
    inflation: float = Field(..., example=4.2)
    unemployment: float = Field(..., example=6.1)
    gdp_per_capita: float = Field(..., example=12000.0)
    population: float = Field(..., example=5000000.0)
    food_production_index: float = Field(..., example=98.0)
    cereal_yield: float = Field(..., example=3200.0)
    cereal_production_tonnes: float = Field(..., example=1500000.0)
    agricultural_land_pct: float = Field(..., example=35.0)
    dietary_energy_supply_adequacy: float = Field(..., example=120.0)
    data_completeness_score: float = Field(..., example=95.0)
    latitude: float = Field(..., example=40.0)
    longitude: float = Field(..., example=20.0)
    food_pressure_score: float = Field(..., example=45.0)
    income_group: str = Field(..., example="Upper middle income")


class PredictionResponse(BaseModel):
    predicted_economic_stress_score: float


# ==================
# API
# ==================
@app.post("/predict", response_model=PredictionResponse)
def predict(data: StressData):
    row = data.model_dump()
    income_group = row.pop("income_group")

    # one-hot encode income_group (same drop_first=True logic as training)
    for category in ["Low income", "Lower middle income", "Not classified", "Upper middle income"]:
        col_name = f"income_group_{category.replace(' ', '_')}"
        row[col_name] = 1 if income_group == category else 0

    df = pd.DataFrame([row])[FEATURE_ORDER]

    prediction = model.predict(df)[0]
    prediction = round(float(prediction), 2)
    return PredictionResponse(predicted_economic_stress_score=prediction)