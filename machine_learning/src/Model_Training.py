# ==================
# IMPORTS
# ==================
import joblib
import pandas as pd

from utils import load_data, evaluate

from sklearn.model_selection import KFold
from xgboost import XGBRegressor

import warnings

warnings.filterwarnings("ignore")


def train_model(input_path: str, output_path: str):
    # ==================
    # DATA PREPARATION
    # ==================
    df = load_data(input_path)

    # ==================
    # SPLIT FEATURES / TARGET
    # ==================
    # We intentionally remove engineered features to:
    # 1. avoid redundancy or potential multicollinearity
    # 2. avoid redundancy or potential multicollinearity
    # 3. compare "raw features vs engineered features" fairly
    drop_cols = [
        "inflation_unemployment",
        "inflation_minus_growth",
        "inflation_squared"
    ]
    df = df.drop(columns=drop_cols)

    TARGET = "economic_stress_score"

    # x = input features used for prediction
    # y = target variable we want to predict (economic stress level)
    x = df.drop(columns=TARGET)
    y = df[TARGET]

    # ==================
    # CV SETUP
    # ==================
    # KFold provides a more reliable evaluation than a single train/test split
    # It reduces variance by training and testing on multiple folds of the dataset
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ==================
    # MODEL
    # ==================
    """
    XGBoost is chosen because:
    - strong performance on tabular data
    - handles non-linear relationships well
    - robust to feature interactions
    
    These values come from notebook experiments:
        Linear Regression:
        RMSE = 6.20 ± 0.22
        R2   = 0.83 ± 0.01
        
        Random Forest:
        RMSE = 4.21 ± 0.11
        R2   = 0.92 ± 0.00
        
        XGBoost (selected model):
        RMSE = 2.40 ± 0.04
        R2   = 0.97 ± 0.00

    Final decision:
    XGBoost is chosen due to best accuracy and lowest variance across folds
    """
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # ==================
    # TRAINING
    # ==================
    # Evaluate model before final training to avoid overfitting bias
    score = evaluate(model, kf, x, y)

    """
    These results represent the final aggregated performance from notebook experiments.
    RMSE ± std:
    
    - Measures average prediction error (lower is better)
    - std shows how stable the model is across different folds
    
    R2 ± std:
    - Measures how much variance the model explains (higher is better)
    - std shows consistency across folds
    
    This is the main metric used to compare all models (Linear / RF / XGB)
    and select the final production model.
    """
    print("\nCROSS-VALIDATION RESULTS")
    print("========================")
    print(f"RMSE: {score[0]:.4f} ± {score[1]:.4f}")
    print(f"R2  : {score[2]:.4f} ± {score[3]:.4f}")

    # ==================
    # FINAL TRAINING
    # ==================
    # Train final model on full dataset after validation
    # This ensures the model learns from all available data before deployment
    model.fit(x, y)

    # ==================
    # SAVING MODEL
    # ==================
    joblib.dump(model, output_path)


# ==================
# RUN PIPELINE
# ==================
if __name__ == "__main__":
    try:
        train_model(
            input_path="../data/processed_data/processed_merged_data.csv",
            output_path="../models/model.pkl"
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise