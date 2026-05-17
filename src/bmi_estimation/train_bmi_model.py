"""
Script to train a machine learning model for BMI estimation.
Since we don't have a real dataset, this generates a synthetic dataset
based on realistic proportions and trains a RandomForestRegressor.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def generate_synthetic_data(n_samples=5000):
    """Generate synthetic body proportion data mapping to BMI."""
    logger.info(f"Generating {n_samples} synthetic training samples...")
    np.random.seed(42)
    
    # Base height in cm (150 to 200)
    heights = np.random.normal(170, 10, n_samples).clip(150, 200)
    
    # Base BMI distribution (16 to 40)
    bmis = np.random.gamma(shape=5, scale=5, size=n_samples).clip(16, 40)
    
    # Simulate features based on BMI
    # Higher BMI generally means higher waist/height, higher fill ratio
    waist_height_ratio = (bmis / 50) + np.random.normal(0, 0.05, n_samples)
    silhouette_fill = (bmis / 60) + 0.3 + np.random.normal(0, 0.05, n_samples)
    shoulder_hip_ratio = np.random.normal(1.2, 0.15, n_samples)
    body_aspect_ratio = (heights / (bmis * 2)) + np.random.normal(0, 0.1, n_samples)
    
    # Introduce some non-linear noise
    head_body_ratio = np.random.normal(0.13, 0.02, n_samples)
    
    df = pd.DataFrame({
        'height_cm': heights,
        'waist_height_ratio': waist_height_ratio,
        'silhouette_fill_ratio': silhouette_fill,
        'shoulder_hip_ratio': shoulder_hip_ratio,
        'body_aspect_ratio': body_aspect_ratio,
        'head_body_ratio': head_body_ratio,
        'target_bmi': bmis
    })
    
    return df


def train_model(output_dir="data/models"):
    """Train and save the Random Forest model."""
    if not HAS_SKLEARN:
        logger.error("scikit-learn is required to train the model. Install via: pip install scikit-learn")
        return

    df = generate_synthetic_data()
    
    features = [
        'height_cm', 'waist_height_ratio', 'silhouette_fill_ratio', 
        'shoulder_hip_ratio', 'body_aspect_ratio', 'head_body_ratio'
    ]
    
    X = df[features]
    y = df['target_bmi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    logger.info(f"Model trained! Validation MAE: {mae:.2f} BMI points")
    
    # Save the model
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    model_file = out_path / "bmi_rf_model.pkl"
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file}")


if __name__ == "__main__":
    train_model()
