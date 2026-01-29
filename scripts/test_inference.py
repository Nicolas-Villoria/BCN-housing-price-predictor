"""
Test inference script to verify model loading and prediction works correctly.
"""

import sys
sys.path.insert(0, '.')

import joblib
import json
from pathlib import Path

# Get paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 60)
print("Testing sklearn Model Inference")
print("=" * 60)

# Load artifacts
print("\n[1/3] Loading model artifacts...")
model = joblib.load(MODELS_DIR / "champion_model.pkl")
transformer = joblib.load(MODELS_DIR / "feature_transformer.pkl")

with open(MODELS_DIR / "model_metadata.json") as f:
    metadata = json.load(f)

print(f"  Model type: {metadata['model_type']}")
print(f"  Training date: {metadata['training_date']}")
print(f"  Features: {transformer.get_feature_names()}")

# Test inference with sample properties
print("\n[2/3] Testing inference with sample properties...")

test_cases = [
    {
        "name": "Small flat in Eixample",
        "size": 60.0,
        "rooms": 2,
        "bathrooms": 1,
        "neighborhood": "la Dreta de l'Eixample",
        "propertyType": "flat",
        "district": "Eixample",
        "avg_income_index": 130.0,
        "density_val": 400.0
    },
    {
        "name": "Large chalet in Sarrià",
        "size": 300.0,
        "rooms": 5,
        "bathrooms": 3,
        "neighborhood": "Sarrià",
        "propertyType": "chalet",
        "district": "Sarrià-Sant Gervasi",
        "avg_income_index": 180.0,
        "density_val": 50.0
    },
    {
        "name": "Studio in Gràcia",
        "size": 35.0,
        "rooms": 1,
        "bathrooms": 1,
        "neighborhood": "la Vila de Gràcia",
        "propertyType": "studio",
        "district": "Gràcia",
        "avg_income_index": 110.0,
        "density_val": 350.0
    }
]

for case in test_cases:
    name = case.pop("name")
    features = transformer.transform(case)
    prediction = model.predict(features)[0]
    print(f"\n  {name}:")
    print(f"    Size: {case['size']}m², Rooms: {case['rooms']}, District: {case['district']}")
    print(f"    Predicted Price: €{prediction:,.2f}")

# Metrics summary
print("\n[3/3] Model performance metrics...")
print(f"  RMSE: €{metadata['metrics']['rmse']:,.2f}")
print(f"  R²: {metadata['metrics']['r2']:.4f}")
print(f"  MAE: €{metadata['metrics']['mae']:,.2f}")
print(f"  Threshold passed: {metadata['threshold_passed']}")

print("\n" + "=" * 60)
print("✓ Inference test completed successfully!")
print("=" * 60)
