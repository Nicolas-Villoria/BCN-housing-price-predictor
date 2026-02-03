"""
train_sklearn.py
-----------------------
Production-Ready sklearn Training Pipeline

This script replaces the PySpark-based ml_training.py for production deployment.
It produces a lightweight, portable model that can be served via FastAPI without JVM overhead.

Key Differences from Spark Version:
- Uses pandas + sklearn instead of PySpark MLlib
- Model serialized as joblib (fast load, no Java dependency)
- Identical feature engineering logic for consistency
- Suitable for datasets that fit in memory (< 1GB)

Output Artifacts:
- models/champion_model.pkl       : Best trained model
- models/feature_transformer.pkl  : Preprocessing pipeline (encoders + scaler)
- models/model_metadata.json      : Version, metrics, feature names
- reports/                        : Visualization artifacts
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/CD
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import FeatureTransformer from separate module for proper serialization
from feature_transformer import FeatureTransformer, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Import model versioning
try:
    from model_versioning import ModelVersionManager
    VERSIONING_AVAILABLE = True
except ImportError:
    VERSIONING_AVAILABLE = False

# MLflow (optional - for experiment tracking)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Metrics will be logged locally only.")

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data_lake" / "gold" / "property_prices_csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# MLflow
MLFLOW_EXPERIMENT_NAME = "bda_project_sklearn"
MLFLOW_TRACKING_URI = f"file:{PROJECT_ROOT / 'mlruns'}"

# Model thresholds 
RMSE_THRESHOLD = 150000  # Maximum acceptable RMSE (€)
R2_THRESHOLD = 0.6       # Minimum acceptable R²

# Target column
TARGET = "price"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_gold_data() -> pd.DataFrame:
    """
    Load the Gold layer data from CSV files.
    
    The Gold layer is produced by exploitation_zone.py and contains
    the final feature-engineered dataset ready for ML training.
    
    Returns:
        pd.DataFrame: Combined dataset from all available CSV partitions
    """
    # Find the latest execution date folder
    csv_folders = sorted([f for f in DATA_PATH.iterdir() if f.is_dir()])
    
    if not csv_folders:
        raise FileNotFoundError(f"No data found in {DATA_PATH}. Run exploitation_zone.py first.")
    
    latest_folder = csv_folders[-1]
    print(f"Loading data from: {latest_folder}")
    
    # Read all CSV parts (Spark writes multiple part files)
    csv_files = list(latest_folder.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {latest_folder}")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data):,} records")
    
    return data


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.
    
    Args:
        df: Raw dataframe from Gold layer
        
    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    # Select only the columns we need
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available_features = [c for c in feature_cols if c in df.columns]
    
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"Warning: Missing features: {missing}")
    
    X = df[available_features].copy()
    y = df[TARGET].copy()
    
    # Handle missing values
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
    
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown")
    
    # Drop any remaining NaN rows
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"Features prepared: {len(X):,} samples, {len(available_features)} features")
    
    return X, y


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def train_models(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """
    Train multiple regression models with hyperparameter tuning.
        
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data for evaluation
        
    Returns:
        Dictionary with best model info and all results
    """
    models_config = [
        {
            "name": "Ridge",
            "model": Ridge(),
            "params": {
                "alpha": [0.01, 0.1, 1.0, 10.0]
            }
        },
        {
            "name": "RandomForest",
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            }
        },
        {
            "name": "GradientBoosting",
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1]
            }
        }
    ]
    
    results = []
    best_model_info = {"rmse": float("inf"), "model": None, "name": None}
    
    for config in models_config:
        print(f"\n{'='*50}")
        print(f"Training: {config['name']}")
        print(f"{'='*50}")
        
        # Grid Search with 3-fold CV (same as Spark version)
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model from grid search
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Best params: {grid_search.best_params_}")
        print(f"Test RMSE: €{rmse:,.2f}")
        print(f"Test R²: {r2:.4f}")
        print(f"Test MAE: €{mae:,.2f}")
        
        result = {
            "name": config["name"],
            "model": best_model,
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "best_params": grid_search.best_params_,
            "predictions": y_pred
        }
        results.append(result)
        
        # Track best model
        if rmse < best_model_info["rmse"]:
            best_model_info = {
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
                "model": best_model,
                "name": config["name"],
                "params": grid_search.best_params_,
                "predictions": y_pred
            }
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"sklearn_{config['name']}"):
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
                mlflow.sklearn.log_model(best_model, "model")
                mlflow.set_tags({
                    "model_family": config["name"],
                    "framework": "sklearn",
                    "cv_folds": 3
                })
    
    return {"best": best_model_info, "all_results": results}


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_metrics_comparison(results: list[dict], output_dir: Path):
    """Generate comparison charts for all models."""
    df = pd.DataFrame([{
        "model": r["name"],
        "rmse": r["rmse"],
        "r2": r["r2"]
    } for r in results])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    sns.barplot(x="model", y="rmse", data=df, palette="Reds_r", ax=ax1, hue="model", legend=False)
    ax1.set_title("RMSE Comparison (Lower is Better)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("RMSE (€)")
    for container in ax1.containers:
        ax1.bar_label(container, fmt='€%.0f', padding=3)
    
    # R² comparison
    sns.barplot(x="model", y="r2", data=df, palette="Greens", ax=ax2, hue="model", legend=False)
    ax2.set_title("R² Comparison (Higher is Better)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("R² Score")
    ax2.set_ylim([0, 1])
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.4f', padding=3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_dir: Path):
    """Generate actual vs predicted scatter plot."""
    plt.figure(figsize=(10, 10))
    
    # Subsample for clarity
    n_samples = min(5000, len(y_true))
    indices = np.random.choice(len(y_true), n_samples, replace=False)
    
    plt.scatter(y_true[indices], y_pred[indices], alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel("Actual Price (€)", fontsize=12)
    plt.ylabel("Predicted Price (€)", fontsize=12)
    plt.title(f"Actual vs Predicted: {model_name}", fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=100)
    plt.close()


def plot_feature_importance(model, feature_names: list[str], output_dir: Path):
    """Generate feature importance chart for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature importance visualization")
        return
    
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        palette="magma",
        hue=[feature_names[i] for i in indices],
        legend=False
    )
    plt.title("Global Feature Importance (Top 15)", fontsize=14, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=100)
    plt.close()


# ==============================================================================
# MODEL SERIALIZATION
# ==============================================================================

def save_model_artifacts(
    model,
    transformer: FeatureTransformer,
    metrics: dict,
    output_dir: Path,
    training_samples: int = 0,
    data_path: Path = None
):
    """
    Save all model artifacts needed for production inference.
    
    Artifacts:
    - champion_model.pkl: The trained sklearn model
    - feature_transformer.pkl: Fitted preprocessing pipeline
    - model_metadata.json: Version, metrics, feature names (with MLOps metadata)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "champion_model.pkl"
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save transformer
    transformer_path = output_dir / "feature_transformer.pkl"
    joblib.dump(transformer, transformer_path)
    print(f"✓ Transformer saved: {transformer_path}")
    
    # Create metadata with enhanced versioning if available
    if VERSIONING_AVAILABLE:
        version_manager = ModelVersionManager()
        metadata = version_manager.create_version_metadata(
            model_type=metrics["name"],
            metrics=metrics,
            training_samples=training_samples,
            data_path=data_path,
            bump="patch",
            description=f"Auto-trained {metrics['name']} model",
            tags=["auto-trained"]
        )
        # Add feature information
        metadata["feature_names"] = transformer.get_feature_names()
        metadata["numeric_features"] = NUMERIC_FEATURES
        metadata["categorical_features"] = CATEGORICAL_FEATURES
        print(f" Enhanced versioning: v{metadata['semantic_version']}")
    else:
        # Fallback to basic metadata
        metadata = {
            "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_type": metrics["name"],
            "metrics": {
                "rmse": round(metrics["rmse"], 2),
                "r2": round(metrics["r2"], 4),
                "mae": round(metrics.get("mae", 0), 2)
            },
            "feature_names": transformer.get_feature_names(),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "training_date": datetime.now().isoformat(),
            "threshold_passed": metrics["rmse"] < RMSE_THRESHOLD and metrics["r2"] > R2_THRESHOLD
        }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")
    
    return metadata


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def main():
    """
    Main training pipeline orchestration.
    
    This function:
    1. Loads data from Gold layer
    2. Prepares features
    3. Trains multiple models with CV
    4. Selects and saves the best model
    5. Generates reports
    """
    print("="*60)
    print("sklearn Training Pipeline")
    print("="*60)
    
    # Setup directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # 1. Load data
    print("\n[1/5] Loading Gold layer data...")
    df = load_gold_data()
    
    # 2. Prepare features
    print("\n[2/5] Preparing features...")
    X, y = prepare_features(df)
    
    # 3. Split data
    print("\n[3/5] Splitting data (80/20)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training: {len(X_train_raw):,} samples")
    print(f"Test: {len(X_test_raw):,} samples")
    
    # 4. Fit transformer and transform data
    print("\n[4/5] Fitting feature transformer...")
    transformer = FeatureTransformer()
    X_train = transformer.fit_transform(X_train_raw)
    X_test = transformer.transform(X_test_raw)
    
    print(f"Feature names: {transformer.get_feature_names()}")
    
    # 5. Train models
    print("\n[5/5] Training models...")
    results = train_models(X_train, y_train, X_test, y_test)
    
    best = results["best"]
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Model: {best['name']}")
    print(f"RMSE: €{best['rmse']:,.2f}")
    print(f"R²: {best['r2']:.4f}")
    
    # Check thresholds
    if best["rmse"] < RMSE_THRESHOLD and best["r2"] > R2_THRESHOLD:
        print(f"\n✓ Model PASSED quality thresholds")
        print(f"  RMSE < €{RMSE_THRESHOLD:,}: ✓")
        print(f"  R² > {R2_THRESHOLD}: ✓")
    else:
        print(f"\n✗ Model FAILED quality thresholds")
        if best["rmse"] >= RMSE_THRESHOLD:
            print(f"  RMSE €{best['rmse']:,.2f} >= €{RMSE_THRESHOLD:,}: ✗")
        if best["r2"] <= R2_THRESHOLD:
            print(f"  R² {best['r2']:.4f} <= {R2_THRESHOLD}: ✗")
    
    # Save artifacts
    print("\nSaving model artifacts...")
    metadata = save_model_artifacts(
        best["model"], 
        transformer, 
        best, 
        MODELS_DIR,
        training_samples=len(X_train_raw),
        data_path=DATA_PATH
    )
    
    # Generate visualizations
    print("\nGenerating reports...")
    plot_metrics_comparison(results["all_results"], REPORT_DIR)
    plot_actual_vs_predicted(y_test.values, best["predictions"], best["name"], REPORT_DIR)
    plot_feature_importance(best["model"], transformer.get_feature_names(), REPORT_DIR)
    
    print(f"\n✓ Reports saved to: {REPORT_DIR}")
    print(f"✓ Models saved to: {MODELS_DIR}")
    
    # Final output for CI/CD
    print("\n" + "="*60)
    print("ARTIFACTS FOR DEPLOYMENT")
    print("="*60)
    print(f"Model:       {MODELS_DIR / 'champion_model.pkl'}")
    print(f"Transformer: {MODELS_DIR / 'feature_transformer.pkl'}")
    print(f"Metadata:    {MODELS_DIR / 'model_metadata.json'}")
    
    return metadata


if __name__ == "__main__":
    main()
