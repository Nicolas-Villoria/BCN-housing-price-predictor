"""
Model loading and inference utilities.

This module handles loading the trained sklearn model and transformer
at application startup, ensuring they're only loaded once.
"""

import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np

import joblib

# Configure logging
logger = logging.getLogger(__name__)

# Paths - can be overridden via environment variables
DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "models"


class ModelService:
    """
    Singleton service for model loading and inference.
    
    Loads the model and transformer once at startup and provides
    thread-safe inference capabilities.
    """
    
    _instance: Optional['ModelService'] = None
    
    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.transformer = None
        self.metadata = None
        self._is_loaded = False
    
    @classmethod
    def get_instance(cls, models_dir: Path = DEFAULT_MODELS_DIR) -> 'ModelService':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(models_dir)
        return cls._instance
    
    def load(self) -> bool:
        """
        Load model artifacts from disk.
        
        Returns:
            True if loading succeeded, False otherwise
        """
        if self._is_loaded:
            logger.info("Model already loaded, skipping reload")
            return True
        
        try:
            # Add scripts directory to path for FeatureTransformer class
            import sys
            scripts_dir = str(self.models_dir.parent / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            
            # Load model
            model_path = self.models_dir / "champion_model.pkl"
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            # Load transformer
            transformer_path = self.models_dir / "feature_transformer.pkl"
            logger.info(f"Loading transformer from {transformer_path}")
            self.transformer = joblib.load(transformer_path)
            
            # Load metadata
            metadata_path = self.models_dir / "model_metadata.json"
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully: {self.metadata['model_type']} v{self.metadata['version']}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, features: dict) -> dict:
        """
        Make a price prediction for a property.
        
        Args:
            features: Dictionary containing property features
            
        Returns:
            Dictionary with prediction, confidence interval, and metadata
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Transform features
        X = self.transformer.transform(features)
        
        # Get prediction
        prediction = float(self.model.predict(X)[0])
        
        # Calculate confidence interval
        # For RandomForest, we can use the individual tree predictions
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([
                tree.predict(X)[0] for tree in self.model.estimators_
            ])
            confidence_low = float(np.percentile(tree_predictions, 10))
            confidence_high = float(np.percentile(tree_predictions, 90))
        else:
            # Fallback: use RMSE-based interval
            rmse = self.metadata['metrics']['rmse']
            confidence_low = prediction - rmse
            confidence_high = prediction + rmse
        
        # Ensure non-negative prices
        prediction = max(0, prediction)
        confidence_low = max(0, confidence_low)
        confidence_high = max(0, confidence_high)
        
        return {
            "predicted_price": round(prediction, 2),
            "confidence_low": round(confidence_low, 2),
            "confidence_high": round(confidence_high, 2),
            "model_version": self.metadata['version']
        }
    
    def get_health_info(self) -> dict:
        """Get health check information."""
        return {
            "model_loaded": self._is_loaded,
            "model_type": self.metadata.get('model_type', 'unknown') if self.metadata else 'not loaded',
            "model_version": self.metadata.get('version', 'unknown') if self.metadata else 'not loaded',
            "metrics": self.metadata.get('metrics', {}) if self.metadata else {}
        }
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded


# Global instance for FastAPI dependency injection
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """
    Dependency injection function for FastAPI.
    
    Returns the singleton ModelService instance.
    """
    global _model_service
    if _model_service is None:
        _model_service = ModelService.get_instance()
        _model_service.load()
    return _model_service
