"""
Feature Transformer for Barcelona Rental Price Model

This module contains the FeatureTransformer class used for preprocessing
input data before model inference. It must be importable for joblib to
deserialize the saved transformer.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Feature configuration (must match training)
NUMERIC_FEATURES = ["size", "rooms", "bathrooms", "avg_income_index", "density_val"]
CATEGORICAL_FEATURES = ["neighborhood", "propertyType", "district"]


class FeatureTransformer:
    """
    Custom feature transformer that handles both numeric and categorical features.
    
    This class encapsulates all preprocessing logic, making it easy to:
    - Save/load the fitted transformer
    - Apply consistent preprocessing in training and inference
    - Expose feature names for explainability
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.feature_names = []
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'FeatureTransformer':
        """Fit encoders and scaler on training data."""
        # Fit label encoders for categorical features
        for col in self.categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                # Add "Unknown" to handle unseen categories during inference
                unique_values = list(X[col].unique()) + ["Unknown"]
                le.fit(unique_values)
                self.label_encoders[col] = le
        
        # Transform categorical features to numeric
        X_transformed = self._encode_categorical(X)
        
        # Fit scaler on numeric features only
        numeric_cols = [c for c in self.numeric_features if c in X_transformed.columns]
        self.scaler.fit(X_transformed[numeric_cols])
        
        # Store feature names for explainability
        self.feature_names = numeric_cols + [f"{c}_encoded" for c in self.categorical_features if c in X.columns]
        
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted encoders and scaler."""
        if not self._is_fitted:
            raise ValueError("FeatureTransformer must be fitted before transform")
        
        # Ensure X is a DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        X = X.copy()
        
        # Encode categorical features
        X_encoded = self._encode_categorical(X)
        
        # Scale numeric features
        numeric_cols = [c for c in self.numeric_features if c in X_encoded.columns]
        X_encoded[numeric_cols] = self.scaler.transform(X_encoded[numeric_cols])
        
        # Get categorical encoded columns
        cat_cols = [f"{c}_encoded" for c in self.categorical_features if c in X.columns]
        
        # Return as numpy array in consistent order
        feature_cols = numeric_cols + cat_cols
        return X_encoded[feature_cols].values
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to categorical columns."""
        X = X.copy()
        
        for col, le in self.label_encoders.items():
            if col in X.columns:
                # Handle unseen categories by mapping to "Unknown"
                X[f"{col}_encoded"] = X[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["Unknown"])[0]
                )
        
        return X
    
    def get_feature_names(self) -> list[str]:
        """Return feature names in the order they appear in transformed output."""
        return self.feature_names
