"""
API Client for Barcelona Rental Price Prediction API.

This module provides a client to communicate with the deployed FastAPI service,
replacing the need for local Spark inference.
"""
import os
import httpx
import streamlit as st
from typing import Optional

# Configuration - can be overridden via environment variable
DEFAULT_API_URL = "https://bcn-housing-price-predictor.onrender.com"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_URL)

# Timeout settings (Render free tier can be slow to wake up)
TIMEOUT_SECONDS = 60.0


class PredictionResponse:
    """Structured response from the prediction API."""
    def __init__(self, data: dict):
        self.predicted_price: float = data.get("predicted_price", 0.0)
        self.confidence_low: float = data.get("confidence_low", 0.0)
        self.confidence_high: float = data.get("confidence_high", 0.0)
        self.currency: str = data.get("currency", "EUR")
        self.model_version: str = data.get("model_version", "unknown")
    
    @property
    def is_valid(self) -> bool:
        return self.predicted_price > 0


class APIClient:
    """Client for the Barcelona Rental Price Prediction API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=TIMEOUT_SECONDS)
    
    def health_check(self) -> dict:
        """Check if the API is healthy and model is loaded."""
        try:
            response = self._client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def predict(self, features: dict) -> Optional[PredictionResponse]:
        """
        Get a price prediction from the API.
        
        Args:
            features: Dictionary containing property features:
                - size (float): Property size in m²
                - rooms (int): Number of bedrooms
                - bathrooms (int): Number of bathrooms
                - district (str): Barcelona district
                - neighborhood (str): Neighborhood name
                - propertyType (str): Type of property
                - has_parking (bool): Has parking
                - has_elevator (bool): Has elevator
                - has_ac (bool): Has air conditioning
                - floor_category (str): Floor category (ground/low/mid/high/penthouse)
                - avg_income_index (float): Income index for neighborhood
                - density_val (float): Population density value
        
        Returns:
            PredictionResponse object or None if request failed
        """
        try:
            # Build request payload - map Streamlit form fields to API schema
            payload = {
                "size": float(features.get("size", 0)),
                "rooms": int(features.get("rooms", 0)),
                "bathrooms": int(features.get("bathrooms", 0)),
                "district": str(features.get("district", "")),
                "neighborhood": str(features.get("neighborhood", "")),
                "propertyType": str(features.get("propertyType", "flat")),
                "has_parking": bool(features.get("has_parking", False)),
                "has_elevator": bool(features.get("has_lift", True)),  # Form uses 'has_lift'
                "has_ac": bool(features.get("has_ac", False)),
                "floor_category": self._get_floor_category(features.get("floor", 1)),
                "avg_income_index": float(features.get("avg_income_index", 100.0)),
                "density_val": float(features.get("density_val", 20000.0)),
            }
            
            response = self._client.post(
                f"{self.base_url}/predict",
                json=payload
            )
            response.raise_for_status()
            return PredictionResponse(response.json())
            
        except httpx.TimeoutException:
            st.error("Request timed out. The API might be waking up - please try again in 30 seconds.")
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                # Validation error - show details
                detail = e.response.json().get("detail", [])
                error_msgs = [f"{err['loc'][-1]}: {err['msg']}" for err in detail]
                st.error(f"Validation error: {', '.join(error_msgs)}")
            else:
                st.error(f"API error: {e.response.status_code}")
            return None
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            return None
    
    def get_model_info(self) -> dict:
        """Get information about the deployed model."""
        try:
            response = self._client.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}
    
    @staticmethod
    def _get_floor_category(floor: int) -> str:
        """Convert numeric floor to category."""
        if floor == 0:
            return "ground"
        elif floor <= 2:
            return "low"
        elif floor <= 5:
            return "mid"
        elif floor <= 10:
            return "high"
        else:
            return "penthouse"
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()


# Cached singleton client for Streamlit
@st.cache_resource
def get_api_client() -> APIClient:
    """Get or create the API client (cached for session)."""
    return APIClient()


def predict_price_via_api(features: dict) -> Optional[float]:
    """
    Convenience function to get a price prediction.
    
    Args:
        features: Property features dictionary from the form
        
    Returns:
        Predicted price as float, or None if failed
    """
    client = get_api_client()
    response = client.predict(features)
    if response and response.is_valid:
        return response.predicted_price
    return None


def check_api_health() -> bool:
    """Check if the API is healthy."""
    client = get_api_client()
    health = client.health_check()
    return health.get("status") == "healthy" and health.get("model_loaded", False)
