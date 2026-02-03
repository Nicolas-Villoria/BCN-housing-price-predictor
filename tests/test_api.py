"""
API Tests for Barcelona Rental Price Predictor

These tests validate the FastAPI endpoints work correctly.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_has_required_fields(self, client):
        """Health response should have all required fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "api_version" in data
    
    def test_health_status_values(self, client):
        """Health status should be 'healthy' or 'degraded'."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded"]


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_200(self, client):
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_has_docs_link(self, client):
        """Root response should include docs link."""
        response = client.get("/")
        data = response.json()
        
        assert "docs" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    @pytest.fixture
    def valid_property(self):
        """A valid property input for testing."""
        return {
            "size": 80,
            "rooms": 2,
            "bathrooms": 1,
            "neighborhood": "el Gòtic",
            "propertyType": "flat",
            "district": "Ciutat Vella",
            "avg_income_index": 15000,
            "density_val": 200,
            "has_lift": True,
            "has_parking": False,
            "has_ac": True,
            "floor": 3
        }
    
    def test_predict_returns_200(self, client, valid_property):
        """Predict endpoint should return 200 for valid input."""
        response = client.post("/predict", json=valid_property)
        # Could be 200 or 503 if model not loaded
        assert response.status_code in [200, 503]
    
    def test_predict_response_structure(self, client, valid_property):
        """Predict response should have expected structure."""
        response = client.post("/predict", json=valid_property)
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_price" in data
            assert "currency" in data
            assert data["currency"] == "EUR"
    
    def test_predict_rejects_invalid_input(self, client):
        """Predict should reject invalid input."""
        invalid_input = {"size": "not a number"}
        response = client.post("/predict", json=invalid_input)
        
        # Should return 422 Unprocessable Entity
        assert response.status_code == 422
    
    def test_predict_rejects_negative_size(self, client, valid_property):
        """Predict should reject negative size."""
        valid_property["size"] = -10
        response = client.post("/predict", json=valid_property)
        
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""
    
    def test_model_info_endpoint_exists(self, client):
        """Model info endpoint should exist."""
        response = client.get("/model/info")
        # 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]


class TestOpenAPI:
    """Tests for API documentation."""
    
    def test_openapi_available(self, client):
        """OpenAPI schema should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
    
    def test_docs_available(self, client):
        """Swagger docs should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
