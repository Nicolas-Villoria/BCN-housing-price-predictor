"""
Test inference to verify model loading and prediction works correctly.
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

class TestInference:
    """Tests for the /predict endpoint."""
    
    def test_predict_returns_200(self, client):
        """Predict endpoint should return 200 OK."""
        payload = {
        "name": "Small flat in Eixample",
        "size": 60.0,
        "rooms": 2,
        "bathrooms": 1,
        "neighborhood": "la Dreta de l'Eixample",
        "propertyType": "flat",
        "district": "Eixample",
        "avg_income_index": 130.0,
        "density_val": 400.0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    
    def test_predict_response_structure(self, client):
        """Predict response should have the correct structure."""
        payload = {
        "name": "Large chalet in Sarrià",
        "size": 300.0,
        "rooms": 5,
        "bathrooms": 3,
        "neighborhood": "Sarrià",
        "propertyType": "chalet",
        "district": "Sarrià-Sant Gervasi",
        "avg_income_index": 180.0,
        "density_val": 50.0
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], (int, float))
        assert "model_version" in data
        assert isinstance(data["model_version"], str)

    