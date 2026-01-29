"""
API Test Script

Tests all API endpoints using httpx.
Run this while the API server is running on port 8000.
"""

import httpx
import json
import sys

BASE_URL = "http://127.0.0.1:8000"


def test_root():
    """Test root endpoint."""
    print("\n[1/4] Testing root endpoint...")
    response = httpx.get(f"{BASE_URL}/")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    assert response.status_code == 200
    print("  ✓ Passed")


def test_health():
    """Test health endpoint."""
    print("\n[2/4] Testing health endpoint...")
    response = httpx.get(f"{BASE_URL}/health")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Response: {json.dumps(data, indent=4)}")
    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    print("  ✓ Passed")


def test_model_info():
    """Test model info endpoint."""
    print("\n[3/4] Testing model info endpoint...")
    response = httpx.get(f"{BASE_URL}/model/info")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Model Type: {data['model_type']}")
    print(f"  Version: {data['version']}")
    print(f"  Metrics: RMSE=€{data['metrics']['rmse']:,.2f}, R²={data['metrics']['r2']:.4f}")
    assert response.status_code == 200
    print("  ✓ Passed")


def test_predict():
    """Test prediction endpoint."""
    print("\n[4/4] Testing prediction endpoint...")
    
    test_cases = [
        {
            "name": "Small flat in Eixample",
            "payload": {
                "size": 60.0,
                "rooms": 2,
                "bathrooms": 1,
                "neighborhood": "la Dreta de l'Eixample",
                "propertyType": "flat",
                "district": "Eixample",
                "avg_income_index": 130.0,
                "density_val": 400.0
            }
        },
        {
            "name": "Large chalet in Sarrià",
            "payload": {
                "size": 300.0,
                "rooms": 5,
                "bathrooms": 3,
                "neighborhood": "Sarrià",
                "propertyType": "chalet",
                "district": "Sarrià-Sant Gervasi",
                "avg_income_index": 180.0,
                "density_val": 50.0
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n  Testing: {case['name']}")
        response = httpx.post(f"{BASE_URL}/predict", json=case["payload"])
        print(f"    Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"    Predicted Price: €{data['predicted_price']:,.2f}")
            print(f"    Confidence: €{data['confidence_low']:,.2f} - €{data['confidence_high']:,.2f}")
            print(f"    Model Version: {data['model_version']}")
            assert data['predicted_price'] > 0
            assert data['confidence_low'] <= data['predicted_price'] <= data['confidence_high']
        else:
            print(f"    Error: {response.json()}")
            sys.exit(1)
    
    print("\n  ✓ All predictions passed")


def main():
    print("=" * 60)
    print("Barcelona Rental Price API - Test Suite")
    print("=" * 60)
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_predict()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except httpx.ConnectError:
        print("\n❌ Error: Could not connect to API server")
        print("  Make sure the server is running:")
        print("  uvicorn api.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
