"""
Barcelona Rental Price Prediction API

Production-ready FastAPI service for real-time property price predictions.

Features:
- /predict: Single property price prediction
- /health: Health check and model status
- /docs: Auto-generated API documentation

Usage:
    uvicorn api.main:app --reload --port 8000
    
Or run directly:
    python -m api.main
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from api import __version__
from api.schemas import (
    PropertyInput,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)
from api.model_loader import ModelService, get_model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# APPLICATION LIFECYCLE
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Loads model at startup and cleans up at shutdown.
    """
    # Startup
    logger.info("Starting Barcelona Rental Price API...")
    
    model_service = ModelService.get_instance()
    if model_service.load():
        logger.info("Model loaded successfully at startup")
    else:
        logger.error("Failed to load model at startup")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# ==============================================================================
# APPLICATION SETUP
# ==============================================================================

app = FastAPI(
    title="Barcelona Rental Price API",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# EXCEPTION HANDLERS
# ==============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={"error": "ValidationError", "detail": str(exc)}
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    """Handle runtime errors (e.g., model not loaded)."""
    return JSONResponse(
        status_code=503,
        content={"error": "ServiceUnavailable", "detail": str(exc)}
    )


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "Barcelona Rental Price API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Check API and model health status"
)
async def health_check(
    model_service: ModelService = Depends(get_model_service)
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the current status of the API and loaded model.
    Useful for container orchestration health probes.
    """
    health_info = model_service.get_health_info()
    
    status = "healthy" if health_info["model_loaded"] else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=health_info["model_loaded"],
        model_type=health_info["model_type"],
        model_version=health_info["model_version"],
        api_version=__version__
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    },
    tags=["Prediction"],
    summary="Predict property price",
    description="Get a price prediction for a Barcelona property based on its features"
)
async def predict_price(
    property_input: PropertyInput,
    model_service: ModelService = Depends(get_model_service)
) -> PredictionResponse:
    """
    Predict the price of a property in Barcelona.
    
    Takes property features as input and returns:
    - **predicted_price**: The model's best estimate
    - **confidence_low**: 10th percentile from ensemble
    - **confidence_high**: 90th percentile from ensemble
    
    The confidence interval comes from the variance across Random Forest trees,
    providing a measure of prediction uncertainty.
    """
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Convert Pydantic model to dict for inference
        features = property_input.model_dump()
        
        # Get prediction
        result = model_service.predict(features)
        
        logger.info(
            f"Prediction: €{result['predicted_price']:,.2f} "
            f"[{result['confidence_low']:,.2f} - {result['confidence_high']:,.2f}] "
            f"for {features['neighborhood']}, {features['size']}m²"
        )
        
        return PredictionResponse(
            predicted_price=result["predicted_price"],
            confidence_low=result["confidence_low"],
            confidence_high=result["confidence_high"],
            currency="EUR",
            model_version=result["model_version"]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get(
    "/model/info",
    tags=["Model"],
    summary="Get model information",
    description="Get detailed information about the loaded model"
)
async def model_info(
    model_service: ModelService = Depends(get_model_service)
):
    """
    Get detailed model information.
    
    Returns model type, version, performance metrics, and feature names.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_service.metadata.get("model_type"),
        "version": model_service.metadata.get("version"),
        "training_date": model_service.metadata.get("training_date"),
        "metrics": model_service.metadata.get("metrics"),
        "features": {
            "numeric": model_service.metadata.get("numeric_features"),
            "categorical": model_service.metadata.get("categorical_features")
        },
        "threshold_passed": model_service.metadata.get("threshold_passed")
    }


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
