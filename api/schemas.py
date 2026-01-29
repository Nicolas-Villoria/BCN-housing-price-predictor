"""
Pydantic schemas for API request/response validation.

These schemas define the contract between the API and its clients,
ensuring type safety and automatic documentation generation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class PropertyType(str, Enum):
    """Valid property types from the training data."""
    FLAT = "flat"
    STUDIO = "studio"
    CHALET = "chalet"
    PENTHOUSE = "penthouse"
    DUPLEX = "duplex"
    COUNTRYHOUSE = "countryHouse"


class District(str, Enum):
    """Barcelona districts."""
    CIUTAT_VELLA = "Ciutat Vella"
    EIXAMPLE = "Eixample"
    SANTS_MONTJUIC = "Sants-Montjuïc"
    LES_CORTS = "Les Corts"
    SARRIA_SANT_GERVASI = "Sarrià-Sant Gervasi"
    GRACIA = "Gràcia"
    HORTA_GUINARDO = "Horta-Guinardó"
    NOU_BARRIS = "Nou Barris"
    SANT_ANDREU = "Sant Andreu"
    SANT_MARTI = "Sant Martí"


class PropertyInput(BaseModel):
    """
    Input schema for property price prediction.
    
    All fields are required for accurate prediction.
    The model uses these features plus derived socioeconomic data.
    """
    size: float = Field(
        ..., 
        gt=0, 
        le=2000,
        description="Property size in square meters",
        examples=[80.0]
    )
    rooms: int = Field(
        ..., 
        ge=0, 
        le=20,
        description="Number of rooms",
        examples=[3]
    )
    bathrooms: int = Field(
        ..., 
        ge=1, 
        le=10,
        description="Number of bathrooms",
        examples=[1]
    )
    neighborhood: str = Field(
        ...,
        min_length=1,
        description="Neighborhood name within Barcelona",
        examples=["la Dreta de l'Eixample"]
    )
    propertyType: str = Field(
        ...,
        description="Type of property",
        examples=["flat"]
    )
    district: str = Field(
        ...,
        description="District of Barcelona",
        examples=["Eixample"]
    )
    avg_income_index: float = Field(
        ...,
        gt=0,
        description="Average income index for the neighborhood (from Open Data BCN)",
        examples=[115.0]
    )
    density_val: float = Field(
        ...,
        gt=0,
        description="Population density value for the neighborhood",
        examples=[350.0]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "size": 80.0,
                    "rooms": 3,
                    "bathrooms": 1,
                    "neighborhood": "la Dreta de l'Eixample",
                    "propertyType": "flat",
                    "district": "Eixample",
                    "avg_income_index": 130.0,
                    "density_val": 400.0
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """
    Response schema for price predictions.
    
    Includes the predicted price, a confidence range, and model metadata.
    """
    predicted_price: float = Field(
        ...,
        description="Predicted rental/sale price in euros",
        examples=[350000.0]
    )
    confidence_low: float = Field(
        ...,
        description="Lower bound of prediction confidence interval (10th percentile)",
        examples=[315000.0]
    )
    confidence_high: float = Field(
        ...,
        description="Upper bound of prediction confidence interval (90th percentile)",
        examples=[385000.0]
    )
    currency: str = Field(
        default="EUR",
        description="Currency of the predicted price"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction",
        examples=["20260127_180000"]
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    model_type: str = Field(..., examples=["RandomForest"])
    model_version: str = Field(..., examples=["20260127_180000"])
    api_version: str = Field(..., examples=["1.0.0"])


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ValidationError",
                    "detail": "size must be greater than 0"
                }
            ]
        }
    }
