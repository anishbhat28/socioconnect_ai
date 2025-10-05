"""
FastAPI application for immigrant community growth prediction and business recommendations.
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime
import uvicorn

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from business.match import BusinessMatcher
from modeling.train_reg import ImmigrantGrowthRegressor
from modeling.train_cls import ImmigrantHotspotClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="SocioConnect AI API",
    description="Predict immigrant community growth and recommend culturally relevant businesses",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


business_matcher = BusinessMatcher()
regressor = ImmigrantGrowthRegressor()
classifier = ImmigrantHotspotClassifier()

# Pydantic models
class HotspotPredictionRequest(BaseModel):
    origin_group: str = Field(..., description="Origin group (e.g., 'india', 'mexico')")
    horizon: int = Field(2, description="Prediction horizon in years", ge=1, le=5)
    min_probability: float = Field(0.5, description="Minimum hotspot probability", ge=0.0, le=1.0)
    limit: int = Field(100, description="Maximum number of results", ge=1, le=1000)

class HotspotPredictionResponse(BaseModel):
    tract_geoid: str
    latitude: float
    longitude: float
    hotspot_probability: float
    growth_prediction: float
    origin_group: str
    horizon: int

class BusinessRecommendationRequest(BaseModel):
    latitude: float = Field(..., description="User latitude", ge=-90, le=90)
    longitude: float = Field(..., description="User longitude", ge=-180, le=180)
    origin_group: str = Field(..., description="Origin group")
    max_distance: float = Field(15.0, description="Maximum distance in kilometers", ge=0.1, le=100.0)
    max_recommendations: int = Field(10, description="Maximum recommendations", ge=1, le=50)
    business_categories: Optional[List[str]] = Field(None, description="Preferred business categories")

class BusinessRecommendation(BaseModel):
    business_name: str
    business_category: str
    latitude: float
    longitude: float
    distance_km: float
    relevance_score: float
    employees: int
    business_age: float
    cuisine_score: Optional[float] = None
    address: Optional[str] = None
    phone: Optional[str] = None

class BusinessRecommendationResponse(BaseModel):
    recommendations: List[BusinessRecommendation]
    user_location: Dict[str, float]
    origin_group: str
    total_found: int

class GrowthPredictionRequest(BaseModel):
    origin_group: str = Field(..., description="Origin group")
    horizon: int = Field(2, description="Prediction horizon in years", ge=1, le=5)
    tract_geoids: Optional[List[str]] = Field(None, description="Specific tract GEOIDs")

class GrowthPredictionResponse(BaseModel):
    tract_geoid: str
    current_density: float
    predicted_growth: float
    confidence_interval: Dict[str, float]
    origin_group: str
    horizon: int

class ModelInfo(BaseModel):
    origin_groups: List[str]
    available_models: List[str]
    last_updated: str
    model_versions: Dict[str, str]

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SocioConnect AI API",
        "version": "1.0.0",
        "description": "Predict immigrant community growth and recommend culturally relevant businesses",
        "docs": "/docs"
    }

@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about available models and origin groups."""
    return ModelInfo(
        origin_groups=[
            'india', 'mexico', 'china', 'philippines', 'vietnam', 'korea',
            'cuba', 'dominican_republic', 'guatemala', 'el_salvador',
            'honduras', 'colombia', 'brazil', 'nigeria', 'ethiopia',
            'jamaica', 'haiti', 'peru', 'ecuador', 'venezuela'
        ],
        available_models=["regression", "classification"],
        last_updated=datetime.now().isoformat(),
        model_versions={"regression": "1.0", "classification": "1.0"}
    )

@app.post("/predict/hotspots", response_model=List[HotspotPredictionResponse])
async def predict_hotspots(request: HotspotPredictionRequest):
    """
    Predict immigrant community hotspots for a specific origin group.
    """
    try:
        logger.info(f"Predicting hotspots for {request.origin_group}")
        
        
        hotspots_df = business_matcher.load_hotspot_predictions(request.origin_group)
        
        
        prob_col = f'{request.origin_group}_hotspot_prob'
        high_prob_hotspots = hotspots_df[hotspots_df[prob_col] >= request.min_probability]
        
        
        high_prob_hotspots = high_prob_hotspots.sort_values(prob_col, ascending=False)
        high_prob_hotspots = high_prob_hotspots.head(request.limit)
        
        # Convert to response format
        predictions = []
        for _, row in high_prob_hotspots.iterrows():
            prediction = HotspotPredictionResponse(
                tract_geoid=row['tract_geoid'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                hotspot_probability=row[prob_col],
                growth_prediction=row[prob_col] * 100,  # Mock growth prediction
                origin_group=request.origin_group,
                horizon=request.horizon
            )
            predictions.append(prediction)
        
        logger.info(f"Returning {len(predictions)} hotspot predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"Error predicting hotspots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/businesses", response_model=BusinessRecommendationResponse)
async def recommend_businesses(request: BusinessRecommendationRequest):
    """
    Get business recommendations for a user location and origin group.
    """
    try:
        logger.info(f"Getting business recommendations for {request.origin_group} at "
                   f"({request.latitude}, {request.longitude})")
        
       
        recommendations_df = business_matcher.get_recommendations(
            user_lat=request.latitude,
            user_lon=request.longitude,
            origin_group=request.origin_group,
            max_distance=request.max_distance,
            max_recommendations=request.max_recommendations
        )
        
        if recommendations_df.empty:
            return BusinessRecommendationResponse(
                recommendations=[],
                user_location={"latitude": request.latitude, "longitude": request.longitude},
                origin_group=request.origin_group,
                total_found=0
            )
        
        
        if request.business_categories:
            recommendations_df = recommendations_df[
                recommendations_df['business_category'].isin(request.business_categories)
            ]
        
        
        recommendations = []
        for _, row in recommendations_df.iterrows():
            recommendation = BusinessRecommendation(
                business_name=row['business_name'],
                business_category=row['business_category'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                distance_km=row['distance_km'],
                relevance_score=row['relevance_score'],
                employees=int(row['employees']) if pd.notna(row['employees']) else 0,
                business_age=float(row['business_age']) if pd.notna(row['business_age']) else 0.0,
                cuisine_score=float(row.get(f'{request.origin_group}_score', 0)) if f'{request.origin_group}_score' in row else None,
                address=row.get('address'),
                phone=row.get('phone')
            )
            recommendations.append(recommendation)
        
        logger.info(f"Returning {len(recommendations)} business recommendations")
        return BusinessRecommendationResponse(
            recommendations=recommendations,
            user_location={"latitude": request.latitude, "longitude": request.longitude},
            origin_group=request.origin_group,
            total_found=len(recommendations)
        )
        
    except Exception as e:
        logger.error(f"Error getting business recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/growth", response_model=List[GrowthPredictionResponse])
async def predict_growth(request: GrowthPredictionRequest):
    """
    Predict immigrant community growth for specific tracts.
    """
    try:
        logger.info(f"Predicting growth for {request.origin_group}")
        
       
        features_df = regressor.load_features()
        
       
        if request.tract_geoids:
            features_df = features_df[features_df['tract_geoid'].isin(request.tract_geoids)]

        predictions = []
        for _, row in features_df.iterrows():
            # Mock current density
            current_density = row.get(f'{request.origin_group}_density', 0)
            
            # Mock growth prediction
            predicted_growth = np.random.uniform(-10, 50)  # Mock growth percentage
            
            prediction = GrowthPredictionResponse(
                tract_geoid=row['tract_geoid'],
                current_density=current_density,
                predicted_growth=predicted_growth,
                confidence_interval={
                    "lower": predicted_growth - 5,
                    "upper": predicted_growth + 5
                },
                origin_group=request.origin_group,
                horizon=request.horizon
            )
            predictions.append(prediction)
        
        logger.info(f"Returning {len(predictions)} growth predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"Error predicting growth: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def get_metrics():
    """Get API usage metrics."""
    return {
        "total_requests": 0,  # Would track in production
        "active_models": 2,
        "last_model_update": datetime.now().isoformat(),
        "uptime": "0 days, 0 hours, 0 minutes"  # Would calculate in production
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
