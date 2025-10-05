# SocioConnect AI API Documentation

## Overview

The SocioConnect AI API provides endpoints for predicting immigrant community growth and recommending culturally relevant businesses. The API is built with FastAPI and provides both REST endpoints and interactive documentation.

## Endpoints

### 1. Get API Information

**GET** `/info`

Returns information about available models and origin groups.

#### Response

```json
{
  "origin_groups": [
    "india", "mexico", "china", "philippines", "vietnam", "korea",
    "cuba", "dominican_republic", "guatemala", "el_salvador",
    "honduras", "colombia", "brazil", "nigeria", "ethiopia",
    "jamaica", "haiti", "peru", "ecuador", "venezuela"
  ],
  "available_models": ["regression", "classification"],
  "last_updated": "2024-12-01T00:00:00Z",
  "model_versions": {
    "regression": "1.0",
    "classification": "1.0"
  }
}
```

### 2. Predict Hotspots

**POST** `/predict/hotspots`

Predicts immigrant community hotspots for a specific origin group.

#### Request Body

```json
{
  "origin_group": "india",
  "horizon": 2,
  "min_probability": 0.5,
  "limit": 100
}
```

#### Parameters

- `origin_group` (string, required): Origin group (e.g., "india", "mexico")
- `horizon` (integer, optional): Prediction horizon in years (1-5, default: 2)
- `min_probability` (float, optional): Minimum hotspot probability (0.0-1.0, default: 0.5)
- `limit` (integer, optional): Maximum number of results (1-1000, default: 100)

#### Response

```json
[
  {
    "tract_geoid": "06037100100",
    "latitude": 37.7749,
    "longitude": -122.4194,
    "hotspot_probability": 0.85,
    "growth_prediction": 42.3,
    "origin_group": "india",
    "horizon": 2
  }
]
```

### 3. Get Business Recommendations

**POST** `/recommend/businesses`

Get business recommendations for a user location and origin group.

#### Request Body

```json
{
  "latitude": 37.7749,
  "longitude": -122.4194,
  "origin_group": "india",
  "max_distance": 15.0,
  "max_recommendations": 10,
  "business_categories": ["food_services", "grocery_stores"]
}
```

#### Parameters

- `latitude` (float, required): User latitude (-90 to 90)
- `longitude` (float, required): User longitude (-180 to 180)
- `origin_group` (string, required): Origin group
- `max_distance` (float, optional): Maximum distance in kilometers (0.1-100.0, default: 15.0)
- `max_recommendations` (integer, optional): Maximum recommendations (1-50, default: 10)
- `business_categories` (array, optional): Preferred business categories

#### Response

```json
{
  "recommendations": [
    {
      "business_name": "Bombay Garden Restaurant",
      "business_category": "food_services",
      "latitude": 37.7849,
      "longitude": -122.4094,
      "distance_km": 1.2,
      "relevance_score": 0.95,
      "employees": 15,
      "business_age": 8.5,
      "cuisine_score": 0.92,
      "address": "123 Main St, San Francisco, CA",
      "phone": "(555) 123-4567"
    }
  ],
  "user_location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "origin_group": "india",
  "total_found": 1
}
```

### 4. Predict Growth

**POST** `/predict/growth`

Predicts immigrant community growth for specific tracts.

#### Request Body

```json
{
  "origin_group": "india",
  "horizon": 2,
  "tract_geoids": ["06037100100", "06037100200"]
}
```

#### Parameters

- `origin_group` (string, required): Origin group
- `horizon` (integer, optional): Prediction horizon in years (1-5, default: 2)
- `tract_geoids` (array, optional): Specific tract GEOIDs

#### Response

```json
[
  {
    "tract_geoid": "06037100100",
    "current_density": 45.2,
    "predicted_growth": 23.5,
    "confidence_interval": {
      "lower": 18.5,
      "upper": 28.5
    },
    "origin_group": "india",
    "horizon": 2
  }
]
```

### 5. Health Check

**GET** `/health`

Returns API health status.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-12-01T12:00:00Z"
}
```

### 6. Get Metrics

**GET** `/metrics`

Returns API usage metrics.

#### Response

```json
{
  "total_requests": 1250,
  "active_models": 2,
  "last_model_update": "2024-12-01T00:00:00Z",
  "uptime": "5 days, 12 hours, 30 minutes"
}
```

## Error Handling

### Error Response Format

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2024-12-01T12:00:00Z"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Endpoint not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Example Error Response

```json
{
  "error": "Validation Error",
  "detail": "origin_group must be one of: india, mexico, china, ...",
  "timestamp": "2024-12-01T12:00:00Z"
}
```

## SDKs and Examples

### Python SDK

```python
import requests

# Predict hotspots
response = requests.post(
    "https://api.socioconnect.ai/v1/predict/hotspots",
    json={
        "origin_group": "india",
        "horizon": 2,
        "min_probability": 0.5,
        "limit": 100
    }
)

hotspots = response.json()
print(f"Found {len(hotspots)} hotspots")
```

### JavaScript SDK

```javascript
// Predict hotspots
const response = await fetch('https://api.socioconnect.ai/v1/predict/hotspots', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    origin_group: 'india',
    horizon: 2,
    min_probability: 0.5,
    limit: 100
  })
});

const hotspots = await response.json();
console.log(`Found ${hotspots.length} hotspots`);
```

### cURL Examples

```bash
# Predict hotspots
curl -X POST "https://api.socioconnect.ai/v1/predict/hotspots" \
  -H "Content-Type: application/json" \
  -d '{
    "origin_group": "india",
    "horizon": 2,
    "min_probability": 0.5,
    "limit": 100
  }'

# Get business recommendations
curl -X POST "https://api.socioconnect.ai/v1/recommend/businesses" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 37.7749,
    "longitude": -122.4194,
    "origin_group": "india",
    "max_distance": 15.0,
    "max_recommendations": 10
  }'
```

## Data Models

### Origin Groups

The API supports the following origin groups:

- `india`: India
- `mexico`: Mexico
- `china`: China
- `philippines`: Philippines
- `vietnam`: Vietnam
- `korea`: South Korea
- `cuba`: Cuba
- `dominican_republic`: Dominican Republic
- `guatemala`: Guatemala
- `el_salvador`: El Salvador
- `honduras`: Honduras
- `colombia`: Colombia
- `brazil`: Brazil
- `nigeria`: Nigeria
- `ethiopia`: Ethiopia
- `jamaica`: Jamaica
- `haiti`: Haiti
- `peru`: Peru
- `ecuador`: Ecuador
- `venezuela`: Venezuela

### Business Categories

- `food_services`: Restaurants and other eating places
- `grocery_stores`: Grocery stores
- `specialty_food`: Specialty food stores
- `clothing_stores`: Clothing stores
- `beauty_salons`: Personal care services
- `laundry_services`: Dry cleaning and laundry services
- `taxi_services`: Taxi and limousine service
- `construction`: Construction
- `healthcare`: Health care and social assistance
- `professional_services`: Professional, scientific, and technical services
- `accommodation`: Accommodation
- `transportation`: Transportation and warehousing
- `retail_trade`: Retail trade
- `wholesale_trade`: Wholesale trade
- `manufacturing`: Manufacturing
- `agriculture`: Agriculture, forestry, fishing and hunting

