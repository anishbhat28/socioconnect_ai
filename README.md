# SocioConnect AI: Immigrant Community Growth Prediction

Predict where immigrant communities will form and grow in the next 1-3 years, then match those places with nearby immigrant-owned or culturally relevant businesses.

## Overview

This system uses official US government data sources to:
1. **Predict** immigrant community growth at the census tract level
2. **Match** predicted hotspots with culturally relevant businesses
3. **Recommend** businesses to families based on their origin group and location

## Data Sources

- **ACS 1-year & 5-year**: Foreign-born demographics, housing, income
- **LEHD LODES**: Employment and commuting patterns
- **HUD**: Fair Market Rents and housing affordability
- **IRS Migration**: County-to-county migration flows
- **ORR**: Refugee admissions and placement data
- **USCIS/DHS**: Naturalizations, visas, asylum grants
- **CBP**: Business establishment counts by industry
- **SBA DSBS**: Small business directory with firm names and locations

## Key Features

- **Spatial-temporal modeling** with census tract-level predictions
- **Multi-origin support** (India, Mexico, China, etc.)
- **Business matching** using official SBA data
- **Fairness-aware** modeling with bias audits
- **API endpoints** for real-time predictions and recommendations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run ETL pipeline
python -m etl.acs
python -m etl.features

# Train models
python -m modeling.train_reg
python -m modeling.train_cls

# Start API server
python -m api.app
```

## Project Structure

```
socioconnect_ai/
├── etl/                 # Data extraction and transformation
├── modeling/            # Machine learning models
├── business/            # Business matching and tagging
├── api/                 # FastAPI endpoints
├── data/                # Raw and processed data
└── docs/                # Documentation and model cards
```

## Ethics & Privacy

- **Aggregation only**: No individual-level inference
- **Bias monitoring**: Regular audits across origin groups and income levels
- **Transparency**: Model cards and explainability features
- **Opt-in**: User consent for personalized recommendations
