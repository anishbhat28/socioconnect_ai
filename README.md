# SocioConnect AI: Immigrant Community Growth Prediction

This project predicts where immigrant communities will form/grow in the next 1-3 years and then match those places with nearby immigrant-owned or culturally relevant businesses.

## Overview

This system uses official US government data sources to:
1. Predict immigrant community growth at the census tract level
2. Match predicted hotspots with culturally relevant businesses
3. Recommend businesses to families based on their origin group and location

## Data Sources

- **ACS 1-year & 5-year**: Foreign-born demographics, housing, income
- **LEHD LODES**: Employment and commuting patterns
- **HUD**: Fair Market Rents and housing affordability
- **IRS Migration**: County-to-county migration flows
- **ORR**: Refugee admissions and placement data
- **USCIS/DHS**: Naturalizations, visas, asylum grants
- **CBP**: Business establishment by industry
- **SBA DSBS**: Small business directory with firm names and locations

  These data sources are in the public domain.


## Key Features

- Spatial-temporal modeling with census tract-level predictions
- Multi-origin support (India, Mexico, China, etc.)
- Business matching using official SBA data
- Fairness-aware modeling with bias audits
- API endpoints for real-time predictions and recommendations

## Quick Start Guide 

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
