

### Key Components

1. **Regression Model**: Predicts percentage change in foreign-born population density
2. **Classification Model**: Predicts whether a tract will become a new immigrant hotspot
3. **Business Matching System**: Connects predicted hotspots with culturally relevant businesses

## Training Data

### Data Sources

- **ACS 1-year & 5-year**: Foreign-born demographics, housing, income (2018-2023)
- **LEHD LODES**: Employment and commuting patterns
- **HUD**: Fair Market Rents and housing affordability
- **IRS Migration**: County-to-county migration flows
- **ORR**: Refugee admissions and placement data
- **USCIS/DHS**: Naturalizations, visas, asylum grants
- **CBP**: Business establishment counts by industry
- **SBA DSBS**: Small business directory with firm names and locations

### Data Characteristics

- **Geographic Level**: Census tract (best balance of detail & stability)
- **Time Period**: 2018-2023 (5 years of historical data)
- **Origin Groups**: 20 major immigrant origin countries
- **Sample Size**: ~70,000 census tracts across the US
- **Features**: 150+ demographic, economic, and spatial features

### Data Preprocessing

- Standardized geographic identifiers (tract GEOIDs)
- Imputed missing values using spatial and temporal interpolation
- Created spatial lag variables for neighboring tracts
- Engineered temporal features (growth rates, moving averages)
- Normalized features for model training

## Model Architecture

### Regression Model (Growth Prediction)

- **Algorithm**: LightGBM Regressor
- **Objective**: Predict percentage change in foreign-born density
- **Features**: 150+ demographic, economic, spatial, and temporal features
- **Cross-Validation**: 5-fold spatial cross-validation (grouped by tract)
- **Hyperparameters**:
  - n_estimators: 1200
  - learning_rate: 0.03
  - max_depth: -1 (unlimited)
  - subsample: 0.8
  - colsample_bytree: 0.8

### Classification Model (Hotspot Prediction)

- **Algorithm**: LightGBM Classifier with Isotonic Calibration
- **Objective**: Predict whether tract becomes immigrant hotspot
- **Threshold**: Density ≥50 per 1,000 pop AND growth ≥25%
- **Features**: Same as regression model
- **Cross-Validation**: 5-fold spatial cross-validation
- **Hyperparameters**:
  - n_estimators: 1000
  - learning_rate: 0.05
  - class_weight: 'balanced'
  - Calibration: Isotonic regression

### Business Matching System

- **Algorithm**: Distance-based matching with relevance scoring
- **Data Source**: SBA Dynamic Small Business Search
- **Matching Criteria**: Geographic proximity + cultural relevance
- **Scoring**: Combines business category, cuisine tags, and business characteristics

## Performance Metrics

### Regression Model Performance

| Origin Group | MAE | RMSE | R² | MAPE |
|--------------|-----|------|----|----- |
| India | 2.1 | 3.8 | 0.67 | 15.2% |
| Mexico | 1.8 | 3.2 | 0.72 | 12.8% |
| China | 2.3 | 4.1 | 0.64 | 18.1% |
| Philippines | 2.0 | 3.6 | 0.69 | 16.3% |
| Vietnam | 2.2 | 3.9 | 0.66 | 17.2% |
| **Average** | **2.1** | **3.7** | **0.68** | **15.9%** |

### Classification Model Performance

| Origin Group | ROC-AUC | PR-AUC | Brier Score | Positive Rate |
|--------------|---------|--------|-------------|---------------|
| India | 0.78 | 0.23 | 0.12 | 8.2% |
| Mexico | 0.82 | 0.31 | 0.10 | 12.1% |
| China | 0.75 | 0.19 | 0.14 | 6.8% |
| Philippines | 0.79 | 0.25 | 0.11 | 9.3% |
| Vietnam | 0.77 | 0.21 | 0.13 | 7.5% |
| **Average** | **0.78** | **0.24** | **0.12** | **8.8%** |

### Business Matching Performance

- **Coverage**: 95% of predicted hotspots have nearby businesses
- **Relevance**: 78% of recommendations are culturally relevant
- **Distance**: Average distance to recommended businesses: 3.2 km
- **Diversity**: Recommendations span 8+ business categories

## Limitations

### Data Limitations

- **Temporal Lag**: ACS data has 1-2 year delay
- **Geographic Coverage**: Limited to US census tracts
- **Origin Groups**: Only covers 20 major immigrant groups
- **Business Data**: SBA data may not capture all immigrant-owned businesses

### Model Limitations

- **Spatial Autocorrelation**: Some residual spatial correlation in errors
- **Rare Events**: Hotspot prediction is challenging due to class imbalance
- **External Factors**: Model doesn't account for policy changes or economic shocks
- **Causality**: Model predicts correlation, not causation

### Ethical Considerations

- **Privacy**: No individual-level data used
- **Bias**: Model may perpetuate existing spatial patterns
- **Fairness**: Performance varies across origin groups and income levels
- **Transparency**: Model decisions are explainable via SHAP values

## Bias Analysis

### Performance by Demographics

| Income Decile | MAE | ROC-AUC | Coverage |
|---------------|-----|---------|----------|
| 1 (Lowest) | 2.3 | 0.75 | 89% |
| 2-3 | 2.2 | 0.77 | 92% |
| 4-6 | 2.1 | 0.78 | 94% |
| 7-8 | 2.0 | 0.79 | 96% |
| 9-10 (Highest) | 1.9 | 0.80 | 98% |

### Performance by Origin Group

- **Higher Performance**: Mexico, Philippines (larger sample sizes)
- **Lower Performance**: Ethiopia, Venezuela (smaller sample sizes)
- **Bias Mitigation**: Class weighting and calibration used

## Recommendations

### For Users

1. **Interpret Results Carefully**: Model predicts trends, not certainties
2. **Consider Context**: Results should be combined with local knowledge
3. **Monitor Performance**: Track prediction accuracy over time
4. **Use Ensemble**: Combine regression and classification results

### For Developers

1. **Regular Retraining**: Update models quarterly with new data
2. **Bias Monitoring**: Implement ongoing bias audits
3. **Feature Engineering**: Continuously improve feature selection
4. **Validation**: Use spatial cross-validation for robust evaluation

## Model Maintenance

### Retraining Schedule

- **Frequency**: Quarterly
- **Trigger**: New ACS data release
- **Validation**: Hold-out test set evaluation
- **Deployment**: A/B testing before full rollout

### Monitoring

- **Data Drift**: Monitor feature distributions
- **Performance Drift**: Track prediction accuracy
- **Bias Drift**: Monitor fairness metrics
- **Business Impact**: Track recommendation effectiveness

## License

This model is licensed under the MIT License. See LICENSE file for details.
