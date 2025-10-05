"""
Regression model training for immigrant community growth prediction.
Predicts percentage change in foreign-born population by origin group.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import lightgbm as lgb
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImmigrantGrowthRegressor:
    """Regression model for predicting immigrant community growth."""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Origin groups for analysis
        self.origin_groups = [
            'india', 'mexico', 'china', 'philippines', 'vietnam', 'korea',
            'cuba', 'dominican_republic', 'guatemala', 'el_salvador',
            'honduras', 'colombia', 'brazil', 'nigeria', 'ethiopia',
            'jamaica', 'haiti', 'peru', 'ecuador', 'venezuela'
        ]
        
        # Model parameters
        self.model_params = {
            'n_estimators': 1200,
            'learning_rate': 0.03,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'huber',
            'alpha': 0.85,
            'metric': 'mae',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
            'num_leaves': 64,
            'min_child_samples': 40,
            'reg_alpha': 0.1,
            'reg_lambda': 0.2
        }
        
        # Feature categories
        self.feature_categories = {
            'demographic': [
                'total_pop', 'foreign_born_density', 'median_age',
                'median_household_income', 'college_educated_rate'
            ],
            'housing': [
                'median_rent', 'rent_income_ratio', 'homeownership_rate',
                'total_housing_units', 'occupied_housing_units'
            ],
            'origin_specific': [f'{origin}_density' for origin in self.origin_groups],
            'language': [
                'spanish_at_home', 'hindi_at_home', 'chinese_at_home',
                'korean_at_home', 'vietnamese_at_home', 'arabic_at_home',
                'language_diversity'
            ],
            'business': [
                'food_services_est', 'grocery_stores_est', 'specialty_food_est',
                'clothing_stores_est', 'beauty_salons_est', 'laundry_services_est'
            ],
            'temporal': [
                'foreign_born_growth_1yr', 'foreign_born_growth_3yr',
                'pop_growth_1yr', 'pop_growth_3yr', 'income_growth_1yr', 'rent_growth_1yr'
            ],
            'spatial': [
                'foreign_born_density_lag_queen', 'foreign_born_density_lag_rook',
                'median_household_income_lag_queen', 'median_rent_lag_queen'
            ],
            'interactions': [
                'rent_income_interaction', 'pop_housing_ratio'
            ],
            'policy': [
                'covid_period', 'post_covid_period', 'decade_2010s', 'decade_2020s'
            ]
        }
    
    def load_features(self) -> pd.DataFrame:
        """Load engineered features."""
        filepath = self.data_dir / "features.parquet"
        if not filepath.exists():
            raise FileNotFoundError(f"Features file not found: {filepath}")
        
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
        """
        Create regression labels (growth rates) for specified horizon.
        
        Args:
            df: Features DataFrame
            horizon: Prediction horizon in years
            
        Returns:
            DataFrame with labels added
        """
        logger.info(f"Creating regression labels for {horizon}-year horizon")
        
        df_labeled = df.copy()
        
        # Sort by tract and year
        df_labeled = df_labeled.sort_values(['tract_geoid', 'year'])
        
        # Create growth rate labels for each origin group
        for origin in self.origin_groups:
            density_col = f'{origin}_density'
            if density_col in df_labeled.columns:
                # Calculate future growth rate
                future_density = df_labeled.groupby('tract_geoid')[density_col].shift(-horizon)
                current_density = df_labeled[density_col]
                
                # Calculate percentage change
                growth_rate = (future_density - current_density) / (current_density + 1e-8) * 100
                df_labeled[f'{origin}_growth_{horizon}yr'] = growth_rate
        
        # Overall foreign-born growth rate
        if 'foreign_born_density' in df_labeled.columns:
            future_density = df_labeled.groupby('tract_geoid')['foreign_born_density'].shift(-horizon)
            current_density = df_labeled['foreign_born_density']
            growth_rate = (future_density - current_density) / (current_density + 1e-8) * 100
            df_labeled[f'foreign_born_growth_{horizon}yr'] = growth_rate
        
        # Remove rows where we don't have future data
        df_labeled = df_labeled.dropna(subset=[f'foreign_born_growth_{horizon}yr'])
        
        logger.info(f"Created labels for {len(df_labeled)} records")
        return df_labeled
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names from DataFrame."""
        feature_cols = []
        for category, features in self.feature_categories.items():
            for feature in features:
                if feature in df.columns:
                    feature_cols.append(feature)
        
        # Add origin-specific features
        for origin in self.origin_groups:
            for suffix in ['_density', '_growth_1yr', '_growth_3yr', '_growth_5yr', 
                          '_density_ma3', '_density_ma5', '_income_interaction', 
                          '_rent_interaction', '_education_interaction']:
                col = f'{origin}{suffix}'
                if col in df.columns:
                    feature_cols.append(col)
        
        # Add year dummies
        year_cols = [col for col in df.columns if col.startswith('year_')]
        feature_cols.extend(year_cols)
        
        # Add spatial lag features
        spatial_lag_cols = [col for col in df.columns if '_lag_' in col]
        feature_cols.extend(spatial_lag_cols)
        
        return feature_cols
    
    def train_model(self, df: pd.DataFrame, origin_group: str, horizon: int = 2) -> Dict:
        """
        Train regression model for specific origin group.
        
        Args:
            df: Labeled features DataFrame
            origin_group: Origin group to predict
            horizon: Prediction horizon in years
            
        Returns:
            Dictionary with model and evaluation metrics
        """
        logger.info(f"Training regression model for {origin_group} ({horizon}-year horizon)")
        
        # Get target variable
        target_col = f'{origin_group}_growth_{horizon}yr'
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df)
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        # Clip extreme growth rates to reduce target outlier impact
        y = y.clip(lower=np.percentile(y, 1), upper=np.percentile(y, 99))
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        # Group by tract for spatial cross-validation
        groups = df['tract_geoid']
        
        # Initialize model
        model = LGBMRegressor(**self.model_params)
        
        # Cross-validation
        gkf = GroupKFold(n_splits=5)
        cv_scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            logger.info(f"Training fold {fold + 1}/5")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
            )
            
            # Predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            cv_scores.append({
                'fold': fold + 1,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            })
            
            models.append(model)
        
        # Train final model on all data
        final_model = LGBMRegressor(**self.model_params)
        final_model.fit(X, y)
        
        # Calculate average CV scores
        avg_scores = {
            'mae': np.mean([score['mae'] for score in cv_scores]),
            'rmse': np.mean([score['rmse'] for score in cv_scores]),
            'r2': np.mean([score['r2'] for score in cv_scores])
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # SHAP values for explainability
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X.sample(min(1000, len(X))))
        
        results = {
            'model': final_model,
            'feature_columns': feature_cols,
            'cv_scores': cv_scores,
            'avg_scores': avg_scores,
            'feature_importance': feature_importance,
            'shap_values': shap_values,
            'shap_explainer': explainer
        }
        
        logger.info(f"Model training complete. MAE: {avg_scores['mae']:.3f}, R²: {avg_scores['r2']:.3f}")
        return results
    
    def train_all_models(self, horizon: int = 2) -> Dict:
        """Train models for all origin groups."""
        logger.info(f"Training regression models for all origin groups ({horizon}-year horizon)")
        
        # Load features
        df = self.load_features()
        
        # Create labels
        df_labeled = self.create_labels(df, horizon)
        
        # Train models for each origin group
        models = {}
        for origin in self.origin_groups:
            try:
                results = self.train_model(df_labeled, origin, horizon)
                models[origin] = results
                
                # Save model
                model_path = self.model_dir / f"regression_{origin}_{horizon}yr.pkl"
                joblib.dump(results, model_path)
                
            except Exception as e:
                logger.error(f"Error training model for {origin}: {e}")
        
        # Save combined results
        combined_path = self.model_dir / f"regression_all_{horizon}yr.pkl"
        joblib.dump(models, combined_path)
        
        logger.info(f"Training complete for {len(models)} origin groups")
        return models
    
    def evaluate_model(self, model_results: Dict, df: pd.DataFrame, origin_group: str, horizon: int = 2) -> Dict:
        """Evaluate model performance with detailed metrics."""
        logger.info(f"Evaluating model for {origin_group}")
        
        model = model_results['model']
        feature_cols = model_results['feature_columns']
        
        # Prepare test data
        X = df[feature_cols].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Calculate additional metrics
        y_true = df[f'{origin_group}_growth_{horizon}yr']
        
        # Error by population size
        pop_buckets = pd.qcut(df['total_pop'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
        error_by_pop = {}
        
        for bucket in pop_buckets.cat.categories:
            mask = pop_buckets == bucket
            if mask.sum() > 0:
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                error_by_pop[bucket] = mae
        
        # Spatial autocorrelation of residuals
        residuals = y_true - y_pred
        # Note: Would need spatial weights for Moran's I calculation
        
        evaluation = {
            'predictions': y_pred,
            'residuals': residuals,
            'error_by_population': error_by_pop,
            'model_results': model_results
        }
        
        return evaluation

def main():
    """Main function to run regression model training."""
    trainer = ImmigrantGrowthRegressor()
    
    try:
        # Train models for 2-year horizon
        models = trainer.train_all_models(horizon=2)
        
        # Print summary
        logger.info("Training Summary:")
        for origin, results in models.items():
            scores = results['avg_scores']
            logger.info(f"{origin}: MAE={scores['mae']:.3f}, R²={scores['r2']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")

if __name__ == "__main__":
    main()
