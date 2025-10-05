"""
Classification model training for immigrant community hotspot prediction.
Predicts whether a tract will become a new immigrant community hotspot.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import lightgbm as lgb
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImmigrantHotspotClassifier:
    """Classification model for predicting immigrant community hotspots."""
    
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
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
            'class_weight': 'balanced',
            'num_leaves': 64,
            'min_child_samples': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 0.2
        }
        
        # Hotspot thresholds
        self.hotspot_thresholds = {
            'density_threshold': 50,  # per 1,000 population
            'growth_threshold': 25,   # percentage increase
            'min_population': 1000    # minimum tract population
        }
    
    def load_features(self) -> pd.DataFrame:
        """Load engineered features."""
        filepath = self.data_dir / "features.parquet"
        if not filepath.exists():
            raise FileNotFoundError(f"Features file not found: {filepath}")
        
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    def create_hotspot_labels(self, df: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
        """
        Create classification labels for hotspot prediction.
        
        Args:
            df: Features DataFrame
            horizon: Prediction horizon in years
            
        Returns:
            DataFrame with hotspot labels added
        """
        logger.info(f"Creating hotspot labels for {horizon}-year horizon")
        
        df_labeled = df.copy()
        
        # Sort by tract and year
        df_labeled = df_labeled.sort_values(['tract_geoid', 'year'])
        
        # Create hotspot labels for each origin group
        for origin in self.origin_groups:
            density_col = f'{origin}_density'
            if density_col in df_labeled.columns:
                # Get future density
                future_density = df_labeled.groupby('tract_geoid')[density_col].shift(-horizon)
                current_density = df_labeled[density_col]
                
                # Calculate growth rate
                growth_rate = (future_density - current_density) / (current_density + 1e-8) * 100
                
                # Create hotspot label
                # Hotspot if: future density >= threshold AND growth >= threshold AND sufficient population
                is_hotspot = (
                    (future_density >= self.hotspot_thresholds['density_threshold']) &
                    (growth_rate >= self.hotspot_thresholds['growth_threshold']) &
                    (df_labeled['total_pop'] >= self.hotspot_thresholds['min_population'])
                )
                
                df_labeled[f'{origin}_hotspot_{horizon}yr'] = is_hotspot.astype(int)
        
        # Overall foreign-born hotspot
        if 'foreign_born_density' in df_labeled.columns:
            future_density = df_labeled.groupby('tract_geoid')['foreign_born_density'].shift(-horizon)
            current_density = df_labeled['foreign_born_density']
            growth_rate = (future_density - current_density) / (current_density + 1e-8) * 100
            
            is_hotspot = (
                (future_density >= self.hotspot_thresholds['density_threshold']) &
                (growth_rate >= self.hotspot_thresholds['growth_threshold']) &
                (df_labeled['total_pop'] >= self.hotspot_thresholds['min_population'])
            )
            
            df_labeled[f'foreign_born_hotspot_{horizon}yr'] = is_hotspot.astype(int)
        
        # Remove rows where we don't have future data
        df_labeled = df_labeled.dropna(subset=[f'foreign_born_hotspot_{horizon}yr'])
        
        # Print class distribution
        for origin in self.origin_groups:
            hotspot_col = f'{origin}_hotspot_{horizon}yr'
            if hotspot_col in df_labeled.columns:
                pos_rate = df_labeled[hotspot_col].mean()
                logger.info(f"{origin} hotspot rate: {pos_rate:.3f}")
        
        logger.info(f"Created labels for {len(df_labeled)} records")
        return df_labeled
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names from DataFrame."""
        feature_cols = []
        
        # Base features
        base_features = [
            'total_pop', 'foreign_born_density', 'median_age',
            'median_household_income', 'college_educated_rate',
            'median_rent', 'rent_income_ratio', 'homeownership_rate',
            'total_housing_units', 'occupied_housing_units',
            'spanish_at_home', 'hindi_at_home', 'chinese_at_home',
            'korean_at_home', 'vietnamese_at_home', 'arabic_at_home',
            'language_diversity', 'foreign_born_growth_1yr', 'foreign_born_growth_3yr',
            'pop_growth_1yr', 'pop_growth_3yr', 'income_growth_1yr', 'rent_growth_1yr',
            'covid_period', 'post_covid_period', 'decade_2010s', 'decade_2020s'
        ]
        
        for feature in base_features:
            if feature in df.columns:
                feature_cols.append(feature)
        
        # Origin-specific features
        for origin in self.origin_groups:
            for suffix in ['_density', '_growth_1yr', '_growth_3yr', '_growth_5yr', 
                          '_density_ma3', '_density_ma5', '_income_interaction', 
                          '_rent_interaction', '_education_interaction']:
                col = f'{origin}{suffix}'
                if col in df.columns:
                    feature_cols.append(col)
        
        # Business features
        business_features = [
            'food_services_est', 'grocery_stores_est', 'specialty_food_est',
            'clothing_stores_est', 'beauty_salons_est', 'laundry_services_est'
        ]
        
        for feature in business_features:
            if feature in df.columns:
                feature_cols.append(feature)
        
        # Year dummies
        year_cols = [col for col in df.columns if col.startswith('year_')]
        feature_cols.extend(year_cols)
        
        # Spatial lag features
        spatial_lag_cols = [col for col in df.columns if '_lag_' in col]
        feature_cols.extend(spatial_lag_cols)
        
        return feature_cols
    
    def train_model(self, df: pd.DataFrame, origin_group: str, horizon: int = 2) -> Dict:
        """
        Train classification model for specific origin group.
        
        Args:
            df: Labeled features DataFrame
            origin_group: Origin group to predict
            horizon: Prediction horizon in years
            
        Returns:
            Dictionary with model and evaluation metrics
        """
        logger.info(f"Training classification model for {origin_group} ({horizon}-year horizon)")
        
        # Get target variable
        target_col = f'{origin_group}_hotspot_{horizon}yr'
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df)
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        # Check class distribution
        pos_rate = y.mean()
        logger.info(f"Positive class rate: {pos_rate:.3f}")
        
        if pos_rate < 0.01:
            logger.warning(f"Very low positive class rate for {origin_group}: {pos_rate:.3f}")
        
        # Group by tract for spatial cross-validation
        groups = df['tract_geoid']
        
        # Initialize model
        model = LGBMClassifier(**self.model_params)
        
        # Cross-validation
        # Stratified by label while respecting groups to stabilize ROC-AUC under imbalance
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
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
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            pr_auc = average_precision_score(y_val, y_pred_proba)
            brier_score = brier_score_loss(y_val, y_pred_proba)
            
            cv_scores.append({
                'fold': fold + 1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'brier_score': brier_score
            })
            
            models.append(model)
        
        # Train final model on all data
        final_model = LGBMClassifier(**self.model_params)
        final_model.fit(X, y)
        
        # Calibrate the model
        calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
        calibrated_model.fit(X, y)
        
        # Calculate average CV scores
        avg_scores = {
            'roc_auc': np.mean([score['roc_auc'] for score in cv_scores]),
            'pr_auc': np.mean([score['pr_auc'] for score in cv_scores]),
            'brier_score': np.mean([score['brier_score'] for score in cv_scores])
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
            'calibrated_model': calibrated_model,
            'feature_columns': feature_cols,
            'cv_scores': cv_scores,
            'avg_scores': avg_scores,
            'feature_importance': feature_importance,
            'shap_values': shap_values,
            'shap_explainer': explainer,
            'class_distribution': pos_rate
        }
        
        logger.info(f"Model training complete. ROC-AUC: {avg_scores['roc_auc']:.3f}, PR-AUC: {avg_scores['pr_auc']:.3f}")
        return results
    
    def train_all_models(self, horizon: int = 2) -> Dict:
        """Train models for all origin groups."""
        logger.info(f"Training classification models for all origin groups ({horizon}-year horizon)")
        
        # Load features
        df = self.load_features()
        
        # Create labels
        df_labeled = self.create_hotspot_labels(df, horizon)
        
        # Train models for each origin group
        models = {}
        for origin in self.origin_groups:
            try:
                results = self.train_model(df_labeled, origin, horizon)
                models[origin] = results
                
                # Save model
                model_path = self.model_dir / f"classification_{origin}_{horizon}yr.pkl"
                joblib.dump(results, model_path)
                
            except Exception as e:
                logger.error(f"Error training model for {origin}: {e}")
        
        # Save combined results
        combined_path = self.model_dir / f"classification_all_{horizon}yr.pkl"
        joblib.dump(models, combined_path)
        
        logger.info(f"Training complete for {len(models)} origin groups")
        return models
    
    def evaluate_model(self, model_results: Dict, df: pd.DataFrame, origin_group: str, horizon: int = 2) -> Dict:
        """Evaluate model performance with detailed metrics."""
        logger.info(f"Evaluating model for {origin_group}")
        
        model = model_results['calibrated_model']
        feature_cols = model_results['feature_columns']
        
        # Prepare test data
        X = df[feature_cols].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # True labels
        y_true = df[f'{origin_group}_hotspot_{horizon}yr']
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        evaluation = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier_score': brier_score,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'precision_recall_curve': (precision, recall, thresholds),
            'model_results': model_results
        }
        
        return evaluation
    
    def get_hotspot_predictions(self, df: pd.DataFrame, origin_group: str, 
                              threshold: float = 0.5, horizon: int = 2) -> pd.DataFrame:
        """Get hotspot predictions for new data."""
        model_path = self.model_dir / f"classification_{origin_group}_{horizon}yr.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model_results = joblib.load(model_path)
        model = model_results['calibrated_model']
        feature_cols = model_results['feature_columns']
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Predictions
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Create results DataFrame
        results = df[['tract_geoid', 'year']].copy()
        results[f'{origin_group}_hotspot_prob'] = probabilities
        results[f'{origin_group}_hotspot_pred'] = predictions
        
        return results

def main():
    """Main function to run classification model training."""
    trainer = ImmigrantHotspotClassifier()
    
    try:
        # Train models for 2-year horizon
        models = trainer.train_all_models(horizon=2)
        
        # Print summary
        logger.info("Training Summary:")
        for origin, results in models.items():
            scores = results['avg_scores']
            class_dist = results['class_distribution']
            logger.info(f"{origin}: ROC-AUC={scores['roc_auc']:.3f}, PR-AUC={scores['pr_auc']:.3f}, "
                       f"Positive rate={class_dist:.3f}")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")

if __name__ == "__main__":
    main()
