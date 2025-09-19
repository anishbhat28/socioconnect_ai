#!/usr/bin/env python3
"""
Main pipeline script for SocioConnect AI - Immigrant Community Growth Prediction.
Runs the complete ETL, modeling, and business matching pipeline.
"""

import sys
import logging
from pathlib import Path
import argparse
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules
from etl.acs import ACSExtractor
from etl.cbp import CBPExtractor
from etl.features import FeatureEngineer
from modeling.train_reg import ImmigrantGrowthRegressor
from modeling.train_cls import ImmigrantHotspotClassifier
from business.sba_ingest import SBADataExtractor
from business.match import BusinessMatcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SocioConnectPipeline:
    """Main pipeline for SocioConnect AI."""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.acs_extractor = ACSExtractor(str(self.data_dir / "acs"))
        self.cbp_extractor = CBPExtractor(str(self.data_dir / "cbp"))
        self.feature_engineer = FeatureEngineer(str(self.data_dir))
        self.regressor = ImmigrantGrowthRegressor(str(self.data_dir), str(self.model_dir))
        self.classifier = ImmigrantHotspotClassifier(str(self.data_dir), str(self.model_dir))
        self.sba_extractor = SBADataExtractor(str(self.data_dir / "sba"))
        self.business_matcher = BusinessMatcher(str(self.data_dir), str(self.model_dir))
    
    def run_etl(self, years: List[int] = None) -> bool:
        """Run ETL pipeline for data extraction and processing."""
        if years is None:
            years = list(range(2018, 2023))
        
        logger.info(f"Running ETL pipeline for years {years}")
        
        try:
            # Extract ACS data
            logger.info("Extracting ACS data...")
            for year in years:
                self.acs_extractor.download_acs_data(year, '5year')
                df = self.acs_extractor.extract_acs_tables(year)
                df_processed = self.acs_extractor.process_acs_data(df)
                self.acs_extractor.save_processed_data(df_processed, year, '5year')
            
            # Extract CBP data
            logger.info("Extracting CBP data...")
            for year in years:
                df = self.cbp_extractor.extract_cbp_data(year)
                if df is not None:
                    df_processed = self.cbp_extractor.process_cbp_data(df)
                    df_county = self.cbp_extractor.aggregate_by_county(df_processed)
                    self.cbp_extractor.save_processed_data(df_county, year)
            
            # Extract SBA data
            logger.info("Extracting SBA data...")
            df = self.sba_extractor.download_sba_data()
            if not df.empty:
                df_processed = self.sba_extractor.process_sba_data(df)
                df_tagged = self.sba_extractor.tag_cuisine_by_name(df_processed)
                df_aggregated = self.sba_extractor.aggregate_by_location(df_tagged)
                self.sba_extractor.save_processed_data(df_tagged)
                
                # Save aggregated data
                aggregated_path = self.data_dir / "sba" / "sba_businesses_aggregated.parquet"
                df_aggregated.to_parquet(aggregated_path, index=False)
            
            logger.info("ETL pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            return False
    
    def run_feature_engineering(self, acs_years: List[int] = None, cbp_years: List[int] = None) -> bool:
        """Run feature engineering pipeline."""
        if acs_years is None:
            acs_years = list(range(2018, 2023))
        if cbp_years is None:
            cbp_years = list(range(2018, 2023))
        
        logger.info("Running feature engineering pipeline...")
        
        try:
            # Engineer features
            features_df = self.feature_engineer.engineer_features(acs_years, cbp_years)
            
            # Save features
            self.feature_engineer.save_features(features_df)
            
            logger.info("Feature engineering pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {e}")
            return False
    
    def run_modeling(self, horizon: int = 2) -> bool:
        """Run modeling pipeline."""
        logger.info(f"Running modeling pipeline for {horizon}-year horizon...")
        
        try:
            # Train regression models
            logger.info("Training regression models...")
            reg_models = self.regressor.train_all_models(horizon)
            
            # Train classification models
            logger.info("Training classification models...")
            cls_models = self.classifier.train_all_models(horizon)
            
            logger.info("Modeling pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Modeling pipeline failed: {e}")
            return False
    
    def run_business_matching(self) -> bool:
        """Run business matching pipeline."""
        logger.info("Running business matching pipeline...")
        
        try:
            # Generate matches for all origin groups
            self.business_matcher.generate_all_matches()
            
            logger.info("Business matching pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Business matching pipeline failed: {e}")
            return False
    
    def run_full_pipeline(self, years: List[int] = None, horizon: int = 2) -> bool:
        """Run the complete pipeline."""
        if years is None:
            years = list(range(2018, 2023))
        
        logger.info("Running full SocioConnect AI pipeline...")
        
        # Run ETL
        if not self.run_etl(years):
            return False
        
        # Run feature engineering
        if not self.run_feature_engineering(years, years):
            return False
        
        # Run modeling
        if not self.run_modeling(horizon):
            return False
        
        # Run business matching
        if not self.run_business_matching():
            return False
        
        logger.info("Full pipeline completed successfully!")
        return True
    
    def run_inference(self, origin_group: str, horizon: int = 2) -> bool:
        """Run inference for new predictions."""
        logger.info(f"Running inference for {origin_group} ({horizon}-year horizon)...")
        
        try:
            # Load features
            features_df = self.feature_engineer.load_features()
            
            # Get hotspot predictions
            hotspots_df = self.business_matcher.load_hotspot_predictions(origin_group)
            
            # Get business recommendations
            recommendations = self.business_matcher.get_recommendations(
                user_lat=37.7749,  # San Francisco
                user_lon=-122.4194,
                origin_group=origin_group,
                max_distance=15.0,
                max_recommendations=10
            )
            
            logger.info(f"Inference completed for {origin_group}")
            logger.info(f"Found {len(hotspots_df)} hotspots and {len(recommendations)} recommendations")
            
            return True
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return False

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="SocioConnect AI Pipeline")
    parser.add_argument("--mode", choices=["etl", "features", "modeling", "matching", "full", "inference"], 
                       default="full", help="Pipeline mode to run")
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020, 2021, 2022],
                       help="Years to process")
    parser.add_argument("--horizon", type=int, default=2, help="Prediction horizon in years")
    parser.add_argument("--origin-group", type=str, default="india", help="Origin group for inference")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = SocioConnectPipeline(args.data_dir, args.model_dir)
    
    # Run pipeline based on mode
    success = False
    
    if args.mode == "etl":
        success = pipeline.run_etl(args.years)
    elif args.mode == "features":
        success = pipeline.run_feature_engineering(args.years, args.years)
    elif args.mode == "modeling":
        success = pipeline.run_modeling(args.horizon)
    elif args.mode == "matching":
        success = pipeline.run_business_matching()
    elif args.mode == "full":
        success = pipeline.run_full_pipeline(args.years, args.horizon)
    elif args.mode == "inference":
        success = pipeline.run_inference(args.origin_group, args.horizon)
    
    if success:
        logger.info("Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
