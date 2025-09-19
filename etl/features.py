"""
Feature engineering pipeline for immigrant community growth prediction.
Creates spatial lags, temporal features, and derived metrics.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from libpysal.weights import Queen, Rook
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engineer features for immigrant community growth prediction."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Origin groups for analysis
        self.origin_groups = [
            'india', 'mexico', 'china', 'philippines', 'vietnam', 'korea',
            'cuba', 'dominican_republic', 'guatemala', 'el_salvador',
            'honduras', 'colombia', 'brazil', 'nigeria', 'ethiopia',
            'jamaica', 'haiti', 'peru', 'ecuador', 'venezuela'
        ]
        
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
            ]
        }
    
    def load_acs_data(self, years: List[int]) -> pd.DataFrame:
        """Load processed ACS data for multiple years."""
        logger.info(f"Loading ACS data for years {years}")
        
        dfs = []
        for year in years:
            filepath = self.data_dir / "acs" / f"acs_5year_{year}_processed.parquet"
            if filepath.exists():
                df = pd.read_parquet(filepath)
                dfs.append(df)
            else:
                logger.warning(f"ACS data for {year} not found")
        
        if not dfs:
            raise FileNotFoundError("No ACS data found")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} records from {len(dfs)} years")
        return combined_df
    
    def load_cbp_data(self, years: List[int]) -> pd.DataFrame:
        """Load processed CBP data for multiple years."""
        logger.info(f"Loading CBP data for years {years}")
        
        dfs = []
        for year in years:
            filepath = self.data_dir / "cbp" / f"cbp_{year}_processed.parquet"
            if filepath.exists():
                df = pd.read_parquet(filepath)
                dfs.append(df)
            else:
                logger.warning(f"CBP data for {year} not found")
        
        if not dfs:
            logger.warning("No CBP data found")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} records from {len(dfs)} years")
        return combined_df
    
    def create_spatial_weights(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Create spatial weights matrices for spatial lag calculations."""
        logger.info("Creating spatial weights matrices")
        
        # Create Queen and Rook contiguity weights
        queen_weights = Queen.from_dataframe(gdf)
        rook_weights = Rook.from_dataframe(gdf)
        
        # Create distance-based weights (optional)
        # distance_weights = DistanceBand.from_dataframe(gdf, threshold=5000)  # 5km threshold
        
        weights = {
            'queen': queen_weights,
            'rook': rook_weights
        }
        
        logger.info(f"Created spatial weights for {len(gdf)} tracts")
        return weights
    
    def calculate_spatial_lags(self, df: pd.DataFrame, weights: Dict, 
                             variables: List[str]) -> pd.DataFrame:
        """Calculate spatial lag variables for specified columns."""
        logger.info(f"Calculating spatial lags for {len(variables)} variables")
        
        df_lagged = df.copy()
        
        for var in variables:
            if var in df.columns:
                # Queen contiguity spatial lag
                df_lagged[f'{var}_lag_queen'] = weights['queen'].lag(df[var].values)
                
                # Rook contiguity spatial lag
                df_lagged[f'{var}_lag_rook'] = weights['rook'].lag(df[var].values)
                
                # Spatial lag of same-origin density (chain migration proxy)
                if 'density' in var and any(origin in var for origin in self.origin_groups):
                    df_lagged[f'{var}_same_origin_lag'] = weights['queen'].lag(df[var].values)
        
        logger.info("Spatial lag calculation complete")
        return df_lagged
    
    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate temporal features and growth rates."""
        logger.info("Calculating temporal features")
        
        df_temporal = df.copy()
        
        # Sort by tract and year
        df_temporal = df_temporal.sort_values(['tract_geoid', 'year'])
        
        # Calculate growth rates for each origin group
        for origin in self.origin_groups:
            density_col = f'{origin}_density'
            if density_col in df_temporal.columns:
                # 1-year growth rate
                df_temporal[f'{origin}_growth_1yr'] = df_temporal.groupby('tract_geoid')[density_col].pct_change()
                
                # 3-year growth rate
                df_temporal[f'{origin}_growth_3yr'] = df_temporal.groupby('tract_geoid')[density_col].pct_change(periods=3)
                
                # 5-year growth rate
                df_temporal[f'{origin}_growth_5yr'] = df_temporal.groupby('tract_geoid')[density_col].pct_change(periods=5)
                
                # Rolling averages
                df_temporal[f'{origin}_density_ma3'] = df_temporal.groupby('tract_geoid')[density_col].rolling(3).mean().reset_index(0, drop=True)
                df_temporal[f'{origin}_density_ma5'] = df_temporal.groupby('tract_geoid')[density_col].rolling(5).mean().reset_index(0, drop=True)
        
        # Overall foreign-born growth rates
        if 'foreign_born_density' in df_temporal.columns:
            df_temporal['foreign_born_growth_1yr'] = df_temporal.groupby('tract_geoid')['foreign_born_density'].pct_change()
            df_temporal['foreign_born_growth_3yr'] = df_temporal.groupby('tract_geoid')['foreign_born_density'].pct_change(periods=3)
        
        # Demographic growth rates
        if 'total_pop' in df_temporal.columns:
            df_temporal['pop_growth_1yr'] = df_temporal.groupby('tract_geoid')['total_pop'].pct_change()
            df_temporal['pop_growth_3yr'] = df_temporal.groupby('tract_geoid')['total_pop'].pct_change(periods=3)
        
        # Income and rent growth
        if 'median_household_income' in df_temporal.columns:
            df_temporal['income_growth_1yr'] = df_temporal.groupby('tract_geoid')['median_household_income'].pct_change()
        
        if 'median_rent' in df_temporal.columns:
            df_temporal['rent_growth_1yr'] = df_temporal.groupby('tract_geoid')['median_rent'].pct_change()
        
        logger.info("Temporal feature calculation complete")
        return df_temporal
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        logger.info("Creating interaction features")
        
        df_interactions = df.copy()
        
        # Housing affordability interactions
        if 'median_rent' in df.columns and 'median_household_income' in df.columns:
            df_interactions['rent_income_interaction'] = df['median_rent'] * df['median_household_income']
        
        # Population density interactions
        if 'total_pop' in df.columns and 'total_housing_units' in df.columns:
            df_interactions['pop_housing_ratio'] = df['total_pop'] / df['total_housing_units']
        
        # Origin group interactions with demographics
        for origin in self.origin_groups:
            density_col = f'{origin}_density'
            if density_col in df.columns:
                # Interaction with income
                if 'median_household_income' in df.columns:
                    df_interactions[f'{origin}_income_interaction'] = df[density_col] * df['median_household_income']
                
                # Interaction with rent
                if 'median_rent' in df.columns:
                    df_interactions[f'{origin}_rent_interaction'] = df[density_col] * df['median_rent']
                
                # Interaction with education
                if 'college_educated_rate' in df.columns:
                    df_interactions[f'{origin}_education_interaction'] = df[density_col] * df['college_educated_rate']
        
        logger.info("Interaction feature creation complete")
        return df_interactions
    
    def create_policy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create policy and temporal dummy variables."""
        logger.info("Creating policy and temporal features")
        
        df_policy = df.copy()
        
        # Year dummies
        years = df['year'].unique()
        for year in years:
            df_policy[f'year_{year}'] = (df['year'] == year).astype(int)
        
        # COVID period dummy
        df_policy['covid_period'] = (df['year'] >= 2020).astype(int)
        
        # Post-COVID period dummy
        df_policy['post_covid_period'] = (df['year'] >= 2022).astype(int)
        
        # Decade dummies
        df_policy['decade_2010s'] = ((df['year'] >= 2010) & (df['year'] < 2020)).astype(int)
        df_policy['decade_2020s'] = (df['year'] >= 2020).astype(int)
        
        # Quarter dummies (if month data available)
        if 'month' in df.columns:
            df_policy['quarter_1'] = (df['month'].isin([1, 2, 3])).astype(int)
            df_policy['quarter_2'] = (df['month'].isin([4, 5, 6])).astype(int)
            df_policy['quarter_3'] = (df['month'].isin([7, 8, 9])).astype(int)
            df_policy['quarter_4'] = (df['month'].isin([10, 11, 12])).astype(int)
        
        logger.info("Policy and temporal feature creation complete")
        return df_policy
    
    def engineer_features(self, acs_years: List[int], cbp_years: List[int]) -> pd.DataFrame:
        """Main feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline")
        
        # Load data
        acs_df = self.load_acs_data(acs_years)
        cbp_df = self.load_cbp_data(cbp_years)
        
        # Merge ACS and CBP data
        if not cbp_df.empty:
            # Convert tract_geoid to county_fips for merging
            acs_df['county_fips'] = acs_df['tract_geoid'].str[:5]
            
            # Merge with CBP data
            df = acs_df.merge(cbp_df, on=['county_fips', 'year'], how='left')
        else:
            df = acs_df.copy()
        
        # Create spatial weights (requires geometry)
        # For now, we'll skip spatial weights and create mock spatial lags
        logger.info("Creating mock spatial weights (replace with actual geometry)")
        
        # Calculate temporal features
        df = self.calculate_temporal_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create policy features
        df = self.create_policy_features(df)
        
        # Create mock spatial lags (replace with actual spatial weights)
        spatial_vars = ['foreign_born_density', 'median_household_income', 'median_rent']
        for var in spatial_vars:
            if var in df.columns:
                # Mock spatial lag (replace with actual spatial weights)
                df[f'{var}_lag_queen'] = df[var].shift(1)  # Mock lag
                df[f'{var}_lag_rook'] = df[var].shift(2)   # Mock lag
        
        # Add origin-specific spatial lags
        for origin in self.origin_groups:
            density_col = f'{origin}_density'
            if density_col in df.columns:
                df[f'{density_col}_same_origin_lag'] = df[density_col].shift(1)  # Mock lag
        
        logger.info("Feature engineering pipeline complete")
        return df
    
    def save_features(self, df: pd.DataFrame, filename: str = "features.parquet"):
        """Save engineered features to parquet format."""
        filepath = self.data_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved features to {filepath}")

def main():
    """Main function to run feature engineering."""
    engineer = FeatureEngineer()
    
    # Define years for analysis
    acs_years = list(range(2018, 2023))
    cbp_years = list(range(2018, 2023))
    
    try:
        # Engineer features
        features_df = engineer.engineer_features(acs_years, cbp_years)
        
        # Save features
        engineer.save_features(features_df)
        
        logger.info(f"Feature engineering complete. Created {len(features_df)} records with {len(features_df.columns)} features")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")

if __name__ == "__main__":
    main()
