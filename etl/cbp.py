"""
County Business Patterns (CBP) data extraction and processing.
Downloads business establishment counts and employment by NAICS industry codes.
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CBPExtractor:
    """Extract and process CBP data for business supply analysis."""
    
    def __init__(self, data_dir: str = "data/cbp"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # NAICS codes relevant to immigrant communities
        self.immigrant_relevant_naics = {
            'food_services': ['7225'],  # Restaurants and other eating places
            'grocery_stores': ['4451'],  # Grocery stores
            'specialty_food': ['4452'],  # Specialty food stores
            'clothing_stores': ['4481'],  # Clothing stores
            'beauty_salons': ['8121'],  # Personal care services
            'laundry_services': ['8123'],  # Dry cleaning and laundry services
            'taxi_services': ['4853'],  # Taxi and limousine service
            'construction': ['23'],  # Construction
            'healthcare': ['62'],  # Health care and social assistance
            'professional_services': ['54'],  # Professional, scientific, and technical services
            'accommodation': ['721'],  # Accommodation
            'transportation': ['48'],  # Transportation and warehousing
            'retail_trade': ['44', '45'],  # Retail trade
            'wholesale_trade': ['42'],  # Wholesale trade
            'manufacturing': ['31', '32', '33'],  # Manufacturing
            'agriculture': ['11'],  # Agriculture, forestry, fishing and hunting
        }
        
        # Detailed NAICS for specific cuisines (proxy for cultural businesses)
        self.cuisine_naics = {
            'indian': ['7225'],  # Will be filtered by name later
            'mexican': ['7225'],
            'chinese': ['7225'],
            'korean': ['7225'],
            'vietnamese': ['7225'],
            'thai': ['7225'],
            'middle_eastern': ['7225'],
            'caribbean': ['7225'],
            'african': ['7225'],
        }
    
    def download_cbp_data(self, year: int) -> str:
        """
        Download CBP data for a given year.
        
        Args:
            year: Data year
            
        Returns:
            Path to downloaded data file
        """
        base_url = "https://www2.census.gov/programs-surveys/cbp/datasets"
        url = f"{base_url}/{year}/cbp{year}co.zip"
        filename = f"cbp_{year}.zip"
        
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"CBP data for {year} already exists")
            return str(filepath)
        
        logger.info(f"Downloading CBP data for {year}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {filename}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to download CBP data for {year}: {e}")
            return None
    
    def extract_cbp_data(self, year: int) -> pd.DataFrame:
        """
        Extract CBP data from downloaded zip file.
        
        Args:
            year: Data year
            
        Returns:
            DataFrame with CBP data
        """
        zip_path = self.download_cbp_data(year)
        if not zip_path:
            return None
        
        logger.info(f"Extracting CBP data for {year}")
        
        # Extract zip file
        extract_dir = self.data_dir / f"cbp_{year}"
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the main data file
        data_files = list(extract_dir.glob("*.txt"))
        if not data_files:
            logger.error(f"No data files found in {extract_dir}")
            return None
        
        # Read the main data file
        data_file = data_files[0]
        df = pd.read_csv(data_file, dtype={'fipstate': str, 'fipscty': str})
        
        # Create county FIPS code
        df['county_fips'] = df['fipstate'].str.zfill(2) + df['fipscty'].str.zfill(3)
        
        # Filter for relevant NAICS codes
        relevant_naics = []
        for category, codes in self.immigrant_relevant_naics.items():
            relevant_naics.extend(codes)
        
        # Create NAICS prefix for filtering
        df['naics_prefix'] = df['naics'].astype(str).str[:2]
        df['naics_3digit'] = df['naics'].astype(str).str[:3]
        df['naics_4digit'] = df['naics'].astype(str).str[:4]
        df['naics_5digit'] = df['naics'].astype(str).str[:5]
        
        # Filter for relevant industries
        mask = (
            df['naics_prefix'].isin([code[:2] for code in relevant_naics]) |
            df['naics_3digit'].isin([code for code in relevant_naics if len(code) == 3]) |
            df['naics_4digit'].isin([code for code in relevant_naics if len(code) == 4]) |
            df['naics_5digit'].isin([code for code in relevant_naics if len(code) == 5])
        )
        
        df_filtered = df[mask].copy()
        
        logger.info(f"Extracted {len(df_filtered)} records for {year}")
        return df_filtered
    
    def process_cbp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw CBP data into analysis-ready format.
        
        Args:
            df: Raw CBP DataFrame
            
        Returns:
            Processed DataFrame with business metrics
        """
        logger.info("Processing CBP data")
        
        # Add year column
        df['year'] = df['year'] if 'year' in df.columns else 2022
        
        # Create business category mappings
        df['business_category'] = 'other'
        
        # Map NAICS to business categories
        for category, codes in self.immigrant_relevant_naics.items():
            for code in codes:
                if len(code) == 2:
                    mask = df['naics_prefix'] == code
                elif len(code) == 3:
                    mask = df['naics_3digit'] == code
                elif len(code) == 4:
                    mask = df['naics_4digit'] == code
                elif len(code) == 5:
                    mask = df['naics_5digit'] == code
                else:
                    mask = df['naics'].astype(str) == code
                
                df.loc[mask, 'business_category'] = category
        
        # Calculate business density metrics
        df['establishments_per_1000'] = df['est'] / df['emp'] * 1000 if 'emp' in df.columns else df['est']
        df['avg_employees_per_establishment'] = df['emp'] / df['est'] if 'emp' in df.columns else 0
        
        # Handle missing values
        df['est'] = df['est'].fillna(0)
        df['emp'] = df['emp'].fillna(0)
        df['ap'] = df['ap'].fillna(0)  # Annual payroll
        
        logger.info("CBP data processing complete")
        return df
    
    def aggregate_by_county(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate CBP data by county and business category.
        
        Args:
            df: Processed CBP DataFrame
            
        Returns:
            County-level aggregated DataFrame
        """
        logger.info("Aggregating CBP data by county")
        
        # Group by county, year, and business category
        agg_cols = {
            'est': 'sum',
            'emp': 'sum',
            'ap': 'sum'
        }
        
        county_agg = df.groupby(['county_fips', 'year', 'business_category']).agg(agg_cols).reset_index()
        
        # Calculate county-level metrics
        county_agg['establishments_per_1000'] = county_agg['est'] / county_agg['emp'] * 1000
        county_agg['avg_employees_per_establishment'] = county_agg['emp'] / county_agg['est']
        county_agg['avg_annual_payroll'] = county_agg['ap'] / county_agg['emp']
        
        # Pivot to wide format for easier analysis
        county_wide = county_agg.pivot_table(
            index=['county_fips', 'year'],
            columns='business_category',
            values=['est', 'emp', 'ap'],
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        county_wide.columns = ['_'.join(col).strip() if col[1] else col[0] 
                              for col in county_wide.columns.values]
        
        logger.info(f"Aggregated data for {len(county_wide)} county-year combinations")
        return county_wide
    
    def save_processed_data(self, df: pd.DataFrame, year: int):
        """Save processed CBP data to parquet format."""
        filename = f"cbp_{year}_processed.parquet"
        filepath = self.data_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")

def main():
    """Main function to run CBP data extraction."""
    extractor = CBPExtractor()
    
    # Extract data for recent years
    for year in range(2018, 2023):
        try:
            # Extract and process
            df = extractor.extract_cbp_data(year)
            if df is not None:
                df_processed = extractor.process_cbp_data(df)
                df_county = extractor.aggregate_by_county(df_processed)
                
                # Save processed data
                extractor.save_processed_data(df_county, year)
            
        except Exception as e:
            logger.error(f"Error processing CBP data for {year}: {e}")

if __name__ == "__main__":
    main()
