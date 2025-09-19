"""
ACS (American Community Survey) data extraction and processing.
Downloads foreign-born population data by place of birth, demographics, and housing.
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

class ACSExtractor:
    """Extract and process ACS data for immigrant community analysis."""
    
    def __init__(self, data_dir: str = "data/acs"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Origin groups mapping (country codes to standardized names)
        self.origin_groups = {
            'India': ['B05006_015E'],  # India
            'Mexico': ['B05006_002E'],  # Mexico
            'China': ['B05006_004E'],   # China
            'Philippines': ['B05006_005E'],  # Philippines
            'Vietnam': ['B05006_006E'],  # Vietnam
            'Korea': ['B05006_007E'],   # South Korea
            'Cuba': ['B05006_008E'],    # Cuba
            'Dominican_Republic': ['B05006_009E'],  # Dominican Republic
            'Guatemala': ['B05006_010E'],  # Guatemala
            'El_Salvador': ['B05006_011E'],  # El Salvador
            'Honduras': ['B05006_012E'],  # Honduras
            'Colombia': ['B05006_013E'],  # Colombia
            'Brazil': ['B05006_014E'],   # Brazil
            'Nigeria': ['B05006_016E'],  # Nigeria
            'Ethiopia': ['B05006_017E'],  # Ethiopia
            'Jamaica': ['B05006_018E'],  # Jamaica
            'Haiti': ['B05006_019E'],    # Haiti
            'Peru': ['B05006_020E'],     # Peru
            'Ecuador': ['B05006_021E'],  # Ecuador
            'Venezuela': ['B05006_022E']  # Venezuela
        }
        
        # Additional demographic variables
        self.demographic_vars = {
            'total_pop': 'B01003_001E',
            'foreign_born_total': 'B05006_001E',
            'median_age': 'B01002_001E',
            'median_household_income': 'B19013_001E',
            'median_rent': 'B25064_001E',
            'total_housing_units': 'B25001_001E',
            'occupied_housing_units': 'B25003_001E',
            'owner_occupied': 'B25003_002E',
            'renter_occupied': 'B25003_003E',
            'vehicles_available': 'B25044_001E',
            'no_vehicles': 'B25044_003E',
            'bachelor_degree': 'B15003_022E',
            'graduate_degree': 'B15003_023E',
            'total_education': 'B15003_001E'
        }
        
        # Language variables
        self.language_vars = {
            'spanish_at_home': 'B16001_003E',
            'hindi_at_home': 'B16001_011E',
            'chinese_at_home': 'B16001_012E',
            'korean_at_home': 'B16001_013E',
            'vietnamese_at_home': 'B16001_014E',
            'arabic_at_home': 'B16001_015E',
            'total_language': 'B16001_001E'
        }
    
    def download_acs_data(self, year: int, survey: str = '5year') -> str:
        """
        Download ACS data for a given year and survey type.
        
        Args:
            year: Survey year
            survey: '1year' or '5year'
            
        Returns:
            Path to downloaded data file
        """
        base_url = "https://www2.census.gov/programs-surveys/acs/data/pums"
        
        if survey == '5year':
            url = f"{base_url}/{year}/5-Year/csv_pus.zip"
            filename = f"acs_5year_{year}.zip"
        else:
            url = f"{base_url}/{year}/1-Year/csv_pus.zip"
            filename = f"acs_1year_{year}.zip"
        
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"ACS data for {year} already exists")
            return str(filepath)
        
        logger.info(f"Downloading ACS {survey} data for {year}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {filename}")
        return str(filepath)
    
    def extract_acs_tables(self, year: int, states: List[str] = None) -> pd.DataFrame:
        """
        Extract ACS tables for demographic and foreign-born data.
        
        Args:
            year: Survey year
            states: List of state FIPS codes (None for all states)
            
        Returns:
            DataFrame with tract-level ACS data
        """
        # This would typically use the census API or cenpy library
        # For now, we'll create a mock structure
        
        logger.info(f"Extracting ACS tables for {year}")
        
        # Mock data structure - in practice, you'd use census API
        columns = ['tract_geoid', 'year'] + list(self.demographic_vars.values()) + \
                 list(self.language_vars.values())
        
        # Add origin group columns
        for origin, vars in self.origin_groups.items():
            columns.extend(vars)
        
        # Create mock data (replace with actual API calls)
        n_tracts = 1000  # Approximate number of tracts
        data = []
        
        for i in range(n_tracts):
            row = {
                'tract_geoid': f"{np.random.randint(1, 100):02d}{np.random.randint(1, 1000):03d}{np.random.randint(1, 100):06d}",
                'year': year
            }
            
            # Add demographic data (mock)
            row.update({
                'B01003_001E': np.random.randint(1000, 10000),  # total_pop
                'B05006_001E': np.random.randint(50, 2000),     # foreign_born_total
                'B01002_001E': np.random.randint(25, 65),       # median_age
                'B19013_001E': np.random.randint(30000, 150000), # median_income
                'B25064_001E': np.random.randint(800, 3000),    # median_rent
                'B25001_001E': np.random.randint(500, 5000),    # total_housing
                'B25003_001E': np.random.randint(400, 4500),    # occupied_housing
                'B25003_002E': np.random.randint(200, 3000),    # owner_occupied
                'B25003_003E': np.random.randint(100, 2000),    # renter_occupied
                'B25044_001E': np.random.randint(300, 4000),    # vehicles_available
                'B25044_003E': np.random.randint(50, 500),      # no_vehicles
                'B15003_022E': np.random.randint(50, 1000),     # bachelor_degree
                'B15003_023E': np.random.randint(20, 500),      # graduate_degree
                'B15003_001E': np.random.randint(800, 8000),    # total_education
            })
            
            # Add language data (mock)
            row.update({
                'B16001_003E': np.random.randint(0, 500),   # spanish_at_home
                'B16001_011E': np.random.randint(0, 100),   # hindi_at_home
                'B16001_012E': np.random.randint(0, 100),   # chinese_at_home
                'B16001_013E': np.random.randint(0, 50),    # korean_at_home
                'B16001_014E': np.random.randint(0, 50),    # vietnamese_at_home
                'B16001_015E': np.random.randint(0, 50),    # arabic_at_home
                'B16001_001E': np.random.randint(500, 5000), # total_language
            })
            
            # Add origin group data (mock)
            for origin, vars in self.origin_groups.items():
                for var in vars:
                    row[var] = np.random.randint(0, 200)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Extracted {len(df)} tracts for {year}")
        return df
    
    def process_acs_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw ACS data into analysis-ready format.
        
        Args:
            df: Raw ACS DataFrame
            
        Returns:
            Processed DataFrame with derived metrics
        """
        logger.info("Processing ACS data")
        
        # Create origin group totals
        for origin, vars in self.origin_groups.items():
            df[f'{origin.lower()}_count'] = df[vars].sum(axis=1)
            df[f'{origin.lower()}_density'] = df[f'{origin.lower()}_count'] / df['B01003_001E'] * 1000
        
        # Calculate derived metrics
        df['foreign_born_density'] = df['B05006_001E'] / df['B01003_001E'] * 1000
        df['rent_income_ratio'] = df['B25064_001E'] * 12 / df['B19013_001E']
        df['homeownership_rate'] = df['B25003_002E'] / df['B25003_001E']
        df['college_educated_rate'] = (df['B15003_022E'] + df['B15003_023E']) / df['B15003_001E']
        df['vehicle_access_rate'] = (df['B25044_001E'] - df['B25044_003E']) / df['B25044_001E']
        
        # Language diversity metrics
        language_cols = [col for col in df.columns if 'at_home' in col and col != 'B16001_001E']
        df['language_diversity'] = (df[language_cols] > 0).sum(axis=1)
        
        logger.info("ACS data processing complete")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, year: int, survey: str = '5year'):
        """Save processed ACS data to parquet format."""
        filename = f"acs_{survey}_{year}_processed.parquet"
        filepath = self.data_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")

def main():
    """Main function to run ACS data extraction."""
    extractor = ACSExtractor()
    
    # Extract data for recent years
    for year in range(2018, 2023):
        try:
            # Download data
            extractor.download_acs_data(year, '5year')
            
            # Extract and process
            df = extractor.extract_acs_tables(year)
            df_processed = extractor.process_acs_data(df)
            
            # Save processed data
            extractor.save_processed_data(df_processed, year, '5year')
            
        except Exception as e:
            logger.error(f"Error processing ACS data for {year}: {e}")

if __name__ == "__main__":
    main()
