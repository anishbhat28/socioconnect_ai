"""
SBA Dynamic Small Business Search (DSBS) data ingestion and processing.
Downloads and processes official SBA business directory data.
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SBADataExtractor:
    """Extract and process SBA DSBS data for business matching."""
    
    def __init__(self, data_dir: str = "data/sba"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # SBA API endpoints
        self.base_url = "https://api.sba.gov/geodata/rest/services"
        self.dsbs_url = "https://api.sba.gov/geodata/rest/services/dsbs"
        
        # NAICS codes for immigrant-relevant businesses
        self.immigrant_naics = {
            'food_services': ['7225', '7223', '7224'],  # Restaurants, bars, cafeterias
            'grocery_stores': ['4451', '4452', '4453'],  # Grocery, specialty food, beer/wine
            'retail_trade': ['44', '45'],  # General retail
            'personal_services': ['8121', '8123', '8129'],  # Beauty, laundry, other personal
            'transportation': ['4853', '4854', '4855'],  # Taxi, bus, other transit
            'construction': ['23'],  # Construction
            'healthcare': ['62'],  # Health care
            'professional_services': ['54'],  # Professional services
            'accommodation': ['721'],  # Hotels, motels
            'manufacturing': ['31', '32', '33'],  # Manufacturing
        }
        
        # Cuisine keywords for name-based tagging
        self.cuisine_keywords = {
            'indian': [
                'indian', 'hindi', 'punjabi', 'bengali', 'tamil', 'telugu', 'gujarati',
                'marathi', 'kannada', 'malayalam', 'oriya', 'assamese', 'kashmiri',
                'bombay', 'delhi', 'mumbai', 'bangalore', 'chennai', 'hyderabad',
                'kolkata', 'pune', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur',
                'curry', 'tandoor', 'masala', 'biryani', 'dal', 'naan', 'roti',
                'samosas', 'pakora', 'tikka', 'korma', 'vindaloo', 'butter chicken',
                'desi', 'spice', 'chai', 'lassi', 'gulab jamun', 'rasgulla'
            ],
            'mexican': [
                'mexican', 'mexico', 'taco', 'burrito', 'quesadilla', 'enchilada',
                'fajita', 'nachos', 'guacamole', 'salsa', 'churro', 'tamale',
                'mole', 'pozole', 'ceviche', 'elote', 'horchata', 'margarita',
                'tequila', 'mezcal', 'carnitas', 'al pastor', 'barbacoa',
                'sopes', 'tostadas', 'flautas', 'empanadas', 'chile relleno'
            ],
            'chinese': [
                'chinese', 'china', 'cantonese', 'szechuan', 'hunan', 'peking',
                'beijing', 'shanghai', 'hong kong', 'taiwan', 'dim sum',
                'lo mein', 'chow mein', 'fried rice', 'kung pao', 'sweet sour',
                'general tso', 'orange chicken', 'beef broccoli', 'moo shu',
                'wonton', 'dumpling', 'spring roll', 'egg roll', 'hot pot',
                'peking duck', 'char siu', 'chow fun', 'chop suey'
            ],
            'korean': [
                'korean', 'korea', 'seoul', 'busan', 'kimchi', 'bulgogi',
                'bibimbap', 'galbi', 'japchae', 'korean bbq', 'soju',
                'makgeolli', 'banchan', 'tteokbokki', 'kimbap', 'ramen',
                'udon', 'pho', 'bun', 'gochujang', 'doenjang', 'sesame oil'
            ],
            'vietnamese': [
                'vietnamese', 'vietnam', 'ho chi minh', 'hanoi', 'pho',
                'banh mi', 'spring roll', 'summer roll', 'bun', 'com tam',
                'bun bo hue', 'cao lau', 'mi quang', 'banh xeo', 'goi cuon',
                'nuoc mam', 'fish sauce', 'sriracha', 'hoisin', 'rice paper'
            ],
            'thai': [
                'thai', 'thailand', 'bangkok', 'pad thai', 'tom yum', 'tom kha',
                'green curry', 'red curry', 'yellow curry', 'massaman',
                'panang', 'larb', 'som tam', 'mango sticky rice', 'thai iced tea',
                'lemongrass', 'galangal', 'kaffir lime', 'fish sauce', 'coconut milk'
            ],
            'middle_eastern': [
                'middle eastern', 'arabic', 'persian', 'iranian', 'lebanese',
                'syrian', 'turkish', 'egyptian', 'moroccan', 'falafel',
                'hummus', 'baba ganoush', 'tabbouleh', 'kebab', 'shawarma',
                'gyro', 'pita', 'naan', 'tahini', 'za\'atar', 'sumac',
                'knafeh', 'baklava', 'halva', 'mint tea', 'turkish coffee'
            ],
            'caribbean': [
                'caribbean', 'jamaican', 'haitian', 'cuban', 'puerto rican',
                'dominican', 'trinidadian', 'barbadian', 'jerk', 'curry goat',
                'ackee', 'saltfish', 'plantain', 'rice peas', 'callaloo',
                'sorrel', 'ginger beer', 'rum', 'coconut', 'tamarind'
            ],
            'african': [
                'african', 'ethiopian', 'nigerian', 'ghanaian', 'senegalese',
                'moroccan', 'south african', 'injera', 'berbere', 'wat',
                'kitfo', 'doro wat', 'tibs', 'fufu', 'jollof rice', 'egusi',
                'suya', 'bobotie', 'bunny chow', 'biltong', 'rooibos tea'
            ]
        }
    
    def download_sba_data(self, state: str = None) -> pd.DataFrame:
        """
        Download SBA DSBS data for specified state or all states.
        
        Args:
            state: State abbreviation (e.g., 'CA') or None for all states
            
        Returns:
            DataFrame with SBA business data
        """
        logger.info(f"Downloading SBA DSBS data for {state or 'all states'}")
        
        # SBA DSBS API endpoint
        url = "https://api.sba.gov/geodata/rest/services/dsbs/MapServer/0/query"
        
        # Query parameters
        params = {
            'where': '1=1',  # Get all records
            'outFields': '*',
            'f': 'json',
            'returnGeometry': 'true'
        }
        
        if state:
            params['where'] = f"STATE='{state}'"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'features' not in data:
                logger.error("No features found in SBA API response")
                return pd.DataFrame()
            
            # Extract business data
            businesses = []
            for feature in data['features']:
                business = feature['attributes']
                if 'geometry' in feature:
                    business['longitude'] = feature['geometry']['x']
                    business['latitude'] = feature['geometry']['y']
                
                businesses.append(business)
            
            df = pd.DataFrame(businesses)
            logger.info(f"Downloaded {len(df)} businesses from SBA")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading SBA data: {e}")
            return pd.DataFrame()
    
    def process_sba_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw SBA data into analysis-ready format.
        
        Args:
            df: Raw SBA DataFrame
            
        Returns:
            Processed DataFrame with business metrics
        """
        logger.info("Processing SBA data")
        
        # Clean and standardize data
        df_processed = df.copy()
        
        # Standardize column names
        column_mapping = {
            'BUSINESS_NAME': 'business_name',
            'NAICS_CODE': 'naics_code',
            'NAICS_DESC': 'naics_description',
            'ADDRESS': 'address',
            'CITY': 'city',
            'STATE': 'state',
            'ZIP': 'zip_code',
            'LATITUDE': 'latitude',
            'LONGITUDE': 'longitude',
            'PHONE': 'phone',
            'WEBSITE': 'website',
            'EMAIL': 'email',
            'YEAR_ESTABLISHED': 'year_established',
            'EMPLOYEES': 'employees',
            'ANNUAL_SALES': 'annual_sales',
            'CERTIFICATION': 'certification',
            'MINORITY_OWNED': 'minority_owned',
            'WOMAN_OWNED': 'woman_owned',
            'VETERAN_OWNED': 'veteran_owned'
        }
        
        # Rename columns
        df_processed = df_processed.rename(columns=column_mapping)
        
        # Clean business names
        df_processed['business_name'] = df_processed['business_name'].str.strip().str.title()
        
        # Create NAICS categories
        df_processed['naics_2digit'] = df_processed['naics_code'].astype(str).str[:2]
        df_processed['naics_3digit'] = df_processed['naics_code'].astype(str).str[:3]
        df_processed['naics_4digit'] = df_processed['naics_code'].astype(str).str[:4]
        df_processed['naics_5digit'] = df_processed['naics_code'].astype(str).str[:5]
        
        # Map to business categories
        df_processed['business_category'] = 'other'
        for category, codes in self.immigrant_naics.items():
            for code in codes:
                if len(code) == 2:
                    mask = df_processed['naics_2digit'] == code
                elif len(code) == 3:
                    mask = df_processed['naics_3digit'] == code
                elif len(code) == 4:
                    mask = df_processed['naics_4digit'] == code
                elif len(code) == 5:
                    mask = df_processed['naics_5digit'] == code
                else:
                    mask = df_processed['naics_code'].astype(str) == code
                
                df_processed.loc[mask, 'business_category'] = category
        
        # Create county FIPS code
        df_processed['county_fips'] = df_processed['state'].str.zfill(2) + '000'  # Simplified
        
        # Handle missing values
        df_processed['employees'] = pd.to_numeric(df_processed['employees'], errors='coerce').fillna(0)
        df_processed['annual_sales'] = pd.to_numeric(df_processed['annual_sales'], errors='coerce').fillna(0)
        df_processed['year_established'] = pd.to_numeric(df_processed['year_established'], errors='coerce').fillna(0)
        
        # Create business age
        current_year = pd.Timestamp.now().year
        df_processed['business_age'] = current_year - df_processed['year_established']
        df_processed['business_age'] = df_processed['business_age'].clip(lower=0)
        
        logger.info("SBA data processing complete")
        return df_processed
    
    def tag_cuisine_by_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tag businesses with cuisine types based on business names.
        
        Args:
            df: Processed SBA DataFrame
            
        Returns:
            DataFrame with cuisine tags added
        """
        logger.info("Tagging businesses with cuisine types")
        
        df_tagged = df.copy()
        
        # Initialize cuisine score columns
        for cuisine in self.cuisine_keywords.keys():
            df_tagged[f'{cuisine}_score'] = 0.0
        
        # Tag businesses based on name
        for idx, row in df_tagged.iterrows():
            business_name = str(row['business_name']).lower()
            
            for cuisine, keywords in self.cuisine_keywords.items():
                score = 0.0
                for keyword in keywords:
                    if keyword.lower() in business_name:
                        score += 1.0
                
                # Normalize score by number of keywords
                df_tagged.at[idx, f'{cuisine}_score'] = score / len(keywords)
        
        # Create primary cuisine tag
        cuisine_cols = [f'{cuisine}_score' for cuisine in self.cuisine_keywords.keys()]
        df_tagged['primary_cuisine'] = df_tagged[cuisine_cols].idxmax(axis=1)
        df_tagged['primary_cuisine'] = df_tagged['primary_cuisine'].str.replace('_score', '')
        
        # Only keep businesses with cuisine scores > 0
        df_tagged['has_cuisine_tag'] = df_tagged[cuisine_cols].max(axis=1) > 0
        
        logger.info(f"Tagged {df_tagged['has_cuisine_tag'].sum()} businesses with cuisine types")
        return df_tagged
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "sba_businesses_processed.parquet"):
        """Save processed SBA data to parquet format."""
        filepath = self.data_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
    
    def aggregate_by_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate business data by location (county/tract level).
        
        Args:
            df: Processed SBA DataFrame
            
        Returns:
            Location-level aggregated DataFrame
        """
        logger.info("Aggregating business data by location")
        
        # Group by county and business category
        county_agg = df.groupby(['county_fips', 'business_category']).agg({
            'business_name': 'count',
            'employees': 'sum',
            'annual_sales': 'sum',
            'business_age': 'mean'
        }).reset_index()
        
        # Rename columns
        county_agg.columns = ['county_fips', 'business_category', 'business_count', 
                             'total_employees', 'total_sales', 'avg_business_age']
        
        # Pivot to wide format
        county_wide = county_agg.pivot_table(
            index='county_fips',
            columns='business_category',
            values=['business_count', 'total_employees', 'total_sales'],
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        county_wide.columns = ['_'.join(col).strip() if col[1] else col[0] 
                              for col in county_wide.columns.values]
        
        # Aggregate cuisine-specific businesses
        cuisine_agg = df[df['has_cuisine_tag']].groupby(['county_fips', 'primary_cuisine']).agg({
            'business_name': 'count',
            'employees': 'sum'
        }).reset_index()
        
        cuisine_wide = cuisine_agg.pivot_table(
            index='county_fips',
            columns='primary_cuisine',
            values=['business_name', 'employees'],
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        cuisine_wide.columns = ['_'.join(col).strip() if col[1] else col[0] 
                               for col in cuisine_wide.columns.values]
        
        # Merge county and cuisine aggregations
        location_agg = county_wide.merge(cuisine_wide, on='county_fips', how='outer')
        
        logger.info(f"Aggregated data for {len(location_agg)} counties")
        return location_agg

def main():
    """Main function to run SBA data extraction."""
    extractor = SBADataExtractor()
    
    try:
        # Download SBA data for all states
        df = extractor.download_sba_data()
        
        if not df.empty:
            # Process data
            df_processed = extractor.process_sba_data(df)
            
            # Tag cuisine types
            df_tagged = extractor.tag_cuisine_by_name(df_processed)
            
            # Aggregate by location
            df_aggregated = extractor.aggregate_by_location(df_tagged)
            
            # Save processed data
            extractor.save_processed_data(df_tagged)
            
            # Save aggregated data
            aggregated_path = extractor.data_dir / "sba_businesses_aggregated.parquet"
            df_aggregated.to_parquet(aggregated_path, index=False)
            
            logger.info(f"Processing complete. {len(df_tagged)} businesses processed, "
                       f"{len(df_aggregated)} counties aggregated")
        
    except Exception as e:
        logger.error(f"Error in SBA data extraction: {e}")

if __name__ == "__main__":
    main()
