"""
Business matching system for connecting predicted immigrant hotspots with relevant businesses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessMatcher:
    """Match predicted immigrant hotspots with culturally relevant businesses."""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        
        # Origin groups
        self.origin_groups = [
            'india', 'mexico', 'china', 'philippines', 'vietnam', 'korea',
            'cuba', 'dominican_republic', 'guatemala', 'el_salvador',
            'honduras', 'colombia', 'brazil', 'nigeria', 'ethiopia',
            'jamaica', 'haiti', 'peru', 'ecuador', 'venezuela'
        ]
        
        # Business categories relevant to each origin group
        self.origin_business_preferences = {
            'india': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'mexico': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'china': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'philippines': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'vietnam': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'korea': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'cuba': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'dominican_republic': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'guatemala': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'el_salvador': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'honduras': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'colombia': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'brazil': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'nigeria': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'ethiopia': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'jamaica': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'haiti': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'peru': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'ecuador': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons'],
            'venezuela': ['food_services', 'grocery_stores', 'specialty_food', 'beauty_salons']
        }
    
    def load_business_data(self) -> pd.DataFrame:
        """Load processed SBA business data."""
        filepath = self.data_dir / "sba" / "sba_businesses_processed.parquet"
        if not filepath.exists():
            raise FileNotFoundError(f"Business data not found: {filepath}")
        
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} businesses")
        return df
    
    def load_hotspot_predictions(self, origin_group: str) -> pd.DataFrame:
        """Load hotspot predictions for specific origin group."""
        # This would typically load from model predictions
        # For now, we'll create mock predictions
        
        logger.info(f"Loading hotspot predictions for {origin_group}")
        
        # Mock hotspot predictions
        n_tracts = 1000
        predictions = []
        
        for i in range(n_tracts):
            tract_geoid = f"{np.random.randint(1, 100):02d}{np.random.randint(1, 1000):03d}{np.random.randint(1, 100):06d}"
            latitude = np.random.uniform(25, 49)  # US latitude range
            longitude = np.random.uniform(-125, -66)  # US longitude range
            
            predictions.append({
                'tract_geoid': tract_geoid,
                'latitude': latitude,
                'longitude': longitude,
                f'{origin_group}_hotspot_prob': np.random.uniform(0, 1),
                f'{origin_group}_hotspot_pred': np.random.choice([0, 1], p=[0.8, 0.2])
            })
        
        df = pd.DataFrame(predictions)
        logger.info(f"Loaded {len(df)} tract predictions for {origin_group}")
        return df
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in kilometers."""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        return c * r
    
    def find_nearby_businesses(self, hotspot_lat: float, hotspot_lon: float, 
                              businesses_df: pd.DataFrame, max_distance: float = 10.0) -> pd.DataFrame:
        """
        Find businesses within specified distance of hotspot.
        
        Args:
            hotspot_lat: Hotspot latitude
            hotspot_lon: Hotspot longitude
            businesses_df: Business DataFrame
            max_distance: Maximum distance in kilometers
            
        Returns:
            DataFrame of nearby businesses
        """
        # Calculate distances
        distances = []
        for _, business in businesses_df.iterrows():
            if pd.notna(business['latitude']) and pd.notna(business['longitude']):
                dist = self.calculate_distance(
                    hotspot_lat, hotspot_lon,
                    business['latitude'], business['longitude']
                )
                distances.append(dist)
            else:
                distances.append(np.inf)
        
        businesses_df = businesses_df.copy()
        businesses_df['distance_km'] = distances
        
        # Filter by distance
        nearby_businesses = businesses_df[businesses_df['distance_km'] <= max_distance]
        
        return nearby_businesses.sort_values('distance_km')
    
    def score_business_relevance(self, business: pd.Series, origin_group: str) -> float:
        """
        Score business relevance for specific origin group.
        
        Args:
            business: Business row from DataFrame
            origin_group: Origin group to match
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        
        # Base score from business category
        if business['business_category'] in self.origin_business_preferences.get(origin_group, []):
            score += 0.3
        
        # Cuisine-specific score
        cuisine_score_col = f'{origin_group}_score'
        if cuisine_score_col in business.index:
            score += business[cuisine_score_col] * 0.5
        
        # Business size score (prefer established businesses)
        if business['employees'] > 0:
            score += 0.1
        
        # Business age score (prefer established businesses)
        if business['business_age'] > 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def match_hotspots_to_businesses(self, origin_group: str, 
                                   max_distance: float = 10.0,
                                   max_businesses: int = 20) -> pd.DataFrame:
        """
        Match predicted hotspots with relevant businesses.
        
        Args:
            origin_group: Origin group to match
            max_distance: Maximum distance in kilometers
            max_businesses: Maximum businesses per hotspot
            
        Returns:
            DataFrame with hotspot-business matches
        """
        logger.info(f"Matching hotspots to businesses for {origin_group}")
        
        # Load data
        hotspots_df = self.load_hotspot_predictions(origin_group)
        businesses_df = self.load_business_data()
        
        # Filter to high-probability hotspots
        hotspot_col = f'{origin_group}_hotspot_prob'
        high_prob_hotspots = hotspots_df[hotspots_df[hotspot_col] > 0.5]
        
        logger.info(f"Found {len(high_prob_hotspots)} high-probability hotspots")
        
        matches = []
        
        for _, hotspot in high_prob_hotspots.iterrows():
            # Find nearby businesses
            nearby_businesses = self.find_nearby_businesses(
                hotspot['latitude'], hotspot['longitude'],
                businesses_df, max_distance
            )
            
            if len(nearby_businesses) == 0:
                continue
            
            # Score business relevance
            nearby_businesses['relevance_score'] = nearby_businesses.apply(
                lambda x: self.score_business_relevance(x, origin_group), axis=1
            )
            
            # Sort by relevance and distance
            nearby_businesses = nearby_businesses.sort_values(
                ['relevance_score', 'distance_km'], ascending=[False, True]
            )
            
            # Take top businesses
            top_businesses = nearby_businesses.head(max_businesses)
            
            # Create matches
            for _, business in top_businesses.iterrows():
                match = {
                    'tract_geoid': hotspot['tract_geoid'],
                    'hotspot_latitude': hotspot['latitude'],
                    'hotspot_longitude': hotspot['longitude'],
                    'hotspot_probability': hotspot[hotspot_col],
                    'business_name': business['business_name'],
                    'business_category': business['business_category'],
                    'business_latitude': business['latitude'],
                    'business_longitude': business['longitude'],
                    'distance_km': business['distance_km'],
                    'relevance_score': business['relevance_score'],
                    'business_employees': business['employees'],
                    'business_age': business['business_age'],
                    'origin_group': origin_group
                }
                
                # Add cuisine-specific information
                cuisine_score_col = f'{origin_group}_score'
                if cuisine_score_col in business.index:
                    match['cuisine_score'] = business[cuisine_score_col]
                
                matches.append(match)
        
        matches_df = pd.DataFrame(matches)
        logger.info(f"Created {len(matches_df)} hotspot-business matches")
        
        return matches_df
    
    def get_recommendations(self, user_lat: float, user_lon: float, 
                          origin_group: str, max_distance: float = 15.0,
                          max_recommendations: int = 10) -> pd.DataFrame:
        """
        Get business recommendations for a user location.
        
        Args:
            user_lat: User latitude
            user_lon: User longitude
            origin_group: User's origin group
            max_distance: Maximum distance in kilometers
            max_recommendations: Maximum number of recommendations
            
        Returns:
            DataFrame with business recommendations
        """
        logger.info(f"Getting recommendations for {origin_group} at ({user_lat}, {user_lon})")
        
        # Load business data
        businesses_df = self.load_business_data()
        
        # Find nearby businesses
        nearby_businesses = self.find_nearby_businesses(
            user_lat, user_lon, businesses_df, max_distance
        )
        
        if len(nearby_businesses) == 0:
            logger.warning("No nearby businesses found")
            return pd.DataFrame()
        
        # Score business relevance
        nearby_businesses['relevance_score'] = nearby_businesses.apply(
            lambda x: self.score_business_relevance(x, origin_group), axis=1
        )
        
        # Sort by relevance and distance
        recommendations = nearby_businesses.sort_values(
            ['relevance_score', 'distance_km'], ascending=[False, True]
        ).head(max_recommendations)
        
        # Add recommendation metadata
        recommendations['user_latitude'] = user_lat
        recommendations['user_longitude'] = user_lon
        recommendations['origin_group'] = origin_group
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def save_matches(self, matches_df: pd.DataFrame, origin_group: str):
        """Save hotspot-business matches to file."""
        filename = f"hotspot_business_matches_{origin_group}.parquet"
        filepath = self.data_dir / filename
        matches_df.to_parquet(filepath, index=False)
        logger.info(f"Saved matches to {filepath}")
    
    def generate_all_matches(self):
        """Generate matches for all origin groups."""
        logger.info("Generating matches for all origin groups")
        
        all_matches = []
        
        for origin_group in self.origin_groups:
            try:
                matches_df = self.match_hotspots_to_businesses(origin_group)
                if not matches_df.empty:
                    all_matches.append(matches_df)
                    self.save_matches(matches_df, origin_group)
            except Exception as e:
                logger.error(f"Error generating matches for {origin_group}: {e}")
        
        if all_matches:
            # Combine all matches
            combined_matches = pd.concat(all_matches, ignore_index=True)
            
            # Save combined matches
            combined_path = self.data_dir / "all_hotspot_business_matches.parquet"
            combined_matches.to_parquet(combined_path, index=False)
            
            logger.info(f"Generated matches for {len(all_matches)} origin groups")
            logger.info(f"Total matches: {len(combined_matches)}")

def main():
    """Main function to run business matching."""
    matcher = BusinessMatcher()
    
    try:
        # Generate matches for all origin groups
        matcher.generate_all_matches()
        
        # Example: Get recommendations for a user
        user_lat = 37.7749  # San Francisco
        user_lon = -122.4194
        origin_group = 'india'
        
        recommendations = matcher.get_recommendations(
            user_lat, user_lon, origin_group, max_distance=15.0, max_recommendations=10
        )
        
        if not recommendations.empty:
            logger.info("Sample recommendations:")
            for _, rec in recommendations.head(5).iterrows():
                logger.info(f"- {rec['business_name']} ({rec['business_category']}) "
                           f"- {rec['distance_km']:.1f}km - Score: {rec['relevance_score']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in business matching: {e}")

if __name__ == "__main__":
    main()
