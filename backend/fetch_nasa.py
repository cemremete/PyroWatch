"\"\"\"
NASA FIRMS Data Fetcher
Fetches real-time fire data from NASA's Fire Information for Resource Management System.

Supported Data Sources:
- VIIRS SNPP (Suomi National Polar-orbiting Partnership)
- VIIRS NOAA-20/21
- MODIS (Aqua/Terra satellites)

API Documentation: https://firms.modaps.eosdis.nasa.gov/api/
\"\"\"

import aiohttp
import asyncio
import csv
from io import StringIO
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

# NASA FIRMS API Configuration
NASA_API_KEY = os.environ.get('NASA_API_KEY', 'DEMO_KEY')
FIRMS_BASE_URL = \"https://firms.modaps.eosdis.nasa.gov/api\"

# Rate limiting: 5000 transactions per 10 minutes
MAX_REQUESTS_PER_WINDOW = 5000
WINDOW_SECONDS = 600


class FIRMSDataFetcher:
    \"\"\"
    Async data fetcher for NASA FIRMS fire data.
    \"\"\"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NASA_API_KEY
        self.request_count = 0
        self.window_start = datetime.now()
    
    def _check_rate_limit(self) -> bool:
        \"\"\"Check if we're within rate limits.\"\"\"
        elapsed = (datetime.now() - self.window_start).total_seconds()
        
        if elapsed > WINDOW_SECONDS:
            # Reset window
            self.window_start = datetime.now()
            self.request_count = 0
            return True
        
        if self.request_count >= MAX_REQUESTS_PER_WINDOW:
            logger.warning(\"Rate limit reached. Please wait before making more requests.\")
            return False
        
        return True
    
    def _parse_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        \"\"\"Parse a CSV row into a fire record.\"\"\"
        try:
            # Handle different column names from different data sources
            lat = float(row.get('latitude', 0))
            lon = float(row.get('longitude', 0))
            
            # Brightness temperature (different column names)
            brightness = row.get('bright_ti4') or row.get('brightness') or row.get('bright_ti5')
            brightness = float(brightness) if brightness else 350.0
            
            # Confidence value
            confidence = row.get('confidence', 'nominal')
            
            # Fire Radiative Power
            frp = row.get('frp')
            frp = float(frp) if frp else None
            
            return {
                'latitude': lat,
                'longitude': lon,
                'brightness': brightness,
                'confidence': confidence,
                'satellite': row.get('satellite', 'VIIRS'),
                'instrument': row.get('instrument', 'VIIRS'),
                'acq_date': row.get('acq_date', datetime.now().strftime('%Y-%m-%d')),
                'acq_time': row.get('acq_time', '0000'),
                'frp': frp,
                'daynight': row.get('daynight', 'D'),
                'scan': float(row.get('scan', 0)) if row.get('scan') else None,
                'track': float(row.get('track', 0)) if row.get('track') else None,
                'version': row.get('version', '2.0'),
            }
        except Exception as e:
            logger.warning(f\"Error parsing row: {e}\")
            return None
    
    async def fetch_global_fires(self, days: int = 1) -> List[Dict[str, Any]]:
        \"\"\"
        Fetch global fire data.
        
        Args:
            days: Number of days to fetch (1-10)
            
        Returns:
            List of fire records
        \"\"\"
        if not self._check_rate_limit():
            return []
        
        url = f\"{FIRMS_BASE_URL}/area/csv/{self.api_key}/VIIRS_SNPP_NRT/world/{days}\"
        
        return await self._fetch_data(url)
    
    async def fetch_country_fires(self, country_code: str, days: int = 1) -> List[Dict[str, Any]]:
        \"\"\"
        Fetch fire data for a specific country.
        
        Args:
            country_code: ISO 3166-1 alpha-3 country code (e.g., 'USA', 'AUS', 'TUR')
            days: Number of days to fetch
            
        Returns:
            List of fire records
        \"\"\"
        if not self._check_rate_limit():
            return []
        
        url = f\"{FIRMS_BASE_URL}/country/csv/{self.api_key}/VIIRS_SNPP_NRT/{country_code}/{days}\"
        
        return await self._fetch_data(url)
    
    async def fetch_area_fires(self, 
                               min_lat: float, min_lon: float,
                               max_lat: float, max_lon: float,
                               days: int = 1) -> List[Dict[str, Any]]:
        \"\"\"
        Fetch fire data for a bounding box area.
        
        Args:
            min_lat, min_lon: Southwest corner coordinates
            max_lat, max_lon: Northeast corner coordinates
            days: Number of days to fetch
            
        Returns:
            List of fire records
        \"\"\"
        if not self._check_rate_limit():
            return []
        
        coords = f\"{min_lon},{min_lat},{max_lon},{max_lat}\"
        url = f\"{FIRMS_BASE_URL}/area/csv/{self.api_key}/VIIRS_SNPP_NRT/world/{days}/{coords}\"
        
        return await self._fetch_data(url)
    
    async def _fetch_data(self, url: str) -> List[Dict[str, Any]]:
        \"\"\"
        Internal method to fetch and parse CSV data.
        
        Args:
            url: FIRMS API URL
            
        Returns:
            List of fire records
        \"\"\"
        fires = []
        
        async with aiohttp.ClientSession() as session:
            try:
                self.request_count += 1
                logger.info(f\"Fetching: {url}\")
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        text = await response.text()
                        
                        # Parse CSV
                        reader = csv.DictReader(StringIO(text))
                        
                        for row in reader:
                            fire = self._parse_row(row)
                            if fire:
                                fires.append(fire)
                        
                        logger.info(f\"Fetched {len(fires)} fire records\")
                    
                    elif response.status == 429:
                        logger.error(\"Rate limit exceeded. Please wait before retrying.\")
                    
                    else:
                        text = await response.text()
                        logger.error(f\"API error {response.status}: {text[:200]}\")
            
            except asyncio.TimeoutError:
                logger.error(\"Request timed out\")
            except Exception as e:
                logger.error(f\"Fetch error: {e}\")
        
        return fires
    
    async def fetch_multiple_sources(self, days: int = 1) -> List[Dict[str, Any]]:
        \"\"\"
        Fetch from multiple satellite sources.
        
        Args:
            days: Number of days to fetch
            
        Returns:
            Combined list of fire records from all sources
        \"\"\"
        all_fires = []
        
        sources = [
            f\"{FIRMS_BASE_URL}/area/csv/{self.api_key}/VIIRS_SNPP_NRT/world/{days}\",
            f\"{FIRMS_BASE_URL}/area/csv/{self.api_key}/VIIRS_NOAA20_NRT/world/{days}\",
        ]
        
        async with aiohttp.ClientSession() as session:
            for url in sources:
                if not self._check_rate_limit():
                    break
                
                try:
                    self.request_count += 1
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                        if response.status == 200:
                            text = await response.text()
                            reader = csv.DictReader(StringIO(text))
                            
                            for row in reader:
                                fire = self._parse_row(row)
                                if fire:
                                    all_fires.append(fire)
                except Exception as e:
                    logger.error(f\"Error fetching from {url}: {e}\")
        
        logger.info(f\"Total fires from all sources: {len(all_fires)}\")
        return all_fires


def parse_confidence(conf_str: str) -> float:
    \"\"\"
    Convert NASA confidence string to numeric value.
    
    Args:
        conf_str: Confidence level ('high', 'nominal', 'low' or numeric)
        
    Returns:
        Numeric confidence (0-100)
    \"\"\"
    conf_map = {
        'high': 90, 'h': 90,
        'nominal': 60, 'n': 60,
        'low': 30, 'l': 30
    }
    
    if conf_str.lower() in conf_map:
        return conf_map[conf_str.lower()]
    
    try:
        return float(conf_str)
    except ValueError:
        return 50.0


async def main():
    \"\"\"Example usage of the FIRMS data fetcher.\"\"\"
    logging.basicConfig(level=logging.INFO)
    
    fetcher = FIRMSDataFetcher()
    
    # Fetch global fires
    fires = await fetcher.fetch_global_fires(days=1)
    print(f\"Fetched {len(fires)} global fires\")
    
    if fires:
        print(\"
Sample fire record:\")
        print(fires[0])


if __name__ == '__main__':
    asyncio.run(main())
"