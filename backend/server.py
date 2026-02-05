from fastapi import FastAPI, APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os, logging, uuid, asyncio, csv, aiohttp
from pathlib import Path
from io import StringIO
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
from datetime import datetime, timezone, timedelta

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB setup
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# NASA FIRMS API - using default key if none provided
NASA_API_KEY = os.environ.get('NASA_API_KEY', 'e609a7bece9ffff29998ea539d309914')
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"

# Initialize FastAPI
app = FastAPI(title="PyroWatch API", description="Wildfire Detection and Monitoring System")
api_router = APIRouter(prefix="/api")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= Data Models =============

class FireRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    latitude: float
    longitude: float
    confidence: float  # 0-100%
    brightness: float  # Kelvin
    satellite: str
    acq_date: str
    acq_time: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frp: Optional[float] = None  # Fire Radiative Power
    daynight: Optional[str] = None
    scan: Optional[float] = None
    track: Optional[float] = None

class FireResponse(BaseModel):
    id: str
    latitude: float
    longitude: float
    confidence: float
    brightness: float
    satellite: str
    acq_date: str
    acq_time: str
    detected_at: str
    frp: Optional[float] = None
    daynight: Optional[str] = None

class AlertSubscription(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    latitude: float
    longitude: float
    radius_km: float = 50.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True

class AlertCreate(BaseModel):
    email: EmailStr
    latitude: float
    longitude: float
    radius_km: float = 50.0

class StatsResponse(BaseModel):
    total_fires: int
    high_confidence: int
    medium_confidence: int
    low_confidence: int
    active_fires_24h: int
    last_updated: str

# ============= Helper Functions =============

def parse_confidence(conf_str: str) -> float:
    """Convert FIRMS confidence text to percentage"""
    if conf_str.lower() in ['high', 'h']:
        return 90.0
    elif conf_str.lower() in ['nominal', 'n']:
        return 60.0
    elif conf_str.lower() in ['low', 'l']:
        return 30.0
    
    # Fallback - try to parse as number
    try:
        return float(conf_str)
    except:
        return 50.0

async def fetch_nasa_firms_data(region: str = "world", days: int = 1) -> List[dict]:
    """Fetch real fire data from NASA FIRMS API"""
    fires = []
    
    # Try multiple data sources for better coverage
    urls = [
        f"{FIRMS_BASE_URL}/area/csv/{NASA_API_KEY}/VIIRS_SNPP_NRT/world/{days}",
        f"{FIRMS_BASE_URL}/country/csv/{NASA_API_KEY}/VIIRS_SNPP_NRT/USA/{days}",
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                print(f"Fetching from: {url}")
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        text = await response.text()
                        reader = csv.DictReader(StringIO(text))
                        
                        for row in reader:
                            try:
                                fire = {
                                    'latitude': float(row.get('latitude', 0)),
                                    'longitude': float(row.get('longitude', 0)),
                                    'brightness': float(row.get('bright_ti4', 0) or row.get('brightness', 300)),
                                    'confidence': parse_confidence(row.get('confidence', 'nominal')),
                                    'satellite': row.get('satellite', 'VIIRS'),
                                    'acq_date': row.get('acq_date', datetime.now().strftime('%Y-%m-%d')),
                                    'acq_time': row.get('acq_time', '0000'),
                                    'frp': float(row.get('frp', 0)) if row.get('frp') else None,
                                    'daynight': row.get('daynight', 'D'),
                                    'scan': float(row.get('scan', 0)) if row.get('scan') else None,
                                    'track': float(row.get('track', 0)) if row.get('track') else None,
                                }
                                fires.append(fire)
                            except Exception as e:
                                logger.warning(f"Error parsing row: {e}")
                                continue
                    else:
                        logger.error(f"NASA FIRMS API error: {response.status}")
            except Exception as e:
                logger.error(f"Error fetching FIRMS data: {e}")
    
    print(f"Total fires fetched: {len(fires)}")
    return fires

async def store_fires_in_db(fires: List[dict]):
    """Store fire records in MongoDB with geospatial indexing"""
    if not fires:
        return 0
    
    # Ensure indexes exist for performance
    try:
        await db.fires.create_index([("location", "2dsphere")])
        await db.fires.create_index([("detected_at", -1)])
        await db.fires.create_index([("confidence", -1)])
    except Exception as e:
        logger.warning(f"Index creation failed (might exist): {e}")
    
    stored = 0
    for fire in fires:
        fire_doc = {
            'id': str(uuid.uuid4()),
            'latitude': fire['latitude'],
            'longitude': fire['longitude'],
            'location': {
                'type': 'Point',
                'coordinates': [fire['longitude'], fire['latitude']]
            },
            'confidence': fire['confidence'],
            'brightness': fire['brightness'],
            'satellite': fire['satellite'],
            'acq_date': fire['acq_date'],
            'acq_time': fire['acq_time'],
            'detected_at': datetime.now(timezone.utc).isoformat(),
            'frp': fire.get('frp'),
            'daynight': fire.get('daynight'),
            'scan': fire.get('scan'),
            'track': fire.get('track'),
        }
        
        # Check for duplicates (same location within 0.01 degrees)
        existing = await db.fires.find_one({
            'latitude': {'$gte': fire['latitude'] - 0.01, '$lte': fire['latitude'] + 0.01},
            'longitude': {'$gte': fire['longitude'] - 0.01, '$lte': fire['longitude'] + 0.01},
            'acq_date': fire['acq_date']
        })
        
        if not existing:
            await db.fires.insert_one(fire_doc)
            stored += 1
    
    logger.info(f"Stored {stored} new fire records")
    return stored

# ============= API Endpoints =============

@api_router.get("/")
async def root():
    return {"message": "PyroWatch API - Wildfire Detection System", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@api_router.get("/fires/recent", response_model=List[FireResponse])
async def get_recent_fires(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back (max 168 = 7 days)"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of fires to return"),
    min_confidence: float = Query(0, ge=0, le=100, description="Minimum confidence level")
):
    """Get fires detected in the last N hours"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    query = {
        'detected_at': {'$gte': cutoff.isoformat()},
        'confidence': {'$gte': min_confidence}
    }
    
    fires = await db.fires.find(query, {"_id": 0}).sort("detected_at", -1).limit(limit).to_list(limit)
    
    # Convert datetime objects to strings for JSON response
    for fire in fires:
        if isinstance(fire.get('detected_at'), datetime):
            fire['detected_at'] = fire['detected_at'].isoformat()
    
    return fires

@api_router.get("/fires/active", response_model=List[FireResponse])
async def get_active_fires(
    limit: int = Query(500, ge=1, le=5000, description="Maximum number of fires"),
    min_confidence: float = Query(50, ge=0, le=100, description="Minimum confidence for active fires")
):
    """Get currently active fires (last 6 hours with high confidence)"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
    
    query = {
        'detected_at': {'$gte': cutoff.isoformat()},
        'confidence': {'$gte': min_confidence}
    }
    
    fires = await db.fires.find(query, {"_id": 0}).sort("confidence", -1).limit(limit).to_list(limit)
    
    for fire in fires:
        if isinstance(fire.get('detected_at'), datetime):
            fire['detected_at'] = fire['detected_at'].isoformat()
    
    return fires

@api_router.get("/fires/location")
async def get_fires_by_location(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    radius_km: float = Query(100, ge=1, le=1000, description="Search radius in kilometers")
):
    """Get fires within a radius of a specific location using geospatial query"""
    # Convert km to radians (Earth radius ≈ 6371 km)
    radius_radians = radius_km / 6371
    
    query = {
        'location': {
            '$geoWithin': {
                '$centerSphere': [[lon, lat], radius_radians]
            }
        }
    }
    
    fires = await db.fires.find(query, {"_id": 0}).limit(1000).to_list(1000)
    
    for fire in fires:
        if isinstance(fire.get('detected_at'), datetime):
            fire['detected_at'] = fire['detected_at'].isoformat()
    
    return {"fires": fires, "count": len(fires), "center": {"lat": lat, "lon": lon}, "radius_km": radius_km}

@api_router.get("/fires/stats", response_model=StatsResponse)
async def get_fire_stats():
    """Get aggregated statistics about fires"""
    now = datetime.now(timezone.utc)
    day_ago = (now - timedelta(hours=24)).isoformat()
    
    total = await db.fires.count_documents({})
    high_conf = await db.fires.count_documents({'confidence': {'$gte': 80}})
    med_conf = await db.fires.count_documents({'confidence': {'$gte': 50, '$lt': 80}})
    low_conf = await db.fires.count_documents({'confidence': {'$lt': 50}})
    active_24h = await db.fires.count_documents({'detected_at': {'$gte': day_ago}})
    
    return StatsResponse(
        total_fires=total,
        high_confidence=high_conf,
        medium_confidence=med_conf,
        low_confidence=low_conf,
        active_fires_24h=active_24h,
        last_updated=now.isoformat()
    )

@api_router.post("/alerts", response_model=AlertSubscription)
async def create_alert(alert: AlertCreate):
    """Subscribe to fire alerts for a specific location"""
    alert_obj = AlertSubscription(
        email=alert.email,
        latitude=alert.latitude,
        longitude=alert.longitude,
        radius_km=alert.radius_km
    )
    
    doc = alert_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.alerts.insert_one(doc)
    
    return alert_obj

@api_router.get("/alerts", response_model=List[AlertSubscription])
async def get_alerts(email: Optional[str] = None):
    """Get all alert subscriptions, optionally filtered by email"""
    query = {}
    if email:
        query['email'] = email
    
    alerts = await db.alerts.find(query, {"_id": 0}).to_list(100)
    return alerts

@api_router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    """Delete an alert subscription"""
    result = await db.alerts.delete_one({'id': alert_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert deleted successfully"}

@api_router.post("/fires/fetch")
async def fetch_and_store_fires(background_tasks: BackgroundTasks, days: int = Query(1, ge=1, le=7)):
    """Trigger fetching of new fire data from NASA FIRMS"""
    background_tasks.add_task(fetch_and_store_task, days)
    return {"message": "Fire data fetch initiated", "days": days}

async def fetch_and_store_task(days: int):
    """Background task to fetch and store fire data"""
    try:
        logger.info(f"Fetching NASA FIRMS data for {days} days...")
        fires = await fetch_nasa_firms_data(days=days)
        logger.info(f"Fetched {len(fires)} fire records from NASA")
        
        if fires:
            stored = await store_fires_in_db(fires)
            logger.info(f"Stored {stored} new fire records in database")
    except Exception as e:
        logger.error(f"Error in fetch_and_store_task: {e}")

@api_router.get("/fires/history")
async def get_fire_history(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    limit: int = Query(5000, ge=1, le=10000)
):
    """Get historical fire data for a date range"""
    query = {
        'acq_date': {'$gte': start_date, '$lte': end_date}
    }
    
    fires = await db.fires.find(query, {"_id": 0}).limit(limit).to_list(limit)
    
    for fire in fires:
        if isinstance(fire.get('detected_at'), datetime):
            fire['detected_at'] = fire['detected_at'].isoformat()
    
    return {"fires": fires, "count": len(fires), "start_date": start_date, "end_date": end_date}

@api_router.get("/fires/heatmap")
async def get_heatmap_data(
    hours: int = Query(24, ge=1, le=168),
    resolution: float = Query(0.5, ge=0.1, le=2.0, description="Grid resolution in degrees")
):
    """Get aggregated fire data for heatmap visualization"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    fires = await db.fires.find(
        {'detected_at': {'$gte': cutoff.isoformat()}},
        {"_id": 0, "latitude": 1, "longitude": 1, "confidence": 1, "brightness": 1}
    ).to_list(10000)
    
    # Calculate intensity for heatmap
    heatmap_points = []
    for fire in fires:
        intensity = (fire['confidence'] / 100) * (fire.get('brightness', 300) / 400)
        heatmap_points.append({
            'lat': fire['latitude'],
            'lng': fire['longitude'],
            'intensity': min(intensity, 1.0)
        })
    
    return {"points": heatmap_points, "count": len(heatmap_points)}

# Include router and setup middleware
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database indexes on startup"""
    logger.info("PyroWatch API starting up...")
    try:
        await db.fires.create_index([("location", "2dsphere")])
        await db.fires.create_index([("detected_at", -1)])
        await db.fires.create_index([("confidence", -1)])
        await db.fires.create_index([("acq_date", 1)])
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
    finally:
        logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
