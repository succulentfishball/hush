"""
Test Google Routes API for MTA Transit Routing
Uses GTFS station IDs and filters for subway/rail only.
"""

import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load .env file (if exists), then check environment variables
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("ROUTES_API_KEY")

# MTA GTFS Stops URL (for getting coordinates from station IDs)
MTA_GTFS_STOPS_URL = "https://data.ny.gov/api/views/39hk-dx4f/rows.csv?accessType=DOWNLOAD"

# Cache for station coordinates
_station_coords_cache = None


def load_station_coordinates():
    """
    Load MTA station coordinates from GTFS data.
    Returns a dict mapping station_id -> (lat, lng, name)
    """
    global _station_coords_cache
    
    if _station_coords_cache is not None:
        return _station_coords_cache
    
    print("📥 Loading MTA station coordinates...")
    
    # Try to load from local cache first
    try:
        with open("mta_stops_cache.json", "r") as f:
            _station_coords_cache = json.load(f)
            print(f"   ✅ Loaded {len(_station_coords_cache)} stations from cache")
            return _station_coords_cache
    except FileNotFoundError:
        pass
    
    # Download fresh GTFS stops data
    try:
        # Use the MTA static GTFS feed
        url = "http://web.mta.info/developers/data/nyct/subway/google_transit.zip"
        print(f"   Downloading from MTA GTFS feed...")
        
        import zipfile
        import io
        import csv
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract stops.txt from the zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open("stops.txt") as f:
                reader = csv.DictReader(io.TextIOWrapper(f, 'utf-8'))
                
                _station_coords_cache = {}
                for row in reader:
                    stop_id = row.get("stop_id", "")
                    lat = float(row.get("stop_lat", 0))
                    lng = float(row.get("stop_lon", 0))
                    name = row.get("stop_name", "")
                    
                    if lat and lng:
                        _station_coords_cache[stop_id] = {
                            "lat": lat,
                            "lng": lng,
                            "name": name
                        }
        
        # Save cache
        with open("mta_stops_cache.json", "w") as f:
            json.dump(_station_coords_cache, f)
        
        print(f"   ✅ Loaded {len(_station_coords_cache)} stations from GTFS")
        return _station_coords_cache
        
    except Exception as e:
        print(f"   ❌ Error loading GTFS: {e}")
        return {}


def get_station_coords(station_id: str) -> dict:
    """
    Get lat/lng coordinates for a GTFS station ID.
    Returns {"latitude": lat, "longitude": lng} or None if not found.
    """
    coords = load_station_coordinates()
    
    # Try exact match first
    if station_id in coords:
        return {
            "latitude": coords[station_id]["lat"],
            "longitude": coords[station_id]["lng"]
        }
    
    # Try without N/S suffix
    base_id = station_id.rstrip("NS")
    if base_id in coords:
        return {
            "latitude": coords[base_id]["lat"],
            "longitude": coords[base_id]["lng"]
        }
    
    # Try with N suffix
    if f"{base_id}N" in coords:
        return {
            "latitude": coords[f"{base_id}N"]["lat"],
            "longitude": coords[f"{base_id}N"]["lng"]
        }
    
    return None


def get_routes_subway_only(origin_id: str, destination_id: str):
    """
    Get transit routes between two MTA stations using Google Routes API.
    Filters for SUBWAY/RAIL only (no buses).
    
    Args:
        origin_id: GTFS station ID (e.g., "127", "R20", "631")
        destination_id: GTFS station ID
        
    Returns:
        List of route dictionaries with transit details
    """
    print("=" * 60)
    print("🚇 Google Routes API - Subway Only")
    print("=" * 60)
    
    # Get coordinates for station IDs
    origin_coords = get_station_coords(origin_id)
    dest_coords = get_station_coords(destination_id)
    
    if not origin_coords:
        print(f"❌ Could not find coordinates for origin station: {origin_id}")
        return []
    
    if not dest_coords:
        print(f"❌ Could not find coordinates for destination station: {destination_id}")
        return []
    
    # Get station names for display
    coords = load_station_coordinates()
    origin_name = coords.get(origin_id, {}).get("name", origin_id)
    dest_name = coords.get(destination_id, {}).get("name", destination_id)
    
    print(f"\n📍 Origin: {origin_name} (ID: {origin_id})")
    print(f"   Coordinates: {origin_coords['latitude']}, {origin_coords['longitude']}")
    print(f"📍 Destination: {dest_name} (ID: {destination_id})")
    print(f"   Coordinates: {dest_coords['latitude']}, {dest_coords['longitude']}")
    print(f"🕐 Departure: Now ({datetime.now().strftime('%H:%M')})")
    
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join([
            "routes.duration",
            "routes.distanceMeters",
            "routes.legs.steps.transitDetails",
            "routes.legs.steps.travelMode",
            "routes.legs.steps.staticDuration"
        ])
    }
    
    body = {
        "origin": {
            "location": {
                "latLng": origin_coords
            }
        },
        "destination": {
            "location": {
                "latLng": dest_coords
            }
        },
        "travelMode": "TRANSIT",
        "computeAlternativeRoutes": True,
        "transitPreferences": {
            # SUBWAY includes metro/subway, RAIL includes commuter rail
            # Excludes BUS, TRAM, etc.
            "allowedTravelModes": ["SUBWAY", "RAIL"]
        }
    }
    
    print("\n⏳ Calling Google Routes API (subway/rail only)...")
    
    try:
        response = requests.post(url, headers=headers, json=body, timeout=15)
        data = response.json()
        
        # Check for errors
        if "error" in data:
            error = data["error"]
            print(f"\n❌ API Error: {error.get('status', 'UNKNOWN')}")
            print(f"   Message: {error.get('message', 'No message')}")
            return []
        
        routes = data.get("routes", [])
        
        if not routes:
            print("❌ No subway/rail routes found")
            return []
        
        print(f"\n✅ Found {len(routes)} route(s)!\n")
        
        # Process and display routes
        processed_routes = []
        
        for i, route in enumerate(routes):
            print(f"{'='*50}")
            print(f"Route {i + 1}")
            print(f"{'='*50}")
            
            # Parse duration (comes as "1234s" string)
            duration_str = route.get("duration", "0s")
            duration_sec = int(duration_str.rstrip("s"))
            duration_min = duration_sec // 60
            
            distance = route.get("distanceMeters", 0)
            
            print(f"  ⏱️  Duration: {duration_min} minutes")
            print(f"  📏 Distance: {distance / 1000:.1f} km ({distance / 1609:.1f} miles)")
            
            # Extract all steps (including walking)
            raw_steps = []
            legs = route.get("legs", [])
            
            for leg in legs:
                steps = leg.get("steps", [])
                
                for step in steps:
                    mode = step.get("travelMode", "?")
                    duration = step.get("staticDuration", "?")
                    
                    # Parse duration
                    dur_sec = int(duration.rstrip("s")) if duration != "?" else 0
                    
                    if mode == "TRANSIT":
                        transit = step.get("transitDetails", {})
                        stopDetails = transit.get("stopDetails", {})
                        departure = stopDetails.get("departureStop", {}).get("name", "?")
                        arrival = stopDetails.get("arrivalStop", {}).get("name", "?")
                        
                        line = transit.get("transitLine", {})
                        line_name = line.get("nameShort") or line.get("name", "?")
                        vehicle_type = line.get("vehicle", {}).get("type", "SUBWAY")
                        
                        num_stops = transit.get("stopCount", "?")
                        
                        raw_steps.append({
                            "type": "transit",
                            "line": line_name,
                            "departure_station": departure,
                            "arrival_station": arrival,
                            "num_stops": num_stops,
                            "duration_seconds": dur_sec,
                            "vehicle_type": vehicle_type
                        })
                        
                    elif mode == "WALK":
                        step_distance = step.get("distanceMeters", 0)
                        raw_steps.append({
                            "type": "walk",
                            "distance_meters": step_distance,
                            "duration_seconds": dur_sec
                        })
            
            # Merge consecutive walks and filter out 0-minute walks
            all_steps = []
            for step in raw_steps:
                if step["type"] == "walk":
                    # Skip 0-minute walks
                    if step["duration_seconds"] < 60:
                        continue
                    # Merge with previous walk if exists
                    if all_steps and all_steps[-1]["type"] == "walk":
                        all_steps[-1]["distance_meters"] += step["distance_meters"]
                        all_steps[-1]["duration_seconds"] += step["duration_seconds"]
                    else:
                        all_steps.append(step)
                else:
                    all_steps.append(step)
            
            # Display cleaned steps
            print(f"\n  📋 Steps:")
            for step_num, step in enumerate(all_steps, 1):
                dur_min = step["duration_seconds"] // 60
                if step["type"] == "transit":
                    print(f"     {step_num}. 🚇 [{step['line']}] {step['departure_station']} → {step['arrival_station']}")
                    print(f"        ({step['num_stops']} stops, {dur_min} min)")
                elif step["type"] == "walk":
                    print(f"     {step_num}. 🚶 Walk {step['distance_meters']}m ({dur_min} min)")
            
            processed_routes.append({
                "duration_minutes": duration_min,
                "distance_meters": distance,
                "steps": all_steps
            })
            
            print()
        
        # Sort by duration and keep only the 3 fastest routes
        processed_routes.sort(key=lambda r: r["duration_minutes"])
        processed_routes = processed_routes[:3]
        
        print(f"🏆 Returning top 3 fastest routes out of {len(routes)} total\n")
        
        # Save full response
        with open("routes_api_response.json", "w") as f:
            json.dump(data, f, indent=2)
        print("💾 Full response saved to routes_api_response.json")
        
        return processed_routes
        
    except requests.RequestException as e:
        print(f"❌ Network Error: {e}")
        return []


def test_api_key():
    """Quick check that we have an API key set."""
    print("\n🔑 Checking API Key...")
    
    if GOOGLE_MAPS_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Please set your Google Maps API key!")
        return False
    
    print(f"✅ API key is set: {GOOGLE_MAPS_API_KEY[:10]}...")
    return True


if __name__ == "__main__":
    print("\n" + "🚇" * 30 + "\n")
    
    if test_api_key():
        # Test with GTFS station IDs
        # Times Square-42 St = various IDs like "127", "725", "902", "R16"
        # Brooklyn Bridge-City Hall = "417", "418", "419", "420"
        
        # Example: Times Square to Brooklyn Bridge
        ORIGIN_STATION_ID = "127"      # Times Square (1/2/3 line)
        DEST_STATION_ID = "420"        # Brooklyn Bridge (4/5/6 line)
        
        routes = get_routes_subway_only(ORIGIN_STATION_ID, DEST_STATION_ID)
        
        if routes:
            print("\n" + "="*50)
            print("📊 TOP 3 FASTEST ROUTES")
            print("="*50)
            for i, r in enumerate(routes):
                transit_count = len([s for s in r['steps'] if s['type'] == 'transit'])
                walk_count = len([s for s in r['steps'] if s['type'] == 'walk'])
                print(f"Route {i+1}: {r['duration_minutes']} min, {transit_count} train(s), {walk_count} walk segment(s)")
    
    print("\n" + "🚇" * 30 + "\n")
