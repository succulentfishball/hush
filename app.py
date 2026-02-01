"""
MTA Sensory-Safe Router
A smart routing app for NYC subway with quiet score ratings (coming soon).
"""

import streamlit as st
import requests
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

GOOGLE_MAPS_API_KEY = "AIzaSyAUHfitfmxyCC4zwWsP-A876BmXhyFpaWM"

# ============================================================================
# STATION DATA
# ============================================================================

@st.cache_data
def load_station_coordinates():
    """Load MTA station coordinates from cached GTFS data."""
    try:
        with open("mta_stops_cache.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Station data not found. Please run test.py first to download station data.")
        return {}


def get_station_list():
    """Get a sorted list of unique station names with their IDs."""
    coords = load_station_coordinates()
    
    # Create a dict of name -> list of IDs (some stations have multiple IDs)
    stations = {}
    for station_id, data in coords.items():
        name = data.get("name", "")
        # Skip directional variants (N/S suffixes)
        if station_id.endswith("N") or station_id.endswith("S"):
            continue
        if name and name not in stations:
            stations[name] = station_id
    
    # Sort by name
    return dict(sorted(stations.items()))


def get_station_coords(station_id: str, coords: dict) -> dict:
    """Get lat/lng coordinates for a GTFS station ID."""
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


# ============================================================================
# ROUTING
# ============================================================================

def get_routes(origin_id: str, destination_id: str, coords: dict):
    """
    Get transit routes between two MTA stations using Google Routes API.
    Returns top 3 fastest routes with subway/rail only.
    """
    origin_coords = get_station_coords(origin_id, coords)
    dest_coords = get_station_coords(destination_id, coords)
    
    if not origin_coords or not dest_coords:
        return None, "Could not find coordinates for stations"
    
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join([
            "routes.duration",
            "routes.distanceMeters",
            "routes.legs.steps.transitDetails",
            "routes.legs.steps.travelMode",
            "routes.legs.steps.staticDuration",
            "routes.legs.steps.distanceMeters"
        ])
    }
    
    body = {
        "origin": {"location": {"latLng": origin_coords}},
        "destination": {"location": {"latLng": dest_coords}},
        "travelMode": "TRANSIT",
        "computeAlternativeRoutes": True,
        "transitPreferences": {
            "allowedTravelModes": ["SUBWAY", "RAIL"]
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=body, timeout=15)
        data = response.json()
        
        if "error" in data:
            return None, data["error"].get("message", "API Error")
        
        routes = data.get("routes", [])
        if not routes:
            return None, "No subway routes found"
        
        # Process routes
        processed_routes = []
        
        for route in routes:
            duration_str = route.get("duration", "0s")
            duration_sec = int(duration_str.rstrip("s"))
            duration_min = duration_sec // 60
            distance = route.get("distanceMeters", 0)
            
            # Extract steps
            raw_steps = []
            for leg in route.get("legs", []):
                for step in leg.get("steps", []):
                    mode = step.get("travelMode", "?")
                    duration = step.get("staticDuration", "?")
                    dur_sec = int(duration.rstrip("s")) if duration != "?" else 0
                    
                    if mode == "TRANSIT":
                        transit = step.get("transitDetails", {})
                        stopDetails = transit.get("stopDetails", {})
                        departure = stopDetails.get("departureStop", {}).get("name", "?")
                        arrival = stopDetails.get("arrivalStop", {}).get("name", "?")
                        line = transit.get("transitLine", {})
                        line_name = line.get("nameShort") or line.get("name", "?")
                        line_color = line.get("color", "#888888")
                        num_stops = transit.get("stopCount", "?")
                        
                        raw_steps.append({
                            "type": "transit",
                            "line": line_name,
                            "color": line_color,
                            "departure": departure,
                            "arrival": arrival,
                            "num_stops": num_stops,
                            "duration_min": dur_sec // 60
                        })
                    
                    elif mode == "WALK":
                        step_dist = step.get("distanceMeters", 0)
                        raw_steps.append({
                            "type": "walk",
                            "distance_m": step_dist,
                            "duration_min": dur_sec // 60
                        })
            
            # Merge consecutive walks and filter 0-minute walks
            steps = []
            for step in raw_steps:
                if step["type"] == "walk":
                    if step["duration_min"] < 1:
                        continue
                    if steps and steps[-1]["type"] == "walk":
                        steps[-1]["distance_m"] += step["distance_m"]
                        steps[-1]["duration_min"] += step["duration_min"]
                    else:
                        steps.append(step)
                else:
                    steps.append(step)
            
            processed_routes.append({
                "duration_min": duration_min,
                "distance_km": distance / 1000,
                "steps": steps,
                "quiet_score": None  # Placeholder for later
            })
        
        # Sort by duration and return top 3
        processed_routes.sort(key=lambda r: r["duration_min"])
        return processed_routes[:3], None
        
    except requests.RequestException as e:
        return None, f"Network error: {e}"


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_route_card(route: dict, index: int):
    """Render a single route as a card."""
    duration = route["duration_min"]
    distance = route["distance_km"]
    steps = route["steps"]
    quiet_score = route.get("quiet_score")
    
    # Count transfers
    transit_count = len([s for s in steps if s["type"] == "transit"])
    transfers = max(0, transit_count - 1)
    
    # Card container
    with st.container():
        # Header row
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### 🚇 Route {index + 1}")
        
        with col2:
            st.metric("Duration", f"{duration} min")
        
        with col3:
            if quiet_score is not None:
                st.metric("Quiet Score", f"{quiet_score}/10")
            else:
                st.markdown("**Quiet Score**")
                st.caption("Coming soon...")
        
        # Route summary
        transfer_text = "Direct" if transfers == 0 else f"{transfers} transfer{'s' if transfers > 1 else ''}"
        st.caption(f"📏 {distance:.1f} km  •  🔄 {transfer_text}")
        
        # Steps
        st.markdown("---")
        
        for i, step in enumerate(steps):
            if step["type"] == "transit":
                line = step["line"]
                color = step.get("color", "#888888")
                
                # Create a colored badge for the line
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin: 8px 0;">
                        <span style="
                            background-color: {color};
                            color: white;
                            padding: 4px 12px;
                            border-radius: 20px;
                            font-weight: bold;
                            margin-right: 10px;
                            min-width: 50px;
                            text-align: center;
                        ">{line}</span>
                        <span>{step['departure']} → {step['arrival']}</span>
                        <span style="color: #666; margin-left: auto;">
                            {step['num_stops']} stops • {step['duration_min']} min
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            elif step["type"] == "walk":
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin: 8px 0; color: #666;">
                        <span style="margin-right: 10px;">🚶</span>
                        <span>Walk {step['distance_m']}m ({step['duration_min']} min)</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="MTA Sensory-Safe Router",
        page_icon="🚇",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
        }
        .route-card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🚇 MTA Sensory-Safe Router")
    st.markdown("*Find the quietest routes through NYC's subway system*")
    st.markdown("---")
    
    # Load station data
    coords = load_station_coordinates()
    stations = get_station_list()
    
    if not stations:
        st.error("Failed to load station data. Please ensure mta_stops_cache.json exists.")
        return
    
    station_names = list(stations.keys())
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Origin")
        origin_name = st.selectbox(
            "Select departure station",
            station_names,
            index=station_names.index("Times Sq-42 St") if "Times Sq-42 St" in station_names else 0,
            key="origin"
        )
    
    with col2:
        st.subheader("🎯 Destination")
        destination_name = st.selectbox(
            "Select arrival station",
            station_names,
            index=station_names.index("Bowling Green") if "Bowling Green" in station_names else 1,
            key="destination"
        )
    
    # Search button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Find Routes", type="primary", use_container_width=True):
        if origin_name == destination_name:
            st.warning("Please select different stations for origin and destination.")
            return
        
        origin_id = stations[origin_name]
        dest_id = stations[destination_name]
        
        with st.spinner("Finding the best routes..."):
            routes, error = get_routes(origin_id, dest_id, coords)
        
        if error:
            st.error(f"❌ {error}")
            return
        
        if not routes:
            st.warning("No routes found between these stations.")
            return
        
        # Display results
        st.markdown("---")
        st.subheader(f"🗺️ Routes from {origin_name} to {destination_name}")
        st.caption(f"Found {len(routes)} route{'s' if len(routes) > 1 else ''} • {datetime.now().strftime('%I:%M %p')}")
        
        # Render route cards
        for i, route in enumerate(routes):
            with st.container(border=True):
                render_route_card(route, i)
    
    # Footer
    st.markdown("---")
    st.caption("💡 Quiet scores will rate routes based on predicted crowding and sensory factors.")


if __name__ == "__main__":
    main()
