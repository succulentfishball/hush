"""
MTA Sensory-Safe Router
A smart routing app for NYC subway with quiet score ratings.
Glassmorphism design with map overlay.
"""

import streamlit as st
import requests
import json
import os
import pydeck as pdk
from datetime import datetime
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("ROUTES_API_KEY")

# NYC center coordinates
NYC_CENTER = [40.7580, -73.9855]  # Midtown Manhattan

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
        return {}


def get_station_list():
    """Get a sorted list of unique station names with their IDs."""
    coords = load_station_coordinates()
    
    stations = {}
    for station_id, data in coords.items():
        name = data.get("name", "")
        if station_id.endswith("N") or station_id.endswith("S"):
            continue
        if name and name not in stations:
            stations[name] = {
                "id": station_id,
                "lat": data.get("lat"),
                "lng": data.get("lng")
            }
    
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
        
        # Deduplicate routes based on the sequence of transit lines used
        def get_route_signature(route):
            """Create a unique signature based on the lines used."""
            transit_steps = [s for s in route["steps"] if s["type"] == "transit"]
            return tuple((s["line"], s["departure"], s["arrival"]) for s in transit_steps)
        
        seen_signatures = set()
        unique_routes = []
        for route in processed_routes:
            sig = get_route_signature(route)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                unique_routes.append(route)
        
        # Sort by duration and return top 3
        unique_routes.sort(key=lambda r: r["duration_min"])
        return unique_routes[:3], None
        
    except requests.RequestException as e:
        return None, f"Network error: {e}"


# ============================================================================
# LINEAR AESTHETIC CSS
# ============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Root variables */
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #111111;
        --bg-elevated: #161616;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-default: rgba(255, 255, 255, 0.1);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.6);
        --text-tertiary: rgba(255, 255, 255, 0.4);
        --accent-green: #00ff88;
        --accent-green-dim: rgba(0, 255, 136, 0.15);
        --accent-red: #ff4444;
        --accent-red-dim: rgba(255, 68, 68, 0.15);
        --accent-purple: #8b5cf6;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Pure black background */
    .stApp {
        background: var(--bg-primary);
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Card with glow effect */
    .linear-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        position: relative;
        transition: all 0.2s ease;
    }
    
    .linear-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 12px;
        background: radial-gradient(ellipse at top, rgba(0, 255, 136, 0.03) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .linear-card:hover {
        border-color: var(--border-default);
        background: var(--bg-elevated);
    }
    
    /* Search container */
    .search-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    
    /* Map container with glow */
    .map-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        height: 85vh;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        position: relative;
        overflow: hidden;
    }
    
    .map-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        height: 50%;
        background: radial-gradient(ellipse, rgba(0, 255, 136, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    
    /* Typography */
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        color: var(--text-tertiary);
        margin-bottom: 24px;
    }
    
    .section-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    /* Override Streamlit selectbox */
    .stSelectbox > div > div {
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--accent-green) !important;
        box-shadow: 0 0 0 1px var(--accent-green) !important;
    }
    
    .stSelectbox label {
        color: var(--text-secondary) !important;
    }
    
    /* Primary button - neon green */
    .stButton > button {
        background: var(--accent-green) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: #00cc6a !important;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Route card */
    .route-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.15s ease;
        position: relative;
    }
    
    .route-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.3), transparent);
        opacity: 0;
        transition: opacity 0.15s ease;
    }
    
    .route-card:hover {
        border-color: var(--border-default);
        background: var(--bg-elevated);
    }
    
    .route-card:hover::before {
        opacity: 1;
    }
    
    /* Duration badge */
    .duration-badge {
        background: var(--bg-primary);
        color: var(--text-primary);
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
        border: 1px solid var(--border-default);
        display: inline-block;
    }
    
    /* Quiet score - neon green for good, red for bad */
    .quiet-badge-good {
        background: var(--accent-green-dim);
        color: var(--accent-green);
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid rgba(0, 255, 136, 0.2);
    }
    
    .quiet-badge-bad {
        background: var(--accent-red-dim);
        color: var(--accent-red);
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid rgba(255, 68, 68, 0.2);
    }
    
    .quiet-badge-pending {
        background: transparent;
        color: var(--text-tertiary);
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.8rem;
        border: 1px solid var(--border-subtle);
    }
    
    /* Line badge */
    .line-badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 36px;
    }
    
    /* Step row */
    .step-row {
        display: flex;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid var(--border-subtle);
        color: var(--text-primary);
        font-size: 0.875rem;
    }
    
    .step-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    
    .step-details {
        flex: 1;
        margin-left: 12px;
        color: var(--text-secondary);
    }
    
    .step-meta {
        color: var(--text-tertiary);
        font-size: 0.8rem;
    }
    
    /* Walk icon */
    .walk-icon {
        width: 36px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        color: var(--text-tertiary);
    }
    
    /* Route header */
    .route-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .route-meta {
        color: var(--text-tertiary);
        font-size: 0.8rem;
        margin-left: 12px;
    }
    
    /* Results header */
    .results-header {
        color: var(--text-tertiary);
        font-size: 0.8rem;
        margin: 16px 0 8px 0;
    }
    
    /* Error state */
    .error-card {
        background: var(--accent-red-dim);
        border: 1px solid rgba(255, 68, 68, 0.2);
        border-radius: 8px;
        padding: 16px;
        color: var(--accent-red);
        text-align: center;
    }
    
    /* Warning/spinner overrides */
    .stSpinner > div {
        border-color: var(--accent-green) !important;
    }
    
    .stWarning {
        background: var(--bg-elevated) !important;
        color: var(--text-secondary) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_route_card(route: dict, index: int):
    """Render a route card with Linear aesthetic."""
    duration = route["duration_min"]
    distance = route["distance_km"]
    steps = route["steps"]
    quiet_score = route.get("quiet_score")
    
    transit_count = len([s for s in steps if s["type"] == "transit"])
    transfers = max(0, transit_count - 1)
    transfer_text = "Direct" if transfers == 0 else f"{transfers} transfer{'s' if transfers > 1 else ''}"
    
    # Build steps HTML - no extra whitespace/newlines
    steps_html = ""
    for step in steps:
        if step["type"] == "transit":
            line = step["line"]
            color = step.get("color", "#888888")
            steps_html += f'<div class="step-row"><span class="line-badge" style="background-color: {color};">{line}</span><span class="step-details">{step["departure"]} → {step["arrival"]}</span><span class="step-meta">{step["num_stops"]} stops · {step["duration_min"]}m</span></div>'
        elif step["type"] == "walk":
            steps_html += f'<div class="step-row"><span class="walk-icon">→</span><span class="step-details">Walk {step["distance_m"]}m</span><span class="step-meta">{step["duration_min"]}m</span></div>'
    
    # Quiet score badge
    if quiet_score is not None:
        if quiet_score >= 7:
            quiet_html = f'<span class="quiet-badge-good">● Quiet {quiet_score}/10</span>'
        elif quiet_score >= 4:
            quiet_html = f'<span class="quiet-badge-pending">○ Moderate {quiet_score}/10</span>'
        else:
            quiet_html = f'<span class="quiet-badge-bad">● Busy {quiet_score}/10</span>'
    else:
        quiet_html = '<span class="quiet-badge-pending">○ Score pending</span>'
    
    # Render the card - single line to avoid whitespace issues
    card_html = f'<div class="route-card"><div class="route-header"><div style="display: flex; align-items: center;"><span class="duration-badge">{duration} min</span><span class="route-meta">{distance:.1f} km · {transfer_text}</span></div>{quiet_html}</div>{steps_html}</div>'
    
    st.markdown(card_html, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Quiet Routes",
        page_icon="🚇",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    inject_custom_css()
    
    # Load data
    coords = load_station_coordinates()
    stations = get_station_list()
    
    if not stations:
        st.markdown("""
        <div class="error-card">Failed to load station data.</div>
        """, unsafe_allow_html=True)
        return
    
    station_names = list(stations.keys())
    
    # Two-column layout
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        # Header
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <div class="app-title">Quiet Routes</div>
            <div class="app-subtitle">Navigate NYC subway with less stress</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Search inputs
        st.markdown('<div class="section-label">From</div>', unsafe_allow_html=True)
        origin_name = st.selectbox(
            "Origin",
            station_names,
            index=station_names.index("Times Sq-42 St") if "Times Sq-42 St" in station_names else 0,
            key="origin",
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="section-label" style="margin-top: 16px;">To</div>', unsafe_allow_html=True)
        destination_name = st.selectbox(
            "Destination",
            station_names,
            index=station_names.index("Bowling Green") if "Bowling Green" in station_names else 1,
            key="destination",
            label_visibility="collapsed"
        )
        
        # Search button
        search_clicked = st.button("Find routes", use_container_width=True)
        
        # Results
        if search_clicked:
            if origin_name == destination_name:
                st.warning("Select different stations")
            else:
                origin_id = stations[origin_name]["id"]
                dest_id = stations[destination_name]["id"]
                
                with st.spinner(""):
                    routes, error = get_routes(origin_id, dest_id, coords)
                
                if error:
                    st.markdown(f"""
                    <div class="error-card">❌ {error}</div>
                    """, unsafe_allow_html=True)
                elif routes:
                    st.markdown(f"""
                    <div class="results-header">
                        {len(routes)} route{'s' if len(routes) > 1 else ''} · {datetime.now().strftime('%H:%M')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, route in enumerate(routes):
                        render_route_card(route, i)
                else:
                    st.markdown("""
                    <div class="error-card">No routes found</div>
                    """, unsafe_allow_html=True)
    
    with right_col:
        # Interactive NYC Subway Map
        # Get origin and destination coordinates for markers
        origin_data = stations.get(origin_name, {})
        dest_data = stations.get(destination_name, {})
        
        # Create marker data
        markers = []
        if origin_data.get("lat") and origin_data.get("lng"):
            markers.append({
                "name": origin_name,
                "lat": origin_data["lat"],
                "lon": origin_data["lng"],
                "color": [0, 255, 136, 200],  # Green
                "type": "origin"
            })
        if dest_data.get("lat") and dest_data.get("lng"):
            markers.append({
                "name": destination_name,
                "lat": dest_data["lat"],
                "lon": dest_data["lng"],
                "color": [255, 68, 68, 200],  # Red
                "type": "destination"
            })
        
        # Calculate map center
        if markers:
            center_lat = sum(m["lat"] for m in markers) / len(markers)
            center_lon = sum(m["lon"] for m in markers) / len(markers)
            zoom = 12
        else:
            center_lat, center_lon = 40.7580, -73.9855  # NYC default
            zoom = 11
        
        # Create pydeck map
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom,
            pitch=0
        )
        
        # Station markers layer
        marker_layer = pdk.Layer(
            "ScatterplotLayer",
            data=markers,
            get_position=["lon", "lat"],
            get_color="color",
            get_radius=150,
            pickable=True
        )
        
        # Render map with Carto dark basemap (free, no API key needed)
        st.pydeck_chart(
            pdk.Deck(
                layers=[marker_layer],
                initial_view_state=view_state,
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                tooltip={"text": "{name}"}
            ),
            height=700
        )


if __name__ == "__main__":
    main()
