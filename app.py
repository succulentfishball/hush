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
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import GNN modules
from gnn_inference import get_predictor
from congestion_scorer import CongestionScorer

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("ROUTES_API_KEY")

# NYC center coordinates
NYC_CENTER = [40.7580, -73.9855]  # Midtown Manhattan

# ============================================================================
# GNN PREDICTION & CONGESTION SCORING
# ============================================================================

@st.cache_resource
def load_gnn_predictor():
    """Load GNN predictor (cached)."""
    return get_predictor()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_tap_in_predictions():
    """
    Get current tap-in predictions from GNN.
    Randomly samples one entry per station matching the current hour.
    """
    try:
        # Load ridership data
        ridership_df = pd.read_csv("data/MTA_Subway_Hourly_Ridership__2020-2024_20260131.csv",
                                   parse_dates=["transit_timestamp"])
        
        # Get current time
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Extract hour from timestamps
        ridership_df['hour'] = ridership_df['transit_timestamp'].dt.hour
        
        # Filter to matching hour
        matching_hour = ridership_df[ridership_df['hour'] == current_hour]
        
        if len(matching_hour) == 0:
            # Fallback if no data for current hour
            matching_hour = ridership_df
        
        # Clean ridership data
        matching_hour = matching_hour.copy()
        matching_hour['ridership'] = pd.to_numeric(
            matching_hour['ridership'].astype(str).str.replace(',', ''),
            errors='coerce'
        ).fillna(0)
        
        # Randomly sample ONE entry per station_complex_id
        current_ridership = (
            matching_hour
            .groupby('station_complex_id')
            .sample(n=1, random_state=None)  # random_state=None for true randomness each time
            [['station_complex_id', 'ridership']]
            .reset_index(drop=True)
        )
        
        # Run GNN inference
        predictor = load_gnn_predictor()
        predictions = predictor.predict(current_ridership, current_time)
        
        return predictions
    
    except Exception as e:
        st.warning(f"Could not load GNN predictions: {e}")
        return {}


def calculate_route_quiet_scores(routes: list) -> list:
    """
    Add quiet scores to routes based on GNN predictions.
    
    Args:
        routes: List of route dicts
    
    Returns:
        Routes with 'quiet_score' field populated
    """
    predictions = get_tap_in_predictions()
    
    if not predictions:
        # If no predictions, return routes with default scores
        for route in routes:
            route['quiet_score'] = 5
        return routes
    
    scorer = CongestionScorer(predictions)
    
    for route in routes:
        quiet_score = scorer.calculate_route_quiet_score(route)
        route['quiet_score'] = quiet_score
    
    return routes

# ============================================================================
# STATION DATA
# ============================================================================

@st.cache_data
def load_station_coordinates():
    """Load MTA station coordinates from cached GTFS data."""
    try:
        with open("data/mta_stops_cache.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@st.cache_data
def load_names_to_coords():
    """Load station name to coordinates mapping."""
    try:
        with open("data/names_to_coords.json", "r") as f:
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


def find_station_coords_by_name(name: str) -> dict:
    """Find station coordinates by name using names_to_coords.json."""
    names_to_coords = load_names_to_coords()
    name_stripped = name.strip()
    
    # First try exact match
    if name_stripped in names_to_coords:
        data = names_to_coords[name_stripped]
        return {"lat": data["lat"], "lng": data["lng"], "name": name_stripped}
    
    # Try case-insensitive exact match
    name_lower = name_stripped.lower()
    for station_name, data in names_to_coords.items():
        if station_name.lower() == name_lower:
            return {"lat": data["lat"], "lng": data["lng"], "name": station_name}
    
    # Try partial match
    for station_name, data in names_to_coords.items():
        station_lower = station_name.lower()
        if name_lower in station_lower or station_lower in name_lower:
            return {"lat": data["lat"], "lng": data["lng"], "name": station_name}
    
    # Try matching without suffixes like "St", "Ave", etc.
    name_clean = name_lower.replace(" st", "").replace(" ave", "").replace(" sq", "")
    for station_name, data in names_to_coords.items():
        station_clean = station_name.lower().replace(" st", "").replace(" ave", "").replace(" sq", "")
        if name_clean in station_clean or station_clean in name_clean:
            return {"lat": data["lat"], "lng": data["lng"], "name": station_name}
    
    return None


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
        top_routes = unique_routes[:3]
        
        # Calculate quiet scores
        top_routes = calculate_route_quiet_scores(top_routes)
        
        return top_routes, None
        
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
        --accent-yellow: #ffd700;
        --accent-yellow-dim: rgba(255, 215, 0, 0.15);
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
    
    /* Best route styling */
    .route-card-best {
        border-color: var(--accent-green) !important;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.15);
        position: relative;
    }
    
    .route-card-best::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 12px;
        background: radial-gradient(ellipse at top, rgba(0, 255, 136, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .best-route-badge {
        position: absolute;
        top: -10px;
        right: 20px;
        background: var(--accent-green);
        color: #000;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        z-index: 10;
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
        background: var(--accent-yellow-dim);
        color: var(--accent-yellow);
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
    
    /* Prediction time banner */
    .prediction-banner {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(0, 255, 136, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 16px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .prediction-icon {
        font-size: 1.5rem;
    }
    
    .prediction-text {
        flex: 1;
    }
    
    .prediction-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--accent-purple);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    
    .prediction-time {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .prediction-hint {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        margin-top: 2px;
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
    
    /* Animated Loading State */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 20px;
        gap: 16px;
    }
    
    .loading-dots {
        display: flex;
        gap: 8px;
    }
    
    .loading-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--accent-green);
        animation: loadingPulse 1.4s ease-in-out infinite;
    }
    
    .loading-dot:nth-child(1) {
        animation-delay: 0s;
    }
    
    .loading-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .loading-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes loadingPulse {
        0%, 80%, 100% {
            transform: scale(0.6);
            opacity: 0.4;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .loading-text {
        color: var(--text-secondary);
        font-size: 0.9rem;
        animation: loadingTextPulse 2s ease-in-out infinite;
    }
    
    @keyframes loadingTextPulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .loading-subtext {
        color: var(--text-tertiary);
        font-size: 0.75rem;
        margin-top: -8px;
    }
    
    .loading-train {
        font-size: 1.5rem;
        animation: trainMove 2s ease-in-out infinite;
    }
    
    @keyframes trainMove {
        0% { transform: translateX(-20px); }
        50% { transform: translateX(20px); }
        100% { transform: translateX(-20px); }
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_route_card(route: dict, index: int, is_best: bool = False):
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
    
    # Quiet score badge - emphasize it's a prediction
    if quiet_score is not None:
        if quiet_score >= 7:
            quiet_html = f'<span class="quiet-badge-good">• Quiet {quiet_score}/10</span>'
        elif quiet_score >= 4:
            quiet_html = f'<span class="quiet-badge-pending">• Moderate {quiet_score}/10</span>'
        else:
            quiet_html = f'<span class="quiet-badge-bad">• Busy {quiet_score}/10</span>'
    else:
        quiet_html = '<span class="quiet-badge-pending">○ Score pending</span>'
    
    # Render the card - single line to avoid whitespace issues
    best_class = ' route-card-best' if is_best else ''
    best_badge = '<span class="best-route-badge">✨ QUIETEST</span>' if is_best else ''
    card_html = f'<div class="route-card{best_class}">{best_badge}<div class="route-header"><div style="display: flex; align-items: center;"><span class="duration-badge">{duration} min</span><span class="route-meta">{distance:.1f} km · {transfer_text}</span></div>{quiet_html}</div>{steps_html}</div>'
    
    st.markdown(card_html, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Hush",
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
            <div class="app-title">Hush</div>
            <div class="app-subtitle">Take the road less travelled: Predict subway congestion for the next hour.</div>
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
                
                # Show custom loading animation
                loading_placeholder = st.empty()
                loading_placeholder.markdown("""
                <div class="loading-container">
                    <div class="loading-train">🚇</div>
                    <div class="loading-dots">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                    <div class="loading-text">Analyzing routes...</div>
                    <div class="loading-subtext">Predicting congestion for the next hour</div>
                </div>
                """, unsafe_allow_html=True)
                
                routes, error = get_routes(origin_id, dest_id, coords)
                
                # Clear loading animation
                loading_placeholder.empty()
                
                if error:
                    st.markdown(f"""
                    <div class="error-card">❌ {error}</div>
                    """, unsafe_allow_html=True)
                elif routes:
                    now = datetime.now()
                    one_hour_later = now + timedelta(hours=1)
                    
                    # Prominent prediction time banner
                    st.markdown(f"""
                    <div class="prediction-banner">
                        <span class="prediction-icon">🔮</span>
                        <div class="prediction-text">
                            <div class="prediction-label">AI Congestion Forecast</div>
                            <div class="prediction-time">{now.strftime('%H:%M')} → {one_hour_later.strftime('%H:%M')}</div>
                            <div class="prediction-hint">Scores reflect predicted crowding for the next hour</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Find best route (highest quiet score)
                    best_route = max(routes, key=lambda r: r.get('quiet_score', 0) or 0)
                    best_idx = routes.index(best_route)
                    
                    # Store in session state for map
                    st.session_state['best_route'] = best_route
                    st.session_state['routes_found'] = True
                    
                    st.markdown(f"""
                    <div class="results-header">
                        {len(routes)} route{'s' if len(routes) > 1 else ''} found · Route {best_idx + 1} recommended
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, route in enumerate(routes):
                        render_route_card(route, i, is_best=(i == best_idx))
                else:
                    st.markdown("""
                    <div class="error-card">No routes found</div>
                    """, unsafe_allow_html=True)
    
    with right_col:
        # Interactive NYC Subway Map
        origin_data = stations.get(origin_name, {})
        dest_data = stations.get(destination_name, {})
        
        # Build station markers
        markers = []
        
        # Origin marker - always at selected station
        if origin_data.get("lat") and origin_data.get("lng"):
            markers.append({
                "name": origin_name,
                "lat": origin_data["lat"],
                "lon": origin_data["lng"],
                "color": [0, 255, 136, 255],
                "radius": 100,
                "type": "origin"
            })
        
        # Destination marker - always at selected station
        if dest_data.get("lat") and dest_data.get("lng"):
            markers.append({
                "name": destination_name,
                "lat": dest_data["lat"],
                "lon": dest_data["lng"],
                "color": [255, 68, 68, 255],
                "radius": 100,
                "type": "destination"
            })
        
        # Intermediate stations from best route
        intermediate_stations = []
        best_route = st.session_state.get('best_route')
        
        if best_route and st.session_state.get('routes_found'):
            transit_steps = [s for s in best_route.get('steps', []) if s['type'] == 'transit']
            seen_coords = set()  # Track by actual coordinates to avoid duplicates
            
            for step in transit_steps:
                # Add departure station if not start/end
                dep_name = step['departure']
                if dep_name.lower() != origin_name.lower() and dep_name.lower() != destination_name.lower():
                    dep_coords = find_station_coords_by_name(dep_name)
                    if dep_coords:
                        # Use coordinates as key to detect true duplicates
                        coord_key = (round(dep_coords["lat"], 5), round(dep_coords["lng"], 5))
                        if coord_key not in seen_coords:
                            intermediate_stations.append({
                                "name": dep_coords["name"],  # Use matched name
                                "lat": dep_coords["lat"],
                                "lon": dep_coords["lng"],
                                "color": [250, 204, 21, 255],  # Yellow
                                "radius": 100,
                                "type": "intermediate"
                            })
                            seen_coords.add(coord_key)
                
                # Add arrival station if not start/end
                arr_name = step['arrival']
                if arr_name.lower() != origin_name.lower() and arr_name.lower() != destination_name.lower():
                    arr_coords = find_station_coords_by_name(arr_name)
                    if arr_coords:
                        # Use coordinates as key to detect true duplicates
                        coord_key = (round(arr_coords["lat"], 5), round(arr_coords["lng"], 5))
                        if coord_key not in seen_coords:
                            intermediate_stations.append({
                                "name": arr_coords["name"],  # Use matched name
                                "lat": arr_coords["lat"],
                                "lon": arr_coords["lng"],
                                "color": [250, 204, 21, 255],  # Yellow
                                "radius": 100,
                                "type": "intermediate"
                            })
                            seen_coords.add(coord_key)
        
        # Calculate map center
        all_points = markers + intermediate_stations
        if all_points:
            center_lat = sum(m["lat"] for m in all_points) / len(all_points)
            center_lon = sum(m["lon"] for m in all_points) / len(all_points)
        else:
            center_lat, center_lon = 40.7580, -73.9855
        
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=12,
            pitch=0,
            bearing=0
        )
        
        layers = []
        
        # Intermediate stations layer (yellow dots)
        if intermediate_stations:
            intermediate_layer = pdk.Layer(
                "ScatterplotLayer",
                data=intermediate_stations,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius="radius",
                pickable=True
            )
            layers.append(intermediate_layer)
        
        # Main markers layer (origin/destination)
        if markers:
            main_markers = pdk.Layer(
                "ScatterplotLayer",
                data=markers,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius="radius",
                pickable=True
            )
            layers.append(main_markers)
        
        # Render map
        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                tooltip={
                    "html": "<b>{name}</b>",
                    "style": {
                        "backgroundColor": "#111111",
                        "color": "white",
                        "border": "1px solid #333",
                        "borderRadius": "8px",
                        "padding": "8px 12px"
                    }
                }
            ),
            height=700
        )
        
        # Map legend
        st.markdown("""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 12px; font-size: 0.75rem; color: rgba(255,255,255,0.6);">
            <span><span style="display: inline-block; width: 10px; height: 10px; background: #00ff88; border-radius: 50%; margin-right: 6px;"></span>Starting Station</span>
            <span><span style="display: inline-block; width: 10px; height: 10px; background: #ff4444; border-radius: 50%; margin-right: 6px;"></span>Final Station</span>
            <span><span style="display: inline-block; width: 10px; height: 10px; background: #facc15; border-radius: 50%; margin-right: 6px;"></span>Intermediate Station</span>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
