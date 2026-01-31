import requests
import streamlit as st
from datetime import datetime
from typing import Dict
from urllib.parse import quote

from .config import TFL_API_BASE, TFL_APP_KEY


def get_api_params() -> dict:
    """Get API parameters including key if available."""
    if TFL_APP_KEY:
        return {"app_key": TFL_APP_KEY}
    return {}


def search_station(query: str) -> list[dict]:
    """Search for stations matching a query string using StopPoint Search."""
    url = f"{TFL_API_BASE}/StopPoint/Search/{query}"
    params = get_api_params()
    params.update({
        "modes": "tube,dlr,overground,elizabeth-line,national-rail,bus",
        "maxResults": 10,
        "faresOnly": "false"
    })

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Format results for our UI
        results = []
        for match in data.get("matches", []):
            results.append({
                "id": match.get("icsId") or match.get("id", ""),
                "name": match.get("name", ""),
                "zone": match.get("zone", "")
            })
        return results
    except requests.RequestException as e:
        st.error(f"Error searching stations: {e}")
        return []


def get_journey_options(from_station: str, to_station: str, time: datetime = None) -> list[dict]:
    """
    Fetch journey options from TfL Journey Planner API.
    Makes multiple requests with different preferences to get route variety.
    """
    from_encoded = quote(from_station, safe='')
    to_encoded = quote(to_station, safe='')

    all_journeys = []
    
    # Try different journey preferences to get variety
    preferences = ["LeastTime", "LeastInterchange", "LeastWalking"]
    
    for preference in preferences:
        url = f"{TFL_API_BASE}/Journey/JourneyResults/{from_encoded}/to/{to_encoded}"
        params = get_api_params()
        params.update({
            "mode": "tube,dlr,overground,elizabeth-line,bus,national-rail",
            "journeyPreference": preference,
            "alternativeCycle": "true",
            "alternativeWalking": "true",
            "maxTransferMinutes": "10",
            "maxWalkingMinutes": "20"
        })

        if time:
            params["time"] = time.strftime("%H%M")
            params["date"] = time.strftime("%Y%m%d")
            params["timeIs"] = "Departing"

        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()

            # Check if TfL returns disambiguation options
            if "toLocationDisambiguation" in data or "fromLocationDisambiguation" in data:
                to_options = data.get("toLocationDisambiguation", {}).get("disambiguationOptions", [])
                from_options = data.get("fromLocationDisambiguation", {}).get("disambiguationOptions", [])

                new_from = from_station
                new_to = to_station

                if from_options and len(from_options) > 0:
                    place = from_options[0].get("place", {})
                    new_from = place.get("icsCode") or place.get("commonName") or from_station

                if to_options and len(to_options) > 0:
                    place = to_options[0].get("place", {})
                    new_to = place.get("icsCode") or place.get("commonName") or to_station

                if new_from != from_station or new_to != to_station:
                    from_encoded = quote(str(new_from), safe='')
                    to_encoded = quote(str(new_to), safe='')
                    url = f"{TFL_API_BASE}/Journey/JourneyResults/{from_encoded}/to/{to_encoded}"
                    response = requests.get(url, params=params, timeout=15)
                    data = response.json()

            response.raise_for_status()
            journeys = data.get("journeys", [])
            
            # Add journeys to our collection, avoiding duplicates
            for journey in journeys[:3]:  # Take top 3 from each preference
                # Create a simple hash to identify unique routes
                leg_summary = tuple(
                    (leg.get("mode", {}).get("name", ""), 
                     leg.get("routeOptions", [{}])[0].get("name", ""))
                    for leg in journey.get("legs", [])
                )
                
                # Check if we already have this route
                is_duplicate = any(
                    tuple(
                        (leg.get("mode", {}).get("name", ""), 
                         leg.get("routeOptions", [{}])[0].get("name", ""))
                        for leg in j.get("legs", [])
                    ) == leg_summary
                    for j in all_journeys
                )
                
                if not is_duplicate:
                    all_journeys.append(journey)
                    
        except requests.RequestException as e:
            st.error(f"Error fetching journeys with {preference}: {e}")
            continue
    
    # Return up to 5 unique routes
    return all_journeys[:5]


def get_crowding_data(naptan_id: str) -> dict:
    """
    Get crowding/capacity data for a station.
    Returns hourly crowding percentages.
    """
    url = f"{TFL_API_BASE}/crowding/{naptan_id}"
    params = get_api_params()

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 404:
            return {"percentageOfBaseline": 0.5}
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"percentageOfBaseline": 0.5}


def get_line_status() -> dict[str, dict]:
    """
    Get current status for all tube/rail lines.
    Returns a dict of line_id -> status info.
    """
    url = f"{TFL_API_BASE}/Line/Mode/tube,dlr,overground,elizabeth-line/Status"
    params = get_api_params()

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        status_map: dict[str, dict] = {}
        for line in data:
            line_id = line.get("id", "")
            statuses = line.get("lineStatuses", [])

            has_severe = any(
                s.get("statusSeverity", 10) <= 5
                for s in statuses
            )

            status_desc = statuses[0].get("statusSeverityDescription", "Unknown") if statuses else "Unknown"
            reason = statuses[0].get("reason", "") if statuses else ""

            status_map[line_id] = {
                "has_severe_delays": has_severe,
                "status": status_desc,
                "reason": reason
            }

        return status_map
    except requests.RequestException as e:
        st.warning(f"Could not fetch line status: {e}")
        return {}


def get_crowding_for_time(naptan_id: str, arrival_time: datetime) -> float:
    """
    Get the crowding level for a specific station at a specific time.
    Returns a value between 0.0 (empty) and 1.0 (packed).
    """
    crowding_data = get_crowding_data(naptan_id)

    if "timeBands" in crowding_data:
        hour = arrival_time.hour
        for band in crowding_data.get("timeBands", []):
            if band.get("hour") == hour:
                return min(band.get("percentageOfBaseline", 50) / 100, 1.0)

    if "percentageOfBaseline" in crowding_data:
        return min(crowding_data["percentageOfBaseline"] / 100, 1.0)

    hour = arrival_time.hour
    day = arrival_time.weekday()

    if day >= 5:
        return 0.4
    elif 7 <= hour <= 9 or 17 <= hour <= 19:
        return 0.85
    elif 10 <= hour <= 16:
        return 0.5
    else:
        return 0.3
