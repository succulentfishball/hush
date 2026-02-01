"""
Script to create a names_to_coords.json file that maps station names to their lat/lng coordinates.
"""

import json


def create_names_to_coords():
    # Load the MTA stops cache
    with open("data/mta_stops_cache.json", "r") as f:
        stops_cache = json.load(f)
    
    # Create a mapping of name -> {lat, lng}
    # Use a dict to deduplicate (same station name may appear multiple times with N/S variants)
    names_to_coords = {}
    
    for stop_id, stop_data in stops_cache.items():
        name = stop_data["name"]
        lat = stop_data["lat"]
        lng = stop_data["lng"]
        
        # Only add if not already in the dict (first occurrence wins)
        # This avoids duplicates from N/S variants
        if name not in names_to_coords:
            names_to_coords[name] = {
                "lat": lat,
                "lng": lng
            }
    
    # Sort by name for easier reading
    sorted_names_to_coords = dict(sorted(names_to_coords.items()))
    
    # Write to file
    with open("data/names_to_coords.json", "w") as f:
        json.dump(sorted_names_to_coords, f, indent=2)
    
    print(f"Created names_to_coords.json with {len(sorted_names_to_coords)} unique stations")
    
    # Print a sample
    print("\nSample entries:")
    for i, (name, coords) in enumerate(sorted_names_to_coords.items()):
        if i >= 5:
            break
        print(f"  {name}: ({coords['lat']}, {coords['lng']})")


if __name__ == "__main__":
    create_names_to_coords()
