import pandas as pd
import networkx as nx

# Load GTFS static data (downloaded MTA GTFS ZIP)
stops = pd.read_csv("stops.txt")
stop_times = pd.read_csv("stop_times.txt")
trips = pd.read_csv("trips.txt")

# Build mapping from trip_id → ordered list of stops
stop_times = stop_times.sort_values(["trip_id","stop_sequence"])
trip_stops = stop_times.groupby("trip_id")["stop_id"].apply(list)

# Build graph
G = nx.Graph()

# Add nodes from stops.txt
for idx, row in stops.iterrows():
    G.add_node(row["stop_id"], name=row["stop_name"])

# For each trip, add edges between consecutive stops
for stops_list in trip_stops:
    for a, b in zip(stops_list[:-1], stops_list[1:]):
        G.add_edge(a, b)

# Write nodes to nodes.py
nodes_data = {}
for node_id, attrs in G.nodes(data=True):
    nodes_data[node_id] = attrs.get("name", "Unknown")

with open("nodes.py", "w") as f:
    f.write("# MTA Graph Nodes (Auto-generated)\n")
    f.write("NODES = {\n")
    for node_id, name in sorted(nodes_data.items()):
        f.write(f'    "{node_id}": "{name}",\n')
    f.write("}\n")

# Write edges to edges.py
edges_list = list(G.edges())

with open("edges.py", "w") as f:
    f.write("# MTA Graph Edges (Auto-generated)\n")
    f.write("EDGES = [\n")
    for a, b in sorted(edges_list):
        f.write(f'    ("{a}", "{b}"),\n')
    f.write("]\n")

print(f"Stations (nodes): {len(G.nodes())}, Connections (edges): {len(G.edges())}")
print("✓ Written nodes to nodes.py")
print("✓ Written edges to edges.py")
