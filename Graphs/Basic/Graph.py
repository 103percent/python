import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# === Step 1: Load spreadsheet ===
edges_df = pd.read_excel("graph_data.xlsx", sheet_name="Edges")
nodes_df = pd.read_excel("graph_data.xlsx", sheet_name="Nodes")  # Optional

# === Step 2: Create graph ===
G = nx.DiGraph()  # Use nx.Graph() for undirected

# Add nodes with attributes
if not nodes_df.empty:
    for _, row in nodes_df.iterrows():
        G.add_node(row["Node"], label=row.get("Label", ""), 
                   group=row.get("Group", 0),
                   color=row.get("Color", "lightblue"),
                   size=row.get("Size", 300))

# Add edges with attributes
for _, row in edges_df.iterrows():
    G.add_edge(row["Source"], row["Target"],
               label=row.get("Label", ""),
               weight=row.get("Weight", 1.0),
               color=row.get("Color", "black"),
               style=row.get("Style", "solid"))

# === Step 3: Draw the graph ===
pos = nx.spring_layout(G)  # or try circular_layout, shell_layout

# Node styling
node_colors = [G.nodes[n].get("color", "lightblue") for n in G.nodes]
node_sizes = [G.nodes[n].get("size", 300) for n in G.nodes]

# Edge styling
edge_colors = [G[u][v].get("color", "black") for u, v in G.edges]
edge_styles = [G[u][v].get("style", "solid") for u, v in G.edges]
edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges]

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

# Draw edges
for style in set(edge_styles):
    edge_subset = [(u, v) for u, v in G.edges if G[u][v].get("style") == style]
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_subset,
        edge_color=[G[u][v]["color"] for u, v in edge_subset],
        width=[G[u][v]["weight"] for u, v in edge_subset],
        style=style
    )

# Labels
nx.draw_networkx_labels(G, pos)
edge_labels = {(u, v): G[u][v].get("label", "") for u, v in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.axis("off")
plt.tight_layout()
plt.show()
