
import os
import pandas as pd
import networkx as nx
import json

def convert_fma_genres():
    # Define paths
    base_dir = "/home/acolombo/music"
    input_csv = os.path.join(base_dir, "data/fma_metadata/genres.csv")
    output_dir = os.path.join(base_dir, "hyperbolic_tree_embeddings/tree_embeddings/data/fma_metadata")
    output_json = os.path.join(output_dir, "fma_metadata.json")
    output_edges = os.path.join(output_dir, "fma_metadata.edges")
    output_mapping = os.path.join(output_dir, "fma_metadata_mapping.json")

    print(f"Reading from {input_csv}...")
    df = pd.read_csv(input_csv)

    # create a list of all unique genre_ids
    unique_ids = df['genre_id'].unique()
    num_genres = len(unique_ids)
    print(f"Found {num_genres} unique genres.")

    # Create mapping: old_id -> new_id
    # We reserve 0 for the virtual root
    # So new_ids will be 1 to num_genres
    # However, if we want the root to be 0 for the algorithm, we should assign 0 to our virtual root.
    
    id_map = {}
    current_new_id = 1
    for old_id in unique_ids:
        id_map[old_id] = current_new_id
        current_new_id += 1
    
    # Virtual root map
    # We will use 0 as the virtual root ID in the new graph
    
    print("Building graph with relabeled nodes...")
    G = nx.DiGraph()
    G.add_node(0) # Virtual root

    for index, row in df.iterrows():
        old_node_id = int(row['genre_id'])
        old_parent_id = int(row['parent'])
        
        new_node_id = id_map[old_node_id]
        
        if old_parent_id != 0:
            if old_parent_id in id_map:
                new_parent_id = id_map[old_parent_id]
                G.add_edge(new_parent_id, new_node_id)
            else:
                print(f"Warning: Parent {old_parent_id} for node {old_node_id} not found in dataset. Attaching to virtual root?")
                # If parent not found, attach to root? Or skip edge?
                # For safety, let's attach to virtual root 0 so it's connected
                print(f"Attaching orphaned node {new_node_id} (old {old_node_id}) to virtual root 0.")
                G.add_edge(0, new_node_id)
                G.add_edge(0, new_node_id)
        else:
            # Node is a top-level node
            # Connect to virtual root 0
            G.add_edge(0, new_node_id)
        
        # Add title attribute
        G.nodes[new_node_id]["title"] = row['title']

    # Add title for virtual root
    G.nodes[0]["title"] = "Root"

    print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print("Checking if graph is a tree/forest...")
    if not nx.is_tree(G):
        print("Graph is not a tree (might be a dag or have cycles).")
    
    # Ensure contiguous IDs (0 to N-1)
    # We used 0 for root, and 1..N for others. This is contiguous.
    assert sorted(list(G.nodes())) == list(range(len(G.nodes()))), "Node IDs are not contiguous!"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Save mapping
    print(f"Saving mapping to {output_mapping}...")
    # Save as string keys for JSON compatibility just in case, or int keys logic
    # JSON keys must be strings
    json_map = {str(k): v for k, v in id_map.items()}
    with open(output_mapping, "w") as f:
        json.dump(json_map, f, indent=4)

    # Save as NetworkX node-link JSON
    print(f"Saving to {output_json}...")
    json_data = nx.node_link_data(G)
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=4)

    # Save as edges list for reference/compatibility
    print(f"Saving to {output_edges}...")
    with open(output_edges, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

    print("Conversion complete.")

if __name__ == "__main__":
    convert_fma_genres()
