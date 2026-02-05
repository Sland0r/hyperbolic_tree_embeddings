
import sys
import os
import networkx as nx

# Add project root to path so we can import tree_embeddings
sys.path.append("/home/acolombo/music/hyperbolic_tree_embeddings")

try:
    from tree_embeddings.trees.file_utils import load_hierarchy
except ImportError:
    # Fallback if running from a different location, though sbatch cwd should handle it
    print("Could not import tree_embeddings from current path.")
    sys.exit(1)

def verify_fma_loading():
    print("Attempting to load FMA hierarchy...")
    try:
        # load_hierarchy expects dataset name (folder) and hierarchy name (filename without .json)
        tree = load_hierarchy("fma_metadata", "fma_metadata")
        print(f"Successfully loaded tree with {tree.number_of_nodes()} nodes and {tree.number_of_edges()} edges.")
        
        is_tree = nx.is_tree(tree)
        is_dag = nx.is_directed_acyclic_graph(tree)
        print(f"Is Tree: {is_tree}")
        print(f"Is DAG: {is_dag}")
        
        if not is_dag:
            print("Warning: Graph is not a DAG (cycles detected?)")
            
    except Exception as e:
        print(f"Failed to load hierarchy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_fma_loading()
