
import os
import random
import matplotlib.pyplot as plt
import networkx as nx
import torch

from tree_embeddings.embeddings.h_mds import h_mds
from tree_embeddings.trees.file_utils import load_hierarchy

def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under CC-BY-SA 4.0.

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)
    root: the root node of current branch
    width: horizontal space allocated for this branch - avoids overlap with other branches
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)

        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def run_h_mds_embeddings(
    dataset: str,
    hierarchy_name: str,
    root: int = 0,
    tau: float = 1.0,
    embedding_dim: int = 20,
):
    # Load hierarchy and turn to directed if necessary
    graph = load_hierarchy(dataset=dataset, hierarchy_name=hierarchy_name)
    if not graph.is_directed():
        graph = nx.bfs_tree(graph, root)
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")

    embeddings, _, _ = h_mds(
        graph=graph,
        dataset=dataset,
        graph_name=hierarchy_name,
        embedding_dim=embedding_dim,
        tau=tau,
        root=root,
    )

    res_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "h_mds",
        dataset,
        hierarchy_name,
    )
    os.makedirs(res_dir, exist_ok=True)
    torch.save(
        embeddings,
        os.path.join(res_dir, "embeddings.pt"),
    )
    
    # Plot and save hierarchy
    plt.figure(figsize=(10, 10))
    try:
        # Re-verify logic for root; assumption is root is 0 after relabeling or passed arg
        # The graph was converted to integers sorted, so root might not be 'root' arg if it was not 0?
        # But convert_node_labels_to_integers with ordering='sorted' maps sorted nodes to 0..N.
        # If root was 0 and nodes were 0..N, it maps 0->0.
        # h_mds assumes root=root.
        pos = hierarchy_pos(graph, root)
    except Exception:
        pos = nx.spring_layout(graph)
    
    nx.draw(graph, pos, with_labels=True, node_size=20, arrowsize=5, font_size=8)
    plt.savefig(os.path.join(res_dir, "hierarchy.png"))
    plt.close()
    
    # Plot and save hyperbolic embeddings (Poincaré disk) manual plot 
    # (h_mds.py already calls plot_embeddings, but constructive_tree_embeddings.py has this extra manual plot)
    # mirroring constructive_tree_embeddings.py exactly:
    if embedding_dim >= 2:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # Draw the unit circle
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
        ax.add_artist(circle)
        
        # Get coordinates
        # embeddings is likely numpy array from h_mds (it returns np.ndarray)
        # convert to numpy if tensor, h_mds returns np.ndarray usually?
        # h_mds.py line 32: -> np.ndarray.
        print(embeddings.shape)
        x_coords = embeddings[:, 0]
        y_coords = embeddings[:, 1]
        
        # Plot nodes
        plt.scatter(x_coords, y_coords, s=20, c='blue', alpha=0.7)
        
        # Draw edges
        for u, v in graph.edges():
            if u < len(embeddings) and v < len(embeddings):
                p1 = embeddings[u]
                p2 = embeddings[v]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=0.5, alpha=0.5)
        
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Hyperbolic Embeddings (2D Projection, dim={embeddings.shape[1]})")
        
        # Add labels for children of root
        try:
            if root in graph:
                children = list(graph.successors(root))
                for child in children:
                    idx = child
                    if idx < len(embeddings):
                        x = embeddings[idx, 0]
                        y = embeddings[idx, 1]
                        label = graph.nodes[child].get('title', str(child))
                        plt.annotate(
                            label, 
                            (x, y), 
                            fontsize=8, 
                            alpha=0.8,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6)
                        )
        except Exception as e:
            print(f"Failed to add labels: {e}")

        plt.savefig(os.path.join(res_dir, "embeddings_poincare.png"))
        plt.close()


