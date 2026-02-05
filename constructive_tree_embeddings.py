from itertools import product
import os
import matplotlib.pyplot as plt
import networkx as nx

import pandas as pd

import torch

from tree_embeddings.embeddings.constructive_method import constructively_embed_tree
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



def run_constructive_tree_embeddings(
    dataset: str,
    hierarchy_name: str,
    root: int = 0,
    gen_type: str = "optim",
    tau: float = 1.0,
    embedding_dim: int = 20,
    nc: int = 1,
    curvature: float = 1.0,
    dtype: torch.dtype = torch.float64,
):
    # Load hierarchy and turn to directed if necessary
    hierarchy = load_hierarchy(dataset=dataset, hierarchy_name=hierarchy_name)
    if not nx.is_directed(hierarchy):
        # Store edge weights, turn into directed graph and reassign weights
        edge_data = {
            (source, target): data
            for source, target, data in nx.DiGraph(hierarchy).edges(data=True)
        }
        hierarchy = nx.bfs_tree(hierarchy, root)
        nx.set_edge_attributes(hierarchy, edge_data)

    embeddings, rel_dist_mean, rel_dist_max = constructively_embed_tree(
        hierarchy=hierarchy,
        dataset=dataset,
        hierarchy_name=hierarchy_name,
        embedding_dim=embedding_dim,
        tau=tau,
        nc=nc,
        curvature=curvature,
        root=root,
        gen_type=gen_type,
        dtype=dtype,
    )

    res_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "constructive_method",
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
        pos = hierarchy_pos(hierarchy, root)
    except Exception:
        pos = nx.spring_layout(hierarchy)
    
    nx.draw(hierarchy, pos, with_labels=True, node_size=20, arrowsize=5, font_size=8)
    plt.savefig(os.path.join(res_dir, "hierarchy.png"))
    plt.close()

    # Plot and save hyperbolic embeddings (Poincaré disk)
    if embeddings.shape[1] >= 2:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # Draw the unit circle
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
        ax.add_artist(circle)
        
        # Get coordinates
        x_coords = embeddings[:, 0].numpy()
        y_coords = embeddings[:, 1].numpy()
        
        # Plot nodes
        plt.scatter(x_coords, y_coords, s=20, c='blue', alpha=0.7)
        
        # Draw edges
        for u, v in hierarchy.edges():
            # hierarchy nodes might be integers 0..N matching embeddings indices
            # verifying this assumption: constructive_tree_embeddings_w_floats.py:L48 hierarchy_embeddings[root] = 0
            # and generic usage seems to map nodes to indices directly if they are integers 0-N.
            # However, if hierarchy nodes are not integers 0..N, we might need a mapping.
            # load_hierarchy usually returns nodes as integers if possible or strings. 
            # constructive_tree_embeddings assumes "currently assume nodes are labeled 0-N" (L47 in _w_floats).
            # So I will assume u and v are valid indices.
            if u < len(embeddings) and v < len(embeddings):
                p1 = embeddings[u].numpy()
                p2 = embeddings[v].numpy()
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=0.5, alpha=0.5)
        
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Hyperbolic Embeddings (2D Projection, dim={embeddings.shape[1]})")
        plt.savefig(os.path.join(res_dir, "embeddings_poincare.png"))
        plt.close()
