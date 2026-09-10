"""Map trajectory windows and sampled points onto a common pseudotime scale."""
from __future__ import annotations

import time
import dictys
import pandas as pd
from dictys.net import stat
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx


class AlignTimeScales:
    """
    standardize the window and sampled points time-scales to S0 or S1 pseudotime values depending on the trajectory range being queried.
    """
    def __init__(self, dictys_dynamic_object=None,
        trajectory_range=(1, 3),
        num_points=40,
        dist=0.001,
        sparsity=0.01,
        total_episodes=8,
        ):
        """
        initialize the aligner with an optional dictys dynamic object.
        """
        self.dictys_dynamic_object = dictys_dynamic_object
        self.trajectory_range = trajectory_range
        self.num_points = num_points
        self.dist = dist
        self.sparsity = sparsity
        self.total_episodes = total_episodes

    def pseudotime_of_windows(self):
        """
        returns - numpy array (of window centroids' pseudotime values)
        S0 or S1 pseudotime values depending on the trajectory range.
        """
        # get 2d array of all nodes' pseudotimes per window
        node_specific_pseudotimes_per_window = self.dictys_dynamic_object.point['s'].dist
        # if traj range starts from 0 return the 0th pseudotime out of 0,1,2,3 and if it starts from 1 return the 1st pseudotime out of 0,1,2,3
        idx_for_querying_pseudotimes = self.trajectory_range[0]
        # return the pseudotime values
        return node_specific_pseudotimes_per_window[:, idx_for_querying_pseudotimes]

    def pseudotime_of_sampled_points(self):
        """
        returns - numpy array of sampled points' pseudotime values. 
        these points are not necessarily the window centroids, they are just equi-spaced points along the trajectory.
        S0 or S1 pseudotime values depending on the trajectory range.
        [should be directly corresponding to window centroid time points]
        """
        # sample equi-spaced points and instantiate smoothing function    
        pts, fsmooth = self.dictys_dynamic_object.linspace(self.trajectory_range[0], self.trajectory_range[1], self.num_points, self.dist)
        # pseudo time values (x axis)
        stat1_x = stat.pseudotime(self.dictys_dynamic_object, pts)
        tmp_x = stat1_x.compute(pts)
        dx = pd.Series(tmp_x[0])
        # return numpy array of pseudotime values
        return dx.to_numpy()


# ---------------------------------------------------------------------------
# Figures: trajectory nodes
# ---------------------------------------------------------------------------

def plot_main_trajectory_nodes(
    adata, n_components=2, comp1=0, comp2=1, fig_size=(8, 6), save_path=None
):
    """
    Plot and label the main trajectory nodes (S0, S1, S2, S3) on the elastic principal graph
    """
    # Use 'Agg' backend for non-interactive environments
    import matplotlib

    matplotlib.use("Agg")

    # Create figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # Plot all EPG edges
    epg = adata.uns["epg"]
    epg_node_pos = nx.get_node_attributes(epg, "pos")

    for edge_i in epg.edges():
        start_pos = epg_node_pos[edge_i[0]]
        end_pos = epg_node_pos[edge_i[1]]

        ax.plot(
            [start_pos[comp1], end_pos[comp1]],
            [start_pos[comp2], end_pos[comp2]],
            "k-",
            alpha=0.3,
            linewidth=1,
        )
        ax.plot(
            [start_pos[comp1], end_pos[comp1]],
            [start_pos[comp2], end_pos[comp2]],
            "ko",
            ms=3,
            alpha=0.3,
        )

    # Get flat tree nodes
    flat_tree = adata.uns["flat_tree"]
    ft_nodes = list(flat_tree.nodes())
    main_nodes = {
        "S0": ft_nodes[0],
        "S3": ft_nodes[1],
        "S2": ft_nodes[2],
        "S1": ft_nodes[3],
    }

    # Plot and label main trajectory nodes
    for label, node_idx in main_nodes.items():
        pos = epg_node_pos[node_idx]
        ax.scatter(pos[comp1], pos[comp2], c="red", s=100, zorder=10)
        ax.text(
            pos[comp1],
            pos[comp2],
            label,
            color="red",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    ax.set_xlabel(f"Dim{comp1+1}")
    ax.set_ylabel(f"Dim{comp2+1}")
    plt.tight_layout()

    # Save the plot
    if save_path is None:
        save_path = "trajectory_nodes.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")
