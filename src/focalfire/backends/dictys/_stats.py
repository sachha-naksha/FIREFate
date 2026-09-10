"""
Custom dictys network stat extensions used by FocalFire.

These subclass ``dictys.net.stat.base`` and live here, rather than being patched
into the installed dictys source, so the analysis depends only on the stock
dictys package.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import dictys.traj
import dictys.net
from dictys.net.stat import base
from dictys.utils.numpy import NDArray


class lcpm_tf(base):
    """
    LogCPM stat specifically for TFs.
    Specifically: log2 (CPM+const) for TFs only
    """
    def __init__(self, d: dictys.net.network, *a, cut: float = 0.01, constant: float = 1, **ka):
        """
        Initialize TF-specific LogCPM stat
        d:          Dataset object
        constant:   Constant to add to CPM before log
        cut:       CPM below cut will be hidden
        """
        self.d = d
        self.cut = cut
        self.const = constant
        super().__init__(*a, **ka)

    def default_names(self):
        """Return TF names only"""
        tf_gene_indices = self.d.nids[0]  # These indices map TFs to their positions in gene list
        return [self.d.nname[tf_gene_indices]]

    def default_label(self):
        """Return label for TF LogCPM"""
        return 'TF Log2 CPM'

    def compute(self, pts):
        """Computes logCPM at each node for TFs only
        Return:
        log2(CPM+const) as np.array(shape=(n_tf,len(pts)))
        """
        if isinstance(pts, dictys.traj.point):
            raise TypeError('lcpm should not be computed at any point. Use existing states or wrap with smooth instead.')

        # Check if all requested TF names exist in network
        t1 = set(self.d.nname)
        t1 = np.nonzero([x not in t1 for x in self.names[0]])[0]  # names[0] is the name for axis 0 in the stat base class
        if len(t1) > 0:
            raise ValueError('TFs not found: {}'.format(','.join(self.names[0][t1])))

        # Get indices for TF names
        tf_gene_indices = self.d.nids[0]  # Maps TFs to their positions in gene list
        tf_names = [self.d.nname[idx] for idx in tf_gene_indices]
        tdict = dict(zip(tf_names, range(len(tf_names))))
        t1 = np.nonzero([x not in tdict for x in self.names[0]])[0]
        if len(t1) > 0:
            raise ValueError('TFs not found: {}'.format(','.join(self.names[0][t1])))

        # Compute logCPM for TFs
        if 'cpm' not in self.d.prop['ns']:
            raise ValueError('CPM results not found. Please recompute.')

        # Extract CPM values only for TF gene indices
        cpm = self.d.prop['ns']['cpm'][tf_gene_indices][:, pts]

        # Apply log transformation
        ans = np.log2(cpm + self.const)
        ans[cpm < self.cut] = np.nan
        return ans


class ChromatinGRNStat(base):
    """
    Stat class for loading and handling chromatin GRN edge weights from external file, as the dynamic.h5 object only has boolean mask
    """
    def __init__(self, stat_file: str, tf_name: str, target_name: str, *a, **ka):
        """
        Initialize ChromatinGRNStat with edge weight data from file.

        Args:
            stat_file: Path to .npy file containing edge weights
            tf_name: Name of the TF (e.g., 'PAX5')
            target_name: Name of the target gene (e.g., 'RUNX2')
        """
        self.tf_name = tf_name
        self.target_name = target_name
        # Load the stat file and handle -inf values
        self.weights = np.load(stat_file)
        self.weights[self.weights == -np.inf] = 0

        # Initialize base class
        super().__init__(*a, **ka)

    def default_names(self) -> list[NDArray[str]]:
        """Define the dimensions of the stat."""
        return [[self.tf_name], [self.target_name]]

    def default_label(self) -> str:
        """Define the label for this stat."""
        return f'Chromatin GRN edge weight ({self.tf_name}->{self.target_name})'

    def compute(self, pts: Union[dictys.traj.point, NDArray[np.int_]]) -> NDArray:
        """
        Compute stat values at each state or point.

        Args:
            pts: Point list instance or state list

        Returns:
            Edge weights as numpy array with shape (1, 1, len(pts))
        """
        if isinstance(pts, dictys.traj.point):
            raise TypeError('ChromatinGRNStat should not be computed at any point. Use existing states or wrap with smooth instead.')

        # Reshape weights to match expected dimensions (1 TF x 1 target x n_timepoints)
        weights = self.weights[pts].reshape(1, 1, -1)
        return weights
