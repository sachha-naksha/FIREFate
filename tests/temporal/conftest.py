"""Shared fixtures for the FocalFire core test-suite.

Everything here is hand-built mock data whose expected values can be worked out
on paper, so that a failing test points at the library and not at the fixture.

The centrepiece is :func:`build_mock_network`, which assembles a *real*
``dictys.net.dynamic_network`` (not a stub) out of a tiny Y-shaped trajectory.
Using the genuine dictys object means the tests exercise the same
``linspace``/``stat``/smoothing machinery the production code relies on, while
still being small enough to reason about exactly.

Mock trajectory
---------------
Four trajectory nodes, three unit-length edges::

    node0 --e0(len 1)--> node1 --e1(len 1)--> node2      (branch "PB")
                            \\--e2(len 1)--> node3      (branch "GC")

Seven windows/states (dictys "subsets") sit on that trajectory::

    idx : 0     1     2     3     4     5     6
    edge: e0    e0    e0    e1    e1    e2    e2
    loc : 0.0   0.5   1.0   0.5   1.0   0.5   1.0
    node: n0          n1          n2          n3

so the distance of each window to node0 is [0, .5, 1, 1.5, 2, 1.5, 2] and to
node1 is [1, .5, 0, .5, 1, .5, 1].  Three cells sit exactly on top of each
window, so the window centroids dictys re-derives from the cells land back on
the same locations.

Genes (8) and regulators (3)::

    idx : 0     1     2      3    4    5    6    7
    name: TFA   TFB   ZNF1   G1   G2   G3   G4   G5
    nids[0] (regulators) = [0, 1, 2]
    nids[1] (targets)    = 0..7  (all genes, as in real dictys networks)

``ZNF1`` exists so that the hard-coded ``ZNF*``/``ZBTB*`` drop in
``EpisodeDynamics.build_episode_grn`` is covered.
"""

import gzip
import os

import numpy as np
import pandas as pd
import pytest

import dictys
import dictys.traj

# --------------------------------------------------------------------------- #
# Mock trajectory / network description                                        #
# --------------------------------------------------------------------------- #

TRAJ_EDGES = np.array([[0, 1], [1, 2], [1, 3]])
TRAJ_LENS = np.array([1.0, 1.0, 1.0])

WINDOW_EDGE = np.array([0, 0, 0, 1, 1, 2, 2])
WINDOW_LOC = np.array([0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0])
N_WINDOWS = len(WINDOW_EDGE)
CELLS_PER_WINDOW = 3

#: window -> window adjacency along the trajectory (a spanning tree)
WINDOW_NEIGHBOURS = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]

#: distance from every window to trajectory node 0 / node 1
WINDOW_PSEUDOTIME_FROM_N0 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 1.5, 2.0])
WINDOW_PSEUDOTIME_FROM_N1 = np.array([1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 1.0])

GENES = np.array(["TFA", "TFB", "ZNF1", "G1", "G2", "G3", "G4", "G5"])
GIDX = {g: i for i, g in enumerate(GENES)}
REGULATORS = ["TFA", "TFB", "ZNF1"]

#: genes whose CPM is constant over all windows -> their smoothed log2(CPM+1)
#: curve must be exactly this constant at *every* pseudotime point, because
#: Gaussian smoothing weights are normalised to sum to one.
CONSTANT_LCPM = {"TFA": 2.0, "ZNF1": 1.0, "G1": 4.0, "G5": 0.0}

#: genes whose log2(CPM+1) rises linearly from 1 to 4 across the windows
RISING_LCPM = ["TFB", "G2"]

#: non-zero edges of the mock GRN: (TF, target) -> per-window weight vector.
#: Everything not listed is exactly zero in every window.
MOCK_EDGES = {
    ("TFA", "G1"): np.full(N_WINDOWS, 2.0),               # constant, positive
    ("TFA", "G2"): np.linspace(1.0, 2.0, N_WINDOWS),      # ramp, positive
    ("TFA", "G3"): np.linspace(3.0, -3.0, N_WINDOWS),     # sign flip in time
    ("TFA", "TFB"): np.full(N_WINDOWS, 1.0),              # TF -> TF edge
    ("TFB", "G4"): np.full(N_WINDOWS, -1.5),              # constant, negative
    ("TFB", "G5"): np.linspace(-0.5, -1.5, N_WINDOWS),    # ramp, negative
    ("ZNF1", "G1"): np.full(N_WINDOWS, 5.0),              # dropped by ZNF filter
}

#: scale factors between the network variants stored on the mock object
VARNAME_SCALE = {"w": 1.0, "w_n": 0.5, "w_in": 2.0}


def _cpm_matrix():
    """CPM values chosen so that ``log2(CPM + 1)`` is a round number."""
    cpm = np.zeros((len(GENES), N_WINDOWS))
    for gene, lcpm in CONSTANT_LCPM.items():
        cpm[GIDX[gene]] = 2.0 ** lcpm - 1.0
    # TFB/G2 rise 1 -> 4, G3 falls 4 -> 1, G4 is a bell curve (in log2 space)
    for gene in RISING_LCPM:
        cpm[GIDX[gene]] = 2.0 ** np.linspace(1.0, 4.0, N_WINDOWS) - 1.0
    cpm[GIDX["G3"]] = 2.0 ** np.linspace(4.0, 1.0, N_WINDOWS) - 1.0
    cpm[GIDX["G4"]] = 2.0 ** np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]) - 1.0
    return cpm


def _net_matrix():
    net = np.zeros((len(REGULATORS), len(GENES), N_WINDOWS))
    for (tf, target), vals in MOCK_EDGES.items():
        net[REGULATORS.index(tf), GIDX[target]] = vals
    return net


def build_mock_network(nan_edge=None):
    """Build a real :class:`dictys.net.dynamic_network` from the mock data.

    Parameters
    ----------
    nan_edge : tuple, optional
        ``((tf, target), window_index)``.  When given, that single per-window
        network entry is set to NaN so the NaN-aware smoothing paths can be
        exercised.
    """
    traj = dictys.traj.trajectory(TRAJ_EDGES, TRAJ_LENS)
    pts_s = dictys.traj.point(traj, WINDOW_EDGE.copy(), WINDOW_LOC.copy())
    pts_c = dictys.traj.point(
        traj,
        np.repeat(WINDOW_EDGE, CELLS_PER_WINDOW),
        np.repeat(WINDOW_LOC, CELLS_PER_WINDOW),
    )

    n_cells = N_WINDOWS * CELLS_PER_WINDOW
    sc_w = np.zeros((N_WINDOWS, n_cells))
    for i in range(N_WINDOWS):
        sc_w[i, i * CELLS_PER_WINDOW:(i + 1) * CELLS_PER_WINDOW] = 1

    neighbours = np.zeros((N_WINDOWS, N_WINDOWS), dtype=int)
    for a, b in WINDOW_NEIGHBOURS:
        neighbours[a, b] = neighbours[b, a] = 1

    net = _net_matrix()
    if nan_edge is not None:
        (tf, target), window = nan_edge
        net[REGULATORS.index(tf), GIDX[target], window] = np.nan

    return dictys.net.dynamic_network(
        cname=np.array([f"cell{i}" for i in range(n_cells)]),
        sname=np.array([f"Subset{i + 1}" for i in range(N_WINDOWS)]),
        nname=GENES.copy(),
        nids=[np.array([GIDX[t] for t in REGULATORS]), np.arange(len(GENES))],
        traj=traj,
        point={"c": pts_c, "s": pts_s},
        scprop={"w": sc_w},
        ssprop={"traj-neighbor": neighbours},
        nsprop={"cpm": _cpm_matrix()},
        esprop={
            "w": net,
            "w_n": net * VARNAME_SCALE["w_n"],
            "w_in": net * VARNAME_SCALE["w_in"],
            "mask": ~np.isnan(net) & (net != 0),
        },
    )


@pytest.fixture(scope="session")
def mock_network():
    """A real dictys dynamic network built from the mock trajectory."""
    return build_mock_network()


@pytest.fixture(scope="session")
def mock_network_nan():
    """Same network but with one NaN entry in the per-window GRN."""
    return build_mock_network(nan_edge=(("TFA", "G1"), 2))


@pytest.fixture(scope="session")
def mock_network_path(tmp_path_factory):
    """The mock network persisted to HDF5, for the process-based runners."""
    path = str(tmp_path_factory.mktemp("net") / "mock_dynamic.h5")
    build_mock_network().to_file(path)
    return path


# --------------------------------------------------------------------------- #
# Curve fixtures (pure numpy, no dictys)                                        #
# --------------------------------------------------------------------------- #

@pytest.fixture
def dx_unit():
    """11 evenly spaced pseudotime points on [0, 1]."""
    return np.linspace(0.0, 1.0, 11)


@pytest.fixture
def four_class_curves(dx_unit):
    """Four curves, one per intended global-activity class.

    Returns ``(dx, DataFrame)`` with rows:

    * ``UP``    - monotone rise 0 -> 1        (large positive terminal logFC)
    * ``DOWN``  - monotone fall 1 -> 0        (large negative terminal logFC)
    * ``BUMP``  - starts and ends at 0, peaks (large positive transient logFC)
    * ``DIP``   - starts and ends at 0, dips  (large negative transient logFC)
    """
    dx = dx_unit
    up = dx.copy()
    down = 1.0 - dx
    bump = np.sin(np.pi * dx)
    dip = -np.sin(np.pi * dx)
    dy = pd.DataFrame([up, down, bump, dip], index=["UP", "DOWN", "BUMP", "DIP"])
    return dx, dy


# --------------------------------------------------------------------------- #
# Chromatin binding fixtures                                                    #
# --------------------------------------------------------------------------- #

#: Per-window binding rows: window -> list of (TF, chromosome, score).
#:
#: The score reported for a TF is the *mean over chromosomes of the per
#: chromosome mean score*, and the count is the *mean over chromosomes of the
#: per-chromosome row count* -- not a global mean/total.  The numbers below are
#: picked so the two differ, which pins the actual aggregation down.
BINDING_ROWS = {
    1: [("TFA", "chr1", 1.0), ("TFA", "chr1", 3.0), ("TFA", "chr2", 10.0),
        ("TFB", "chr1", 4.0)],
    2: [("TFA", "chr1", 2.0), ("TFA", "chr2", 6.0),
        ("TFC", "chr1", 7.0), ("TFC", "chr1", 9.0)],
    3: [("TFA", "chr1", 5.0), ("TFB", "chr1", 2.0), ("TFB", "chr2", 8.0)],
}

#: expected (score, count) per TF per window, derived by hand from BINDING_ROWS
EXPECTED_BINDING = {
    "TFA": {1: (6.0, 1.5), 2: (4.0, 1.0), 3: (5.0, 1.0)},
    "TFB": {1: (4.0, 1.0), 3: (5.0, 1.0)},
    "TFC": {2: (8.0, 2.0)},
}


def write_binding_windows(base_path, rows_by_window):
    """Write ``Subset<i>/binding.tsv.gz`` files in the dictys binding format."""
    for window, rows in rows_by_window.items():
        folder = os.path.join(base_path, f"Subset{window}")
        os.makedirs(folder, exist_ok=True)
        df = pd.DataFrame(
            [
                {"TF": tf, "loc": f"{chrom}:{100 + i}:{200 + i}", "score": score}
                for i, (tf, chrom, score) in enumerate(rows)
            ]
        )
        with gzip.open(os.path.join(folder, "binding.tsv.gz"), "wt") as f:
            df.to_csv(f, sep="\t", index=False)


@pytest.fixture
def binding_dir(tmp_path):
    """Three windows of mock TF binding data on disk."""
    base = tmp_path / "tmp_dynamic"
    base.mkdir()
    write_binding_windows(str(base), BINDING_ROWS)
    return str(base)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: test spawns worker processes / is comparatively slow"
    )
