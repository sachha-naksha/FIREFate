"""Trajectory topology declared once (ISSUES.md #23, stage 5).

``TrajectorySegments`` holds one :class:`TemporalManager` (hence one
:class:`TFForceSource`) per named segment and derives everything that spans
segments from that.  A linear trajectory is one segment.  Phases stay what they
are on any segment: cell-state switches that link peaks are binned into.
"""
import os

import numpy as np
import pandas as pd
import pytest

from focalfire.temporal import (
    EpisodeDynamics,
    ForceWavePhases,
    TemporalManager,
    TFForceSource,
    TFForceValidation,
    TFForceWaves,
    TrajectorySegments,
)
from focalfire.temporal import manager as manager_module

SETTINGS = dict(num_points=10, dist=0.3, sparsity=0.1)
BRANCHES = {"PB": (0, 2), "GC": (0, 3)}       # mock: node 1 branches to nodes 2 and 3
LINKS = [("TFA", "G1"), ("TFB", "G4")]


@pytest.fixture
def traj(mock_network, tmp_path):
    return TrajectorySegments(mock_network, BRANCHES, output_dir=str(tmp_path), **SETTINGS)


@pytest.fixture
def linear(mock_network):
    return TrajectorySegments(mock_network, {"lin": (0, 2)}, **SETTINGS)


class TestDeclaration:
    def test_one_manager_per_segment_with_its_range_and_shared_settings(self, traj, tmp_path):
        assert traj.names == ["PB", "GC"]
        assert len(traj) == 2
        for name, rng in BRANCHES.items():
            mgr = traj[name]
            assert isinstance(mgr, TemporalManager)
            assert mgr._traj_range == rng
            assert (mgr._num_points, mgr._dist, mgr._sparsity) == (10, 0.3, 0.1)
            assert mgr.output_dir == os.path.join(str(tmp_path), name)

    def test_no_output_dir_stays_in_memory(self, linear):
        assert linear["lin"].output_dir is None

    def test_unknown_segment_and_empty_declaration_raise(self, traj, mock_network):
        with pytest.raises(KeyError, match="Segments"):
            traj["nope"]
        with pytest.raises(ValueError):
            TrajectorySegments(mock_network, {})

    def test_one_source_per_segment(self, traj):
        sources = traj.sources()
        assert list(sources) == ["PB", "GC"]
        for name in BRANCHES:
            assert isinstance(sources[name], TFForceSource)
            assert sources[name] is traj[name].force_source()
            assert sources[name].trajectory_range == BRANCHES[name]
        assert sources["PB"] is not sources["GC"]


class TestWavesAcrossSegments:
    def test_waves_are_built_once_on_each_segment_s_source(self, traj):
        waves = traj.waves()
        assert traj.waves() is waves
        for name in BRANCHES:
            assert waves[name].source is traj.sources()[name]
            assert waves[name].trajectory_range == BRANCHES[name]

    def test_segment_waves_equal_a_standalone_branch(self, traj, mock_network):
        gc = traj.waves()["GC"]
        gc.compute_forces(LINKS)
        solo = TFForceWaves(mock_network, trajectory_range=(0, 3), **SETTINGS)
        solo.compute_forces(LINKS)
        pd.testing.assert_frame_equal(gc.force_curves, solo.force_curves)

    def test_selector_spans_the_segments(self, traj):
        sel = traj.selector()
        assert sel.branches == ["PB", "GC"]
        assert sel.network_type == "w_in"
        assert traj.selector(network_type="w").network_type == "w"
        combined = sel.combined_abs_max_force(LINKS)
        assert set(combined) == set(LINKS)
        assert {d["branch"] for d in combined.values()} <= {"PB", "GC"}

    def test_linear_selector_is_one_branch(self, linear):
        assert linear.selector().branches == ["lin"]


class TestValidationAcrossSegments:
    def test_mode_defaults_to_combined_for_several_segments(self, traj):
        v = traj.validate(LINKS)
        assert isinstance(v, TFForceValidation)
        assert v.mode == "combined"
        assert v.network_type == "w_in"
        assert v.selector.branches == ["PB", "GC"]
        assert v.selector.waves_by_branch["PB"] is traj.waves()["PB"]

    def test_mode_defaults_to_lineage_for_one_segment(self, linear):
        assert linear.validate(LINKS).mode == "lineage"

    def test_explicit_mode_and_network_type(self, traj):
        v = traj.validate(LINKS, network_type="w", mode="lineage")
        assert (v.mode, v.network_type, v.selector.network_type) == ("lineage", "w", "w")

    def test_compare_sets_delegates_with_the_selector(self, traj, monkeypatch):
        seen = {}
        def fake(cls, branches, enriched_sets, **kw):
            seen.update(branches=branches, enriched_sets=enriched_sets, kw=kw)
            return "table"
        monkeypatch.setattr(TFForceValidation, "compare_sets", classmethod(fake))
        sets = {"A": LINKS[:1], "B": LINKS[1:]}
        assert traj.compare_sets(sets, network_type="w", random_state=3) == "table"
        assert isinstance(seen["branches"], TFForceWaves.ForceSelector)
        assert seen["branches"].branches == ["PB", "GC"]
        assert seen["branches"].network_type == "w"
        assert seen["enriched_sets"] is sets
        assert seen["kw"] == {"network_type": "w", "random_state": 3}

    def test_compare_sets_by_phase_delegates_and_checks_segment_names(self, traj, monkeypatch):
        seen = {}
        def fake(cls, branches, enriched_sets, switch_pseudotimes, **kw):
            seen.update(branches=branches, switches=switch_pseudotimes, kw=kw)
            return "table"
        monkeypatch.setattr(TFForceValidation, "compare_sets_by_phase", classmethod(fake))
        switches = {"PB": [0.6, 1.4]}
        assert traj.compare_sets_by_phase({"A": LINKS}, switches, top_k=3) == "table"
        assert seen["branches"].branches == ["PB", "GC"]
        assert seen["switches"] is switches
        assert seen["kw"] == {"network_type": "w_in", "top_k": 3}
        with pytest.raises(KeyError, match="unknown segment"):
            traj.compare_sets_by_phase({"A": LINKS}, {"XX": [0.5]})


class TestPhasesAlongOneSegment:
    def test_state_frequency_takes_the_segment_s_range(self, traj, monkeypatch):
        seen = {}
        class FakeSF:
            def __init__(self, net, cell_labels, **kw):
                seen.update(net=net, cell_labels=cell_labels, kw=kw)
        monkeypatch.setattr(manager_module, "StateFrequency", FakeSF)
        sf = traj.state_frequency("GC", "labels.csv", cluster_column="state")
        assert isinstance(sf, FakeSF)
        assert seen["net"] is traj.dictys_dynamic_object
        assert seen["cell_labels"] == "labels.csv"
        assert seen["kw"] == {"trajectory_range": (0, 3), "cluster_column": "state"}

    def test_phases_bin_link_peaks_on_that_segment(self, traj):
        traj.waves()["PB"].compute_forces(LINKS)
        fwp = traj.phases("PB", [0.6, 1.4])
        assert isinstance(fwp, ForceWavePhases)
        assert fwp.waves is traj.waves()["PB"]
        df = fwp.classify_phases()
        assert len(df) == len(traj.waves()["PB"].force_curves)
        assert set(df["phase"]) <= {1, 2, 3}

    def test_linear_trajectory_has_phases_from_its_switches(self, linear):
        linear.waves()["lin"].compute_forces(LINKS)
        fwp = linear.phases("lin", [0.6, 1.4])
        assert fwp.n_phases == 3
        peaks = fwp.link_peak_pseudotimes()
        for link, info in peaks.items():
            assert fwp.phase_of(info["pseudotime"]) == int(np.digitize(info["pseudotime"], [0.6, 1.4], right=True)) + 1


class TestEpisodesPerSegment:
    def test_episodes_build_on_the_segment_s_source(self, traj, mock_network, tmp_path):
        grn = traj["GC"].build_episode(1, slice(0, 5), percentile=0, n_processes=1)
        assert grn.attrs["network_type"] == "w"
        epi = EpisodeDynamics(mock_network, str(tmp_path / "solo"), trajectory_range=(0, 3), **SETTINGS)
        epi.compute_expression_curves()
        epi.build_episode_grn(time_slice=slice(0, 5))
        epi.filter_edges(n_processes=1)
        epi.compute_tf_expression()
        epi.calculate_forces(n_processes=1)
        pd.testing.assert_frame_equal(grn, epi.select_top_edges(0))
        assert traj["GC"]._episode_dynamics("w").source is traj.sources()["GC"]
