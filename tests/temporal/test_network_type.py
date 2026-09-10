"""``network_type``: which dictys network variable supplies beta (ISSUES.md #23, stage 2).

Each context chooses its beta network on purpose and exposes the choice as a
parameter with a context-specific default: episodic construction uses the
direct-effect network ``"w"``, force waves / phases / validation use the
total-effect network ``"w_in"``.  The shared force source itself has no default.
These tests pin the defaults, that the override reaches the beta curves in every
context, that the choice is stamped on the outputs, and that the two contexts
agree exactly on the force of a link once they are given the same network.
"""
import numpy as np
import pandas as pd
import pytest

from firefate.temporal import (
    EpisodeDynamics,
    SmoothedCurvesGRN,
    TemporalManager,
    TFForceValidation,
    TFForceWaves,
)
from firefate.temporal.manager import run_episodic_construction
from conftest import VARNAME_SCALE

TRAJ = (0, 2)
NUM_POINTS = 10
DIST = 0.3
SPARSITY = 0.1
LINK = ("TFA", "G1")          # constant beta 2.0 in every window, survives the episodic filter
EPISODE = slice(0, 5)
KW = dict(trajectory_range=TRAJ, num_points=NUM_POINTS, dist=DIST, sparsity=SPARSITY)


def _episode(mock_network, tmp_path, **kw):
    epi = EpisodeDynamics(mock_network, output_folder=str(tmp_path), **KW, **kw)
    epi.compute_expression_curves()
    epi.build_episode_grn(time_slice=EPISODE)
    epi.filter_edges(n_processes=1)
    epi.compute_tf_expression()
    epi.calculate_forces(n_processes=1)
    epi.select_top_edges(0)
    return epi


def _waves(mock_network):
    return TFForceWaves(mock_network, **KW)


# --------------------------------------------------------------------------- #
# episodic construction: default "w", override reaches beta, stamped on outputs #
# --------------------------------------------------------------------------- #

class TestEpisodicNetworkType:
    def test_default_is_the_direct_effect_network(self, mock_network, tmp_path):
        assert EpisodeDynamics(mock_network, str(tmp_path), **KW).network_type == "w"

    def test_default_beta_equals_an_explicit_w(self, mock_network, tmp_path):
        default = _episode(mock_network, tmp_path / "d").episode_beta_dcurve
        explicit = _episode(mock_network, tmp_path / "e", network_type="w").episode_beta_dcurve
        pd.testing.assert_frame_equal(default, explicit)

    @pytest.mark.parametrize("network_type,scale", sorted(VARNAME_SCALE.items()))
    def test_override_selects_the_network_variant(self, mock_network, tmp_path,
                                                  network_type, scale):
        beta = _episode(mock_network, tmp_path, network_type=network_type).episode_beta_dcurve
        assert beta.loc[LINK].values == pytest.approx(np.full(5, 2.0 * scale))

    def test_override_changes_the_forces(self, mock_network, tmp_path):
        direct = _episode(mock_network, tmp_path / "w").avg_force_df
        total = _episode(mock_network, tmp_path / "w_in", network_type="w_in").avg_force_df
        assert direct.loc[LINK, "avg_force"] != pytest.approx(total.loc[LINK, "avg_force"])

    @pytest.mark.parametrize("network_type", ["w", "w_in"])
    def test_every_output_frame_is_stamped(self, mock_network, tmp_path, network_type):
        epi = _episode(mock_network, tmp_path, network_type=network_type)
        for frame in (epi.episode_beta_dcurve, epi.force_curves, epi.avg_force_df,
                      epi.episodic_grn_edges):
            assert frame.attrs["network_type"] == network_type
        epi.select_top_activating_and_repressing_edges(50, 50)
        assert epi.episodic_grn_edges.attrs["network_type"] == network_type

    def test_stamp_survives_the_parquet_round_trip(self, mock_network_path, tmp_path):
        path = run_episodic_construction(
            1, mock_network_path, str(tmp_path), trajectory_range=TRAJ,
            num_points=NUM_POINTS, time_slice_start=0, time_slice_end=5,
            dist=DIST, sparsity=SPARSITY, percentile=0, network_type="w_in",
        )
        assert pd.read_parquet(path).attrs["network_type"] == "w_in"
        assert pd.read_parquet(tmp_path / "avg_force_episode_1.parquet").attrs["network_type"] == "w_in"

    def test_runner_default_is_w(self, mock_network_path, tmp_path):
        path = run_episodic_construction(
            1, mock_network_path, str(tmp_path), trajectory_range=TRAJ,
            num_points=NUM_POINTS, time_slice_start=0, time_slice_end=5,
            dist=DIST, sparsity=SPARSITY, percentile=0,
        )
        assert pd.read_parquet(path).attrs["network_type"] == "w"


# --------------------------------------------------------------------------- #
# force waves / selector / validation: default "w_in", override reaches beta    #
# --------------------------------------------------------------------------- #

class TestWavesNetworkType:
    def test_default_is_the_total_effect_network(self, mock_network):
        waves = _waves(mock_network)
        waves.compute_forces([LINK])
        explicit, _ = waves.curves.get_beta_curves([LINK], network_type="w_in")
        pd.testing.assert_frame_equal(waves.beta_curves, explicit)
        assert waves.network_type == "w_in"
        assert waves.force_curves.attrs["network_type"] == "w_in"

    @pytest.mark.parametrize("network_type,scale", sorted(VARNAME_SCALE.items()))
    def test_override_selects_the_network_variant(self, mock_network, network_type, scale):
        waves = _waves(mock_network)
        waves.compute_forces([LINK], network_type=network_type)
        assert waves.beta_curves.loc[LINK].values == pytest.approx(np.full(NUM_POINTS, 2.0 * scale))
        assert waves.force_curves.attrs["network_type"] == network_type

    def test_selector_default_and_override(self, mock_network):
        waves = _waves(mock_network)
        assert TFForceWaves.ForceSelector(waves).network_type == "w_in"
        sel = TFForceWaves.ForceSelector(waves, network_type="w")
        fc, _ = sel.force_curves([LINK])
        assert fc.attrs["network_type"] == "w"
        expected = SmoothedCurvesGRN.calculate_force_curves(
            waves.curves.get_beta_curves([LINK], network_type="w")[0],
            waves.curves.get_smoothed_curves(mode="tf_expression")[0].loc[["TFA"]],
        )
        pd.testing.assert_frame_equal(fc, expected)

    def test_validation_default_and_override(self, mock_network):
        waves = _waves(mock_network)
        assert TFForceValidation(waves, [LINK]).network_type == "w_in"
        v = TFForceValidation(waves, [LINK], network_type="w")
        assert v.network_type == "w"
        assert v.selector.network_type == "w"


# --------------------------------------------------------------------------- #
# manager threads the parameter through with the context defaults              #
# --------------------------------------------------------------------------- #

class TestManagerNetworkType:
    @pytest.fixture
    def mgr(self, mock_network, tmp_path):
        return TemporalManager(mock_network, output_dir=str(tmp_path), **KW)

    def test_build_episode_default_is_w(self, mgr):
        grn = mgr.build_episode(1, EPISODE, percentile=0, n_processes=1)
        assert grn.attrs["network_type"] == "w"

    def test_build_episode_override(self, mgr):
        grn = mgr.build_episode(1, EPISODE, percentile=0, n_processes=1, network_type="w_in")
        assert grn.attrs["network_type"] == "w_in"

    def test_enrich_episode_accepts_the_parameter(self, mgr):
        enr = mgr.enrich_episode(1, EPISODE, ["G1", "G2"], percentile=0, n_processes=1,
                                 network_type="w_in")
        assert isinstance(enr, pd.DataFrame)

    def test_transition_window_forces_match_the_waves(self, mgr, mock_network):
        # ISSUES.md #24: used to pass the LAST expression column as a Series and
        # raise; now pairs the full expression frame, like compute_forces.
        beta, forces, dtime = mgr.build_transition_window([LINK])
        waves = _waves(mock_network)
        waves.compute_forces([LINK])
        pd.testing.assert_frame_equal(beta, waves.beta_curves)
        pd.testing.assert_frame_equal(forces, waves.force_curves)
        assert forces.attrs["network_type"] == "w_in"
        _, forces_w, _ = mgr.build_transition_window([LINK], network_type="w")
        waves.compute_forces([LINK], network_type="w")
        pd.testing.assert_frame_equal(forces_w, waves.force_curves)

    def test_validate_forwards_the_parameter(self, mgr):
        waves = mgr.waves()
        assert mgr.validate(waves, [LINK]).network_type == "w_in"
        assert mgr.validate(waves, [LINK], network_type="w").network_type == "w"


# --------------------------------------------------------------------------- #
# cross-context contract: same network + same window -> same force              #
# --------------------------------------------------------------------------- #

class TestContextsAgreeOnTheForce:
    @pytest.mark.parametrize("network_type", ["w", "w_in"])
    def test_episodic_force_equals_the_wave_force_on_the_same_network(
            self, mock_network, tmp_path, network_type):
        epi = _episode(mock_network, tmp_path, network_type=network_type)
        waves = _waves(mock_network)
        waves.compute_forces([LINK], network_type=network_type)
        episodic = epi.force_curves.loc[LINK].values
        wave = waves.force_curves.loc[LINK].values[EPISODE]
        assert episodic == pytest.approx(wave)

    def test_default_contexts_differ_because_the_networks_differ(self, mock_network, tmp_path):
        epi = _episode(mock_network, tmp_path)               # "w"
        waves = _waves(mock_network)
        waves.compute_forces([LINK])                          # "w_in"
        assert epi.force_curves.attrs["network_type"] != waves.force_curves.attrs["network_type"]
        assert not np.allclose(epi.force_curves.loc[LINK].values,
                               waves.force_curves.loc[LINK].values[EPISODE])
