"""Tests for the rollout (ungrounded) mode of analysis/linear_probe.py.

The failure mode this guards against is silent: if probe_train and probe_test rows
come from the same episode, every R^2 in the output is inflated and nothing errors.
So the disjointness check is a hard assertion on episode IDENTITIES (val.indices),
not on positional slots, and it is tested here in both directions.

Run with either:
    python tests/test_linear_probe_rollout.py
    pytest tests/test_linear_probe_rollout.py     # from the repo root (ts env)
"""

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "linear_probe", REPO / "analysis" / "linear_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBE = load_probe_module()


class FakeVal:
    """Stands in for a TrajSubset: positional slot -> underlying trajectory id."""

    def __init__(self, ids):
        self.indices = ids


def test_horizon_counts_only_predicted_frames():
    # rollout() returns len(act) + 1 frames: the first num_hist are real encodings and the
    # last predicts a step past the final observed state, so it has no ground truth.
    assert PROBE.rollout_horizon(101, 3) == 98
    assert PROBE.rollout_horizon(4, 3) == 1
    assert PROBE.rollout_horizon(20, 3) == 17     # the shape that exposed the bug


def test_tmax_default_drops_nothing():
    limit, kept, dropped = PROBE.choose_tmax({0: 9, 1: 5, 2: 12})
    assert limit == 5
    assert kept == [0, 1, 2]
    assert dropped == []


def test_tmax_override_drops_and_reports_short_episodes():
    limit, kept, dropped = PROBE.choose_tmax({0: 9, 1: 5, 2: 12}, 10)
    assert limit == 10
    assert kept == [2]
    assert dropped == [0, 1]


def test_impossible_tmax_raises():
    with pytest.raises(RuntimeError):
        PROBE.choose_tmax({0: 9, 1: 5}, 20)
    with pytest.raises(RuntimeError):
        PROBE.choose_tmax({})


def test_episode_ids_use_the_underlying_trajectory():
    val = FakeVal([7, 7, 9, 11])
    assert PROBE.episode_ids(val, [0, 2]) == [7, 9]
    # a loader without .indices falls back to positions rather than crashing
    assert PROBE.episode_ids(object(), [3, 5]) == [3, 5]


def test_shared_trajectory_behind_two_splits_is_rejected():
    val = FakeVal([7, 7, 9, 11])
    with pytest.raises(AssertionError):
        PROBE.assert_disjoint_episodes(val, [0], [1])           # same trajectory 7
    with pytest.raises(AssertionError):
        PROBE.assert_disjoint_episodes(val, [0, 2], [1, 3])     # 7 on both sides


def test_disjoint_episodes_pass_and_are_returned():
    val = FakeVal([7, 8, 9, 11])
    train_eps, test_eps = PROBE.assert_disjoint_episodes(val, [0, 2], [1, 3])
    assert train_eps == [7, 9] and test_eps == [8, 11]


def test_retention_is_nan_safe():
    assert PROBE.retention(0.8, 0.4) == pytest.approx(0.5)
    assert PROBE.retention(0.0, 0.4) != PROBE.retention(0.0, 0.4)    # nan
    assert PROBE.retention(None, 0.4) != PROBE.retention(None, 0.4)  # nan


def test_channel_table_has_four_arms_per_env():
    table = PROBE.models_for("channel")
    for env in ("point_maze", "point_maze_medium", "pusht"):
        assert env in table, env
        assert len(table[env]) == 4, (env, len(table[env]))
        for variant, path in table[env].items():
            assert path.startswith("checkpoints/test/"), (env, variant, path)
            assert "projchannel" in path, (env, variant, path)
    assert PROBE.models_for("projglobal") is PROBE.MODELS
    with pytest.raises(RuntimeError):
        PROBE.models_for("nonsense")


def test_shape_sanity_is_equality_not_plus_one():
    """obs and act describe the SAME number of model steps.
    The val items are raw trajectories: obs/states are frameskip-strided and act groups
    frameskip raw actions, so the sketch obs == act + 1 does not hold for this loader."""
    src = (REPO / "analysis" / "linear_probe.py").read_text()
    assert "episode time bases disagree" in src
    assert "assert n_obs == act_ms.shape[0]" in src

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
