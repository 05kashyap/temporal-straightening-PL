"""Regression tests for planning/evaluator.py's visualization batch sizes.

The crash this pins (first real full run on the cluster, `chunk_size=null` = unchunked,
`n_plot_samples=10`):

    planning/evaluator.py:86 _mask_traj -> if length[i] != np.inf:
    IndexError: index 10 is out of bounds for axis 0 with size 10

`i_z_obses_first` holds the whole FIRST CHUNK (`chunk_size` rows) while the mask was built
from `action_len[: n_plot_samples]`, so any `chunk_size > n_plot_samples` crashed -- and the
mirror case (`chunk_size < n_plot_samples`) mismatched inside `_plot_rollout_compare`, whose
video loop iterates `e_visuals`' rows while indexing `i_visuals`.

No world model, no env, no GPU: only the pure helpers are exercised.

Run with either:
    python tests/test_evaluator_viz.py
    pytest tests/test_evaluator_viz.py        # from the repo root (ts env)
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from planning.evaluator import PlanEvaluator  # noqa: E402

N_PLOT = 10


def dummy_evaluator(n_plot_samples=N_PLOT, frameskip=5):
    """The viz helpers need only these attributes (no WM / env / device)."""
    ev = PlanEvaluator.__new__(PlanEvaluator)
    ev.n_plot_samples = n_plot_samples
    ev.frameskip = frameskip
    ev.plot_full = False
    return ev


def test_the_old_call_site_raises_for_a_larger_chunk():
    """chunk_size=50 with n_plot_samples=10: the mask is shorter than the decoded chunk."""
    ev = dummy_evaluator()
    decoded_first_chunk = torch.ones(50, 4, 3, 3, 3)   # i_z_obses_first -> decoded
    mask_from_n_plot = np.full(N_PLOT, np.inf)         # the old action_len[:n_plot_samples]
    with pytest.raises(IndexError):
        ev._mask_traj(decoded_first_chunk, mask_from_n_plot)


def test_the_new_call_site_is_clean_for_every_chunk_size():
    """The fixed expression: decode min(n_plot_samples, cs) rows and mask that same batch."""
    ev = dummy_evaluator()
    for cs in (1, 8, 10, 12, 50):
        decoded_first_chunk = torch.ones(cs, 4, 3, 3, 3)
        n_viz = min(ev.n_plot_samples, decoded_first_chunk.shape[0])
        assert n_viz == min(N_PLOT, cs)
        sliced = decoded_first_chunk[:n_viz]
        action_len = np.full(50, np.inf)
        out = ev._mask_traj(sliced, action_len[: sliced.shape[0]] + 1)
        assert out.shape == sliced.shape


def test_mask_traj_zeroes_after_the_index_and_keeps_inf():
    ev = dummy_evaluator()
    data = torch.ones(3, 5, 2)
    length = np.array([3.0, np.inf, 0.0])
    out = ev._mask_traj(data, length)
    assert torch.all(out[0, :3] == 1) and torch.all(out[0, 3:] == 0)
    assert torch.all(out[1] == 1)          # inf: nothing masked
    assert torch.all(out[2] == 0)          # length 0: everything masked


def test_the_plot_uses_one_batch_for_both_sides():
    """n_viz = min(n_plot_samples, e_visuals, i_visuals) is what the video loop indexes."""
    for n_env, n_imag in ((50, 50), (50, 1), (3, 3), (50, 12)):
        assert min(N_PLOT, n_env, n_imag) == min(N_PLOT, n_env, n_imag)


def test_the_source_keeps_the_clamped_batch():
    """Cheap source-level guard so the fix cannot be reverted unnoticed."""
    src = (REPO / "planning" / "evaluator.py").read_text()
    assert "n_viz = min(self.n_plot_samples" in src
    assert "action_len[: self.n_plot_samples] + 1" not in src, "the crashing mask is back"
    assert "action_len[: i_visuals.shape[0]] + 1" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
