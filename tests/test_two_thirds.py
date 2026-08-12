"""Unit tests for VWorldModel._two_thirds_residual / total_two_thirds_loss.

Hand-built tensors only: no dataset, no encoder, no GPU required.

Run with either:
    python tests/test_two_thirds.py
    pytest tests/test_two_thirds.py        # from the repo root (ts env)
"""

import math
import os
import sys

import torch

# Make `models` importable when this file is run as a plain script from
# anywhere (pytest from the repo root handles this on its own).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.visual_world_model import VWorldModel


class _StubEncoder:
    """Bare stand-in carrying only the agg() pooling head the loss needs."""

    @staticmethod
    def agg(tokens):
        return tokens.mean(dim=1)


def _model():
    """VWorldModel instance without running __init__ (no encoder/config needed)."""
    m = object.__new__(VWorldModel)
    m.encoder = _StubEncoder()
    return m


def _loss(z, mode="cos"):
    return _model().total_two_thirds_loss(z, mode=mode)


def test_law_satisfying_trajectory_has_near_zero_loss():
    """A trajectory built to obey log s + (1/3) log kappa = const -> ~zero loss.

    Construction: fix a target residual r_const and a curvature (turn-angle)
    sequence theta1, theta2. The power law s^(2/3) * theta^(1/3) = exp(r_const)
    fixes the average step length per window: s_t = exp(1.5 r_const) * theta_t^-0.5.
    Pick three step lengths so each window's average hits its target, then lay
    the points out in the plane with the prescribed turns.
    """
    r_const = -1.0
    theta1, theta2 = 0.5, 0.8

    c = math.exp(1.5 * r_const)
    s1 = c * theta1 ** -0.5
    s2 = c * theta2 ** -0.5
    u2_len = 0.3
    u1_len = 2 * s1 - u2_len
    u3_len = 2 * s2 - u2_len
    assert u1_len > 0 and u3_len > 0

    u1 = torch.tensor([u1_len, 0.0])
    u2 = torch.tensor([u2_len * math.cos(theta1), u2_len * math.sin(theta1)])
    u3 = torch.tensor(
        [u3_len * math.cos(theta1 + theta2), u3_len * math.sin(theta1 + theta2)]
    )

    z0 = torch.zeros(2)
    z = torch.stack([z0, z0 + u1, z0 + u1 + u2, z0 + u1 + u2 + u3])
    z = z.view(1, 4, 1, 2).double()

    loss = _loss(z, mode="cos")
    assert loss < 1e-4, f"law-satisfying trajectory should be ~0, got {loss.item()}"


def test_law_violating_trajectory_matches_hand_computation():
    """Hand-derived example: Var([r1, r2]) ~= 0.0052868.

    z = (0,0), (1,0), (2,0.5), (2.5,1.5). Steps u1=(1,0), u2=(1,0.5), u3=(0.5,1).
    r_t = log(s_t) + (1/3) log(kappa_t), kappa_t = theta_t / s_t
      window 1: s = (1 + sqrt(1.25))/2 = 1.059017, theta = acos(1/sqrt(1.25))
      window 2: s = sqrt(1.25) = 1.118034,         theta = acos(0.8)
      -> r1 = -0.2179828, r2 = -0.0725627
    Var([r1, r2]) (biased) = ((r1 - r2)/2)^2 = 0.0052868
    """
    z = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.5]], [[2.5, 1.5]]]],
        dtype=torch.float64,
    )
    expected = 0.0052868  # eps/clamp terms shift this by < 1e-6
    loss = _loss(z, mode="cos")
    assert abs(loss.item() - expected) < 2e-4, (loss.item(), expected)


def test_straight_constant_speed_is_finite():
    """Collinear, evenly spaced points (kappa ~ 0): exercises eps/clamp paths.

    Smoke test only -- the 'correct' value at kappa -> 0 is conventional given
    the eps floor, so we only assert finiteness (no NaN/Inf).
    """
    t = torch.arange(4, dtype=torch.float64)
    z = torch.stack([t, torch.zeros(4, dtype=torch.float64)], dim=-1).view(1, 4, 1, 2)
    loss = _loss(z, mode="cos")
    assert torch.isfinite(loss), loss


def test_near_duplicate_frame_is_finite():
    """One ~zero-length step (nearly identical consecutive frames).

    This is exactly the case clamp_min(step_thresh) exists to guard.
    """
    z = torch.tensor(
        [[[[0.0, 0.0]],
          [[1.0, 0.0]],
          [[1.0 + 1e-8, 0.0]],  # ~duplicate of the previous frame
          [[2.0, 0.0]]]],
        dtype=torch.float64,
    )
    loss = _loss(z, mode="cos")
    assert torch.isfinite(loss), loss


def test_gradient_reaches_the_encoder():
    """Loss must be differentiable w.r.t. the encoder output (no silent detach)."""
    def encoder_stub(x):
        return x  # identity encoder

    x = torch.randn(1, 4, 1, 2, dtype=torch.float64, requires_grad=True)
    loss = _loss(encoder_stub(x), mode="cos")
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0.0  # a real graph dependency, not a dead edge


def test_aggcos_equals_cos_for_single_patch():
    """With p=1, mean-pooled aggcos and patch-wise cos must agree exactly."""
    z = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.5]], [[2.5, 1.5]]]],
        dtype=torch.float64,
    )
    m = _model()
    loss_cos = m.total_two_thirds_loss(z, mode="cos")
    loss_agg = m.total_two_thirds_loss(z, mode="aggcos")
    assert torch.allclose(loss_cos, loss_agg, atol=1e-12)


if __name__ == "__main__":
    import traceback

    tests = [
        (k, v)
        for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
