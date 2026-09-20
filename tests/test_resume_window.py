"""Unit tests for train.epoch_window -- the resume / target-epoch arithmetic.

Pure arithmetic: no dataset, no encoder, no GPU.

Run with either:
    python tests/test_resume_window.py
    pytest tests/test_resume_window.py        # from the repo root (ts env)
"""

import os
import sys

import pytest

# Make `train` importable when this file is run as a plain script from anywhere
# (pytest from the repo root handles this on its own).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import epoch_window


TARGET_CASES = [
    # (saved_epoch, current_iter, epochs, mode, expected)
    (0, 0, 20, "target", (1, 20)),      # fresh run: unchanged by this feature
    (0, 0, 1, "target", (1, 1)),
    (12, 0, 20, "target", (13, 20)),    # between epochs: finish the run, do not add 20 more
    (12, 57, 20, "target", (12, 20)),   # mid-epoch: finish epoch 12 first (dataloader start_iter)
    (20, 0, 20, "target", None),        # already at target
    (20, 5, 20, "target", (20, 20)),  # epoch 20 was interrupted mid-pass: finish it
    (25, 0, 20, "target", None),        # saved past the target
    (32, 0, 20, "target", None),        # overshot before this change: never train backwards
    (20, 0, 30, "target", (21, 30)),    # top up by raising the target
    (0, 0, 20, "additional", (1, 20)),  # legacy modes agree on a fresh run
    (12, 0, 20, "additional", (13, 32)),
    (12, 57, 20, "additional", (12, 31)),
    (20, 0, 20, "additional", (21, 40)),
]


@pytest.mark.parametrize("saved,current_iter,epochs,mode,expected", TARGET_CASES)
def test_epoch_window(saved, current_iter, epochs, mode, expected):
    assert epoch_window(saved, current_iter, epochs, mode) == expected


def test_default_mode_is_target():
    """The launcher and old configs omit the key -- that must mean target, not legacy."""
    assert epoch_window(12, 0, 20) == (13, 20)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown epochs_mode"):
        epoch_window(5, 0, 20, "nonsense")


def test_window_is_contiguous_and_bounded():
    for saved, ci, epochs, mode, expected in TARGET_CASES:
        if expected is None:
            continue
        first, last = expected
        assert first <= last
        assert last - first + 1 == max(1, last - first + 1)
        assert first >= 1
        if mode == "target":
            assert last == epochs


if __name__ == "__main__":
    failures = 0
    for case in TARGET_CASES:
        saved, ci, epochs, mode, expected = case
        got = epoch_window(saved, ci, epochs, mode)
        ok = got == expected
        failures += 0 if ok else 1
        print(("ok  " if ok else "FAIL") + f" saved={saved} current_iter={ci} epochs={epochs} mode={mode}"
              + f" -> {got} (expected {expected})")
    try:
        epoch_window(5, 0, 20, "nonsense")
    except ValueError as e:
        print("ok   unknown mode raises:", e)
    else:
        failures += 1
        print("FAIL unknown mode did not raise")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)
