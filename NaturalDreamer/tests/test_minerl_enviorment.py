# NaturalDreamer/tests/test_minerl_environment.py
import logging

import numpy as np
import pytest
import gymnasium as gym

from NaturalDreamer.envs import make_env
from NaturalDreamer.utils import loadConfig, seedEverything

logger = logging.getLogger(__name__)


# ---------- helpers ----------
def _flat_noop(space: gym.spaces.Box) -> np.ndarray:
    """Valid no-op for our flattened Box action:
    - continuous part = zeros
    - each discrete segment = one-hot with index 0 ('off')
    """
    cont_dim   = getattr(space, "_cont_dim", space.shape[0])
    segments   = getattr(space, "_disc_segments", [])
    total_disc = sum(segments)
    a = np.zeros(cont_dim + total_disc, dtype=np.float32)

    # mark index 0 in each discrete segment
    pos = cont_dim
    for n in segments:
        if n > 0:
            a[pos] = 1.0
        pos += n
    return a


def _segment_starts(segments):
    starts, s = [], 0
    for n in segments:
        starts.append(s)
        s += n
    return starts


def _press_segment(space: gym.spaces.Box, seg_idx: int, choice: int = 1) -> np.ndarray:
    """Return a flat action that presses a specific discrete segment (one-hot at 'choice'),
    everything else remains 'off', continuous stays zero."""
    cont_dim = getattr(space, "_cont_dim", space.shape[0])
    segments = getattr(space, "_disc_segments", [])
    a = _flat_noop(space)

    # bounds check
    assert 0 <= seg_idx < len(segments), f"seg_idx {seg_idx} out of range (len={len(segments)})"
    n = segments[seg_idx]
    assert n >= 1, f"segment {seg_idx} has invalid size {n}"

    # compute segment start in the flat vector tail
    starts = _segment_starts(segments)
    start = cont_dim + starts[seg_idx]

    # clear that segment and set desired option (clamped to valid range)
    a[start:start+n] = 0.0
    a[start + min(choice, n - 1)] = 1.0
    return a


def _with_camera_delta(space: gym.spaces.Box, yaw_delta: float = 0.0, pitch_delta: float = 0.0) -> np.ndarray:
    """Continuous camera deltas in the prefix (assumes camera is shape (2,))."""
    cont_dim = getattr(space, "_cont_dim", space.shape[0])
    a = _flat_noop(space)
    if cont_dim >= 2:
        # convention: index 0 = pitch, index 1 = yaw (MineRL 'camera' order)
        a[0] += float(pitch_delta)
        a[1] += float(yaw_delta)
    return a


# ---------- pytest fixtures ----------
@pytest.fixture(scope="module")
def env():
    cfg = loadConfig("mine-rl.yml")
    seedEverything(cfg.seed)
    e, _ = make_env(cfg.environmentName)
    yield e
    e.close()


# ---------- tests ----------
def test_noop_runs(env):
    """Step the env with a valid no-op for 50 ticks."""
    obs = env.reset()
    for _ in range(50):
        a = _flat_noop(env.action_space)
        obs, reward, done = env.step(a)
        if done:
            obs = env.reset()


def test_camera_sweep(env):
    """Sweep yaw over 50 ticks using the continuous prefix."""
    obs = env.reset()
    steps = 50
    yaw_per_step = 180.0 / steps  # well within MineRL camera bounds [-180, 180]
    for _ in range(steps):
        a = _with_camera_delta(env.action_space, yaw_delta=yaw_per_step, pitch_delta=0.0)
        obs, reward, done = env.step(a)
        if done:
            obs = env.reset()


def test_each_discrete_key(env):
    """For every discrete segment, press option 1 (if exists) for 50 ticks."""
    segments = getattr(env.action_space, "_disc_segments", [])
    if not segments:
        pytest.skip("No discrete segments exposed by the flattened action space.")
    obs = env.reset()
    for seg_idx, n in enumerate(segments):
        # obs = env.reset()
        # press this key for 50 ticks (use choice=1 when n>=2, else 0)
        choice = 1 if n >= 2 else 0
        for i in range(50):
            print(f"Testing action {seg_idx+1} step {i}")
            a = _press_segment(env.action_space, seg_idx, choice=choice)
            obs, reward, done = env.step(a)
            if done:
                print("done")
                obs = env.reset()
