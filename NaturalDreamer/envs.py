import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TransformAction

import malmoenv
import numpy as np

def getEnvProperties(env):
    assert env.observation_space.shape == (3, 64, 64), "The observation space is incorrect"
    assert isinstance(env.action_space, gym.spaces.Box), "Expected flat Box action space"
    observationShape = env.observation_space.shape

    # Read hybrid meta if present; otherwise assume all continuous.
    cont_dim = getattr(env.action_space, "_cont_dim", env.action_space.shape[0])
    disc_segments = getattr(env.action_space, "_disc_segments", [])
    disc_n = sum(disc_segments) #TODO disc_n is not used

    low  = env.action_space.low.reshape(-1).tolist()
    high = env.action_space.high.reshape(-1).tolist()
    cont_low  = low[:cont_dim]
    cont_high = high[:cont_dim]
    return observationShape, disc_segments, (cont_dim, cont_low, cont_high)

class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation
    
class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs


class FlatDictActionSpace(gym.ActionWrapper):
    """
    Flatten Dict action space into a single Box:
      [all continuous (from Box keys) | concatenated one-hot segments (from Discrete keys)]

    Exposes:
      action_space._cont_dim  = total continuous length
      action_space._disc_n    = total discrete one-hot length
    """
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Dict), "Expects Dict action space"
        super().__init__(env)

        spaces = env.action_space.spaces
        self.continuous = [(k, sp) for k, sp in spaces.items() if isinstance(sp, gym.spaces.Box)]
        self.discrete   = [(k, sp) for k, sp in spaces.items() if isinstance(sp, gym.spaces.Discrete)]
        assert len(self.continuous) + len(self.discrete) == len(spaces), "Only Box and Discrete actions are supported"

        # Sizes
        self.continuous_dim = sum(math.prod(sp.shape) for _, sp in self.continuous)
        self.discrete_segments = [sp.n for _, sp in self.discrete]
        self.discrete_dim      = sum(self.discrete_segments)
        self.action_space._disc_segments = self.discrete_segments

        # Bounds: continuous lows/highs, then zeros/ones for the one-hot tail
        continuous_lows = [sp.low.ravel() for _, sp in self.continuous]
        continuous_highs = [sp.high.ravel() for _, sp in self.continuous]
        discrete_low = [np.zeros(self.discrete_dim, dtype=np.float32)]
        discrete_high = [np.ones(self.discrete_dim, dtype=np.float32)]
        low  = np.concatenate(continuous_lows + discrete_low)
        high = np.concatenate(continuous_highs + discrete_high)

        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space._cont_dim = self.continuous_dim
        self.action_space._disc_n   = self.discrete_dim
        self.action_space._disc_segments  = list(self.discrete_segments)

    # --- helpers ---
    def _validate_one_hot_index(self, segment: np.ndarray, idx: int, key: str) -> int:
        ref = np.zeros_like(segment); ref[idx] = 1.0
        if not np.allclose(segment, ref):
            raise ValueError(f"Invalid one-hot action for '{key}': {segment}")

    # --- Box(flat) -> original MineRL Dict ---
    def action(self, a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        assert a.shape[0] == self.continuous_dim + self.discrete_dim, f"Invalid flat action length: got {a.shape[0]}, expected {self.continuous_dim + self.discrete_dim}"

        out, pos = {}, 0

        # continuous prefix
        for key, sp in self.continuous:
            size = int(np.prod(sp.shape))
            out[key] = a[pos:pos+size].reshape(sp.shape)
            pos += size
        # discrete tail (concatenated segments)
        macro, pos = a[pos:], 0
        for key, sp in self.discrete:
            segment = macro[pos:pos+sp.n]
            idx = int(np.argmax(segment))
            self._validate_one_hot_index(segment, idx, key)
            out[key] = idx
            pos += sp.n
        return out


class MalmoAdapter(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, malmo_env):
        super().__init__()
        self.env = malmo_env
        self.action_space = malmo_env.action_space
        self.observation_space = malmo_env.observation_space

    def reset(self, *, seed=None, options=None): return self.env.reset(), {}  # Gymnasium requires (obs, info)
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, bool(done), False, info #The info is likely to be json string, not dict
    def render(self): return self.env.render()
    def close(self): self.env.close()

# --- factory ---
def make_env(config):
    environment_name = config.environmentName

    if environment_name == "CarRacing-v3":
        base = gym.make(environment_name)
        eval_base = gym.make(environment_name, render_mode="rgb_array")

    if environment_name in {"Minecraft"}:
        xml = Path("../Malmo/MalmoEnv/missions/findthegoal.xml").read_text()
        base = malmoenv.make()
        base.init(
            xml,
            10000,
            server="127.0.0.1",
            server2=None,
            port2=None,  #
            role=0,
            exp_uid="test1",
            episode=0,
            resync=0, reshape=True) #TODO disable info via step_options

        base = TransformAction(
            MalmoAdapter(base),
            func=lambda a: a["actions"],
            action_space=spaces.Dict({"actions": base.action_space}),
        )
        base = FlatDictActionSpace(base)
        eval_base = base

    # TODO Different resize for minecraft
    env = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(base, (64, 64))))
    env_eval = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(eval_base, (64, 64))))
    return env, env_eval
