import math
import pathlib
import warnings
from typing import Dict, Tuple

import cv2
import gymnasium as gym
import numpy as np

import malmoenv

DEFAULT_ACTIONS: dict[str, int] = {
    "move": 0,
    "jump": 0,
    "sneak": 0,
    "sprint": 0,
    "attack": 0,
    "use": 0,
    "moveMouse": 0,
}

BASIC_ACTIONS: dict[str, dict] = {
    # movement / combat
    "noop": {},
    "attack": {"attack": 1},
    "use": {"use": 1},
    "forward": {"move": 1},
    "back": {"move": 2},
    "right": {"move": 3},
    "left": {"move": 4},
    "jump": {"jump": 1},
    "jump_forward": {"jump": 1, "move": 1},
    "sneak": {"sneak": 1},
    "sprint": {"sprint": 1},

    # mouse (explicit indices 0..8 from your CommandParser)
    "moveMouse 0 0":   {"moveMouse": 0},
    "moveMouse 100 0": {"moveMouse": 1},
    "moveMouse -100 0":{"moveMouse": 2},
    "moveMouse 0 50":  {"moveMouse": 3},
    "moveMouse 0 -50": {"moveMouse": 4},
    "moveMouse 70 35": {"moveMouse": 5},
    "moveMouse -70 35": {"moveMouse": 6},
    "moveMouse 70 -35": {"moveMouse": 7},
    "moveMouse -70 -35": {"moveMouse": 8},

    # hotbar (0..8 -> slots 1..9). Default is “not sent”, so slot 1
    # is only sent when explicitly chosen.
    "hotbar_1": {"hotbar": 0},
    "hotbar_2": {"hotbar": 1},
    "hotbar_3": {"hotbar": 2},
}



class _MalmoAdapter(gym.Env):
    """Thin adapter around malmoenv.Env to expose Gymnasium 5-tuple and RGB frames."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, mission_xml: str, reshape: bool = True):
        super().__init__()
        self._env = malmoenv.make()  # malmoenv.core.Env()
        self._env.init(
            mission_xml,
            port=10000,
            server="127.0.0.1",
            role=0,
            exp_uid=None,
            episode=0,
            action_filter={},
            resync=0,
            reshape=reshape,
            synchronous=True,
        )
        # Spaces come from malmoenv
        self.action_space = self._env.action_space        # Dict[str, Discrete]
        self.observation_space = self._env.observation_space  # Box(H,W,3|4,uint8)

        self._last_obs = None
        self._first = True

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._env.set_episode_seed(seed)
        obs, info = self._env.reset()
        self._last_obs = obs
        self._first = True
        return obs, info or {}

    def step(self, action_dict: Dict[str, int]):
        obs, reward, done, info = self._env.step(action_dict)
        terminated = bool(done)
        truncated = False
        self._last_obs = obs
        # Compose info dict; keep original keys, add flags.
        info = {} if info is None else (info if isinstance(info, dict) else {"raw_info": info})
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class MalmoMinecraft(gym.Env):
    """Dreamer-friendly wrapper:
       - Discrete macro action space (BASIC_ACTIONS)
       - Action repeat
       - Time limit with Gymnasium terminated/truncated
       - 64x64 RGB frames (uint8 HxW×3)
       - Adds is_first/is_last/is_terminal; derives health from info when present
    """

    def __init__(
        self,
        mission_xml_path: str,
        repeat: int = 1,
        size: Tuple[int, int] = (64, 64),
        time_limit: int | None = None,
    ):
        super().__init__()
        self._repeat = int(repeat)
        self._time_limit = int(time_limit) if time_limit else None
        self._step_count = 0

        mission_xml = pathlib.Path(mission_xml_path).read_text()
        self._env = _MalmoAdapter(mission_xml, reshape=True)

        # Observation space: force RGB only (H,W,3)
        H, W, C = self._env.observation_space.shape
        self.size = size
        self._need_resize = (H, W) != size
        if self._need_resize:
            warnings.warn(f"Mission video size {H}x{W} != requested {size}")
        assert C == 3, "Mission must output RGB"
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (size[0], size[1], 3), dtype=np.uint8),
                "depth": gym.spaces.Box(0.0, np.inf, (size[0], size[1]), dtype=np.float32),
                "model_view_metrix": gym.spaces.Box(-np.inf, np.inf, (4, 4), dtype=np.float32),
                "projection_metrix": gym.spaces.Box(-np.inf, np.inf, (4, 4), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, (), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.uint8),
                "health": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            }
        )

        # Discrete macro actions
        self._action_names = tuple(BASIC_ACTIONS.keys())
        self._action_values = tuple(BASIC_ACTIONS.values())
        self.action_space = gym.spaces.Discrete(len(self._action_values))
        self._first = True

    # ---- Gymnasium API ----
    def reset(self, *, seed: int | None = None, options=None):
        obs, info = self._env.reset(seed=seed)
        self._step_count = 0
        self._first = True
        return self._format_obs(obs, info, is_first=True, is_last=False, is_terminal=False), {}

    def step(self, action: int):
        # Expand macro to low-level command dict
        low = self._expand_actions(self._action_values[int(action)])
        reward_sum = 0.0
        terminated = False
        truncated = False
        info_last = {}

        # Repeat action if requested
        for i in range(self._repeat):
            obs, rew, term, trunc, info = self._env.step(low)
            reward_sum += float(rew)
            info_last = info or {}
            terminated = bool(term) or terminated
            truncated = bool(trunc) or truncated
            if terminated or truncated:
                break

        self._step_count += 1
        if (self._time_limit is not None) and (self._step_count >= self._time_limit) and not (terminated or truncated):
            truncated = True

        out_obs = self._format_obs(obs, info_last, is_first=False,
                                   is_last=terminated or truncated,
                                   is_terminal=terminated)

        return out_obs, reward_sum, terminated, truncated, info_last

    # ---- Helpers ----
    def _expand_actions(self, macro: dict) -> dict[str, int]:
        actions = DEFAULT_ACTIONS.copy()
        actions.update((k,v) for k, v in macro.items())
        return actions

    def _format_obs(self, rgb: np.ndarray, info: dict, *, is_first: bool, is_last: bool, is_terminal: bool):
        depth = info.get("depth")
        if self._need_resize:
            rgb = cv2.resize(rgb, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)

        health = np.float32([(info.get("life", 20) / 20.0)])
        out = {
            "image": rgb.astype(np.uint8),
            "depth": depth.astype(np.float32),
            "model_view_metrix": info.get("model_view_metrix").astype(np.float32),
            "projection_metrix": info.get("projection_metrix").astype(np.float32),
            "is_first": np.array(is_first, np.uint8),
            "is_last": np.array(is_last, np.uint8),
            "is_terminal": np.array(is_terminal, np.uint8),
            "health": health,
        }
        return out

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
