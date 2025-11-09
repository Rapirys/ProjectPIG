import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml
import os
import attridict
import gymnasium as gym
import csv
import pandas as pd
import plotly.graph_objects as pgo
import time

from malmoenv.world_tracking.utils import (
    WorldUpdate,
    ChunkLoad,
    ChunkUnload,
    SectionBlob,
)


def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def findFile(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def loadConfig(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = findFile(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return attridict(config)


def getEnvProperties(env):
    observationShape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discreteActionBool = True
        actionSize = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discreteActionBool = False
        actionSize = env.action_space.shape[0]
    else:
        raise Exception
    return observationShape, discreteActionBool, actionSize


def saveLossesToCSV(filename, metrics):
    fileAlreadyExists = os.path.isfile(filename + ".csv")
    with open(filename + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not fileAlreadyExists:
            writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


def plotMetrics(filename, title="", savePath="metricsPlot", window=10):
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    data = pd.read_csv(filename)
    fig = pgo.Figure()

    colors = [
        "gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
    num_colors = len(colors)

    for idx, column in enumerate(data.columns):
        if column in ["envSteps", "gradientSteps"]:
            continue
        
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))
        
        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2)))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=30),
            yanchor='top'
        ),
        xaxis=dict(
            title="Gradient Steps",
            showgrid=True,
            zeroline=False,
            position=0
        ),
        yaxis_title="Value",
        template="plotly_dark",
        height=1080,
        width=1920,
        margin=dict(t=60, l=40, r=40, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="White",
            borderwidth=2,
            font=dict(size=12)
        )
    )

    if not savePath.endswith(".html"):
        savePath += ".html"
    fig.write_html(savePath)


def sequentialModel1D(inputSize, hiddenSizes, outputSize, activationFunction="Tanh", finishWithActivation=False):
    activationFunction = getattr(nn, activationFunction)()
    layers = []
    currentInputSize = inputSize

    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(activationFunction)
        currentInputSize = hiddenSize
    
    layers.append(nn.Linear(currentInputSize, outputSize))
    if finishWithActivation:
        layers.append(activationFunction)

    return nn.Sequential(*layers)


def computeLambdaValues(rewards, values, continues, lambda_=0.95):
    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    for i in reversed(range(rewards.shape[-1])):
        returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
        bootstrap = returns[:, i]
    return returns


def ensureParentFolders(*paths):
    for path in paths:
        parentFolder = os.path.dirname(path)
        if parentFolder and not os.path.exists(parentFolder):
            os.makedirs(parentFolder, exist_ok=True)


class Moments(nn.Module):
    def __init__( self, device, decay = 0.99, min_=1, percentileLow = 0.05, percentileHigh = 0.95):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_)
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        self.low = self._decay*self.low + (1 - self._decay)*low
        self.high = self._decay*self.high + (1 - self._decay)*high
        inverseScale = torch.max(self._min, self.high - self.low)
        return self.low.detach(), inverseScale.detach()

def symlog(x):
    # sign(x) * log(1 + |x|)
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _now_sync():
    """Wall-clock time with a CUDA sync so GPU kernels are accounted for."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


import numpy as np

def collapse_world_updates(updates: list[WorldUpdate]) -> WorldUpdate:
    #TODO horrible and aggly function
    """
    Collapse several consecutive WorldUpdate objects into one:
      • Chunks: keep only the final event per (cx, cz) in the window.
        - If final event is load  -> emit ChunkLoad with its latest sections.
        - If final event is unload-> emit ChunkUnload.
      • Blocks: last-write-wins by (x, y, z).
    """
    if not updates:
        return WorldUpdate([], [], None)

    last_chunk_event: dict[tuple[int, int], tuple[str, list[SectionBlob] | None]] = {}
    for upd in updates:
        for ev in upd.chunk_unloads:
            last_chunk_event[(ev.cx, ev.cz)] = ("unload", None)
        for ev in upd.chunk_loads:
            last_chunk_event[(ev.cx, ev.cz)] = ("load", ev.sections)

    chunk_loads = [
        ChunkLoad(cx=cx, cz=cz, sections=sections)
        for (cx, cz), (kind, sections) in last_chunk_event.items()
        if kind == "load"
    ]
    chunk_unloads = [
        ChunkUnload(cx, cz)
        for (cx, cz), (kind, _) in last_chunk_event.items()
        if kind == "unload"
    ]

    # ---- Collapse block updates (last write wins) ----
    arrs = [u.block_updates for u in updates if (u.block_updates is not None and u.block_updates.size > 0)]
    dtype = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("state", "<u2")])
    blocks = np.concatenate(arrs, axis=0) if arrs else np.empty(0, dtype=dtype)

    # final-unloaded chunk coordinates as a structured array
    unloaded_pairs = np.array(
        [(cx, cz) for (cx, cz), (kind, _) in last_chunk_event.items() if kind == "unload"],
        dtype=[("cx", "<i4"), ("cz", "<i4")],
    )

    # vectorized mask: keep blocks whose (cx,cz) is NOT in finally-unloaded set
    pairs = np.empty(blocks.shape[0], dtype=unloaded_pairs.dtype)
    pairs["cx"] = (blocks["x"] >> 4).astype(np.int32)
    pairs["cz"] = (blocks["z"] >> 4).astype(np.int32)
    keep = ~np.isin(pairs, unloaded_pairs)  # works even if unloaded_pairs is empty
    blocks = blocks[keep]

    # last-write-wins per (x,y,z): reverse, unique on keys, map indices back
    rev_blocks = blocks[::-1]
    rev_keys = rev_blocks[["x", "y", "z"]]
    _, first_idx_rev = np.unique(rev_keys, return_index=True)
    sel = (rev_blocks.shape[0] - 1) - first_idx_rev
    sel.sort()
    block_updates = blocks[sel]

    if block_updates.size == 0:
        block_updates = None

    return WorldUpdate(chunk_loads=chunk_loads, chunk_unloads=chunk_unloads, block_updates=block_updates)
