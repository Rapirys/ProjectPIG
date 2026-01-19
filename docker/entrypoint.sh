#!/usr/bin/env bash
set -e

export PYTHONPATH="/workspace/ProjectPIG:/workspace/ProjectPIG/Malmo/MalmoEnv:${PYTHONPATH:-}"
export MINERL_HEADLESS=1

cd /workspace/ProjectPIG
exec python dreamerv3-torch/dreamer.py --configs minecraft --task CustomMinecraft_findgoal --logdir ./logdir/minecraft-3d
