#!/usr/bin/env bash
set -e

export PYTHONPATH="/workspace/ProjectPIG:/workspace/ProjectPIG/Malmo/MalmoEnv:${PYTHONPATH:-}"
export MINERL_HEADLESS=1

cd /workspace/ProjectPIG/Malmo/Minecraft
./launchClient.sh -port 10000 -env >/tmp/minecraft.log 2>&1 &

/workspace/ProjectPIG/Malmo/Minecraft/wait_for_port.sh 10000 127.0.0.1

cd /workspace/ProjectPIG/dreamerv3-torch
exec python dreamer.py --configs minecraft --task CustomMinecraft_findgoal --logdir ./logdir/minecraft
