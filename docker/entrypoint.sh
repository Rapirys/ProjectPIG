#!/usr/bin/env bash
set -e

export PYTHONPATH="/workspace/ProjectPIG:/workspace/ProjectPIG/Malmo/MalmoEnv:${PYTHONPATH:-}"
export MINERL_HEADLESS=1

cd /workspace/ProjectPIG/Malmo/Minecraft
./launchClient.sh -port 10000 -env >/tmp/minecraft.log 2>&1 &

cd /workspace/ProjectPIG/NaturalDreamer
exec python main.py --config minecraft.yml
