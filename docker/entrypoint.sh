#!/usr/bin/env bash
# docker/entrypoint.sh
set -euo pipefail

export PYTHONPATH="/workspace/ProjectPIG:/workspace/ProjectPIG/Malmo/MalmoEnv:${PYTHONPATH:-}"

# --- Software OpenGL (Mesa/llvmpipe) + LWJGL flags
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=2.1
export LIBGL_DRI3_DISABLE=1
export _JAVA_OPTIONS="${_JAVA_OPTIONS:-} -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true"

# --- Start Minecraft (Malmo) headless via Xvfb
MC_DIR="/workspace/ProjectPIG/Malmo/Minecraft"
MCLOG="/tmp/minecraft.log"
: > "$MCLOG"

pushd "$MC_DIR" >/dev/null

# Keep your version.properties behavior
(echo -n "malmomod.version=" && cat ../VERSION) > ./src/main/resources/version.properties || true

XVFB_ARGS="-screen 0 1280x720x24 -ac +extension GLX +extension RANDR +render -noreset"
xvfb-run -a -s "$XVFB_ARGS" bash -lc 'xrandr -q | head -n 30' || true
nohup xvfb-run -a -s "$XVFB_ARGS" ./launchClient.sh -port 10000 -env >>"$MCLOG" 2>&1 </dev/null &
MC_PID=$!
popd >/dev/null

trap 'kill "$MC_PID" >/dev/null 2>&1 || true' EXIT INT TERM

# --- Wait for Malmo port
for _ in $(seq 1 200); do
  if nc -z 127.0.0.1 10000 >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

# --- Run NaturalDreamer
cd /workspace/ProjectPIG/NaturalDreamer
exec python main.py --config minecraft.yml
