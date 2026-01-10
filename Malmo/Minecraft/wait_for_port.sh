#!/bin/bash
PORT="${1:-10000}"
HOST="${2:-127.0.0.1}"

echo >&1 "waiting for ${HOST}:${PORT} to be open"
while true; do
  nc -z "$HOST" "$PORT" >/dev/null 2>&1
  if [ $? -eq 0 ]; then
    break
  else
    echo >&1 "${HOST}:${PORT} is still closed"
    sleep 1
  fi
done

# add an extra sleep because we may be too fast detecting the port, and JVM crashes
sleep 3
