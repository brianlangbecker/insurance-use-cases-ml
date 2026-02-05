#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/docker.sh up        # build + run in background (Docker Compose)
  ./scripts/docker.sh dev       # dev mode (uvicorn --reload, bind-mount repo)
  ./scripts/docker.sh down      # stop and remove containers
  ./scripts/docker.sh logs      # tail logs
  ./scripts/docker.sh ps        # show running services

Environment variables:
  HOST_PORT=<port>  Host port to publish the app on (container always listens on 8080).

Notes:
  - If HOST_PORT is not set, this script will try a few common ports (8080, 18080, 3000, 5000)
    and pick the first free one.
EOF
}

port_in_use() {
  local port="$1"

  # On macOS, ports published by Docker Desktop may not always show up in `lsof`.
  # Check both:
  #   1) OS-level listeners
  #   2) Other Docker containers already publishing the port
  if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    return 0
  fi

  # If Docker isn't running, don't fail the script—just treat as "not in use".
  if docker ps --format '{{.Ports}}' 2>/dev/null | grep -Eq "(^|[ ,])([0-9.]*:)?${port}->"; then
    return 0
  fi

  return 1
}

pick_host_port() {
  local candidates=(8080 18080 3000 5000)
  local port

  for port in "${candidates[@]}"; do
    if ! port_in_use "$port"; then
      echo "$port"
      return 0
    fi
  done

  # Fallback: pick a port in the ephemeral range and hope it's free.
  for port in $(seq 49152 49252); do
    if ! port_in_use "$port"; then
      echo "$port"
      return 0
    fi
  done

  echo "No free port found" >&2
  return 1
}

cmd="${1:-up}"

case "$cmd" in
  -h|--help|help)
    usage
    exit 0
    ;;
  up)
    HOST_PORT="${HOST_PORT:-$(pick_host_port)}"
    if port_in_use "$HOST_PORT"; then
      echo "HOST_PORT=${HOST_PORT} is already in use. Pick another one, e.g.:" >&2
      echo "  HOST_PORT=18080 ./scripts/docker.sh up" >&2
      exit 2
    fi

    HOST_PORT="$HOST_PORT" docker compose up --build -d

    echo "App is running:" >&2
    echo "  http://localhost:${HOST_PORT}" >&2
    echo "Health check:" >&2
    echo "  curl -fsS http://localhost:${HOST_PORT}/healthz" >&2
    ;;
  dev)
    HOST_PORT="${HOST_PORT:-$(pick_host_port)}"
    if port_in_use "$HOST_PORT"; then
      echo "HOST_PORT=${HOST_PORT} is already in use. Pick another one, e.g.:" >&2
      echo "  HOST_PORT=18080 ./scripts/docker.sh dev" >&2
      exit 2
    fi

    echo "Starting dev server on http://localhost:${HOST_PORT} (ctrl-c to stop)" >&2
    HOST_PORT="$HOST_PORT" docker compose --profile dev up --build app-dev
    ;;
  down)
    docker compose down
    ;;
  logs)
    docker compose logs -f --tail=200
    ;;
  ps)
    docker compose ps
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
