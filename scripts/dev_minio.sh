#!/usr/bin/env bash
# Start / stop / status local MinIO for development.
#
#   scripts/dev_minio.sh start   # start in background, create buckets if missing
#   scripts/dev_minio.sh stop    # stop the running MinIO process
#   scripts/dev_minio.sh status  # show whether MinIO is running
#   scripts/dev_minio.sh logs    # tail the MinIO log
#
# Requires:
#   brew install minio/stable/minio minio/stable/mc

set -euo pipefail

DATA_DIR="${HOME}/cortyze-minio-data"
LOG_FILE="${HOME}/cortyze-minio.log"
PID_FILE="${HOME}/.cortyze-minio.pid"
API_PORT=9000
CONSOLE_PORT=9001
ROOT_USER="minioadmin"
ROOT_PASS="minioadmin"
BUCKETS=(cortyze-uploads cortyze-predictions)

cmd="${1:-start}"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "$cmd" in
  start)
    if is_running; then
      echo "MinIO already running (pid $(cat "$PID_FILE"))"
    else
      mkdir -p "$DATA_DIR"
      MINIO_ROOT_USER="$ROOT_USER" MINIO_ROOT_PASSWORD="$ROOT_PASS" \
        nohup minio server "$DATA_DIR" --console-address ":$CONSOLE_PORT" \
        > "$LOG_FILE" 2>&1 &
      echo $! > "$PID_FILE"
      echo "MinIO starting (pid $(cat "$PID_FILE")); waiting for health..."
      for _ in {1..20}; do
        if curl -fs "http://localhost:$API_PORT/minio/health/live" > /dev/null; then
          echo "MinIO healthy on http://localhost:$API_PORT (console: http://localhost:$CONSOLE_PORT)"
          break
        fi
        sleep 0.5
      done
    fi
    mc alias set local "http://localhost:$API_PORT" "$ROOT_USER" "$ROOT_PASS" > /dev/null
    for bucket in "${BUCKETS[@]}"; do
      mc mb -p "local/$bucket" 2>/dev/null || true
    done
    echo "Buckets: ${BUCKETS[*]}"
    ;;
  stop)
    if is_running; then
      pid=$(cat "$PID_FILE")
      kill "$pid"
      rm -f "$PID_FILE"
      echo "MinIO stopped (pid $pid)"
    else
      echo "MinIO is not running"
      rm -f "$PID_FILE"
    fi
    ;;
  status)
    if is_running; then
      echo "MinIO running (pid $(cat "$PID_FILE")) on http://localhost:$API_PORT"
    else
      echo "MinIO is not running"
    fi
    ;;
  logs)
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "usage: $0 {start|stop|status|logs}" >&2
    exit 1
    ;;
esac
