#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [gpu|cpu] [dev|prod] [compose args...]" >&2
  echo "" >&2
  echo "  gpu dev   base + gpu overlay + dev override (auto-loaded)" >&2
  echo "  cpu dev   base + cpu overlay + dev override (auto-loaded)" >&2
  echo "  gpu prod  base + gpu overlay (no dev mounts)" >&2
  echo "  cpu prod  base + cpu overlay (no dev mounts)" >&2
  exit 1
}

DEVICE="${1:-}"
ENV="${2:-}"

[[ -z "$DEVICE" || -z "$ENV" ]] && usage

shift 2

FILES=(-f docker-compose.yaml)

case "$DEVICE" in
  gpu) FILES+=(-f docker-compose.gpu.yaml) ;;
  cpu) FILES+=(-f docker-compose.cpu.yaml) ;;
  *) usage ;;
esac

case "$ENV" in
  dev)  FILES+=(-f docker-compose.override.yaml) ;;
  prod) ;;
  *) usage ;;
esac

exec docker compose "${FILES[@]}" "$@"
