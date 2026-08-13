#!/usr/bin/env bash
set -e

docker compose \
  -f docker-compose.yaml \
  -f docker-compose.override.yaml \
  -f docker-compose.cpu.yaml \
  up --build