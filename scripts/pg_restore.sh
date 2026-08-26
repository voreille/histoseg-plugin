#!/usr/bin/env bash
# Restores the histoseg PostgreSQL database from a SQL dump.
# Usage: ./scripts/pg_restore.sh <backup_file.sql>
# WARNING: this drops and recreates all data in the database.
set -euo pipefail

BACKUP_FILE="${1:-}"
if [[ -z "$BACKUP_FILE" ]]; then
  echo "Usage: $0 <backup_file.sql>" >&2
  exit 1
fi

if [[ ! -f "$BACKUP_FILE" ]]; then
  echo "File not found: $BACKUP_FILE" >&2
  exit 1
fi

read -rp "This will overwrite the database. Continue? [y/N] " confirm
[[ "$confirm" =~ ^[yY]$ ]] || { echo "Aborted."; exit 0; }

docker exec histoseg-db psql -U histoseg -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;" histoseg
docker exec -i histoseg-db psql -U histoseg histoseg < "$BACKUP_FILE"

echo "Restore complete from $BACKUP_FILE"
