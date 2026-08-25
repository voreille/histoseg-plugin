#!/usr/bin/env bash
# Creates a timestamped SQL dump of the histoseg PostgreSQL database.
# Usage: ./scripts/pg_backup.sh [output_dir]
#   output_dir defaults to ./backups
set -euo pipefail

OUTPUT_DIR="${1:-./backups}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTPUT_DIR/histoseg_${TIMESTAMP}.sql"

docker exec histoseg-db pg_dump -U histoseg histoseg > "$OUTFILE"

echo "Backup written to $OUTFILE"
