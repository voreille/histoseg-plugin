# scripts/smoke_job.sh
#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8090}"
SLIDE_URI="${1:?Usage: scripts/smoke_job.sh file:///path/to/slide.svs}"
MODEL_ID="${MODEL_ID:-default}"

echo "Submitting job..."

response="$(
  curl -sS -X POST "$API_URL/jobs" \
    -H "Content-Type: application/json" \
    -d "{
      \"items\": [
        {
          \"slide_uri\": \"$SLIDE_URI\",
          \"model_id\": \"$MODEL_ID\"
        }
      ]
    }"
)"

echo "$response" | jq .

job_id="$(echo "$response" | jq -r '.job_id')"

echo "Polling job $job_id..."

while true; do
  status_response="$(curl -sS "$API_URL/jobs/$job_id")"
  echo "$status_response" | jq .

  status="$(echo "$status_response" | jq -r '.status')"

  if [[ "$status" == "completed" || "$status" == "failed" || "$status" == "partial" ]]; then
    break
  fi

  sleep 2
done

echo "Final status: $status"