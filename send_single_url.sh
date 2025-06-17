#!/bin/bash

# Configuration
API_URL="http://127.0.0.1:4050/api/login_candidates"
TASK_ID="101"
URL="aswad546.github.io"
FULL_URL="https://${URL}"
SCAN_DOMAIN="${URL}"
ID_NUM=1

# Create the JSON payload
JSON_PAYLOAD=$(cat <<EOF
{
  "task_id": "${TASK_ID}",
  "candidates": [
    {
      "id": ${ID_NUM},
      "url": "${FULL_URL}",
      "actions": null,
      "scan_domain": "${SCAN_DOMAIN}"
    }
  ]
}
EOF
)

echo "Sending URL: ${FULL_URL}"
echo "Payload: ${JSON_PAYLOAD}"

# Send the request
response=$(curl -s -w "\n%{http_code}" \
  -X POST \
  -H "Content-Type: application/json" \
  -d "${JSON_PAYLOAD}" \
  "${API_URL}")

# Extract response body and status code
http_code=$(echo "$response" | tail -n1)
response_body=$(echo "$response" | head -n -1)

echo "HTTP Status Code: ${http_code}"
echo "Response: ${response_body}"

# Check if successful
if [ "$http_code" -eq 200 ]; then
    echo "✅ Successfully sent URL!"
else
    echo "❌ Failed to send URL. HTTP Status: ${http_code}"
    exit 1
fi