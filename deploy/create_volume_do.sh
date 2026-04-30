#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing deploy/.env. Copy deploy/env.example to deploy/.env and fill values."
  exit 1
fi

source "$ENV_FILE"

EXISTING="$(doctl compute volume list --format ID,Name,Region --no-header | awk -v name="$VOLUME_NAME" '$2 == name {print $1}' | head -n 1 || true)"

if [ -n "$EXISTING" ]; then
  echo "Volume already exists: $VOLUME_NAME ($EXISTING)"
  exit 0
fi

echo "Creating volume: $VOLUME_NAME"
echo "Region: $DO_REGION"
echo "Size: $VOLUME_SIZE"

doctl compute volume create "$VOLUME_NAME" \
  --region "$DO_REGION" \
  --size "$VOLUME_SIZE" \
  --desc "Persistent storage for personalized reading assistant"

echo "Volume created."
doctl compute volume list
