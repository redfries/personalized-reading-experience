#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing deploy/.env."
  exit 1
fi

source "$ENV_FILE"

if ! doctl compute droplet get "$DROPLET_NAME" >/dev/null 2>&1; then
  echo "No Droplet found with name: $DROPLET_NAME"
  echo "Nothing to destroy."
  exit 0
fi

echo "This will destroy GPU Droplet: $DROPLET_NAME"
echo "The persistent volume '$VOLUME_NAME' will NOT be deleted."
read -p "Type DESTROY to continue: " CONFIRM

if [ "$CONFIRM" != "DESTROY" ]; then
  echo "Cancelled."
  exit 0
fi

doctl compute droplet delete "$DROPLET_NAME" --force

echo "Droplet destroyed. GPU billing for this Droplet has stopped."
echo "Persistent volume still exists: $VOLUME_NAME"
