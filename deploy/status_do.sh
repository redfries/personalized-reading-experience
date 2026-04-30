#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing deploy/.env."
  exit 1
fi

source "$ENV_FILE"

if doctl compute droplet get "$DROPLET_NAME" >/dev/null 2>&1; then
  doctl compute droplet get "$DROPLET_NAME" --format ID,Name,PublicIPv4,Status,Memory,VCPUs,Disk,Region
  IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header)"
  echo "App URL: http://$IP"
else
  echo "No Droplet found with name: $DROPLET_NAME"
fi

echo ""
echo "Volume:"
doctl compute volume list --format ID,Name,Size,Region,DropletIDs | grep "$VOLUME_NAME" || true
