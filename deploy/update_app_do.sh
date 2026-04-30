#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing deploy/.env."
  exit 1
fi

source "$ENV_FILE"

IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header)"

if [ -z "$IP" ]; then
  echo "No running Droplet found."
  exit 1
fi

ssh root@"$IP" "cd /opt/personalized-reading-experience && git pull origin ${APP_BRANCH} && source .venv/bin/activate && pip install -r requirements.txt && systemctl restart reading-app && systemctl status reading-app --no-pager"

echo "Updated and restarted. App URL: http://$IP"
