#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
TEMPLATE_FILE="${SCRIPT_DIR}/cloud-init.template.yaml"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing deploy/.env. Copy deploy/env.example to deploy/.env and fill values."
  exit 1
fi

if [ ! -f "$TEMPLATE_FILE" ]; then
  echo "Missing deploy/cloud-init.template.yaml."
  exit 1
fi

source "$ENV_FILE"

if doctl compute droplet get "$DROPLET_NAME" >/dev/null 2>&1; then
  IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header)"
  echo "Droplet already exists: $DROPLET_NAME"
  echo "App URL: http://$IP"
  exit 0
fi

VOLUME_ID="$(doctl compute volume list --format ID,Name,Region --no-header | awk -v name="$VOLUME_NAME" -v region="$DO_REGION" '$2 == name && $3 == region {print $1}' | head -n 1 || true)"

if [ -z "$VOLUME_ID" ]; then
  echo "Volume not found: $VOLUME_NAME in region $DO_REGION"
  echo "Run this first: bash deploy/create_volume_do.sh"
  exit 1
fi

TMP_USER_DATA="$(mktemp)"

sed \
  -e "s|__APP_REPO__|${APP_REPO}|g" \
  -e "s|__APP_BRANCH__|${APP_BRANCH}|g" \
  -e "s|__EMBEDDING_MODEL_NAME__|${EMBEDDING_MODEL_NAME}|g" \
  -e "s|__VOLUME_NAME__|${VOLUME_NAME}|g" \
  "$TEMPLATE_FILE" > "$TMP_USER_DATA"

echo "Creating GPU Droplet: $DROPLET_NAME"
echo "Region: $DO_REGION"
echo "Size: $DO_SIZE"
echo "Image: $DO_IMAGE"
echo "Volume: $VOLUME_NAME ($VOLUME_ID)"
echo "Model: $EMBEDDING_MODEL_NAME"

doctl compute droplet create "$DROPLET_NAME" \
  --region "$DO_REGION" \
  --size "$DO_SIZE" \
  --image "$DO_IMAGE" \
  --ssh-keys "$DO_SSH_KEY" \
  --volumes "$VOLUME_ID" \
  --user-data-file "$TMP_USER_DATA" \
  --wait

rm -f "$TMP_USER_DATA"

IP="$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header)"

echo ""
echo "Droplet created."
echo "IP: $IP"
echo "App URL: http://$IP"
echo ""
echo "First startup can take several minutes because packages and models may download."
echo "To watch setup logs:"
echo "ssh root@$IP"
echo "tail -f /root/setup-reading-app.log"
echo ""
echo "To watch app logs after setup:"
echo "journalctl -u reading-app -f"
