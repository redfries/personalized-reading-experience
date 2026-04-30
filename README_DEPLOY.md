# DigitalOcean disposable GPU deployment for Personalized Reading Assistant

This pack lets you create a temporary DigitalOcean GPU Droplet, attach a persistent Volume, deploy the app from GitHub, and destroy the GPU Droplet when done.

The Volume remains, so Hugging Face model cache, profiles, outputs, and logs can survive between deployments.

## Files

- `deploy/env.example`: copy to `deploy/.env` and fill your DigitalOcean values.
- `deploy/cloud-init.template.yaml`: setup script that runs automatically on first boot.
- `deploy/create_volume_do.sh`: creates the persistent Volume once.
- `deploy/deploy_do.sh`: creates the GPU Droplet and deploys the app.
- `deploy/status_do.sh`: shows current Droplet and Volume info.
- `deploy/update_app_do.sh`: pulls latest GitHub code and restarts the app.
- `deploy/destroy_do.sh`: destroys the GPU Droplet but keeps the Volume.

## Normal workflow

```bash
cp deploy/env.example deploy/.env
nano deploy/.env

bash deploy/create_volume_do.sh
bash deploy/deploy_do.sh
bash deploy/status_do.sh
```

Open the app URL printed by the script.

When finished:

```bash
bash deploy/destroy_do.sh
```

## Important

Do not commit `deploy/.env` because it contains your real configuration.
