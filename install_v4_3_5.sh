#!/usr/bin/env bash
set -euo pipefail

if [ ! -f "app.py" ]; then
  echo "ERROR: Run this from the project repo root where app.py exists."
  exit 1
fi

echo "Verifying v4.3.5 files..."
ls backend/main.py backend/sample_data.py backend/model_service.py backend/pdf_utils.py backend/profile_engine.py backend/analysis_engine.py backend/llm_note.py >/dev/null
ls frontend/package.json frontend/src/main.jsx frontend/src/api.js frontend/src/styles.css >/dev/null

echo "Installing Python requirements..."
pip install -r requirements_v4.txt

echo "Building React frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Done. Run:"
echo "uvicorn backend.main:app --host 0.0.0.0 --port 7860"
