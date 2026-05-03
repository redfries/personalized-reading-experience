# v4.2.1 Complete Real Backend Package

This zip includes all required files. In particular, it includes:

```text
backend/__init__.py
backend/main.py
backend/sample_data.py
backend/model_service.py
backend/pdf_utils.py
backend/profile_engine.py
backend/analysis_engine.py
frontend/package.json
frontend/index.html
frontend/src/api.js
frontend/src/main.jsx
frontend/src/styles.css
requirements_v4.txt
```

## Copy into repo

Copy `backend/`, `frontend/`, and `requirements_v4.txt` into your repo root.

Make sure you are in:

```bash
/teamspace/studios/this_studio/personalized-reading-experience
```

not only:

```bash
/teamspace/studios/this_studio
```

## Verify files

```bash
pwd
ls backend
```

You must see:

```text
__init__.py
main.py
sample_data.py
model_service.py
pdf_utils.py
profile_engine.py
analysis_engine.py
```

## Run

```bash
cd /teamspace/studios/this_studio/personalized-reading-experience
pip install -r requirements_v4.txt

cd frontend
npm install
npm run build
cd ..

export EMBEDDING_DEVICE=cuda
export EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
export APP_RUNTIME_DIR=/teamspace/studios/this_studio/personalized_reading_runtime_v4

uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

Open Lightning Port Viewer on port 7860.

## Test

1. Profile Builder
2. Choose at least two sources, for example topics + keywords
3. Save real profile
4. Reader
5. Refresh profiles
6. Select saved profile
7. Upload target PDF
8. Analyze paper
