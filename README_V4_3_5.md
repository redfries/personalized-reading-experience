# v4.3.5 Dynamic Highlight Explanations Complete Package

This version fixes the repeated explanation problem.

## What was wrong before

The old explanation text used only the profile source list:

- keywords
- topics
- seed papers
- research statement

So every selected sentence got almost the same reason.

## What is new

Each highlight explanation now uses:

- the exact selected sentence
- paragraph context
- section name
- reading tag
- actual profile keywords found in that sentence/paragraph
- actual selected topics found in that sentence/paragraph
- the closest saved profile source based on per-source embedding similarity
- whether it is a normal highlight or a weak bridge-point highlight

## Preserved from previous versions

- v4.3.1 quality filtering for headings, references, tables, duplicates
- v4.3.2 Concise / Standard / Expanded / rescue mode behavior
- v4.3.4 grounded Gemini PDF evidence note
- React PDF.js viewer and clickable highlight insight panel

## Install and run

```bash
cd /teamspace/studios/this_studio/personalized-reading-experience

pip install -r requirements_v4.txt

cd frontend
npm install
npm run build
cd ..

uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

## Test

1. Select a saved profile.
2. Upload a PDF.
3. Analyze.
4. Click several highlights.
5. The "Why this relates to your work" explanation should now change based on the selected sentence.
