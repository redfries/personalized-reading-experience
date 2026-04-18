from pathlib import Path
import csv
import hashlib
import os
import re
from datetime import datetime, timezone

import fitz
import gradio as gr
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Repo paths
BASE_DIR = Path(__file__).resolve().parent

# Runtime-writable paths for Hugging Face Spaces
RUNTIME_DIR = Path(os.environ.get("APP_RUNTIME_DIR", "/tmp/personalized_reading_runtime"))
CACHE_DIR = RUNTIME_DIR / "cache"
LOGS_DIR = RUNTIME_DIR / "logs"
HF_CACHE_DIR = RUNTIME_DIR / "hf_cache"
APP_RUNS_FILE = LOGS_DIR / "app_runs.csv"

for d in [RUNTIME_DIR, CACHE_DIR, LOGS_DIR, HF_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR))

# Read profile text from either root or data/ if available
PROFILE_CANDIDATES = [
    BASE_DIR / "topic_profile.txt",
    BASE_DIR / "data" / "topic_profile.txt",
]
PROFILE_PATH = next((p for p in PROFILE_CANDIDATES if p.exists()), None)

DEFAULT_PROFILE = (
    PROFILE_PATH.read_text(encoding="utf-8").strip()
    if PROFILE_PATH is not None
    else "Describe your research interests here. Example: AI tools for reading scientific papers, personalized highlighting, human-computer interaction, and reading support systems."
)

MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"
OFF_PROFILE_LIMIT = 0.40
model = None


def get_model():
    global model
    if model is None:
        print("Loading sentence-transformer model...", flush=True)
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print("Sentence-transformer model loaded.", flush=True)
    return model


def compute_doc_hash(file_bytes: bytes) -> str:
    h = hashlib.sha1()
    h.update(file_bytes)
    return h.hexdigest()


def get_cache_paths(doc_hash: str):
    sent_path = CACHE_DIR / f"doc_{doc_hash}_sentences.csv"
    emb_path = CACHE_DIR / f"doc_{doc_hash}_embeddings.npz"
    return sent_path, emb_path


def cache_exists(doc_hash: str) -> bool:
    sent_path, emb_path = get_cache_paths(doc_hash)
    return sent_path.exists() and emb_path.exists()


def load_cache(doc_hash: str):
    sent_path, emb_path = get_cache_paths(doc_hash)
    sentences = []
    with sent_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences.append((row["sentence_id"], row["sentence_text"]))
    data = np.load(emb_path)
    embeddings = torch.tensor(data["embeddings"], dtype=torch.float32, device=DEVICE)
    return sentences, embeddings


def save_cache(doc_hash: str, sentences, embeddings: torch.Tensor):
    sent_path, emb_path = get_cache_paths(doc_hash)
    with sent_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence_id", "sentence_text"])
        writer.writerows(sentences)
    np.savez_compressed(emb_path, embeddings=embeddings.detach().cpu().numpy())


def extract_pdf_bytes_and_name(pdf_file):
    if pdf_file is None:
        return None, None

    path = getattr(pdf_file, "name", None)
    if path is None and isinstance(pdf_file, dict):
        path = pdf_file.get("name")
    if path is None:
        return None, None

    with open(path, "rb") as f:
        return f.read(), Path(path).name


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def strip_references_section(text: str) -> str:
    patterns = [r"\nreferences\b", r"\nreference\b", r"\nbibliography\b"]
    lowered = text.lower()
    indices = []
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            indices.append(match.start())
    if indices:
        return text[: min(indices)]
    return text


def is_probable_section_title(s: str) -> bool:
    words = s.split()
    if len(words) > 5:
        return False
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return uppercase_ratio > 0.8


def simple_sentence_split(text: str):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def clean_sentences(text: str, min_words=5, max_words=60):
    raw_sentences = simple_sentence_split(text)
    clean = []
    for sent in raw_sentences:
        num_words = len(sent.split())
        lowered = sent.lower()

        if num_words < min_words or num_words > max_words:
            continue
        if re.match(r"^\[\d+\]", sent):
            continue
        if any(tok in lowered for tok in ["doi", "ieee access", "http://", "https://", "vol.", "volume "]):
            continue
        if is_probable_section_title(sent):
            continue
        if lowered.startswith(("figure ", "fig. ", "table ")):
            continue

        clean.append(sent)
    return clean


def make_sentence_table(sentences):
    rows = []
    for i, sent in enumerate(sentences, start=1):
        rows.append((f"s{i:03d}", sent))
    return rows


def score_sentences(profile_text: str, sentence_rows, sentence_embeddings: torch.Tensor | None = None):
    active_model = get_model()
    profile_emb = active_model.encode(profile_text, convert_to_tensor=True, device=DEVICE)
    profile_emb = profile_emb.to(DEVICE)

    if sentence_embeddings is None:
        texts = [s for _, s in sentence_rows]
        sentence_embeddings = active_model.encode(texts, convert_to_tensor=True, device=DEVICE)
        sentence_embeddings = sentence_embeddings.to(DEVICE)

    scores = util.cos_sim(profile_emb, sentence_embeddings)[0].detach().cpu().numpy()
    records = []
    for (sentence_id, sentence_text), score in zip(sentence_rows, scores):
        records.append({
            "sentence_id": sentence_id,
            "sentence_text": sentence_text,
            "score": float(score),
        })
    records.sort(key=lambda x: x["score"], reverse=True)
    return records, sentence_embeddings


def select_highlights(df_sorted, top_k: int, threshold: float):
    if not df_sorted:
        return "No usable sentences found.", []

    total_sentences = len(df_sorted)
    max_score = df_sorted[0]["score"]
    num_above_threshold = sum(row["score"] >= threshold for row in df_sorted)

    if max_score < OFF_PROFILE_LIMIT:
        mode = "off_profile"
        selected = df_sorted[:top_k]
        summary = (
            f"Mode: Off profile (topic mismatch)\\n"
            f"Total sentences: {total_sentences}\\n"
            f"Max similarity: {max_score:.3f}\\n"
            f"Threshold ignored in this mode\\n"
            f"Showing top {len(selected)} best guesses"
        )
        highlights = [(f"[{row['score']:.3f}] {row['sentence_text']}", "Best guess") for row in selected]
    else:
        mode = "profile"
        filtered = [row for row in df_sorted if row["score"] >= threshold]
        if filtered:
            selected = filtered[:top_k]
            note = "Threshold applied successfully"
        else:
            selected = df_sorted[:top_k]
            note = "No sentences met threshold, so top results were shown"
        summary = (
            f"Mode: Profile match\\n"
            f"Total sentences: {total_sentences}\\n"
            f"Max similarity: {max_score:.3f}\\n"
            f"Sentences above threshold: {num_above_threshold}\\n"
            f"Showing: {len(selected)}\\n"
            f"{note}"
        )
        highlights = [(f"[{row['score']:.3f}] {row['sentence_text']}", "Relevant") for row in selected]

    return summary, highlights


def append_run_log(pdf_name: str, top_k: int, threshold: float, summary: str):
    file_exists = APP_RUNS_FILE.exists()
    with APP_RUNS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_utc", "pdf_name", "top_k", "threshold", "summary"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            pdf_name,
            top_k,
            threshold,
            summary.replace("\n", " | "),
        ])


def analyze_pdf(profile_text, pdf_file, top_k, threshold):
    if not pdf_file:
        return "Please upload a PDF.", []
    if not profile_text or not profile_text.strip():
        return "Please enter your research profile text.", []

    file_bytes, pdf_name = extract_pdf_bytes_and_name(pdf_file)
    if not file_bytes:
        return "Could not read the uploaded PDF.", []

    doc_hash = compute_doc_hash(file_bytes)

    if cache_exists(doc_hash):
        sentence_rows, sentence_embeddings = load_cache(doc_hash)
    else:
        raw_text = extract_text_from_pdf_bytes(file_bytes)
        if len(raw_text.strip()) < 50:
            return "The PDF seems empty or scanned with no extractable text.", []
        clean_text = strip_references_section(raw_text)
        sentences = clean_sentences(clean_text)
        if len(sentences) < 5:
            return "The PDF did not yield enough usable text.", []
        sentence_rows = make_sentence_table(sentences)
        _, sentence_embeddings = score_sentences(profile_text, sentence_rows, sentence_embeddings=None)
        save_cache(doc_hash, sentence_rows, sentence_embeddings)

    df_sorted, _ = score_sentences(profile_text, sentence_rows, sentence_embeddings=sentence_embeddings)
    summary, highlights = select_highlights(df_sorted, int(top_k), float(threshold))
    append_run_log(pdf_name or "uploaded.pdf", int(top_k), float(threshold), summary)
    return summary, highlights


def build_demo():
    with gr.Blocks(title="Personalized Reading Experience") as demo:
        gr.Markdown("# Personalized Reading Experience")
        gr.Markdown("Upload a paper, compare its sentences against your research profile, and see the most relevant highlights.")

        with gr.Row():
            with gr.Column(scale=1):
                profile_input = gr.Textbox(
                    label="Research profile",
                    lines=8,
                    value=DEFAULT_PROFILE,
                    placeholder="Describe your research interests here...",
                )
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                top_k_input = gr.Slider(1, 20, value=8, step=1, label="Top-K sentences")
                threshold_input = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="Similarity threshold")
                run_button = gr.Button("Analyze paper")

            with gr.Column(scale=1):
                summary_output = gr.Textbox(label="Summary", lines=7)
                highlights_output = gr.HighlightedText(
                    label="Highlighted sentences",
                    combine_adjacent=False,
                    color_map={"Relevant": "#ffd54f", "Best guess": "#90caf9"},
                )

        run_button.click(
            fn=analyze_pdf,
            inputs=[profile_input, pdf_input, top_k_input, threshold_input],
            outputs=[summary_output, highlights_output],
        )

    return demo


print("===== Application Startup =====", flush=True)
demo = build_demo()

if __name__ == "__main__":
    print("Launching Gradio app...", flush=True)
    demo.launch()
