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

# Portable project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = DATA_DIR / "logs"
PROFILE_PATH = DATA_DIR / "topic_profile.txt"
APP_RUNS_FILE = LOGS_DIR / "app_runs.csv"

for d in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_PROFILE = (
    PROFILE_PATH.read_text(encoding="utf-8").strip()
    if PROFILE_PATH.exists()
    else "Describe your research interests here. Example: AI tools for reading scientific papers, personalized highlighting, human-computer interaction, and reading support systems."
)

MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"  # safest for first deployment
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
OFF_PROFILE_LIMIT = 0.40


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
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def is_junk_sentence(s: str) -> bool:
    num_words = len(s.split())
    if num_words < 5 or num_words > 60:
        return True

    lower = s.lower().strip()
    junk_markers = [
        "doi",
        "issn",
        "copyright",
        "creative commons",
        "ieee access",
        "http://",
        "https://",
        "arxiv:",
    ]
    if any(marker in lower for marker in junk_markers):
        return True
    if lower.startswith("[") and "]" in lower[:10]:
        return True
    if lower.startswith(("figure ", "fig. ", "table ")):
        return True
    if is_probable_section_title(s):
        return True
    return False


def pdf_to_clean_sentences(pdf_file):
    file_bytes, _ = extract_pdf_bytes_and_name(pdf_file)
    if file_bytes is None:
        return None, []

    doc_hash = compute_doc_hash(file_bytes)
    raw_text = extract_text_from_pdf_bytes(file_bytes)
    if len(raw_text.strip()) < 100:
        return doc_hash, []

    text = strip_references_section(raw_text)
    raw_sentences = simple_sentence_split(text)

    sentences = []
    sid_counter = 1
    for s in raw_sentences:
        s = s.strip()
        if not s or is_junk_sentence(s):
            continue
        sid = f"s{sid_counter:03d}"
        sentences.append((sid, s))
        sid_counter += 1
    return doc_hash, sentences


def get_sentences_and_embeddings(pdf_file):
    file_bytes, _ = extract_pdf_bytes_and_name(pdf_file)
    if file_bytes is None:
        return None, [], None

    doc_hash = compute_doc_hash(file_bytes)
    if cache_exists(doc_hash):
        sentences, embeddings = load_cache(doc_hash)
        return doc_hash, sentences, embeddings

    doc_hash, sentences = pdf_to_clean_sentences(pdf_file)
    if doc_hash is None or not sentences:
        return doc_hash, [], None

    sentence_texts = [text for _, text in sentences]
    embeddings = model.encode(sentence_texts, convert_to_tensor=True, device=DEVICE)
    save_cache(doc_hash, sentences, embeddings)
    return doc_hash, sentences, embeddings


def score_sentences_for_profile(sentences, embeddings, profile_text: str):
    if not sentences or embeddings is None:
        return np.array([])
    profile_emb = model.encode(profile_text, convert_to_tensor=True, device=DEVICE)
    scores = util.cos_sim(profile_emb, embeddings)[0]
    return scores.detach().cpu().numpy()


def select_highlights(sentences, scores, top_k: int, threshold: float):
    if len(sentences) == 0 or len(scores) == 0:
        return "Error: no usable text found in this PDF.", []

    rows = [
        {"sid": sid, "text": text, "score": float(score)}
        for (sid, text), score in zip(sentences, scores)
    ]
    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)

    total_sentences = len(rows_sorted)
    max_score = rows_sorted[0]["score"]
    num_above_threshold = sum(1 for row in rows_sorted if row["score"] >= threshold)
    mode = "off_profile" if max_score < OFF_PROFILE_LIMIT else "profile"

    highlights = []
    if mode == "profile":
        filtered = [row for row in rows_sorted if row["score"] >= threshold]
        used_threshold = bool(filtered)
        selected = (filtered if filtered else rows_sorted)[:top_k]
        summary_lines = [
            "Mode: Profile match",
            f"Total sentences: {total_sentences}",
            f"Max similarity: {max_score:.3f}",
            f"Threshold: {threshold:.2f}",
            f"Sentences above threshold: {num_above_threshold}",
            f"Sentences shown: {len(selected)}",
        ]
        if not used_threshold:
            summary_lines.append("Note: threshold was too strict, so top results were shown instead.")
        label = "Relevant"
    else:
        selected = rows_sorted[:top_k]
        summary_lines = [
            "Mode: Off profile",
            f"Total sentences: {total_sentences}",
            f"Max similarity: {max_score:.3f}",
            f"Threshold is ignored in off-profile mode.",
            f"Sentences shown: {len(selected)}",
        ]
        label = "Best guess"

    for row in selected:
        display_text = f"[{row['score']:.3f}] {row['text']}"
        highlights.append((display_text, label))

    return "\n".join(summary_lines), highlights


def log_app_run(user_id: str, paper_name: str, mode: str, max_score: float, top_k: int, threshold: float, num_sentences: int, num_shown: int):
    header = [
        "timestamp_utc",
        "user_id",
        "paper_name",
        "mode",
        "max_score",
        "top_k",
        "threshold",
        "num_sentences",
        "num_shown",
    ]
    file_exists = APP_RUNS_FILE.exists()
    with APP_RUNS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            user_id,
            paper_name,
            mode,
            f"{max_score:.3f}",
            top_k,
            f"{threshold:.2f}",
            num_sentences,
            num_shown,
        ])


def run_analysis(pdf_file, profile_text, top_k, threshold, user_id):
    if pdf_file is None:
        return "Please upload a PDF.", []

    profile_text = (profile_text or "").strip()
    if not profile_text:
        return "Please enter your profile text.", []

    _, paper_name = extract_pdf_bytes_and_name(pdf_file)
    if paper_name is None:
        return "Could not read the uploaded PDF.", []

    doc_hash, sentences, embeddings = get_sentences_and_embeddings(pdf_file)
    if doc_hash is None or not sentences or embeddings is None:
        return "No usable text was extracted from this PDF. It may be scanned, image-only, or too noisy.", []

    scores = score_sentences_for_profile(sentences, embeddings, profile_text)
    if len(scores) == 0:
        return "Could not compute similarity scores.", []

    summary, highlights = select_highlights(sentences, scores, int(top_k), float(threshold))

    max_score = float(scores.max())
    mode = "profile" if max_score >= OFF_PROFILE_LIMIT else "off_profile"
    user_id_clean = (user_id or "anonymous").strip() or "anonymous"

    try:
        log_app_run(
            user_id=user_id_clean,
            paper_name=paper_name,
            mode=mode,
            max_score=max_score,
            top_k=int(top_k),
            threshold=float(threshold),
            num_sentences=len(sentences),
            num_shown=len(highlights),
        )
    except Exception as exc:
        print(f"Logging failed: {exc}")

    return summary, highlights


def create_app():
    with gr.Blocks(title="Personalized Paper Highlighter") as demo:
        gr.Markdown(
            "# Personalized Reading Highlights\n"
            "Upload a research paper PDF, enter your research profile, and view the most relevant sentences."
        )

        with gr.Row():
            with gr.Column(scale=1):
                user_id = gr.Textbox(
                    label="User ID for logs (optional)",
                    value="u1",
                    placeholder="u1, initials, or leave as default",
                )
                profile_box = gr.Textbox(
                    label="Research profile",
                    value=DEFAULT_PROFILE,
                    lines=8,
                )
                pdf_input = gr.File(label="Upload a PDF", file_types=[".pdf"])
                top_k_slider = gr.Slider(
                    label="Number of sentences to show",
                    minimum=5,
                    maximum=50,
                    step=1,
                    value=15,
                )
                threshold_slider = gr.Slider(
                    label="Similarity threshold",
                    minimum=0.0,
                    maximum=0.6,
                    step=0.02,
                    value=0.20,
                )
                analyze_btn = gr.Button("Analyze paper")

            with gr.Column(scale=1):
                summary_out = gr.Textbox(label="Summary", lines=8)
                highlight_out = gr.HighlightedText(
                    label="Highlighted sentences",
                    combine_adjacent=True,
                    color_map={"Relevant": "green", "Best guess": "orange"},
                )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[pdf_input, profile_box, top_k_slider, threshold_slider, user_id],
            outputs=[summary_out, highlight_out],
        )

    return demo


demo = create_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
    )
