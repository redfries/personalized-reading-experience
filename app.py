from pathlib import Path
import csv
import hashlib
import json
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

# Runtime-writable paths. Lightning.ai can use /teamspace; local/demo can use /tmp.
LIGHTNING_BASE_DIR = Path("/teamspace/studios/this_studio")
DEFAULT_RUNTIME_DIR = (
    LIGHTNING_BASE_DIR / "personalized_reading_runtime"
    if LIGHTNING_BASE_DIR.exists()
    else Path("/tmp/personalized_reading_runtime")
)
RUNTIME_DIR = Path(os.environ.get("APP_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)))
CACHE_DIR = RUNTIME_DIR / "cache"
LOGS_DIR = RUNTIME_DIR / "logs"
HF_CACHE_DIR = RUNTIME_DIR / "hf_cache"
PROFILES_DIR = RUNTIME_DIR / "profiles"
APP_RUNS_FILE = LOGS_DIR / "app_runs.csv"

for d in [RUNTIME_DIR, CACHE_DIR, LOGS_DIR, HF_CACHE_DIR, PROFILES_DIR]:
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

# Strong default for GPU machines. Override on smaller machines, for example:
# EMBEDDING_MODEL_NAME=BAAI/bge-m3 or EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")
DEVICE = os.environ.get("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
OFF_PROFILE_LIMIT = 0.40
MAX_SEED_PAPERS = 3
MAX_SEED_SENTENCES_PER_PAPER = 60
SEED_CHUNK_SENTENCES = 8
SCHEMA_VERSION = "1.1"
PROFILE_EMBEDDING_STRATEGY = "weighted_mean_of_source_embeddings"
model = None

RESEARCH_TOPICS = [
    "Machine Learning",
    "Natural Language Processing",
    "Information Retrieval",
    "Recommender Systems",
    "Human-Computer Interaction",
    "Education Technology",
    "Healthcare AI",
    "Explainable AI",
    "Computer Vision",
    "Semantic Search",
    "Academic Reading Support",
    "Personalized Learning",
]


def get_model():
    global model
    if model is None:
        print(f"Loading sentence-transformer model: {MODEL_NAME} on {DEVICE}...", flush=True)
        try:
            if "Qwen3-Embedding" in MODEL_NAME and DEVICE == "cuda":
                model = SentenceTransformer(
                    MODEL_NAME,
                    device=DEVICE,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    tokenizer_kwargs={"padding_side": "left"},
                )
            else:
                model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        except TypeError:
            # Compatibility fallback for older sentence-transformers versions.
            model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print("Sentence-transformer model loaded.", flush=True)
    return model


def encode_texts(texts, is_query=False, convert_to_tensor=True):
    active_model = get_model()
    encode_kwargs = {
        "convert_to_tensor": convert_to_tensor,
        "device": DEVICE,
        "show_progress_bar": False,
    }
    prompts = getattr(active_model, "prompts", {}) or {}
    if is_query and "query" in prompts:
        encode_kwargs["prompt_name"] = "query"
    return active_model.encode(texts, **encode_kwargs)


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
    if path is None and isinstance(pdf_file, str):
        path = pdf_file
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
    return [(f"s{i:03d}", sent) for i, sent in enumerate(sentences, start=1)]


def score_sentences(profile_text: str, sentence_rows, sentence_embeddings: torch.Tensor | None = None, profile_embedding: torch.Tensor | None = None):
    if profile_embedding is None:
        profile_embedding = encode_texts(profile_text, is_query=True, convert_to_tensor=True)
    profile_embedding = profile_embedding.to(DEVICE)

    if sentence_embeddings is None:
        texts = [s for _, s in sentence_rows]
        sentence_embeddings = encode_texts(texts, is_query=False, convert_to_tensor=True)
        sentence_embeddings = sentence_embeddings.to(DEVICE)

    scores = util.cos_sim(profile_embedding, sentence_embeddings)[0].detach().cpu().numpy()
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
        selected = df_sorted[:top_k]
        summary = (
            f"Mode: Off profile (topic mismatch)\n"
            f"Total sentences: {total_sentences}\n"
            f"Max similarity: {max_score:.3f}\n"
            f"Threshold ignored in this mode\n"
            f"Showing top {len(selected)} best guesses"
        )
        highlights = [(f"[{row['score']:.3f}] {row['sentence_text']}", "Best guess") for row in selected]
    else:
        filtered = [row for row in df_sorted if row["score"] >= threshold]
        if filtered:
            selected = filtered[:top_k]
            note = "Threshold applied successfully"
        else:
            selected = df_sorted[:top_k]
            note = "No sentences met threshold, so top results were shown"
        summary = (
            f"Mode: Profile match\n"
            f"Total sentences: {total_sentences}\n"
            f"Max similarity: {max_score:.3f}\n"
            f"Sentences above threshold: {num_above_threshold}\n"
            f"Showing: {len(selected)}\n"
            f"{note}"
        )
        highlights = [(f"[{row['score']:.3f}] {row['sentence_text']}", "Relevant") for row in selected]

    return summary, highlights


def append_run_log(pdf_name: str, top_k: int, threshold: float, summary: str):
    file_exists = APP_RUNS_FILE.exists()
    with APP_RUNS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_utc", "pdf_name", "top_k", "threshold", "summary", "embedding_model"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            pdf_name,
            top_k,
            threshold,
            summary.replace("\n", " | "),
            MODEL_NAME,
        ])


def load_profile_embedding(profile_id):
    if not profile_id:
        return None, "No saved profile embedding selected."

    profile_dir = PROFILES_DIR / profile_id
    profile_json = profile_dir / "profile.json"
    if not profile_json.exists():
        return None, "Saved profile JSON was not found."

    try:
        with profile_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return None, f"Could not read saved profile metadata: {exc}"

    saved_model = data.get("embedding_model")
    if saved_model != MODEL_NAME:
        return None, f"Saved embedding was created with {saved_model}; current model is {MODEL_NAME}. Re-encoding profile text."

    embedding_file = data.get("profile_embedding_file", "profile_embedding.npy")
    embedding_path = profile_dir / embedding_file
    if not embedding_path.exists():
        return None, "Saved profile embedding file was not found. Re-encoding profile text."

    try:
        embedding = np.load(embedding_path)
        return torch.tensor(embedding, dtype=torch.float32, device=DEVICE), "Using saved profile embedding."
    except Exception as exc:
        return None, f"Could not load saved profile embedding: {exc}. Re-encoding profile text."


def analyze_pdf(profile_text, loaded_profile_id, loaded_profile_text, pdf_file, top_k, threshold):
    if not pdf_file:
        return "Please upload a PDF.", []
    if not profile_text or not profile_text.strip():
        return "Please enter your research profile text or load a saved profile.", []

    profile_embedding = None
    embedding_status = "Encoding profile text for this analysis."
    if loaded_profile_id and loaded_profile_text and profile_text.strip() == loaded_profile_text.strip():
        profile_embedding, embedding_status = load_profile_embedding(loaded_profile_id)

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
        _, sentence_embeddings = score_sentences(
            profile_text,
            sentence_rows,
            sentence_embeddings=None,
            profile_embedding=profile_embedding,
        )
        save_cache(doc_hash, sentence_rows, sentence_embeddings)

    df_sorted, _ = score_sentences(
        profile_text,
        sentence_rows,
        sentence_embeddings=sentence_embeddings,
        profile_embedding=profile_embedding,
    )
    summary, highlights = select_highlights(df_sorted, int(top_k), float(threshold))
    summary = f"{summary}\nProfile embedding: {embedding_status}\nModel: {MODEL_NAME}"
    append_run_log(pdf_name or "uploaded.pdf", int(top_k), float(threshold), summary)
    return summary, highlights


def normalize_file_list(files):
    if not files:
        return []
    if isinstance(files, list):
        return files
    return [files]


def parse_keywords(keyword_text: str):
    if not keyword_text or not keyword_text.strip():
        return []
    raw_keywords = re.split(r"[,;\n]+", keyword_text)
    keywords = []
    seen = set()
    for kw in raw_keywords:
        cleaned = re.sub(r"\s+", " ", kw).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            keywords.append(cleaned)
    return keywords


def clean_free_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slugify_profile_name(profile_name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", profile_name.strip().lower()).strip("_")
    if not base:
        base = "research_profile"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}"


def get_profile_strength(source_count: int) -> str:
    if source_count <= 0:
        return "Not enough information"
    if source_count == 1:
        return "Weak"
    if source_count == 2:
        return "Good"
    if source_count == 3:
        return "Strong"
    return "Very strong"


def validate_profile_sources(selected_topics, seed_papers, keyword_text, free_profile_text):
    selected_topics = selected_topics or []
    seed_files = normalize_file_list(seed_papers)
    keywords = parse_keywords(keyword_text)
    free_text = clean_free_text(free_profile_text)

    sources_used = []
    if selected_topics:
        sources_used.append("Topics")
    if seed_files:
        sources_used.append("Seed papers")
    if keywords:
        sources_used.append("Keywords")
    if free_text:
        sources_used.append("Research statement")

    source_count = len(sources_used)
    can_preview = source_count >= 1
    can_save = source_count >= 2 and len(seed_files) <= MAX_SEED_PAPERS
    strength = get_profile_strength(source_count)

    warnings = []
    if source_count == 0:
        warnings.append("Add at least one source to preview the profile.")
    elif source_count == 1:
        warnings.append("This is a weak profile. Add at least one more source before saving.")
    if len(seed_files) > MAX_SEED_PAPERS:
        warnings.append(f"Upload at most {MAX_SEED_PAPERS} seed papers for this MVP version.")

    return {
        "selected_topics": selected_topics,
        "seed_files": seed_files,
        "keywords": keywords,
        "free_profile_text": free_text,
        "sources_used": sources_used,
        "source_count": source_count,
        "profile_strength": strength,
        "can_preview": can_preview,
        "can_save": can_save,
        "warnings": warnings,
    }


def chunk_sentences(sentences, chunk_size=SEED_CHUNK_SENTENCES):
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def extract_seed_paper_evidence(seed_papers):
    seed_files = normalize_file_list(seed_papers)
    evidence_blocks = []
    seed_chunks = []
    seed_metadata = []
    errors = []

    for seed_file in seed_files[:MAX_SEED_PAPERS]:
        file_bytes, pdf_name = extract_pdf_bytes_and_name(seed_file)
        if not file_bytes:
            errors.append("Could not read one uploaded seed paper.")
            continue

        try:
            raw_text = extract_text_from_pdf_bytes(file_bytes)
        except Exception as exc:
            errors.append(f"Could not extract text from {pdf_name or 'seed paper'}: {exc}")
            continue

        if len(raw_text.strip()) < 50:
            errors.append(f"{pdf_name or 'Seed paper'} appears empty or scanned with no extractable text.")
            continue

        clean_text = strip_references_section(raw_text)
        sentences = clean_sentences(clean_text, min_words=6, max_words=80)
        selected_sentences = sentences[:MAX_SEED_SENTENCES_PER_PAPER]

        if not selected_sentences:
            errors.append(f"{pdf_name or 'Seed paper'} did not provide enough usable text.")
            continue

        chunks = chunk_sentences(selected_sentences)
        evidence_text = " ".join(selected_sentences)
        evidence_blocks.append(f"Evidence from seed paper '{pdf_name}':\n{evidence_text}")
        for idx, chunk in enumerate(chunks, start=1):
            seed_chunks.append({
                "paper_name": pdf_name or "uploaded_seed_paper.pdf",
                "chunk_index": idx,
                "text": f"Relevant seed paper evidence from {pdf_name or 'uploaded_seed_paper.pdf'}: {chunk}",
            })
        seed_metadata.append({
            "file_name": pdf_name or "uploaded_seed_paper.pdf",
            "usable_sentences": len(sentences),
            "sentences_used_in_profile": len(selected_sentences),
            "num_chunks": len(chunks),
            "num_extracted_words_used": len(evidence_text.split()),
        })

    return evidence_blocks, seed_chunks, seed_metadata, errors


def build_profile_source_items(profile_name, selected_topics, seed_papers, keyword_text, free_profile_text):
    validation = validate_profile_sources(selected_topics, seed_papers, keyword_text, free_profile_text)
    evidence_blocks, seed_chunks, seed_metadata, seed_errors = extract_seed_paper_evidence(seed_papers)

    source_items = []

    # Seed evidence is placed first to protect the richest signal if any downstream model truncates text.
    seed_total_weight = 2.0 if seed_chunks else 0.0
    seed_chunk_weight = seed_total_weight / len(seed_chunks) if seed_chunks else 0.0
    for chunk in seed_chunks:
        source_items.append({
            "source_type": "seed_paper",
            "label": f"{chunk['paper_name']} chunk {chunk['chunk_index']}",
            "weight": seed_chunk_weight,
            "text": chunk["text"],
        })

    if validation["selected_topics"]:
        topic_text = ", ".join(validation["selected_topics"])
        source_items.append({
            "source_type": "topics",
            "label": "selected research topics",
            "weight": 1.0,
            "text": "The user selected the following research areas as interest signals: " f"{topic_text}.",
        })

    if validation["keywords"]:
        keyword_text_clean = ", ".join(validation["keywords"])
        source_items.append({
            "source_type": "keywords",
            "label": "user keywords",
            "weight": 1.0,
            "text": "The user provided the following keywords as additional interest signals: " f"{keyword_text_clean}.",
        })

    if validation["free_profile_text"]:
        source_items.append({
            "source_type": "research_statement",
            "label": "manual research statement",
            "weight": 1.25,
            "text": "The user provided the following research statement or existing project description: " f"{validation['free_profile_text']}",
        })

    sections = []
    if profile_name and profile_name.strip():
        sections.append(f"Profile name: {profile_name.strip()}.")
    if evidence_blocks:
        sections.append(
            "Seed paper evidence provided by the user:\n\n" + "\n\n".join(evidence_blocks)
        )
    for item in source_items:
        if item["source_type"] != "seed_paper":
            sections.append(item["text"])

    if not sections:
        sections.append("No profile evidence was provided.")

    combined_profile_text = "\n\n".join(sections).strip()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "profile_name": profile_name.strip() if profile_name else "Untitled Profile",
        "selected_topics": validation["selected_topics"],
        "keywords": validation["keywords"],
        "free_profile_text": validation["free_profile_text"],
        "sources_used": validation["sources_used"],
        "source_count": validation["source_count"],
        "profile_strength": validation["profile_strength"],
        "seed_papers": seed_metadata,
        "warnings": validation["warnings"] + seed_errors,
        "source_embedding_strategy": PROFILE_EMBEDDING_STRATEGY,
    }
    return combined_profile_text, metadata, source_items


def create_profile_embedding(source_items):
    if not source_items:
        return None, None, [], []

    source_texts = [item["text"] for item in source_items]
    weights = np.array([float(item["weight"]) for item in source_items], dtype=np.float32)
    labels = [f"{item['source_type']}::{item['label']}" for item in source_items]

    source_embeddings_tensor = encode_texts(source_texts, is_query=True, convert_to_tensor=True)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE).view(-1, 1)
    profile_embedding_tensor = (source_embeddings_tensor * weight_tensor).sum(dim=0) / weight_tensor.sum()

    source_embeddings = source_embeddings_tensor.detach().cpu().float().numpy()
    profile_embedding = profile_embedding_tensor.detach().cpu().float().numpy()
    return profile_embedding, source_embeddings, labels, weights


def format_profile_preview(profile_name, combined_profile_text, metadata, source_items):
    warnings = metadata.get("warnings", [])
    warning_text = ""
    if warnings:
        warning_text = "\n\n**Warnings:**\n" + "\n".join(f"- {w}" for w in warnings)

    topics = metadata.get("selected_topics", [])
    keywords = metadata.get("keywords", [])
    seed_papers = metadata.get("seed_papers", [])
    free_text = metadata.get("free_profile_text", "")

    topic_text = ", ".join(topics) if topics else "None"
    keyword_text = ", ".join(keywords) if keywords else "None"
    free_text_status = "Provided" if free_text else "None"
    seed_text = "\n".join(
        f"- {paper['file_name']} ({paper['sentences_used_in_profile']} sentences, {paper['num_chunks']} chunks)"
        for paper in seed_papers
    ) or "None"
    source_items_text = "\n".join(
        f"- {item['source_type']}: {item['label']} | weight={item['weight']:.3f}"
        for item in source_items
    ) or "None"

    return (
        f"## Profile Preview\n\n"
        f"**Profile name:** {profile_name or 'Untitled Profile'}\n\n"
        f"**Profile strength:** {metadata.get('profile_strength', 'Unknown')}\n\n"
        f"**Sources used:** {', '.join(metadata.get('sources_used', [])) or 'None'}\n\n"
        f"**Selected topics:** {topic_text}\n\n"
        f"**Keywords:** {keyword_text}\n\n"
        f"**Research statement:** {free_text_status}\n\n"
        f"**Seed papers:**\n{seed_text}\n\n"
        f"**Embedding source items:**\n{source_items_text}"
        f"{warning_text}\n\n"
        f"### Combined profile text\n\n"
        f"```text\n{combined_profile_text}\n```"
    )


def preview_profile(profile_name, selected_topics, seed_papers, keyword_text, free_profile_text):
    validation = validate_profile_sources(selected_topics, seed_papers, keyword_text, free_profile_text)
    if not validation["can_preview"]:
        return "Please provide at least one source: topics, keywords, seed papers, or a research statement."

    combined_profile_text, metadata, source_items = build_profile_source_items(
        profile_name,
        selected_topics,
        seed_papers,
        keyword_text,
        free_profile_text,
    )
    return format_profile_preview(profile_name, combined_profile_text, metadata, source_items)


def save_profile(profile_name, selected_topics, seed_papers, keyword_text, free_profile_text):
    if not profile_name or not profile_name.strip():
        return "Please enter a profile name before saving.", gr.update(), DEFAULT_PROFILE, "Profile was not loaded.", None, ""

    validation = validate_profile_sources(selected_topics, seed_papers, keyword_text, free_profile_text)
    if not validation["can_save"]:
        warning_text = "\n".join(f"- {w}" for w in validation["warnings"])
        return (
            "Profile not saved. Please provide at least two profile sources before saving.\n\n"
            f"{warning_text}",
            gr.update(),
            DEFAULT_PROFILE,
            "Profile was not loaded.",
            None,
            "",
        )

    combined_profile_text, metadata, source_items = build_profile_source_items(
        profile_name,
        selected_topics,
        seed_papers,
        keyword_text,
        free_profile_text,
    )

    if not source_items:
        return "Profile not saved because no usable profile source items were generated.", gr.update(), DEFAULT_PROFILE, "Profile was not loaded.", None, ""

    profile_id = slugify_profile_name(profile_name)
    profile_dir = PROFILES_DIR / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_embedding, source_embeddings, source_labels, source_weights = create_profile_embedding(source_items)
    if profile_embedding is None:
        return "Profile not saved because the embedding could not be created.", gr.update(), DEFAULT_PROFILE, "Profile was not loaded.", None, ""

    embedding_path = profile_dir / "profile_embedding.npy"
    source_embeddings_path = profile_dir / "source_embeddings.npz"
    np.save(embedding_path, profile_embedding)
    np.savez_compressed(
        source_embeddings_path,
        embeddings=source_embeddings,
        labels=np.array(source_labels),
        weights=source_weights,
    )

    profile_data = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "profile_name": profile_name.strip(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_topics": metadata["selected_topics"],
        "keywords": metadata["keywords"],
        "free_profile_text": metadata["free_profile_text"],
        "seed_papers": metadata["seed_papers"],
        "sources_used": metadata["sources_used"],
        "source_count": metadata["source_count"],
        "profile_strength": metadata["profile_strength"],
        "combined_profile_text": combined_profile_text,
        "embedding_model": MODEL_NAME,
        "embedding_device": DEVICE,
        "profile_embedding_file": embedding_path.name,
        "source_embeddings_file": source_embeddings_path.name,
        "source_embedding_strategy": PROFILE_EMBEDDING_STRATEGY,
        "source_items": [
            {
                "source_type": item["source_type"],
                "label": item["label"],
                "weight": item["weight"],
                "text_preview": item["text"][:500],
            }
            for item in source_items
        ],
        "metadata": {
            "profile_version": "v1.1",
            "storage_location": str(profile_dir),
            "runtime_dir": str(RUNTIME_DIR),
            "note": "Saved locally in the application runtime directory. Use persistent storage for production deployment.",
        },
    }

    with (profile_dir / "profile.json").open("w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)

    status = (
        "Profile saved successfully.\n\n"
        f"Profile name: {profile_data['profile_name']}\n"
        f"Profile strength: {profile_data['profile_strength']}\n"
        f"Sources used: {', '.join(profile_data['sources_used'])}\n"
        f"Embedding model: {MODEL_NAME}\n"
        f"Embedding strategy: {PROFILE_EMBEDDING_STRATEGY}\n"
        f"Saved profile ID: {profile_id}"
    )
    load_status = (
        f"Loaded saved profile: {profile_data['profile_name']}\n"
        f"Profile strength: {profile_data['profile_strength']}\n"
        f"Sources used: {', '.join(profile_data['sources_used'])}"
    )
    return (
        status,
        gr.update(choices=get_saved_profile_choices(), value=profile_id),
        combined_profile_text,
        load_status,
        profile_id,
        combined_profile_text,
    )


def get_saved_profile_choices():
    choices = []
    for profile_json in sorted(PROFILES_DIR.glob("*/profile.json")):
        try:
            with profile_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            profile_id = data.get("profile_id", profile_json.parent.name)
            profile_name = data.get("profile_name", profile_id)
            strength = data.get("profile_strength", "Unknown")
            model_name = data.get("embedding_model", "unknown model")
            label = f"{profile_name} ({strength}, {model_name})"
            choices.append((label, profile_id))
        except Exception:
            continue
    return choices


def refresh_saved_profiles():
    return gr.update(choices=get_saved_profile_choices(), value=None), None, ""


def load_saved_profile_text(profile_id):
    if not profile_id:
        return DEFAULT_PROFILE, "No saved profile selected. You can use the manual profile text instead.", None, ""

    profile_json = PROFILES_DIR / profile_id / "profile.json"
    if not profile_json.exists():
        return DEFAULT_PROFILE, "Could not find the selected saved profile. Try refreshing the profile list.", None, ""

    with profile_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    profile_text = data.get("combined_profile_text", "") or DEFAULT_PROFILE
    status = (
        f"Loaded saved profile: {data.get('profile_name', profile_id)}\n"
        f"Profile strength: {data.get('profile_strength', 'Unknown')}\n"
        f"Sources used: {', '.join(data.get('sources_used', []))}\n"
        f"Embedding model: {data.get('embedding_model', 'Unknown')}"
    )
    return profile_text, status, profile_id, profile_text


def build_demo():
    with gr.Blocks(title="Personalized Reading Experience") as demo:
        gr.Markdown("# Personalized Reading Experience")
        gr.Markdown(
            "Build a reusable research profile, then use it to highlight the most relevant sentences in an academic paper."
        )
        gr.Markdown(f"**Active embedding model:** `{MODEL_NAME}` on `{DEVICE}`")

        loaded_profile_id_state = gr.State(value=None)
        loaded_profile_text_state = gr.State(value="")

        with gr.Tabs():
            with gr.Tab("Build User Profile"):
                gr.Markdown(
                    "Create a profile from low-effort academic signals. Preview is allowed with one source, but saving requires at least two sources."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        profile_name_input = gr.Textbox(
                            label="Profile name",
                            placeholder="Example: AI Reading Assistant Profile",
                        )
                        topics_input = gr.CheckboxGroup(
                            choices=RESEARCH_TOPICS,
                            label="Select research topics",
                        )
                        seed_papers_input = gr.File(
                            label=f"Upload seed papers, optional, max {MAX_SEED_PAPERS} PDFs",
                            file_types=[".pdf"],
                            file_count="multiple",
                        )
                        keywords_input = gr.Textbox(
                            label="Optional keywords",
                            lines=3,
                            placeholder="Example: personalization, semantic search, paper highlighting",
                        )
                        free_profile_text_input = gr.Textbox(
                            label="Optional research statement or existing project text",
                            lines=5,
                            placeholder="Paste a short proposal paragraph, advisor topic, thesis idea, or manual profile text here...",
                        )
                        with gr.Row():
                            preview_button = gr.Button("Preview Profile")
                            save_button = gr.Button("Save Profile")
                    with gr.Column(scale=1):
                        profile_preview_output = gr.Markdown(label="Profile preview")
                        save_status_output = gr.Textbox(label="Save status", lines=9)

            with gr.Tab("Analyze Paper"):
                gr.Markdown(
                    "Choose a saved profile or edit the manual profile text, then upload a paper for personalized highlighting."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        saved_profile_dropdown = gr.Dropdown(
                            label="Choose saved profile",
                            choices=get_saved_profile_choices(),
                            value=None,
                        )
                        refresh_profiles_button = gr.Button("Refresh saved profiles")
                        profile_load_status = gr.Textbox(label="Profile load status", lines=4)
                        profile_input = gr.Textbox(
                            label="Research profile used for analysis",
                            lines=8,
                            value=DEFAULT_PROFILE,
                            placeholder="Describe your research interests here...",
                        )
                        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                        top_k_input = gr.Slider(1, 20, value=8, step=1, label="Top-K sentences")
                        threshold_input = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="Similarity threshold")
                        run_button = gr.Button("Analyze paper")

                    with gr.Column(scale=1):
                        summary_output = gr.Textbox(label="Summary", lines=9)
                        highlights_output = gr.HighlightedText(
                            label="Highlighted sentences",
                            combine_adjacent=False,
                            color_map={"Relevant": "#ffd54f", "Best guess": "#90caf9"},
                        )

        preview_button.click(
            fn=preview_profile,
            inputs=[profile_name_input, topics_input, seed_papers_input, keywords_input, free_profile_text_input],
            outputs=[profile_preview_output],
        )
        save_button.click(
            fn=save_profile,
            inputs=[profile_name_input, topics_input, seed_papers_input, keywords_input, free_profile_text_input],
            outputs=[
                save_status_output,
                saved_profile_dropdown,
                profile_input,
                profile_load_status,
                loaded_profile_id_state,
                loaded_profile_text_state,
            ],
        )
        refresh_profiles_button.click(
            fn=refresh_saved_profiles,
            inputs=None,
            outputs=[saved_profile_dropdown, loaded_profile_id_state, loaded_profile_text_state],
        )
        saved_profile_dropdown.change(
            fn=load_saved_profile_text,
            inputs=[saved_profile_dropdown],
            outputs=[profile_input, profile_load_status, loaded_profile_id_state, loaded_profile_text_state],
        )
        run_button.click(
            fn=analyze_pdf,
            inputs=[profile_input, loaded_profile_id_state, loaded_profile_text_state, pdf_input, top_k_input, threshold_input],
            outputs=[summary_output, highlights_output],
        )

    return demo


print("===== Application Startup =====", flush=True)
demo = build_demo()

if __name__ == "__main__":
    print("Launching Gradio app...", flush=True)
    demo.launch()
