from pathlib import Path
import csv
import hashlib
import json
import os
import re
import shutil
import traceback
from datetime import datetime, timezone
from html import escape

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
OUTPUTS_DIR = RUNTIME_DIR / "outputs"
APP_RUNS_FILE = LOGS_DIR / "app_runs.csv"

for d in [RUNTIME_DIR, CACHE_DIR, LOGS_DIR, HF_CACHE_DIR, PROFILES_DIR, OUTPUTS_DIR]:
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
SCHEMA_VERSION = "1.6"
PROFILE_EMBEDDING_STRATEGY = "weighted_mean_of_source_embeddings"
PDF_RENDER_DPI = int(os.environ.get("PDF_RENDER_DPI", "140"))
OUTPUT_CLEANUP_AGE_SECONDS = int(os.environ.get("OUTPUT_CLEANUP_AGE_SECONDS", "3600"))
TARGET_SENTENCE_MAX_WORDS = int(os.environ.get("TARGET_SENTENCE_MAX_WORDS", "80"))
CACHE_SCHEMA_VERSION = "v3_7_product_polish_records"

# Compact section choices shown in the Analyze tab. Unknown is included by default
# because many PDFs do not expose headings cleanly, but references are never used.
HIGHLIGHT_SECTION_CHOICES = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Background",
    "Methodology",
    "Experiments",
    "Results",
    "Discussion",
    "Conclusion",
    "Limitations",
    "Future Work",
    "Unknown",
]
DEFAULT_ALLOWED_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Background",
    "Methodology",
    "Experiments",
    "Results",
    "Discussion",
    "Conclusion",
    "Limitations",
    "Future Work",
    "Unknown",
]

# Rhetorical/reading-role tags used in the ranked table. These are simple
# heuristic labels, not LLM-generated claims. They help the app behave like a
# reading assistant by explaining whether a segment is mainly background, method,
# result, limitation, future work, or general relevance.
READING_TAGS = [
    "Background",
    "Method",
    "Result",
    "Limitation",
    "Future Work",
    "General Relevance",
]

PARAGRAPH_SCORE_BEST_WEIGHT = 0.70
PARAGRAPH_SCORE_TOP2_WEIGHT = 0.30
PARAGRAPH_CONTEXT_MAX_CHARS = int(os.environ.get("PARAGRAPH_CONTEXT_MAX_CHARS", "650"))
BROAD_PARAGRAPH_HIGHLIGHT_MIN_SUPPORT = int(os.environ.get("BROAD_PARAGRAPH_HIGHLIGHT_MIN_SUPPORT", "2"))
BROAD_PARAGRAPH_HIGHLIGHT_MAX_SENTENCES = int(os.environ.get("BROAD_PARAGRAPH_HIGHLIGHT_MAX_SENTENCES", "4"))
BROAD_PARAGRAPH_SUPPORT_MARGIN = float(os.environ.get("BROAD_PARAGRAPH_SUPPORT_MARGIN", "0.08"))
TABLE_NUMERIC_TOKEN_RATIO_LIMIT = float(os.environ.get("TABLE_NUMERIC_TOKEN_RATIO_LIMIT", "0.45"))


SECTION_ALIASES_TARGET = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "literature review": "Related Work",
    "background": "Background",
    "method": "Methodology",
    "methods": "Methodology",
    "methodology": "Methodology",
    "approach": "Methodology",
    "proposed method": "Methodology",
    "proposed approach": "Methodology",
    "system design": "Methodology",
    "experiments": "Experiments",
    "experimental setup": "Experiments",
    "evaluation": "Experiments",
    "experimental results": "Results",
    "results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "limitations": "Limitations",
    "limitation": "Limitations",
    "future work": "Future Work",
    "references": "References",
    "bibliography": "References",
}

NON_BODY_PREFIXES = (
    "figure ", "fig. ", "fig ", "table ", "algorithm ", "equation ",
    "appendix ", "copyright ", "©", "isbn", "issn", "keywords ", "keyword ", "index terms",
)



# Highlight density controls how many PDF-visible highlights the system tries to create.
# Expanded mode uses a wider semantic candidate boundary: it can consider lower-ranked
# sentences that are still close to the profile in embedding space, instead of only
# the strict top-k list. Off-profile papers remain capped so best guesses do not
# look like confident recommendations.
HIGHLIGHT_DENSITY_SETTINGS = {
    # Concise uses the previous practical focused behavior: a clean set of
    # high-confidence reading segments without making the PDF look empty.
    "Concise": {
        "target_pdf_matches": 12,
        "candidate_limit": 45,
        "threshold_relax": 0.05,
        "score_boundary_margin": 0.12,
        "minimum_score_floor": 0.30,
        "allow_paragraph_highlight": False,
        "prefer_full_sentence": True,
        "allow_partial_sentence_fallback": True,
    },
    # Standard is the recommended default. It expands the semantic boundary
    # enough for a finished reading view, but it prefers full-sentence matches
    # instead of small phrase fragments.
    "Standard": {
        "target_pdf_matches": 20,
        "candidate_limit": 90,
        "threshold_relax": 0.15,
        "score_boundary_margin": 0.24,
        "minimum_score_floor": 0.20,
        "allow_paragraph_highlight": False,
        "prefer_full_sentence": True,
        "allow_partial_sentence_fallback": False,
    },
    # Expanded is the strongest reading-assistant mode. It searches farther
    # inside the model's semantic boundary and can highlight a paragraph region
    # when several surrounding sentences in that paragraph support relevance.
    "Expanded": {
        "target_pdf_matches": 30,
        "candidate_limit": 150,
        "threshold_relax": 0.28,
        "score_boundary_margin": 0.38,
        "minimum_score_floor": 0.12,
        "allow_paragraph_highlight": True,
        "prefer_full_sentence": True,
        "allow_partial_sentence_fallback": False,
    },
}
# Tunable source weights for profile construction and future ablation studies.
# Seed papers are treated as one evidence source with a fixed total weight,
# so adding more papers refines that source rather than overwhelming topics,
# keywords, and the user research statement.
SOURCE_WEIGHTS = {
    "topics": 1.0,
    "keywords": 1.0,
    "research_statement": 1.25,
    "seed_papers_total": 2.0,
}

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


def cleanup_old_output_dirs(max_age_seconds=OUTPUT_CLEANUP_AGE_SECONDS):
    """Remove old analysis folders so repeated demos do not fill the runtime disk."""
    if max_age_seconds <= 0 or not OUTPUTS_DIR.exists():
        return

    cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
    for path in OUTPUTS_DIR.iterdir():
        if not path.is_dir() or not path.name.startswith("analysis_"):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                shutil.rmtree(path)
        except OSError as exc:
            print(f"Could not clean old output folder {path}: {exc}", flush=True)


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

    embeddings = active_model.encode(texts, **encode_kwargs)

    # Qwen and other large models may return bfloat16 on GPU.
    # Standardize all runtime embeddings to float32 so cosine similarity,
    # weighted averaging, and cache saving never mix float32 with bfloat16.
    if convert_to_tensor and isinstance(embeddings, torch.Tensor):
        return embeddings.to(device=DEVICE, dtype=torch.float32)
    if not convert_to_tensor:
        return np.asarray(embeddings, dtype=np.float32)
    return embeddings


def compute_doc_hash(file_bytes: bytes) -> str:
    h = hashlib.sha1()
    h.update(file_bytes)
    return h.hexdigest()


def get_model_cache_key() -> str:
    model_key = f"{MODEL_NAME}|{DEVICE}|{CACHE_SCHEMA_VERSION}"
    return hashlib.sha1(model_key.encode("utf-8")).hexdigest()[:12]


def get_cache_paths(doc_hash: str):
    model_key = get_model_cache_key()
    sent_path = CACHE_DIR / f"doc_{doc_hash}_{model_key}_sentences.csv"
    emb_path = CACHE_DIR / f"doc_{doc_hash}_{model_key}_embeddings.npz"
    return sent_path, emb_path


def cache_exists(doc_hash: str) -> bool:
    sent_path, emb_path = get_cache_paths(doc_hash)
    return sent_path.exists() and emb_path.exists()


def load_cache(doc_hash: str):
    sent_path, emb_path = get_cache_paths(doc_hash)
    records = []
    with sent_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bbox_text = row.get("block_bbox", "[]")
            try:
                block_bbox = json.loads(bbox_text) if bbox_text else []
            except json.JSONDecodeError:
                block_bbox = []
            try:
                page = int(row.get("page", "0"))
            except ValueError:
                page = 0
            records.append({
                "sentence_id": row.get("sentence_id", ""),
                "sentence_text": row.get("sentence_text", ""),
                "clean_sentence_text": row.get("clean_sentence_text", row.get("sentence_text", "")),
                "paragraph_id": row.get("paragraph_id", ""),
                "paragraph_text": row.get("paragraph_text", ""),
                "page": page,
                "section": row.get("section") or "Unknown",
                "content_type": row.get("content_type") or "body",
                "reading_tag": row.get("reading_tag") or "General Relevance",
                "block_bbox": block_bbox,
            })
    data = np.load(emb_path)
    embeddings = torch.tensor(data["embeddings"], dtype=torch.float32, device=DEVICE)
    return records, embeddings


def save_cache(doc_hash: str, sentence_records, embeddings: torch.Tensor):
    sent_path, emb_path = get_cache_paths(doc_hash)
    with sent_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sentence_id", "sentence_text", "clean_sentence_text", "paragraph_id",
            "paragraph_text", "page", "section", "content_type", "reading_tag", "block_bbox"
        ])
        for record in sentence_records:
            writer.writerow([
                record.get("sentence_id", ""),
                record.get("sentence_text", ""),
                record.get("clean_sentence_text", record.get("sentence_text", "")),
                record.get("paragraph_id", ""),
                record.get("paragraph_text", ""),
                record.get("page", 0),
                record.get("section", "Unknown"),
                record.get("content_type", "body"),
                record.get("reading_tag", "General Relevance"),
                json.dumps(record.get("block_bbox", [])),
            ])
    np.savez_compressed(emb_path, embeddings=embeddings.detach().cpu().float().numpy())


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


def clean_sentences(text: str, min_words=5, max_words=TARGET_SENTENCE_MAX_WORDS):
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


def normalize_target_heading(text: str) -> str:
    text = text or ""
    text = text.replace(" ", " ")
    text = re.sub(r"^[\s\d.IVXivx\-–—:]+", "", text)
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def canonical_section_from_heading(text: str):
    normalized = normalize_target_heading(text)
    if not normalized or len(normalized.split()) > 8:
        return None
    if normalized in SECTION_ALIASES_TARGET:
        return SECTION_ALIASES_TARGET[normalized]
    for alias, canonical in SECTION_ALIASES_TARGET.items():
        if normalized.endswith(f" {alias}"):
            return canonical
    return None


def looks_like_title_heading(text: str, median_size: float = 0.0, body_size_hint: float = 0.0) -> bool:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return True
    if canonical_section_from_heading(text):
        return True

    words = text.split()
    if not words:
        return True

    has_sentence_punctuation = bool(re.search(r"[.!?]$", text))
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True

    uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    title_case_ratio = sum(1 for w in words if w[:1].isupper()) / max(1, len(words))
    bigger_than_body = bool(body_size_hint and median_size and median_size >= body_size_hint + 1.0)

    if len(words) <= 8 and not has_sentence_punctuation:
        if uppercase_ratio > 0.60 or title_case_ratio >= 0.70 or bigger_than_body:
            return True

    if len(words) <= 25 and bigger_than_body and not has_sentence_punctuation:
        return True

    if len(words) <= 12 and title_case_ratio >= 0.85 and not has_sentence_punctuation:
        return True

    return False

def looks_like_table_like_text(text: str) -> bool:
    """Detect simple table/statistic blocks so they are not scored or highlighted.

    This is intentionally conservative. It catches rows with many numeric tokens,
    repeated separators, and very fragmented short tokens. Body sentences that
    merely mention a table or figure are still allowed unless they are captions.
    """
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return True
    tokens = cleaned.split()
    if len(tokens) < 4:
        return False

    numeric_tokens = sum(1 for tok in tokens if re.search(r"\d", tok))
    numeric_ratio = numeric_tokens / max(1, len(tokens))
    separator_count = len(re.findall(r"[\|•·]{1,}| {3,}", text or ""))
    short_token_ratio = sum(1 for tok in tokens if len(tok) <= 2) / max(1, len(tokens))

    no_sentence_punctuation = not re.search(r"[.!?]", cleaned)
    bullet_like = bool(re.search(r"[•·●▪]", cleaned))
    yes_no_tokens = sum(1 for tok in tokens if tok.lower().strip(";:,.()[]") in {"yes", "no"})
    titleish_tokens = sum(1 for tok in tokens if re.match(r"^[A-Z][A-Za-z-]+$", tok))
    titleish_ratio = titleish_tokens / max(1, len(tokens))
    method_table_terms = [
        "content based filtering", "collaborative filtering", "hybrid approach",
        "title and abstract", "whole content", "citation context",
        "limited features", "single level", "purely content based",
    ]
    lower_cleaned = cleaned.lower()

    if len(tokens) <= 40 and numeric_ratio >= TABLE_NUMERIC_TOKEN_RATIO_LIMIT and no_sentence_punctuation:
        return True
    if separator_count >= 2 and numeric_ratio >= 0.25:
        return True
    if len(tokens) >= 8 and short_token_ratio >= 0.55 and numeric_ratio >= 0.25:
        return True
    if bullet_like and no_sentence_punctuation and len(tokens) <= 28:
        return True
    if yes_no_tokens >= 1 and no_sentence_punctuation and titleish_ratio >= 0.35 and len(tokens) <= 35:
        return True
    if sum(1 for term in method_table_terms if term in lower_cleaned) >= 2 and len(tokens) <= 40:
        return True
    if re.match(r"^step\s*\d+\s*[:.-]", lower_cleaned) and len(tokens) <= 18 and no_sentence_punctuation:
        return True
    return False


def looks_like_non_body_text(text: str) -> bool:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return True
    lowered = text.lower()

    if lowered.startswith(NON_BODY_PREFIXES):
        return True
    if any(tok in lowered for tok in ["doi", "http://", "https://", "ieee access", "arxiv:"]):
        return True
    if re.match(r"^\[\d+\]", text):
        return True
    if looks_like_table_like_text(text):
        return True

    if len(text.split()) <= 8 and not re.search(r"[.!?]$", text) and looks_like_title_heading(text):
        return True
    return False

def extract_text_from_block(block) -> tuple[str, float]:
    parts = []
    sizes = []
    for line in block.get("lines", []):
        line_parts = []
        for span in line.get("spans", []):
            span_text = (span.get("text") or "").strip()
            if span_text:
                line_parts.append(span_text)
                try:
                    sizes.append(float(span.get("size", 0.0)))
                except Exception:
                    pass
        if line_parts:
            parts.append(" ".join(line_parts))
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    median_size = float(np.median(sizes)) if sizes else 0.0
    return text, median_size


def strip_leading_section_heading_prefix(text: str):
    """Remove merged section prefixes like '1 Introduction This paper...' from body text.

    Returns (cleaned_text, detected_section). The detected section can update the
    nearest-section metadata, while the cleaned text is used for scoring and PDF search.
    """
    original = re.sub(r"\s+", " ", text or "").strip()
    if not original:
        return "", None
    aliases = sorted(SECTION_ALIASES_TARGET.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, canonical in aliases:
        if canonical == "References":
            continue
        alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
        pattern = re.compile(
            rf"^\s*(?:\d+(?:\.\d+)*\s*|[IVXLC]+\.?\s*)?{alias_pattern}\s*(?:[:;,.\-–—]+\s*|\s+)(?P<rest>.+)$",
            re.IGNORECASE,
        )
        match = pattern.match(original)
        if not match:
            continue
        rest = re.sub(r"\s+", " ", match.group("rest")).strip()
        if len(rest.split()) >= 5:
            return rest, canonical
    return original, None


def clean_body_sentence(sentence: str):
    sentence = re.sub(r"\s+", " ", sentence or "").strip()
    sentence, detected_section = strip_leading_section_heading_prefix(sentence)

    for alias, canonical in sorted(SECTION_ALIASES_TARGET.items(), key=lambda item: len(item[0]), reverse=True):
        if canonical == "References":
            continue
        alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
        sentence = re.sub(
            rf"^\s*(?:\d+(?:\.\d+)*\s*|[IVXLC]+\.?\s*)?{alias_pattern}\s*[:;,.\-–—]*\s+",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip()

    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence, detected_section

def is_heading_only_sentence(sentence: str) -> bool:
    sentence = re.sub(r"\s+", " ", sentence or "").strip()
    if not sentence:
        return True
    if canonical_section_from_heading(sentence):
        return True
    if len(sentence.split()) <= 8 and not re.search(r"[.!?]$", sentence) and looks_like_title_heading(sentence):
        return True
    return False


def infer_reading_tag(section: str, sentence: str, paragraph: str = "") -> str:
    text = f"{section or ''} {sentence or ''} {paragraph or ''}".lower()
    section = section or "Unknown"
    if section == "Future Work" or any(k in text for k in ["future work", "in the future", "we plan", "could be extended", "future research"]):
        return "Future Work"
    if section == "Limitations" or any(k in text for k in ["limitation", "limitations", "however", "challenge", "fails", "cannot", "restricted", "constraint"]):
        return "Limitation"
    if section in {"Results", "Experiments", "Discussion"} or any(k in text for k in ["outperform", "improve", "achieve", "accuracy", "precision", "recall", "f1", "experiment", "evaluation", "result"]):
        return "Result"
    if section == "Methodology" or any(k in text for k in ["we propose", "our method", "architecture", "framework", "pipeline", "algorithm", "model", "approach", "method"]):
        return "Method"
    if section in {"Abstract", "Introduction", "Related Work", "Background"} or any(k in text for k in ["prior work", "existing", "previous", "motivation", "problem", "challenge", "background"]):
        return "Background"
    return "General Relevance"


def truncate_context(text: str, max_chars: int = PARAGRAPH_CONTEXT_MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def extract_structured_sentence_records(file_bytes: bytes):
    """Extract body sentence records with paragraph context, page and section metadata."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    records = []
    current_section = "Unknown"
    stop_after_references = False
    sentence_counter = 1
    paragraph_counter = 1

    size_samples = []
    try:
        for page_index in range(min(len(doc), 3)):
            page_dict = doc[page_index].get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                block_text, median_size = extract_text_from_block(block)
                if median_size and len(block_text.split()) >= 8:
                    size_samples.append(median_size)
    except Exception:
        pass
    body_size_hint = float(np.median(size_samples)) if size_samples else 0.0

    for page_index, page in enumerate(doc):
        if stop_after_references:
            break
        try:
            page_dict = page.get_text("dict")
        except Exception:
            continue

        for block in page_dict.get("blocks", []):
            if stop_after_references:
                break
            if block.get("type") != 0:
                continue

            block_text, median_size = extract_text_from_block(block)
            if not block_text:
                continue

            heading = canonical_section_from_heading(block_text)
            if heading:
                if heading == "References":
                    stop_after_references = True
                    break
                current_section = heading
                continue

            if looks_like_title_heading(block_text, median_size, body_size_hint):
                continue

            stripped_block_text, detected_section = strip_leading_section_heading_prefix(block_text)
            if detected_section:
                current_section = detected_section
                block_text = stripped_block_text

            if looks_like_non_body_text(block_text):
                continue

            raw_sentences = simple_sentence_split(block_text)
            paragraph_sentences = []
            paragraph_section = current_section
            for raw_sentence in raw_sentences:
                clean_sentence, sentence_section = clean_body_sentence(raw_sentence)
                if sentence_section:
                    paragraph_section = sentence_section
                if is_heading_only_sentence(clean_sentence):
                    continue
                if looks_like_non_body_text(clean_sentence):
                    continue
                num_words = len(clean_sentence.split())
                if num_words < 5 or num_words > TARGET_SENTENCE_MAX_WORDS:
                    continue
                paragraph_sentences.append(clean_sentence)

            if not paragraph_sentences:
                continue

            paragraph_text = re.sub(r"\s+", " ", " ".join(paragraph_sentences)).strip()
            if len(paragraph_text.split()) < 8:
                continue

            paragraph_id = f"p{paragraph_counter:03d}"
            paragraph_counter += 1
            for sentence in paragraph_sentences:
                records.append({
                    "sentence_id": f"s{sentence_counter:03d}",
                    "sentence_text": sentence,
                    "clean_sentence_text": sentence,
                    "paragraph_id": paragraph_id,
                    "paragraph_text": paragraph_text,
                    "page": page_index + 1,
                    "section": paragraph_section,
                    "content_type": "body",
                    "reading_tag": infer_reading_tag(paragraph_section, sentence, paragraph_text),
                    "block_bbox": list(block.get("bbox", [])),
                })
                sentence_counter += 1

    doc.close()
    return records


def fallback_sentence_records_from_text(text: str):
    clean_text = strip_references_section(text)
    sentences = clean_sentences(clean_text)
    records = []
    paragraph_counter = 1
    sentence_counter = 1
    for i in range(0, len(sentences), 3):
        paragraph_sentences = []
        for raw in sentences[i:i + 3]:
            cleaned, _ = clean_body_sentence(raw)
            if cleaned and not is_heading_only_sentence(cleaned):
                paragraph_sentences.append(cleaned)
        if not paragraph_sentences:
            continue
        paragraph_text = " ".join(paragraph_sentences)
        paragraph_id = f"p{paragraph_counter:03d}"
        paragraph_counter += 1
        for sentence in paragraph_sentences:
            records.append({
                "sentence_id": f"s{sentence_counter:03d}",
                "sentence_text": sentence,
                "clean_sentence_text": sentence,
                "paragraph_id": paragraph_id,
                "paragraph_text": paragraph_text,
                "page": 0,
                "section": "Unknown",
                "content_type": "body",
                "reading_tag": infer_reading_tag("Unknown", sentence, paragraph_text),
                "block_bbox": [],
            })
            sentence_counter += 1
    return records


def filter_records_by_sections(sentence_records, sentence_embeddings, allowed_sections):
    allowed = set(allowed_sections or DEFAULT_ALLOWED_SECTIONS)
    selected_records = []
    selected_indices = []
    for idx, record in enumerate(sentence_records):
        if record.get("content_type", "body") != "body":
            continue
        if record.get("section", "Unknown") not in allowed:
            continue
        selected_records.append(dict(record))
        selected_indices.append(idx)

    if sentence_embeddings is None:
        return selected_records, None
    if not selected_indices:
        return [], sentence_embeddings[:0]
    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=sentence_embeddings.device)
    return selected_records, sentence_embeddings.index_select(0, index_tensor)


def summarize_sections(sentence_records):
    counts = {}
    for record in sentence_records:
        section = record.get("section", "Unknown")
        counts[section] = counts.get(section, 0) + 1
    if not counts:
        return "None"
    return ", ".join(f"{name} ({count})" for name, count in sorted(counts.items()))


def score_sentence_records(profile_text: str, sentence_records, sentence_embeddings: torch.Tensor | None = None, profile_embedding: torch.Tensor | None = None):
    if profile_embedding is None:
        profile_embedding = encode_texts(profile_text, is_query=True, convert_to_tensor=True)
    profile_embedding = profile_embedding.to(device=DEVICE, dtype=torch.float32)

    if sentence_embeddings is None:
        texts = [record.get("clean_sentence_text") or record.get("sentence_text", "") for record in sentence_records]
        sentence_embeddings = encode_texts(texts, is_query=False, convert_to_tensor=True)
    sentence_embeddings = sentence_embeddings.to(device=DEVICE, dtype=torch.float32)

    if profile_embedding.ndim == 1:
        profile_embedding = profile_embedding.unsqueeze(0)
    if sentence_embeddings.ndim == 1:
        sentence_embeddings = sentence_embeddings.unsqueeze(0)

    if profile_embedding.shape[-1] != sentence_embeddings.shape[-1]:
        raise ValueError(
            "Embedding dimension mismatch. This usually means cached PDF embeddings were created "
            "with a different model. The app now uses a model-aware cache, but you may need to "
            "clear old runtime cache once. Profile dim="
            f"{profile_embedding.shape[-1]}, sentence dim={sentence_embeddings.shape[-1]}."
        )

    scores = util.cos_sim(profile_embedding, sentence_embeddings)[0].detach().cpu().numpy()
    records = []
    for idx, (record, score) in enumerate(zip(sentence_records, scores)):
        enriched = dict(record)
        enriched["score"] = float(score)
        enriched["sentence_score"] = float(score)
        enriched["embedding_index"] = idx
        records.append(enriched)
    records.sort(key=lambda x: x["score"], reverse=True)
    return records, sentence_embeddings


def rank_paragraph_segments(scored_sentence_records):
    """Aggregate sentence relevance into paragraph-level reading segments.

    Sentences are scored for precision, but paragraphs are ranked to preserve
    reading context. The best sentence in each selected paragraph is highlighted
    in conservative modes. Expanded mode can highlight a paragraph region when the
    surrounding sentences are also semantically relevant.
    """
    grouped = {}
    for record in scored_sentence_records:
        pid = record.get("paragraph_id") or record.get("sentence_id")
        grouped.setdefault(pid, []).append(record)

    paragraph_records = []
    for pid, records in grouped.items():
        records_sorted = sorted(records, key=lambda r: r.get("sentence_score", r.get("score", 0.0)), reverse=True)
        best = dict(records_sorted[0])
        top_scores = [float(r.get("sentence_score", r.get("score", 0.0))) for r in records_sorted[:2]]
        best_score = top_scores[0]
        top2_avg = float(np.mean(top_scores)) if top_scores else best_score
        paragraph_score = (PARAGRAPH_SCORE_BEST_WEIGHT * best_score) + (PARAGRAPH_SCORE_TOP2_WEIGHT * top2_avg)
        paragraph_text = best.get("paragraph_text") or best.get("sentence_text", "")

        support_boundary = max(0.0, best_score - BROAD_PARAGRAPH_SUPPORT_MARGIN)
        support_records = [
            r for r in records_sorted
            if float(r.get("sentence_score", r.get("score", 0.0))) >= support_boundary
        ][:BROAD_PARAGRAPH_HIGHLIGHT_MAX_SENTENCES]

        best["score"] = float(paragraph_score)
        best["paragraph_score"] = float(paragraph_score)
        best["sentence_score"] = float(best_score)
        best["paragraph_context"] = paragraph_text
        best["paragraph_context_short"] = truncate_context(paragraph_text)
        best["supporting_sentence_count"] = len(records_sorted)
        best["semantically_close_sentence_count"] = len(support_records)
        best["paragraph_highlight_sentences"] = [r.get("sentence_text", "") for r in support_records if r.get("sentence_text")]
        best["paragraph_highlight_ready"] = len(support_records) >= BROAD_PARAGRAPH_HIGHLIGHT_MIN_SUPPORT
        best["highlight_scope"] = "sentence"
        best["reading_tag"] = infer_reading_tag(best.get("section", "Unknown"), best.get("sentence_text", ""), paragraph_text)
        paragraph_records.append(best)

    paragraph_records.sort(key=lambda x: x["paragraph_score"], reverse=True)
    return paragraph_records

def summarize_top_sections(paragraph_records, limit=3):
    section_scores = {}
    for record in paragraph_records:
        section = record.get("section", "Unknown")
        section_scores.setdefault(section, []).append(float(record.get("paragraph_score", record.get("score", 0.0))))
    if not section_scores:
        return "None"
    ranked = []
    for section, scores in section_scores.items():
        ranked.append((section, float(np.mean(sorted(scores, reverse=True)[:5]))))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(f"{section} ({score:.3f})" for section, score in ranked[:limit])


def get_highlight_density_config(highlight_density: str):
    return HIGHLIGHT_DENSITY_SETTINGS.get(highlight_density, HIGHLIGHT_DENSITY_SETTINGS["Standard"])


def compute_candidate_score_boundary(max_score: float, threshold: float, config: dict) -> float:
    """Return a dynamic semantic boundary for candidate expansion.

    The boundary is model-score based, not random. Expanded mode relaxes the user
    threshold and also includes sentences close to the top score in vector space.
    """
    relaxed_threshold = max(0.0, float(threshold) - float(config["threshold_relax"]))
    near_top_boundary = float(max_score) - float(config["score_boundary_margin"])
    return max(float(config["minimum_score_floor"]), min(relaxed_threshold, near_top_boundary))


def select_highlight_candidates(df_sorted, top_k: int, threshold: float, highlight_density: str):
    if not df_sorted:
        return "No usable paragraph segments found.", [], [], {
            "mode": "none",
            "target_pdf_matches": 0,
            "candidate_limit": 0,
            "score_boundary": None,
            "highlight_density": highlight_density,
        }

    config = get_highlight_density_config(highlight_density)
    total_segments = len(df_sorted)
    max_score = float(df_sorted[0]["score"])
    num_above_threshold = sum(row["score"] >= threshold for row in df_sorted)

    top_k = int(top_k)
    base_target = int(config["target_pdf_matches"])
    target_pdf_matches = max(top_k, base_target)
    candidate_limit = max(int(config["candidate_limit"]), target_pdf_matches * 3)
    score_boundary = None

    if max_score < OFF_PROFILE_LIMIT:
        target_pdf_matches = min(top_k, 8)
        candidate_limit = min(candidate_limit, 25)
        candidate_records = [dict(row, label="Best guess", highlight_scope="sentence") for row in df_sorted[:candidate_limit]]
        summary = (
            f"Mode: Off profile (topic mismatch)\n"
            f"Ranking unit: paragraph context with sentence-level PDF highlights\n"
            f"Total paragraph segments: {total_segments}\n"
            f"Max paragraph similarity: {max_score:.3f}\n"
            f"Threshold ignored in this mode\n"
            f"Highlight density: {highlight_density}\n"
            f"Target PDF-visible best guesses: {target_pdf_matches}\n"
            f"Candidate search limit: {len(candidate_records)}\n"
            f"Best guesses are shown in blue, not yellow, to avoid overclaiming relevance."
        )
        mode = "off_profile"
    else:
        score_boundary = compute_candidate_score_boundary(max_score, threshold, config)
        boundary_candidates = [row for row in df_sorted if row["score"] >= score_boundary]
        if not boundary_candidates:
            boundary_candidates = df_sorted[:top_k]
            note = "No segments met the semantic boundary, so top segments were used."
        else:
            note = "Semantic boundary expansion used to find enough PDF-visible paragraph segments."

        candidate_records = []
        paragraph_highlight_candidates = 0
        for row in boundary_candidates[:candidate_limit]:
            row_copy = dict(row, label="Relevant")
            if (
                highlight_density == "Expanded"
                and config.get("allow_paragraph_highlight")
                and row_copy.get("paragraph_highlight_ready")
                and row_copy.get("block_bbox")
            ):
                row_copy["highlight_scope"] = "paragraph"
                paragraph_highlight_candidates += 1
            else:
                row_copy["highlight_scope"] = "sentence"
            candidate_records.append(row_copy)

        summary = (
            f"Mode: Profile match\n"
            f"Ranking unit: paragraph context with sentence-level PDF highlights\n"
            f"Total paragraph segments: {total_segments}\n"
            f"Max paragraph similarity: {max_score:.3f}\n"
            f"Segments above user threshold: {num_above_threshold}\n"
            f"Highlight density: {highlight_density}\n"
            f"Semantic candidate boundary: {score_boundary:.3f}\n"
            f"Target PDF-visible highlights: {target_pdf_matches}\n"
            f"Candidate search limit: {len(candidate_records)}\n"
            f"Most relevant sections: {summarize_top_sections(candidate_records)}\n"
            f"Expanded paragraph-ready segments: {paragraph_highlight_candidates if highlight_density == 'Expanded' else 0}\n"
            f"{note}"
        )
        mode = "profile"

    preview_records = candidate_records[:top_k]
    highlights = [(f"[{row['score']:.3f}] {row['sentence_text']}", row["label"]) for row in preview_records]
    metadata = {
        "mode": mode,
        "target_pdf_matches": target_pdf_matches,
        "candidate_limit": len(candidate_records),
        "score_boundary": score_boundary,
        "highlight_density": highlight_density,
        "prefer_full_sentence": bool(config.get("prefer_full_sentence", True)),
        "allow_partial_sentence_fallback": bool(config.get("allow_partial_sentence_fallback", True)),
        "allow_paragraph_highlight": bool(config.get("allow_paragraph_highlight", False)),
    }
    return summary, highlights, candidate_records, metadata

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


def safe_filename_stem(name: str) -> str:
    stem = Path(name or "uploaded_paper").stem
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem).strip("_")
    return cleaned or "uploaded_paper"


def normalize_for_pdf_search(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\u00ad": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\ufb01": "fi",
        "\ufb02": "fl",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_inline_citations(text: str) -> str:
    text = re.sub(r"\s*\[[0-9,;\-\s]+\]", "", text)
    text = re.sub(r"\s*\([A-Z][A-Za-z]+(?: et al\.)?,?\s*\d{4}[a-z]?\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_search_candidates(sentence: str, allow_partial_fallback=True):
    base_variants = []
    normalized = normalize_for_pdf_search(sentence)
    no_cites = remove_inline_citations(normalized)
    for item in [normalized, no_cites]:
        if item and item not in base_variants:
            base_variants.append(item)

    candidates = []
    for base in base_variants:
        words = base.split()
        raw_parts = [base]

        if allow_partial_fallback:
            for size in [18, 14, 10]:
                if len(words) >= size:
                    raw_parts.append(" ".join(words[:size]))
                    middle_start = max(0, (len(words) - size) // 2)
                    raw_parts.append(" ".join(words[middle_start:middle_start + size]))
                    raw_parts.append(" ".join(words[-size:]))

        for part in raw_parts:
            part = normalize_for_pdf_search(part)
            if len(part.split()) >= 5 and len(part) >= 25 and part not in candidates:
                candidates.append(part)
    return candidates

def rect_from_match(match):
    try:
        return fitz.Rect(match.rect)
    except Exception:
        try:
            return fitz.Rect(match)
        except Exception:
            return None


def rect_overlap_ratio(rect_a, rect_b):
    if rect_a is None or rect_b is None:
        return 0.0
    inter = fitz.Rect(rect_a)
    inter.intersect(rect_b)
    inter_area = max(0.0, inter.get_area())
    base_area = max(1.0, fitz.Rect(rect_a).get_area())
    return inter_area / base_area


def filter_matches_to_body_block(matches, block_bbox, min_overlap=0.15):
    if not block_bbox or len(block_bbox) != 4:
        return matches
    block_rect = fitz.Rect(block_bbox)
    filtered = []
    for match in matches:
        match_rect = rect_from_match(match)
        if match_rect is None:
            continue
        if rect_overlap_ratio(match_rect, block_rect) >= min_overlap:
            filtered.append(match)
    return filtered


def add_annotation_for_matches(page, matches, color):
    try:
        annot = page.add_highlight_annot(matches)
        try:
            annot.set_colors(stroke=color)
        except Exception:
            pass
        annot.update()
    except Exception:
        for match in matches:
            annot = page.add_highlight_annot(match)
            try:
                annot.set_colors(stroke=color)
            except Exception:
                pass
            annot.update()


def add_paragraph_bbox_highlight(doc, page_index, block_bbox, color=(1, 1, 0)):
    if page_index is None or page_index < 0 or page_index >= len(doc):
        return {
            "matched": False,
            "page_index": None,
            "page": "-",
            "matched_text": "",
            "num_boxes": 0,
            "highlight_scope": "paragraph",
        }
    if not block_bbox or len(block_bbox) != 4:
        return {
            "matched": False,
            "page_index": None,
            "page": "-",
            "matched_text": "",
            "num_boxes": 0,
            "highlight_scope": "paragraph",
        }

    page = doc[page_index]
    rect = fitz.Rect(block_bbox)
    try:
        annot = page.add_highlight_annot(rect)
        try:
            annot.set_colors(stroke=color)
            annot.set_opacity(0.22)
        except Exception:
            pass
        annot.update()
        return {
            "matched": True,
            "page_index": page_index,
            "page": page_index + 1,
            "matched_text": "paragraph region",
            "num_boxes": 1,
            "highlight_scope": "paragraph",
        }
    except Exception:
        return {
            "matched": False,
            "page_index": None,
            "page": "-",
            "matched_text": "",
            "num_boxes": 0,
            "highlight_scope": "paragraph",
        }


def add_highlight_for_sentence(
    doc,
    sentence: str,
    color=(1, 1, 0),
    preferred_page_index=None,
    preferred_block_bbox=None,
    allow_partial_fallback=True,
):
    candidates = build_search_candidates(sentence, allow_partial_fallback=allow_partial_fallback)
    if preferred_page_index is not None and 0 <= preferred_page_index < len(doc):
        page_order = [preferred_page_index] + [idx for idx in range(len(doc)) if idx != preferred_page_index]
    else:
        page_order = list(range(len(doc)))

    for candidate in candidates:
        for page_index in page_order:
            page = doc[page_index]
            try:
                matches = page.search_for(candidate, quads=True)
            except TypeError:
                matches = page.search_for(candidate)
            except Exception:
                matches = []

            if matches and page_index == preferred_page_index:
                matches = filter_matches_to_body_block(matches, preferred_block_bbox)

            if matches:
                add_annotation_for_matches(page, matches, color)
                return {
                    "matched": True,
                    "page_index": page_index,
                    "page": page_index + 1,
                    "matched_text": candidate,
                    "num_boxes": len(matches),
                    "highlight_scope": "sentence",
                }
    return {
        "matched": False,
        "page_index": None,
        "page": "-",
        "matched_text": "",
        "num_boxes": 0,
        "highlight_scope": "sentence",
    }


def add_highlight_for_record(doc, row, color=(1, 1, 0), allow_partial_fallback=True):
    preferred_page_index = int(row.get("page", 0)) - 1 if str(row.get("page", "")).isdigit() else None
    block_bbox = row.get("block_bbox")

    if row.get("highlight_scope") == "paragraph":
        result = add_paragraph_bbox_highlight(doc, preferred_page_index, block_bbox, color=color)
        if result.get("matched"):
            return result

    return add_highlight_for_sentence(
        doc,
        row.get("sentence_text", ""),
        color=color,
        preferred_page_index=preferred_page_index,
        preferred_block_bbox=block_bbox,
        allow_partial_fallback=allow_partial_fallback,
    )

def render_pages_to_gallery(doc, page_indices, output_dir: Path):
    gallery_items = []
    for page_index in sorted(page_indices):
        if page_index is None or page_index < 0 or page_index >= len(doc):
            continue
        page = doc[page_index]
        image_path = output_dir / f"page_{page_index + 1:03d}.png"
        try:
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI, annots=True)
        except TypeError:
            zoom = PDF_RENDER_DPI / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pix.save(str(image_path))
        gallery_items.append((str(image_path), f"Page {page_index + 1}"))
    return gallery_items


def get_annotation_color(label: str):
    # Yellow for confident profile-match relevance; blue for off-profile best guesses.
    if label == "Best guess":
        return (0.25, 0.55, 1.0)
    return (1.0, 1.0, 0.0)


def create_highlighted_pdf_outputs(file_bytes: bytes, pdf_name: str, candidate_records, target_pdf_matches: int, table_min_rows: int, allow_partial_sentence_fallback=True):
    if not candidate_records:
        return None, [], [], [], "PDF highlights created: 0 / 0\nRendered highlighted pages: 0"

    output_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = OUTPUTS_DIR / f"analysis_{output_id}_{compute_doc_hash(file_bytes)[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_pdf_path = output_dir / f"{safe_filename_stem(pdf_name)}_highlighted.pdf"
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    match_results = []
    displayed_records = []
    highlighted_pages = set()
    matched_count = 0
    paragraph_highlight_count = 0
    attempted_count = 0
    table_min_rows = max(int(table_min_rows), int(target_pdf_matches))

    for candidate_rank, row in enumerate(candidate_records, start=1):
        if matched_count >= target_pdf_matches and len(displayed_records) >= table_min_rows:
            break

        attempted_count += 1
        row_copy = dict(row)
        row_copy["candidate_rank"] = candidate_rank
        color = get_annotation_color(row_copy.get("label", "Relevant"))

        result = add_highlight_for_record(
            doc,
            row_copy,
            color=color,
            allow_partial_fallback=allow_partial_sentence_fallback,
        )
        result["rank"] = len(displayed_records) + 1
        result["candidate_rank"] = candidate_rank

        should_show = result.get("matched") or len(displayed_records) < table_min_rows
        if should_show:
            displayed_records.append(row_copy)
            match_results.append(result)

        if result.get("matched"):
            matched_count += 1
            if result.get("highlight_scope") == "paragraph":
                paragraph_highlight_count += 1
            highlighted_pages.add(result.get("page_index"))

    gallery_items = []
    pdf_output_path = None

    if matched_count > 0:
        gallery_items = render_pages_to_gallery(doc, highlighted_pages, output_dir)
        doc.save(str(annotated_pdf_path), garbage=4, deflate=True)
        pdf_output_path = str(annotated_pdf_path)
    doc.close()

    status = (
        f"PDF highlights created: {matched_count} / {target_pdf_matches}\n"
        f"Paragraph-region highlights created: {paragraph_highlight_count}\n"
        f"Sentence fallback partial chunks allowed: {'Yes' if allow_partial_sentence_fallback else 'No'}\n"
        f"Candidates attempted for PDF matching: {attempted_count} / {len(candidate_records)}\n"
        f"Rows shown in ranked table: {len(displayed_records)}\n"
        f"Rendered highlighted pages: {len(gallery_items)}\n"
        f"Render DPI: {PDF_RENDER_DPI}"
    )
    if matched_count == 0:
        status += "\nNo annotated PDF was created because none of the selected candidate sentences could be matched in the original PDF text."

    return pdf_output_path, gallery_items, match_results, displayed_records, status

def source_type_display_name(source_type: str) -> str:
    mapping = {
        "seed_paper": "Seed papers",
        "topics": "Topics",
        "keywords": "Keywords",
        "research_statement": "Research statement",
    }
    return mapping.get(source_type, source_type.replace("_", " ").title())


def load_profile_source_data(profile_id):
    if not profile_id:
        return None
    profile_dir = PROFILES_DIR / profile_id
    profile_json = profile_dir / "profile.json"
    if not profile_json.exists():
        return None
    try:
        with profile_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if data.get("embedding_model") != MODEL_NAME:
        return None
    source_path = profile_dir / data.get("source_embeddings_file", "source_embeddings.npz")
    if not source_path.exists():
        return None
    try:
        source_npz = np.load(source_path, allow_pickle=True)
        embeddings = torch.tensor(source_npz["embeddings"], dtype=torch.float32, device=DEVICE)
        items = data.get("source_items", [])
        if len(items) != embeddings.shape[0]:
            return None
        return {"embeddings": embeddings, "items": items}
    except Exception:
        return None


def compact_source_detail(item):
    source_type = item.get("source_type", "profile")
    preview = re.sub(r"\s+", " ", item.get("text_preview", "") or "").strip()
    label = item.get("label", "") or source_type_display_name(source_type)

    if source_type == "topics":
        match = re.search(r"interest signals:\s*(.+?)\.?$", preview, flags=re.IGNORECASE)
        return match.group(1).strip() if match else "selected research topics"
    if source_type == "keywords":
        match = re.search(r"interest signals:\s*(.+?)\.?$", preview, flags=re.IGNORECASE)
        return match.group(1).strip() if match else "user keywords"
    if source_type == "seed_paper":
        return re.sub(r"\s+chunk\s+\d+$", "", label).strip() or "uploaded seed paper evidence"
    if source_type == "research_statement":
        return "the user research statement"
    return label


def explain_highlight(row, sentence_embeddings, source_data):
    if source_data is None or sentence_embeddings is None:
        return "Manual profile", "This highlight was selected by semantic similarity to the profile text provided for this analysis."
    idx = row.get("embedding_index")
    if idx is None:
        return "Profile", "This highlight was selected by profile-to-segment semantic similarity."
    try:
        sentence_embedding = sentence_embeddings[int(idx)].view(1, -1).to(device=DEVICE, dtype=torch.float32)
        source_embeddings = source_data["embeddings"].to(device=DEVICE, dtype=torch.float32)
        scores = util.cos_sim(sentence_embedding, source_embeddings)[0].detach().cpu().numpy()
    except Exception:
        return "Profile", "This highlight was selected by profile-to-segment semantic similarity."

    ranked_indices = list(np.argsort(scores)[::-1])
    selected = []
    seen_types = set()
    for source_idx in ranked_indices:
        item = source_data["items"][int(source_idx)]
        source_type = item.get("source_type", "profile")
        if source_type in seen_types:
            continue
        seen_types.add(source_type)
        selected.append({
            "type": source_type,
            "display": source_type_display_name(source_type),
            "detail": compact_source_detail(item),
            "score": float(scores[int(source_idx)]),
        })
        if len(selected) >= 3:
            break

    if not selected:
        return "Profile", "This highlight was selected by profile-to-segment semantic similarity."

    matched_sources = ", ".join(item["display"] for item in selected)
    reason_bits = []
    for item in selected:
        source_type = item["type"]
        detail = item["detail"]
        if source_type == "keywords":
            reason_bits.append(f"it is close to your keywords ({detail})")
        elif source_type == "topics":
            reason_bits.append(f"it aligns with your selected topics ({detail})")
        elif source_type == "seed_paper":
            reason_bits.append(f"it is similar to evidence from your seed paper ({detail})")
        elif source_type == "research_statement":
            reason_bits.append("it matches your research statement")
        else:
            reason_bits.append(f"it matches {item['display'].lower()}")

    section = row.get("section", "Unknown")
    tag = row.get("reading_tag", "General Relevance")
    explanation = "This highlight relates to your work because " + "; ".join(reason_bits) + "."
    if section and section != "Unknown":
        explanation += f" It appears in the {section} section as {tag.lower()} content."
    return matched_sources, explanation

def make_highlight_table(displayed_records, match_results, sentence_embeddings=None, source_data=None):
    rows = []
    for rank, row in enumerate(displayed_records, start=1):
        match = match_results[rank - 1] if rank - 1 < len(match_results) else {}
        matched = bool(match.get("matched"))
        matched_sources, explanation = explain_highlight(row, sentence_embeddings, source_data)
        rows.append([
            rank,
            round(float(row.get("score", 0.0)), 4),
            round(float(row.get("sentence_score", row.get("score", 0.0))), 4),
            row.get("section", "Unknown"),
            row.get("reading_tag", "General Relevance"),
            row.get("label", "Relevant"),
            (match.get("highlight_scope") or row.get("highlight_scope") or "sentence").title(),
            "Yes" if matched else "No",
            match.get("page", row.get("page", "-")) or "-",
            matched_sources,
            explanation,
            row.get("sentence_text", ""),
            row.get("paragraph_context_short") or truncate_context(row.get("paragraph_context") or row.get("paragraph_text", "")),
        ])
    return rows


def make_highlight_cards(highlight_rows, max_cards=8):
    if not highlight_rows:
        return "<div style='padding:12px;border:1px solid #e5e7eb;border-radius:12px;'>No highlight insights to show yet.</div>"

    cards = []
    for row in highlight_rows[:max_cards]:
        rank, p_score, s_score, section, tag, label, scope, pdf_match, page, sources, explanation, sentence, context = row
        border = "#facc15" if label == "Relevant" else "#60a5fa"
        tag_color = "#eef2ff"
        cards.append(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid {border};border-radius:14px;padding:14px 16px;margin:12px 0;background:#ffffff;box-shadow:0 1px 4px rgba(15,23,42,0.06);">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
                <div style="font-weight:700;font-size:15px;color:#111827;">Highlight {int(rank)} · {escape(str(section))}</div>
                <div style="font-size:12px;background:{tag_color};border-radius:999px;padding:4px 9px;color:#3730a3;white-space:nowrap;">{escape(str(tag))}</div>
            </div>
            <div style="font-size:12px;color:#6b7280;margin-top:4px;">Page {escape(str(page))} · {escape(str(scope))} highlight · PDF match: {escape(str(pdf_match))}</div>
            <div style="margin-top:10px;font-size:13px;color:#374151;"><b>Why it relates to your profile:</b><br>{escape(str(explanation))}</div>
            <div style="margin-top:10px;font-size:13px;color:#374151;"><b>Matched profile sources:</b> {escape(str(sources))}</div>
            <div style="margin-top:10px;padding:10px 12px;background:#fffbeb;border-radius:10px;font-size:13px;color:#111827;"><b>Highlighted sentence</b><br>{escape(str(sentence))}</div>
            <details style="margin-top:10px;font-size:13px;color:#374151;">
                <summary style="cursor:pointer;font-weight:600;">Show paragraph context</summary>
                <div style="margin-top:8px;line-height:1.45;">{escape(str(context))}</div>
            </details>
            <details style="margin-top:8px;font-size:12px;color:#6b7280;">
                <summary style="cursor:pointer;">Advanced scores</summary>
                <div style="margin-top:6px;">Paragraph score: {p_score} · Sentence score: {s_score}</div>
            </details>
        </div>
        """)
    if len(highlight_rows) > max_cards:
        cards.append(f"<div style='font-size:12px;color:#6b7280;margin-top:8px;'>Showing {max_cards} insight cards. See Advanced ranked results for all {len(highlight_rows)} rows.</div>")
    return "<div>" + "\n".join(cards) + "</div>"


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


def empty_analysis_response(message):
    return message, [], None, "", []


def analyze_pdf(profile_text, loaded_profile_id, loaded_profile_text, pdf_file, top_k, threshold, highlight_density, allowed_sections):
    try:
        return analyze_pdf_impl(profile_text, loaded_profile_id, loaded_profile_text, pdf_file, top_k, threshold, highlight_density, allowed_sections)
    except Exception as exc:
        error_trace = traceback.format_exc()
        print(error_trace, flush=True)
        return empty_analysis_response(
            "Analysis failed.\n\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n\n"
            "Check the terminal log for the full traceback."
        )


def analyze_pdf_impl(profile_text, loaded_profile_id, loaded_profile_text, pdf_file, top_k, threshold, highlight_density, allowed_sections):
    if not pdf_file:
        return empty_analysis_response("Please upload a PDF.")
    if not profile_text or not profile_text.strip():
        return empty_analysis_response("Please enter your research profile text or load a saved profile.")

    allowed_sections = allowed_sections or DEFAULT_ALLOWED_SECTIONS
    profile_embedding = None
    source_data = None
    embedding_status = "Encoding profile text for this analysis."
    if loaded_profile_id and loaded_profile_text and profile_text.strip() == loaded_profile_text.strip():
        profile_embedding, embedding_status = load_profile_embedding(loaded_profile_id)
        source_data = load_profile_source_data(loaded_profile_id)

    file_bytes, pdf_name = extract_pdf_bytes_and_name(pdf_file)
    if not file_bytes:
        return empty_analysis_response("Could not read the uploaded PDF.")

    doc_hash = compute_doc_hash(file_bytes)

    if cache_exists(doc_hash):
        all_sentence_records, all_sentence_embeddings = load_cache(doc_hash)
    else:
        raw_text = extract_text_from_pdf_bytes(file_bytes)
        if len(raw_text.strip()) < 50:
            return empty_analysis_response("The PDF seems empty or scanned with no extractable text.")
        all_sentence_records = extract_structured_sentence_records(file_bytes)
        if len(all_sentence_records) < 5:
            all_sentence_records = fallback_sentence_records_from_text(raw_text)
        if len(all_sentence_records) < 5:
            return empty_analysis_response("The PDF did not yield enough usable body text.")
        _, all_sentence_embeddings = score_sentence_records(
            profile_text,
            all_sentence_records,
            sentence_embeddings=None,
            profile_embedding=profile_embedding,
        )
        save_cache(doc_hash, all_sentence_records, all_sentence_embeddings)

    sentence_records, sentence_embeddings = filter_records_by_sections(
        all_sentence_records,
        all_sentence_embeddings,
        allowed_sections,
    )
    if len(sentence_records) < 5:
        section_summary = summarize_sections(all_sentence_records)
        return empty_analysis_response(
            "The selected sections did not yield enough usable body sentences.\n\n"
            f"Sections detected: {section_summary}\n"
            "Try including Unknown or more sections in the section filter."
        )

    scored_sentence_records, sentence_embeddings = score_sentence_records(
        profile_text,
        sentence_records,
        sentence_embeddings=sentence_embeddings,
        profile_embedding=profile_embedding,
    )
    paragraph_records = rank_paragraph_segments(scored_sentence_records)
    summary, _, candidate_records, selection_meta = select_highlight_candidates(paragraph_records, int(top_k), float(threshold), highlight_density)

    highlighted_pdf_path, gallery_items, match_results, displayed_records, pdf_status = create_highlighted_pdf_outputs(
        file_bytes,
        pdf_name or "uploaded.pdf",
        candidate_records,
        selection_meta["target_pdf_matches"],
        int(top_k),
        allow_partial_sentence_fallback=selection_meta.get("allow_partial_sentence_fallback", True),
    )
    highlights_table = make_highlight_table(displayed_records, match_results, sentence_embeddings, source_data)
    highlight_cards = make_highlight_cards(highlights_table)

    summary = (
        f"{summary}\n"
        f"Profile embedding: {embedding_status}\n"
        f"Model: {MODEL_NAME}\n"
        f"Cache key: {get_model_cache_key()}\n"
        f"Body extraction: section-aware paragraph heuristic\n"
        f"Usable body sentences after section filter: {len(sentence_records)}\n"
        f"Paragraph segments after aggregation: {len(paragraph_records)}\n"
        f"Sections detected: {summarize_sections(all_sentence_records)}\n"
        f"Sections included for scoring: {summarize_sections(sentence_records)}\n"
        f"Allowed section filter: {', '.join(allowed_sections)}\n"
        f"Explanations: {'source-level profile evidence' if source_data else 'manual profile similarity only'}\n"
        f"{pdf_status}"
    )
    append_run_log(pdf_name or "uploaded.pdf", int(top_k), float(threshold), summary)
    return summary, gallery_items, highlighted_pdf_path, highlight_cards, highlights_table


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


SECTION_HEADING_NAMES = [
    "abstract",
    "keywords",
    "index terms",
    "introduction",
    "background",
    "related work",
    "literature review",
    "method",
    "methods",
    "methodology",
    "approach",
    "proposed method",
    "system design",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "acknowledgment",
    "acknowledgement",
    "references",
]


def normalize_heading_line(line: str) -> str:
    line = re.sub(r"[^A-Za-z ]+", " ", line or "")
    line = re.sub(r"\s+", " ", line).strip().lower()
    return line


def is_known_section_heading(line: str) -> bool:
    normalized = normalize_heading_line(line)
    if not normalized or len(normalized.split()) > 5:
        return False
    return any(normalized == name or normalized.endswith(f" {name}") for name in SECTION_HEADING_NAMES)


def extract_section_blocks_by_headings(text: str, wanted_names):
    """Extract simple section blocks by line headings.

    This is intentionally heuristic. It improves seed evidence quality when
    common headings are present and falls back safely when they are not.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks = []
    found_names = []
    i = 0
    while i < len(lines):
        normalized = normalize_heading_line(lines[i])
        matched_name = None
        for name in wanted_names:
            if normalized == name or normalized.endswith(f" {name}"):
                matched_name = name
                break
        if not matched_name:
            i += 1
            continue

        collected = []
        j = i + 1
        while j < len(lines):
            if is_known_section_heading(lines[j]):
                break
            collected.append(lines[j])
            j += 1
        block = " ".join(collected).strip()
        if len(block.split()) >= 25:
            blocks.append(block)
            found_names.append(matched_name)
        i = max(j, i + 1)
    return blocks, found_names


def extract_abstract_inline(text: str):
    pattern = re.compile(
        r"(?is)(?:^|\n)\s*abstract\s*[—:.-]?\s*(.*?)(?=\n\s*(?:keywords|index terms|1\s+introduction|i\.\s+introduction|introduction)\b)",
    )
    match = pattern.search(text)
    if not match:
        return ""
    block = re.sub(r"\s+", " ", match.group(1)).strip()
    return block if len(block.split()) >= 25 else ""


def extract_preferred_seed_text(clean_text: str):
    wanted = ["abstract", "introduction", "conclusion", "conclusions"]
    blocks, found = extract_section_blocks_by_headings(clean_text, wanted)

    inline_abstract = extract_abstract_inline(clean_text)
    if inline_abstract and "abstract" not in found:
        blocks.insert(0, inline_abstract)
        found.insert(0, "abstract")

    if blocks:
        return "\n\n".join(blocks[:3]), sorted(set(found)), "section_aware"
    return clean_text, [], "fallback_first_clean_sentences"


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
        preferred_text, sections_found, extraction_method = extract_preferred_seed_text(clean_text)
        sentences = clean_sentences(preferred_text, min_words=6, max_words=80)
        if not sentences and extraction_method != "fallback_first_clean_sentences":
            extraction_method = "fallback_first_clean_sentences"
            sections_found = []
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
            "seed_extraction_method": extraction_method,
            "preferred_sections_found": sections_found,
        })

    return evidence_blocks, seed_chunks, seed_metadata, errors


def build_profile_source_items(profile_name, selected_topics, seed_papers, keyword_text, free_profile_text):
    validation = validate_profile_sources(selected_topics, seed_papers, keyword_text, free_profile_text)
    evidence_blocks, seed_chunks, seed_metadata, seed_errors = extract_seed_paper_evidence(seed_papers)

    source_items = []

    # Seed evidence is placed first to protect the richest signal if any downstream model truncates text.
    seed_total_weight = SOURCE_WEIGHTS["seed_papers_total"] if seed_chunks else 0.0
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
            "weight": SOURCE_WEIGHTS["topics"],
            "text": "The user selected the following research areas as interest signals: " f"{topic_text}.",
        })

    if validation["keywords"]:
        keyword_text_clean = ", ".join(validation["keywords"])
        source_items.append({
            "source_type": "keywords",
            "label": "user keywords",
            "weight": SOURCE_WEIGHTS["keywords"],
            "text": "The user provided the following keywords as additional interest signals: " f"{keyword_text_clean}.",
        })

    if validation["free_profile_text"]:
        source_items.append({
            "source_type": "research_statement",
            "label": "manual research statement",
            "weight": SOURCE_WEIGHTS["research_statement"],
            "text": "The user provided the following research statement or existing project description: " f"{validation['free_profile_text']}",
        })

    sections = []
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
        "source_weights": SOURCE_WEIGHTS,
        "seed_weight_policy": "fixed_total_weight_split_across_seed_chunks",
    }
    return combined_profile_text, metadata, source_items


def create_profile_embedding(source_items):
    if not source_items:
        return None, None, [], []

    source_texts = [item["text"] for item in source_items]
    weights = np.array([float(item["weight"]) for item in source_items], dtype=np.float32)
    labels = [f"{item['source_type']}::{item['label']}" for item in source_items]

    source_embeddings_tensor = encode_texts(source_texts, is_query=True, convert_to_tensor=True)
    source_embeddings_tensor = source_embeddings_tensor.to(device=DEVICE, dtype=torch.float32)
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
        f"- {paper['file_name']} ({paper['sentences_used_in_profile']} sentences, "
        f"{paper['num_chunks']} chunks, method={paper.get('seed_extraction_method', 'unknown')}, "
        f"sections={', '.join(paper.get('preferred_sections_found', [])) or 'none'})"
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
        "source_weights": SOURCE_WEIGHTS,
        "seed_weight_policy": "fixed_total_weight_split_across_seed_chunks",
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
            "profile_version": "v1.4",
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
            "Build a reusable research profile, then use it to highlight relevant paper segments with paragraph context."
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
                    "Choose a saved profile, upload a paper, and review personalized highlights with profile-based explanations."
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
                        top_k_input = gr.Slider(1, 20, value=8, step=1, label="Top-K reading segments")
                        threshold_input = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="Similarity threshold")
                        highlight_density_input = gr.Dropdown(
                            choices=list(HIGHLIGHT_DENSITY_SETTINGS.keys()),
                            value="Standard",
                            label="Highlight style",
                            info="Concise shows a clean high-confidence set, Standard is the recommended default, and Expanded can use paragraph-region highlighting when surrounding sentences are also relevant.",
                        )
                        with gr.Accordion("Reading filters", open=False):
                            allowed_sections_input = gr.Dropdown(
                                choices=HIGHLIGHT_SECTION_CHOICES,
                                value=DEFAULT_ALLOWED_SECTIONS,
                                multiselect=True,
                                label="Sections to consider",
                                info="Headings, captions, figures, tables, and references are excluded automatically. Unknown is useful for PDFs where section headings are hard to detect.",
                            )
                        run_button = gr.Button("Analyze paper")

                    with gr.Column(scale=1):
                        summary_output = gr.Textbox(label="Summary", lines=12)
                        highlighted_pages_gallery = gr.Gallery(
                            label="Highlighted paper view",
                            columns=1,
                            object_fit="contain",
                            height=650,
                        )
                        highlighted_pdf_output = gr.File(label="Download annotated PDF")
                        highlight_cards_output = gr.HTML(label="Highlight insights")
                        with gr.Accordion("Advanced ranked results", open=False):
                            highlights_table_output = gr.Dataframe(
                                headers=["Rank", "Paragraph Score", "Sentence Score", "Section", "Tag", "Label", "Scope", "PDF Match", "Page", "Matched Sources", "Explanation", "Highlighted Sentence", "Paragraph Context"],
                                datatype=["number", "number", "number", "str", "str", "str", "str", "str", "str", "str", "str", "str", "str"],
                                label="Technical ranked results",
                                interactive=False,
                                wrap=True,
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
            inputs=[profile_input, loaded_profile_id_state, loaded_profile_text_state, pdf_input, top_k_input, threshold_input, highlight_density_input, allowed_sections_input],
            outputs=[summary_output, highlighted_pages_gallery, highlighted_pdf_output, highlight_cards_output, highlights_table_output],
        )

    return demo


cleanup_old_output_dirs()
print("===== Application Startup =====", flush=True)
demo = build_demo()

if __name__ == "__main__":
    print("Launching Gradio app...", flush=True)
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )
