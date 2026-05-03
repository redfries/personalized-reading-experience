import re
from pathlib import Path
import fitz

SECTION_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "literature review": "Related Work",
    "method": "Methodology",
    "methods": "Methodology",
    "methodology": "Methodology",
    "approach": "Methodology",
    "proposed method": "Methodology",
    "experiments": "Experiments",
    "experiment": "Experiments",
    "evaluation": "Experiments",
    "results": "Results",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "limitation": "Limitations",
    "future work": "Future Work",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "references": "References",
    "reference": "References",
    "bibliography": "References",
    "works cited": "References",
}

REFERENCE_HEADINGS = {"references", "reference", "bibliography", "works cited"}

JUNK_PREFIXES = (
    "figure", "fig.", "table", "algorithm", "equation", "appendix",
    "copyright", "doi", "http", "https", "www.", "arxiv",
    "isbn", "issn", "received", "accepted", "published",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\x00", " ")).strip()


def clean_heading_text(text: str) -> str:
    clean = normalize_ws(text)
    clean = re.sub(r"^(\d+(\.\d+)*|[IVX]+)[.)]?\s+", "", clean, flags=re.I).strip()
    clean = re.sub(r"^[A-Z]\.\s+", "", clean).strip()
    return clean


def looks_like_references_heading(text: str) -> bool:
    clean = clean_heading_text(text).lower().strip(":.- ")
    if clean in REFERENCE_HEADINGS:
        return True
    if len(clean.split()) <= 4 and any(clean.startswith(h) for h in REFERENCE_HEADINGS):
        return True
    return False


def looks_like_section_heading(text: str) -> str | None:
    clean = clean_heading_text(text)
    clean_lower = clean.lower().strip(":.- ")
    if clean_lower in SECTION_ALIASES:
        return SECTION_ALIASES[clean_lower]
    if len(clean.split()) <= 5 and clean_lower in SECTION_ALIASES:
        return SECTION_ALIASES[clean_lower]
    return None


def looks_like_heading_fragment(text: str) -> bool:
    clean = normalize_ws(text)
    if not clean:
        return True

    if looks_like_section_heading(clean):
        return True

    words = clean.split()
    punctuation_count = len(re.findall(r"[.!?]", clean))
    if len(words) <= 8 and punctuation_count == 0:
        upper_or_title = sum(1 for w in words if w.isupper() or w[:1].isupper())
        if upper_or_title >= max(3, int(0.7 * len(words))):
            return True

    if len(words) <= 6 and clean.isupper():
        return True

    return False


def strip_leading_section_prefix(text: str) -> str:
    clean = normalize_ws(text)
    for raw in sorted(SECTION_ALIASES.keys(), key=len, reverse=True):
        pattern = rf"^((\d+(\.\d+)*|[IVX]+)[.)]?\s+)?{re.escape(raw)}[:.]?\s+"
        clean = re.sub(pattern, "", clean, flags=re.I).strip()
    return clean


def looks_like_caption_or_junk(text: str) -> bool:
    clean = normalize_ws(text)
    lower = clean.lower()
    if not clean:
        return True
    if lower.startswith(JUNK_PREFIXES):
        return True
    if "@" in clean and len(clean.split()) < 20:
        return True
    if looks_like_references_heading(clean):
        return True
    return False


def looks_like_reference_entry(text: str) -> bool:
    clean = normalize_ws(text)
    lower = clean.lower()
    words = clean.split()
    if len(words) < 6:
        return False

    if re.match(r"^\[\d+\]\s+", clean):
        return True
    if re.match(r"^\d{1,3}\.\s+[A-Z][A-Za-z-]+", clean):
        return True
    if re.search(r"\b(19|20)\d{2}\b", clean) and re.search(r"\bdoi\b|https?://|www\.|proceedings|conference|journal|transactions|vol\.|no\.", lower):
        return True
    if re.search(r"\b[A-Z][A-Za-z-]+,\s+[A-Z]\.", clean) and re.search(r"\b(19|20)\d{2}\b", clean):
        return True
    if lower.count(",") >= 4 and re.search(r"\b(19|20)\d{2}\b", lower):
        return True
    return False


def line_stats(raw_text: str) -> dict:
    lines = [normalize_ws(line) for line in (raw_text or "").splitlines() if normalize_ws(line)]
    word_counts = [len(line.split()) for line in lines]
    punctuation_counts = [len(re.findall(r"[.!?]", line)) for line in lines]
    return {
        "lines": lines,
        "line_count": len(lines),
        "avg_words": sum(word_counts) / max(len(word_counts), 1),
        "short_line_ratio": sum(1 for c in word_counts if c <= 4) / max(len(word_counts), 1),
        "punctuation_total": sum(punctuation_counts),
    }


def looks_like_table_fragment(text: str, raw_text: str | None = None) -> bool:
    clean = normalize_ws(text)
    if not clean:
        return True

    words = clean.split()
    if len(words) < 5:
        return True

    stats = line_stats(raw_text or text)

    bullet_count = clean.count("•") + clean.count("|") + clean.count("\t")
    yes_no_count = len(re.findall(r"\b(yes|no)\b", clean, flags=re.I))
    number_count = len(re.findall(r"\b\d+(\.\d+)?\b", clean))
    punctuation_count = len(re.findall(r"[.!?]", clean))
    comma_count = clean.count(",")

    if stats["line_count"] >= 4 and stats["avg_words"] <= 5 and stats["punctuation_total"] <= 1:
        return True
    if stats["line_count"] >= 5 and stats["short_line_ratio"] >= 0.65 and stats["punctuation_total"] <= 2:
        return True
    if bullet_count >= 2 and punctuation_count == 0:
        return True
    if yes_no_count >= 2 and punctuation_count == 0:
        return True
    if number_count >= 6 and punctuation_count <= 1:
        return True

    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    if len(words) >= 8 and avg_word_len < 4 and punctuation_count == 0:
        return True

    title_caseish = sum(1 for w in words if w[:1].isupper())
    if len(words) <= 14 and title_caseish >= max(6, int(0.7 * len(words))) and punctuation_count == 0:
        return True

    table_terms = len(re.findall(r"\b(method|approach|dataset|feature|title|abstract|yes|no|profile|metadata|baseline|accuracy|precision|recall)\b", clean, flags=re.I))
    if table_terms >= 5 and punctuation_count <= 1 and comma_count <= 2:
        return True

    return False


def split_sentences(paragraph: str) -> list[str]:
    text = normalize_ws(paragraph)
    if not text:
        return []
    parts = SENTENCE_SPLIT_RE.split(text)
    cleaned = []
    for part in parts:
        part = strip_leading_section_prefix(part)
        part = normalize_ws(part)
        words = part.split()
        if (
            6 <= len(words) <= 90
            and not looks_like_caption_or_junk(part)
            and not looks_like_heading_fragment(part)
            and not looks_like_table_fragment(part)
            and not looks_like_reference_entry(part)
        ):
            cleaned.append(part)
    return cleaned


def extract_seed_text(pdf_path: Path, max_sentences: int = 70) -> list[str]:
    sentences = []
    current_section = "Unknown"

    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                raw_text = block[4] or ""
                text = normalize_ws(raw_text)
                if not text:
                    continue

                if looks_like_references_heading(text):
                    return sentences[:max_sentences]

                section = looks_like_section_heading(text)
                if section:
                    current_section = section
                    if section == "References":
                        return sentences[:max_sentences]
                    continue

                if current_section == "References":
                    return sentences[:max_sentences]

                if (
                    looks_like_caption_or_junk(text)
                    or looks_like_table_fragment(text, raw_text)
                    or looks_like_reference_entry(text)
                    or looks_like_heading_fragment(text)
                ):
                    continue

                for sentence in split_sentences(text):
                    sentences.append(sentence)
                    if len(sentences) >= max_sentences:
                        return sentences

    return sentences[:max_sentences]


def extract_paragraph_records(pdf_path: Path) -> list[dict]:
    records = []
    current_section = "Unknown"
    paragraph_id = 0
    reference_like_streak = 0

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_rect = page.rect
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

            for block in blocks:
                x0, y0, x1, y1, raw_text, *_ = block
                text = normalize_ws(raw_text)
                if not text:
                    continue

                if looks_like_references_heading(text):
                    return records

                section = looks_like_section_heading(text)
                if section:
                    current_section = section
                    if section == "References":
                        return records
                    continue

                if current_section == "References":
                    return records

                clean_text = strip_leading_section_prefix(text)

                if looks_like_reference_entry(clean_text):
                    reference_like_streak += 1
                    if reference_like_streak >= 2:
                        return records
                    continue
                else:
                    reference_like_streak = 0

                if (
                    looks_like_caption_or_junk(clean_text)
                    or looks_like_table_fragment(clean_text, raw_text)
                    or looks_like_heading_fragment(clean_text)
                ):
                    continue

                sentences = split_sentences(clean_text)
                if not sentences:
                    continue

                paragraph_id += 1
                records.append({
                    "paragraph_id": f"p{paragraph_id:04d}",
                    "page": page_index,
                    "section": current_section,
                    "paragraph_text": clean_text,
                    "sentences": sentences,
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "page_width": float(page_rect.width),
                    "page_height": float(page_rect.height),
                })

    return records


def bbox_to_percent(bbox: list[float], page_width: float, page_height: float) -> dict:
    x0, y0, x1, y1 = bbox
    return {
        "x": round(100 * x0 / page_width, 3),
        "y": round(100 * y0 / page_height, 3),
        "width": round(100 * (x1 - x0) / page_width, 3),
        "height": round(100 * (y1 - y0) / page_height, 3),
    }


def rect_area(rect: dict) -> float:
    return max(rect.get("width", 0), 0) * max(rect.get("height", 0), 0)


def overlap_ratio(a: dict, b: dict) -> float:
    ax0, ay0 = a["x"], a["y"]
    ax1, ay1 = a["x"] + a["width"], a["y"] + a["height"]
    bx0, by0 = b["x"], b["y"]
    bx1, by1 = b["x"] + b["width"], b["y"] + b["height"]

    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = ix * iy
    smaller = max(min(rect_area(a), rect_area(b)), 1e-6)
    return intersection / smaller


def dedupe_rectangles(rects: list[dict], threshold: float = 0.72) -> list[dict]:
    cleaned = []
    for rect in rects:
        if rect_area(rect) <= 0:
            continue
        if any(rect.get("page") == prev.get("page") and overlap_ratio(rect, prev) >= threshold for prev in cleaned):
            continue
        cleaned.append(rect)
    return cleaned


def search_sentence_rects(pdf_path: Path, page_number: int, sentence: str, paragraph_bbox: list[float] | None = None) -> list[dict]:
    candidates = []
    sentence = normalize_ws(sentence)

    if sentence:
        candidates.append(sentence)

    words = sentence.split()
    if len(words) >= 16:
        candidates.append(" ".join(words[:16]))
        mid = len(words) // 2
        candidates.append(" ".join(words[max(0, mid - 8): mid + 8]))
        candidates.append(" ".join(words[-16:]))

    rects_percent = []
    seen = set()

    with fitz.open(pdf_path) as doc:
        if page_number < 1 or page_number > len(doc):
            return []

        page = doc[page_number - 1]
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)

        for candidate in candidates:
            if len(candidate.split()) < 5:
                continue

            for rect in page.search_for(candidate):
                bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                key = tuple(round(v, 1) for v in bbox)
                if key in seen:
                    continue
                seen.add(key)

                if paragraph_bbox:
                    px0, py0, px1, py1 = paragraph_bbox
                    overlap_x = max(0, min(bbox[2], px1) - max(bbox[0], px0))
                    overlap_y = max(0, min(bbox[3], py1) - max(bbox[1], py0))
                    overlap_area = overlap_x * overlap_y
                    rect_area_abs = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
                    if overlap_area / rect_area_abs < 0.55:
                        continue

                rect_percent = bbox_to_percent(bbox, page_w, page_h)
                rect_percent["page"] = page_number
                rects_percent.append(rect_percent)

            if rects_percent:
                break

    return dedupe_rectangles(rects_percent)


def paragraph_rect(record: dict) -> dict:
    rect = bbox_to_percent(record["bbox"], record["page_width"], record["page_height"])
    rect["page"] = record["page"]
    return rect
