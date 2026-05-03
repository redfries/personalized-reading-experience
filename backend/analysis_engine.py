import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .model_service import encode_texts, get_model_name, get_device
from .llm_note import generate_llm_note
from .pdf_utils import (
    extract_paragraph_records,
    paragraph_rect,
    search_sentence_rects,
    overlap_ratio,
    dedupe_rectangles,
)

RESCUE_MESSAGE = "This paper appears weakly related to your profile. I expanded the search to find possible bridge points, but these are weaker signals."


def cosine_scores(profile_embedding: np.ndarray, sentence_embeddings: torch.Tensor) -> np.ndarray:
    profile = torch.tensor(profile_embedding, dtype=torch.float32, device=sentence_embeddings.device).unsqueeze(0)
    sentence_embeddings = sentence_embeddings.float()
    scores = torch.mm(profile, sentence_embeddings.T)[0].detach().cpu().numpy()
    return scores.astype("float32")


def normalize_for_dedupe(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\[[0-9,\s-]+\]", " ", text)
    text = re.sub(r"\(([a-z]+ et al\.,?\s*)?(19|20)\d{2}[a-z]?\)", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words[:28])


def token_similarity(a: str, b: str) -> float:
    aset = set(a.split())
    bset = set(b.split())
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / max(len(aset | bset), 1)


def reading_tag(section: str, text: str) -> str:
    lower = (section + " " + text).lower()

    if "future work" in lower or "in the future" in lower or "we plan" in lower:
        return "Future Work"
    if "limitation" in lower or "however" in lower or "challenge" in lower or "cannot" in lower:
        return "Limitation"
    if section in {"Methodology", "Experiments"} or any(w in lower for w in ["we propose", "our method", "framework", "algorithm", "model computes", "pipeline"]):
        return "Method"
    if section in {"Results", "Discussion"} or any(w in lower for w in ["outperform", "improve", "achieve", "precision", "recall", "f1", "accuracy", "result"]):
        return "Result"
    if section in {"Introduction", "Background", "Related Work", "Abstract"}:
        return "Background"
    return "General Relevance"


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "using", "used",
    "use", "are", "was", "were", "has", "have", "had", "can", "could", "should",
    "paper", "papers", "article", "articles", "research", "system", "systems",
    "model", "models", "method", "methods", "approach", "approaches",
}


def clip_for_reason(text: str, limit: int = 210) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def extract_present_keywords(keywords: list[str], text: str) -> list[str]:
    lower = (text or "").lower()
    found = []
    for keyword in keywords or []:
        key = str(keyword).strip()
        if not key:
            continue
        key_lower = key.lower()
        key_tokens = [t for t in re.findall(r"[a-z0-9]+", key_lower) if len(t) > 2]
        if key_lower in lower or any(t in lower for t in key_tokens):
            found.append(key)
    return found[:5]


def extract_present_topics(topics: list[str], text: str) -> list[str]:
    lower_tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    found = []
    for topic in topics or []:
        topic_text = str(topic).strip()
        topic_tokens = [
            t for t in re.findall(r"[a-z0-9]+", topic_text.lower())
            if len(t) > 2 and t not in STOPWORDS
        ]
        if topic_tokens and any(t in lower_tokens for t in topic_tokens):
            found.append(topic_text)
    return found[:4]


def source_type_label(source_type: str) -> str:
    labels = {
        "topics": "Topics",
        "keywords": "Keywords",
        "research_statement": "Research statement",
        "seed_paper": "Seed paper",
    }
    return labels.get(source_type, source_type.replace("_", " ").title())


def best_profile_sources(profile: dict, source_scores: list[float] | None) -> list[dict]:
    source_items = profile.get("source_items", []) or []
    if not source_scores or not source_items:
        return []

    rows = []
    for idx, item in enumerate(source_items):
        if idx >= len(source_scores):
            break
        rows.append({
            "source_type": item.get("source_type", "profile"),
            "source_label": item.get("source_label", source_type_label(item.get("source_type", "profile"))),
            "score": float(source_scores[idx]),
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:3]


def tag_specific_reason(tag: str, section: str, sentence: str, paragraph: str) -> str:
    text = f"{sentence} {paragraph}".lower()

    if tag == "Result":
        if any(term in text for term in ["outperform", "improve", "precision", "recall", "f1", "accuracy", "result", "evaluation"]):
            return "It is useful because it describes evidence, evaluation, or performance signals from the paper."
        return "It is useful because it appears to summarize an outcome or finding."

    if tag == "Method":
        if any(term in text for term in ["framework", "algorithm", "compute", "extract", "feature", "model", "pipeline", "proposed"]):
            return "It is useful because it explains how the paper's method, model, features, or pipeline works."
        return "It is useful because it describes method-level content rather than only background."

    if tag == "Limitation":
        return "It is useful because it points to a constraint, challenge, or missing piece that may affect how relevant the paper is."

    if tag == "Future Work":
        return "It is useful because it suggests where this line of work could go next."

    if tag == "Background":
        return "It is useful because it gives background context that helps connect the paper to your research direction."

    if section and section != "Unknown":
        return f"It is useful because it appears in the {section} section and may help you decide whether to read that part closely."

    return "It is useful because it is one of the closest semantic matches to your saved profile."


def explanation_from_profile(
    profile: dict,
    section: str,
    tag: str,
    sentence: str,
    paragraph: str,
    source_scores: list[float] | None = None,
    rescue_mode: bool = False,
) -> tuple[list[str], str]:
    """Create a sentence-specific explanation.

    Earlier versions used only the profile source list, so every highlight sounded the same.
    This version uses the selected sentence, paragraph text, lexical matches, section/tag,
    and per-source semantic scores from the saved profile.
    """
    topics = profile.get("selected_topics", []) or []
    keywords = profile.get("keywords", []) or []
    text = f"{sentence} {paragraph}"

    found_keywords = extract_present_keywords(keywords, text)
    found_topics = extract_present_topics(topics, text)
    top_sources = best_profile_sources(profile, source_scores)

    matched_sources = []
    if found_keywords:
        matched_sources.append("Keywords: " + ", ".join(found_keywords[:3]))
    if found_topics:
        matched_sources.append("Topics: " + ", ".join(found_topics[:3]))

    for source in top_sources[:2]:
        label = source_type_label(source["source_type"])
        source_label = clip_for_reason(str(source["source_label"]), 55)
        if source["score"] >= 0.20:
            matched_sources.append(f"{label}: {source_label}")

    if not matched_sources:
        matched_sources = ["Profile semantic match"]

    parts = []

    exact_sentence = clip_for_reason(sentence, 260)
    if exact_sentence:
        parts.append(f'The selected evidence says: "{exact_sentence}"')

    if found_keywords:
        parts.append("It directly overlaps with your profile keywords: " + ", ".join(found_keywords[:4]) + ".")
    elif keywords:
        parts.append("It is semantically close to your keyword profile even though the exact keyword wording is not repeated.")

    if found_topics:
        parts.append("It connects to your selected topic area through: " + ", ".join(found_topics[:3]) + ".")
    elif topics:
        best_topic_source = next((s for s in top_sources if s["source_type"] == "topics"), None)
        if best_topic_source and best_topic_source["score"] >= 0.20:
            parts.append("Its wording is semantically close to your selected research topics.")

    seed_source = next((s for s in top_sources if s["source_type"] == "seed_paper"), None)
    if seed_source and seed_source["score"] >= 0.20:
        parts.append(
            "Among your saved profile sources, the closest supporting source is the seed paper "
            f'"{clip_for_reason(seed_source["source_label"], 80)}".'
        )

    statement_source = next((s for s in top_sources if s["source_type"] == "research_statement"), None)
    if statement_source and statement_source["score"] >= 0.20:
        parts.append("It also matches your written research statement more strongly than the other profile sources.")

    parts.append(tag_specific_reason(tag, section, sentence, paragraph))

    if section and section != "Unknown":
        parts.append(f"Section context: {section}. Reading tag: {tag or 'General Relevance'}.")

    if rescue_mode:
        prefix = "This is a weaker bridge-point highlight, not a strong relevance claim. "
    else:
        prefix = "This highlight was selected for this specific sentence. "

    explanation = prefix + " ".join(parts)

    return matched_sources[:5], explanation

def density_config(density: str) -> dict:
    density = (density or "Standard").lower()

    # v4.3.5: modes are intentionally visibly different.
    # Concise = fewer, stronger, mostly sentence-level highlights.
    # Standard = balanced, broader, allows paragraph context.
    # Expanded = broad scan, widest search, weak bridge support.
    if density == "concise":
        return {
            "top_k": 7,
            "threshold": 0.32,
            "expanded": False,
            "min_highlights_before_rescue": 3,
            "label": "Concise",
        }
    if density == "expanded":
        return {
            "top_k": 35,
            "threshold": 0.16,
            "expanded": True,
            "min_highlights_before_rescue": 0,
            "label": "Expanded",
        }
    return {
        "top_k": 18,
        "threshold": 0.22,
        "expanded": True,
        "min_highlights_before_rescue": 5,
        "label": "Standard",
    }


def rescue_config() -> dict:
    return {"top_k": 28, "threshold": 0.16, "expanded": True, "min_highlights_before_rescue": 0, "label": "Expanded rescue"}


def paragraph_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    ordered = sorted(scores, reverse=True)
    best = ordered[0]
    top2_avg = sum(ordered[:2]) / min(len(ordered), 2)
    return 0.7 * best + 0.3 * top2_avg


def classify_profile_match(all_candidates: list[dict]) -> dict:
    if not all_candidates:
        return {
            "label": "Off-topic",
            "top_score": 0.0,
            "top_5_average": 0.0,
            "candidate_count": 0,
            "strong_candidate_count": 0,
            "message": "This paper does not appear to contain enough usable text that matches the selected profile.",
        }

    ordered = sorted(all_candidates, key=lambda x: (x["paragraph_score"], x["sentence_score"]), reverse=True)
    top_score = float(max(item["sentence_score"] for item in ordered))
    top_5 = ordered[:5]
    top_5_average = float(sum(item["sentence_score"] for item in top_5) / max(len(top_5), 1))
    strong_candidate_count = sum(1 for item in ordered if item["sentence_score"] >= 0.34 or item["paragraph_score"] >= 0.34)
    moderate_candidate_count = sum(1 for item in ordered if item["sentence_score"] >= 0.26 or item["paragraph_score"] >= 0.26)

    if (top_score >= 0.45 and top_5_average >= 0.34) or strong_candidate_count >= 6:
        label = "Strong match"
        message = "This paper appears strongly related to the selected profile."
    elif top_score >= 0.34 or top_5_average >= 0.28 or strong_candidate_count >= 2:
        label = "Moderate match"
        message = "This paper appears moderately related to the selected profile."
    elif top_score >= 0.24 or top_5_average >= 0.20 or moderate_candidate_count >= 2:
        label = "Weak match"
        message = RESCUE_MESSAGE
    else:
        label = "Off-topic"
        message = "This paper appears off-topic for the selected profile. I found only very weak bridge points, so treat any highlights as low-confidence."

    return {
        "label": label,
        "top_score": round(top_score, 4),
        "top_5_average": round(top_5_average, 4),
        "candidate_count": len(ordered),
        "strong_candidate_count": strong_candidate_count,
        "message": message,
    }


def rects_are_duplicate(rects: list[dict], used_rects_by_page: dict[int, list[dict]], threshold: float = 0.70) -> bool:
    if not rects:
        return False
    duplicate_count = 0
    for rect in rects:
        used = used_rects_by_page.get(rect.get("page"), [])
        if any(overlap_ratio(rect, prev) >= threshold for prev in used):
            duplicate_count += 1
    return duplicate_count == len(rects)


def add_used_rects(rects: list[dict], used_rects_by_page: dict[int, list[dict]]) -> None:
    for rect in rects:
        page = rect.get("page")
        used_rects_by_page.setdefault(page, []).append(rect)


def filter_new_rects(rects: list[dict], used_rects_by_page: dict[int, list[dict]], threshold: float = 0.70) -> list[dict]:
    cleaned = []
    for rect in rects:
        used = used_rects_by_page.get(rect.get("page"), [])
        if any(overlap_ratio(rect, prev) >= threshold for prev in used):
            continue
        cleaned.append(rect)
    return dedupe_rectangles(cleaned)


def dedupe_ranked_candidates(candidates: list[dict], top_k: int) -> list[dict]:
    selected = []
    seen_keys = []

    for item in candidates:
        key = normalize_for_dedupe(item["best_sentence"])
        if len(key.split()) < 5:
            continue

        if any(key == existing or token_similarity(key, existing) >= 0.82 for existing in seen_keys):
            continue

        paragraph_key = (
            item["record"]["page"],
            round(item["record"]["bbox"][0], 1),
            round(item["record"]["bbox"][1], 1),
            round(item["record"]["bbox"][2], 1),
            round(item["record"]["bbox"][3], 1),
        )

        if any(prev.get("paragraph_key") == paragraph_key for prev in selected):
            continue

        item["dedupe_key"] = key
        item["paragraph_key"] = paragraph_key
        selected.append(item)
        seen_keys.append(key)

        if len(selected) >= top_k:
            break

    return selected


def select_candidates(candidates: list[dict], cfg: dict) -> list[dict]:
    filtered = [
        item for item in candidates
        if item["sentence_score"] >= cfg["threshold"] or item["paragraph_score"] >= cfg["threshold"]
    ]
    filtered = sorted(filtered, key=lambda x: (x["paragraph_score"], x["sentence_score"]), reverse=True)
    return dedupe_ranked_candidates(filtered, cfg["top_k"])


def build_highlights(selected: list[dict], profile: dict, pdf_path: Path, cfg: dict, rescue_mode: bool) -> list[dict]:
    highlights = []
    used_rects_by_page = {}

    for item in selected:
        record = item["record"]
        tag = reading_tag(record["section"], item["best_sentence"] + " " + record["paragraph_text"])

        scope = "Sentence"
        rects = search_sentence_rects(
            pdf_path,
            record["page"],
            item["best_sentence"],
            paragraph_bbox=record["bbox"],
        )

        if cfg["expanded"] and len([s for s in item["all_scores"] if s >= max(0.20, item["sentence_score"] - 0.08)]) >= 2:
            scope = "Paragraph"
            rects = [paragraph_rect(record)]

        if not rects and cfg["expanded"]:
            rects = [paragraph_rect(record)]
            scope = "Paragraph"

        rects = filter_new_rects(rects, used_rects_by_page)
        if rects_are_duplicate(rects, used_rects_by_page):
            continue

        add_used_rects(rects, used_rects_by_page)

        matched_sources, explanation = explanation_from_profile(
            profile=profile,
            section=record["section"],
            tag=tag,
            sentence=item["best_sentence"],
            paragraph=record["paragraph_text"],
            source_scores=item.get("source_scores", []),
            rescue_mode=rescue_mode,
        )

        highlights.append({
            "highlight_id": f"h{len(highlights) + 1:03d}",
            "rank": len(highlights) + 1,
            "page": record["page"],
            "section": record["section"],
            "tag": tag,
            "scope": scope,
            "label": "Bridge point" if rescue_mode else "Relevant",
            "sentence_score": round(float(item["sentence_score"]), 4),
            "paragraph_score": round(float(item["paragraph_score"]), 4),
            "matched_sources": matched_sources,
            "explanation": explanation,
            "highlighted_sentence": item["best_sentence"],
            "paragraph_context": record["paragraph_text"],
            "rects": rects,
            "pdf_match": bool(rects),
            "low_confidence": bool(rescue_mode),
        })

    return highlights


def build_llm_evidence(pdf_path: Path, records: list[dict], candidates: list[dict], highlights: list[dict]) -> dict:
    """Build compact, structured PDF evidence for Gemini.

    This does not send the whole PDF blindly. It sends:
    - abstract if detected
    - small section-wise paragraph samples
    - top embedding candidates with exact sentences and paragraph context
    - final highlight evidence
    """
    section_priority = [
        "Abstract",
        "Introduction",
        "Background",
        "Related Work",
        "Methodology",
        "Experiments",
        "Results",
        "Discussion",
        "Limitations",
        "Future Work",
        "Conclusion",
        "Unknown",
    ]

    def clip(text: str, limit: int) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    abstract_parts = []
    for record in records:
        if record.get("section") == "Abstract":
            abstract_parts.append(record.get("paragraph_text", ""))
            if len(" ".join(abstract_parts)) > 1800:
                break

    by_section: dict[str, list[str]] = {}
    for record in records:
        section = record.get("section", "Unknown")
        if section == "References":
            continue
        by_section.setdefault(section, [])
        if len(by_section[section]) < 2:
            by_section[section].append(clip(record.get("paragraph_text", ""), 900))

    section_samples = []
    seen = set()
    for section in section_priority:
        if section in by_section and section not in seen:
            section_samples.append({
                "section": section,
                "paragraphs": by_section[section],
            })
            seen.add(section)

    # Include any non-priority sections if they exist.
    for section, paragraphs in by_section.items():
        if section not in seen and len(section_samples) < 8:
            section_samples.append({
                "section": section,
                "paragraphs": paragraphs[:2],
            })
            seen.add(section)

    top_candidates = []
    ordered = sorted(candidates, key=lambda x: (x["paragraph_score"], x["sentence_score"]), reverse=True)[:12]
    for idx, item in enumerate(ordered, start=1):
        record = item["record"]
        top_candidates.append({
            "rank": idx,
            "section": record.get("section", "Unknown"),
            "page": record.get("page"),
            "sentence_score": round(float(item.get("sentence_score", 0.0)), 4),
            "paragraph_score": round(float(item.get("paragraph_score", 0.0)), 4),
            "sentence": clip(item.get("best_sentence", ""), 600),
            "paragraph": clip(record.get("paragraph_text", ""), 1000),
        })

    highlight_evidence = []
    for item in highlights[:10]:
        highlight_evidence.append({
            "rank": item.get("rank"),
            "section": item.get("section"),
            "page": item.get("page"),
            "tag": item.get("tag"),
            "label": item.get("label"),
            "low_confidence": item.get("low_confidence", False),
            "sentence_score": item.get("sentence_score"),
            "paragraph_score": item.get("paragraph_score"),
            "highlighted_sentence": clip(item.get("highlighted_sentence", ""), 600),
            "paragraph_context": clip(item.get("paragraph_context", ""), 1000),
        })

    return {
        "paper_title": pdf_path.name,
        "abstract": clip(" ".join(abstract_parts), 1800),
        "section_samples": section_samples[:8],
        "top_embedding_candidates": top_candidates,
        "highlight_evidence": highlight_evidence,
    }


def analyze_pdf_with_profile(pdf_path: Path, profile: dict, profiles_dir: Path, analysis_dir: Path, density: str = "Standard") -> dict:
    profile_id = profile["profile_id"]
    profile_dir = profiles_dir / profile_id
    embedding_path = profile_dir / "profile_embedding.npy"

    if not embedding_path.exists():
        raise FileNotFoundError("Saved profile embedding not found. Re-save the profile in v4.3.5.")

    profile_embedding = np.load(embedding_path).astype("float32")

    records = extract_paragraph_records(pdf_path)
    if not records:
        raise ValueError("No usable body paragraphs were extracted from this PDF. The file may be scanned, reference-only, or table-heavy.")

    sentence_rows = []
    sentence_texts = []
    for record in records:
        for sentence in record["sentences"]:
            sentence_rows.append((record, sentence))
            sentence_texts.append(sentence)

    if not sentence_texts:
        raise ValueError("No usable body sentences were extracted from this PDF.")

    sentence_embeddings = encode_texts(sentence_texts, batch_size=16)
    scores = cosine_scores(profile_embedding, sentence_embeddings)

    source_score_matrix = None
    source_items = profile.get("source_items", []) or []
    source_embedding_path = profile_dir / "source_embeddings.npz"
    if source_embedding_path.exists() and source_items:
        try:
            source_embeddings = np.load(source_embedding_path)["embeddings"].astype("float32")
            if source_embeddings.shape[0] == len(source_items):
                source_tensor = torch.tensor(source_embeddings, dtype=torch.float32, device=sentence_embeddings.device)
                source_score_matrix = torch.mm(sentence_embeddings.float(), source_tensor.T).detach().cpu().numpy()
        except Exception:
            source_score_matrix = None

    grouped = {}
    for idx, ((record, sentence), score) in enumerate(zip(sentence_rows, scores)):
        pid = record["paragraph_id"]
        grouped.setdefault(pid, {"record": record, "items": []})
        source_scores = source_score_matrix[idx].tolist() if source_score_matrix is not None else []
        grouped[pid]["items"].append({
            "sentence": sentence,
            "score": float(score),
            "source_scores": source_scores,
        })

    candidates = []
    for _, data in grouped.items():
        items = sorted(data["items"], key=lambda x: x["score"], reverse=True)
        score_values = [item["score"] for item in items]
        candidates.append({
            "record": data["record"],
            "best_sentence": items[0]["sentence"],
            "sentence_score": items[0]["score"],
            "paragraph_score": paragraph_score(score_values),
            "all_scores": score_values,
            "source_scores": items[0].get("source_scores", []),
        })

    profile_match = classify_profile_match(candidates)
    cfg = density_config(density)
    selected = select_candidates(candidates, cfg)

    rescue_mode = False
    analysis_mode_used = cfg["label"]

    if cfg["label"] != "Expanded":
        too_few = len(selected) < cfg["min_highlights_before_rescue"]
        weak_match = profile_match["label"] in {"Weak match", "Off-topic"}
        if too_few or weak_match:
            rescue_mode = True
            cfg = rescue_config()
            selected = select_candidates(candidates, cfg)
            analysis_mode_used = "Expanded rescue"
            profile_match["message"] = RESCUE_MESSAGE
    else:
        rescue_mode = profile_match["label"] in {"Weak match", "Off-topic"}
        analysis_mode_used = "Expanded broad scan"

    highlights = build_highlights(selected, profile, pdf_path, cfg, rescue_mode=rescue_mode)

    top_sections = []
    for h in highlights:
        if h["section"] not in top_sections:
            top_sections.append(h["section"])

    analysis_id = "analysis_" + uuid.uuid4().hex[:10]
    out_dir = analysis_dir / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_pdf = out_dir / pdf_path.name
    if pdf_path.resolve() != saved_pdf.resolve():
        saved_pdf.write_bytes(pdf_path.read_bytes())

    pdf_matches = sum(1 for h in highlights if h["rects"])
    llm_evidence = build_llm_evidence(pdf_path, records, candidates, highlights)
    warnings = []
    if rescue_mode:
        warnings.append(RESCUE_MESSAGE)
    if highlights and pdf_matches / max(len(highlights), 1) < 0.5:
        warnings.append("Low PDF match rate: some relevant sentences were ranked but could not be placed on the PDF layout.")
    if not highlights:
        warnings.append("No highlight candidates passed the current threshold. Try Expanded mode or use a broader profile.")

    result = {
        "analysis_id": analysis_id,
        "profile_id": profile_id,
        "profile_name": profile.get("profile_name", "Saved profile"),
        "paper_title": pdf_path.name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_name": get_model_name(),
        "device": get_device(),
        "pdf_url": f"/files/analyses/{analysis_id}/{pdf_path.name}",
        "profile_match": profile_match,
        "analysis_mode_used": analysis_mode_used,
        "llm_evidence": llm_evidence,
        "summary": {
            "mode": "Profile match",
            "density": density,
            "visible_highlights": len(highlights),
            "pdf_matches": pdf_matches,
            "top_sections": top_sections[:5],
            "warnings": warnings,
            "mode_config": {
                "threshold": cfg["threshold"],
                "top_k": cfg["top_k"],
                "expanded_context": cfg["expanded"],
            },
            "filters": [
                "references_removed",
                "table_like_blocks_removed",
                "duplicate_candidates_removed",
                "overlapping_rectangles_removed",
                "adaptive_rescue_enabled",
                "dynamic_explanations_enabled",
            ],
        },
        "highlights": highlights,
    }

    result["llm_note"] = generate_llm_note(profile, result)

    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
