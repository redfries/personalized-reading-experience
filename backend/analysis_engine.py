import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .model_service import encode_texts, get_model_name, get_device
from .pdf_utils import extract_paragraph_records, paragraph_rect, search_sentence_rects


def cosine_scores(profile_embedding: np.ndarray, sentence_embeddings: torch.Tensor) -> np.ndarray:
    profile = torch.tensor(profile_embedding, dtype=torch.float32, device=sentence_embeddings.device).unsqueeze(0)
    sentence_embeddings = sentence_embeddings.float()
    scores = torch.mm(profile, sentence_embeddings.T)[0].detach().cpu().numpy()
    return scores.astype("float32")


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


def explanation_from_profile(profile: dict, section: str, tag: str) -> tuple[list[str], str]:
    sources = profile.get("sources_used", [])
    topics = profile.get("selected_topics", [])
    keywords = profile.get("keywords", [])

    parts = []
    if "Keywords" in sources and keywords:
        parts.append("it is close to your keyword signal (" + ", ".join(keywords[:5]) + ")")
    if "Topics" in sources and topics:
        parts.append("it aligns with your selected topics (" + ", ".join(topics[:4]) + ")")
    if "Seed papers" in sources:
        parts.append("it is semantically related to evidence extracted from your seed papers")
    if "Research statement" in sources:
        parts.append("it is related to your written research statement")

    if not parts:
        parts.append("it is semantically close to the selected research profile")

    sentence = "This highlight relates to your work because " + ", and ".join(parts) + "."
    if section and section != "Unknown":
        sentence += f" It appears in the {section} section"
        if tag:
            sentence += f" as {tag.lower()} content"
        sentence += "."

    return sources, sentence


def density_config(density: str) -> dict:
    density = (density or "Standard").lower()
    if density == "concise":
        return {"top_k": 8, "threshold": 0.35, "expanded": False}
    if density == "expanded":
        return {"top_k": 30, "threshold": 0.20, "expanded": True}
    return {"top_k": 15, "threshold": 0.28, "expanded": False}


def paragraph_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    ordered = sorted(scores, reverse=True)
    best = ordered[0]
    top2_avg = sum(ordered[:2]) / min(len(ordered), 2)
    return 0.7 * best + 0.3 * top2_avg


def analyze_pdf_with_profile(pdf_path: Path, profile: dict, profiles_dir: Path, analysis_dir: Path, density: str = "Standard") -> dict:
    profile_id = profile["profile_id"]
    profile_dir = profiles_dir / profile_id
    embedding_path = profile_dir / "profile_embedding.npy"

    if not embedding_path.exists():
        raise FileNotFoundError("Saved profile embedding not found. Re-save the profile in v4.2.1.")

    profile_embedding = np.load(embedding_path).astype("float32")

    records = extract_paragraph_records(pdf_path)
    if not records:
        raise ValueError("No usable body paragraphs were extracted from this PDF.")

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

    grouped = {}
    for (record, sentence), score in zip(sentence_rows, scores):
        pid = record["paragraph_id"]
        grouped.setdefault(pid, {"record": record, "items": []})
        grouped[pid]["items"].append({"sentence": sentence, "score": float(score)})

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
        })

    cfg = density_config(density)
    candidates = [
        item for item in candidates
        if item["sentence_score"] >= cfg["threshold"] or item["paragraph_score"] >= cfg["threshold"]
    ]
    candidates = sorted(candidates, key=lambda x: (x["paragraph_score"], x["sentence_score"]), reverse=True)
    selected = candidates[:cfg["top_k"]]

    matched_sources, _ = explanation_from_profile(profile, "", "")

    highlights = []
    for idx, item in enumerate(selected, start=1):
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

        _, explanation = explanation_from_profile(profile, record["section"], tag)

        highlights.append({
            "highlight_id": f"h{idx:03d}",
            "rank": idx,
            "page": record["page"],
            "section": record["section"],
            "tag": tag,
            "scope": scope,
            "label": "Relevant",
            "sentence_score": round(float(item["sentence_score"]), 4),
            "paragraph_score": round(float(item["paragraph_score"]), 4),
            "matched_sources": matched_sources,
            "explanation": explanation,
            "highlighted_sentence": item["best_sentence"],
            "paragraph_context": record["paragraph_text"],
            "rects": rects,
        })

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

    result = {
        "analysis_id": analysis_id,
        "profile_id": profile_id,
        "profile_name": profile.get("profile_name", "Saved profile"),
        "paper_title": pdf_path.name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_name": get_model_name(),
        "device": get_device(),
        "pdf_url": f"/files/analyses/{analysis_id}/{pdf_path.name}",
        "summary": {
            "mode": "Profile match",
            "density": density,
            "visible_highlights": len(highlights),
            "pdf_matches": sum(1 for h in highlights if h["rects"]),
            "top_sections": top_sections[:5],
        },
        "highlights": highlights,
    }

    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
