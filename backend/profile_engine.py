import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import UploadFile

from .model_service import encode_texts, get_model_name, get_device
from .pdf_utils import extract_seed_text


def parse_keywords(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,\n;]+", text or "") if item.strip()]


def get_profile_strength(source_count: int) -> str:
    if source_count <= 0:
        return "Incomplete"
    if source_count == 1:
        return "Weak"
    if source_count == 2:
        return "Good"
    if source_count == 3:
        return "Strong"
    return "Very strong"


async def save_uploads(files: list[UploadFile], target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for upload in (files or [])[:3]:
        if not upload.filename:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", upload.filename)
        path = target_dir / safe_name
        content = await upload.read()
        path.write_bytes(content)
        paths.append(path)

    return paths


def build_source_items(profile_name: str, selected_topics: list[str], keyword_text: str, research_statement: str, seed_paper_texts: list[dict]) -> list[dict]:
    keyword_list = parse_keywords(keyword_text)
    items = []

    if selected_topics:
        items.append({
            "source_type": "topics",
            "source_label": ", ".join(selected_topics),
            "text": "The user is interested in research areas including " + ", ".join(selected_topics) + ".",
            "weight": 1.0,
        })

    if keyword_list:
        items.append({
            "source_type": "keywords",
            "source_label": ", ".join(keyword_list),
            "text": "The user provided keyword signals including " + ", ".join(keyword_list) + ".",
            "weight": 1.0,
        })

    if research_statement.strip():
        items.append({
            "source_type": "research_statement",
            "source_label": "Research statement",
            "text": research_statement.strip(),
            "weight": 1.25,
        })

    if seed_paper_texts:
        total_seed_weight = 2.0
        per_seed_weight = total_seed_weight / max(len(seed_paper_texts), 1)
        for seed in seed_paper_texts:
            evidence = " ".join(seed["sentences"][:70])
            if evidence.strip():
                items.append({
                    "source_type": "seed_paper",
                    "source_label": seed["filename"],
                    "text": evidence,
                    "weight": per_seed_weight,
                })

    return items


def build_combined_profile_text(source_items: list[dict]) -> str:
    seed_items = [item for item in source_items if item["source_type"] == "seed_paper"]
    other_items = [item for item in source_items if item["source_type"] != "seed_paper"]
    ordered = seed_items + other_items
    return "\\n\\n".join(item["text"] for item in ordered if item["text"].strip())


def weighted_average_embeddings(source_items: list[dict]):
    texts = [item["text"] for item in source_items]
    weights = np.array([float(item.get("weight", 1.0)) for item in source_items], dtype="float32")

    embeddings = encode_texts(texts, batch_size=8).detach().cpu().numpy().astype("float32")
    weighted = embeddings * weights[:, None]
    profile_embedding = weighted.sum(axis=0) / max(weights.sum(), 1e-6)
    norm = np.linalg.norm(profile_embedding)
    if norm > 0:
        profile_embedding = profile_embedding / norm

    return profile_embedding.astype("float32"), embeddings.astype("float32")


def make_preview(profile_name: str, selected_topics: list[str], keywords: str, research_statement: str, seed_filenames: list[str], seed_sentence_counts: list[int] | None = None) -> dict:
    keyword_list = parse_keywords(keywords)
    sources_used = []
    if selected_topics:
        sources_used.append("Topics")
    if keyword_list:
        sources_used.append("Keywords")
    if research_statement.strip():
        sources_used.append("Research statement")
    if seed_filenames:
        sources_used.append("Seed papers")

    parts = []
    if seed_filenames:
        if seed_sentence_counts:
            paper_bits = [f"{name} ({count} extracted sentences)" for name, count in zip(seed_filenames, seed_sentence_counts)]
        else:
            paper_bits = seed_filenames
        parts.append("Seed paper evidence was extracted from " + ", ".join(paper_bits) + ".")

    if selected_topics:
        parts.append("The user selected research areas related to " + ", ".join(selected_topics) + ".")

    if keyword_list:
        parts.append("The user provided keyword signals including " + ", ".join(keyword_list) + ".")

    if research_statement.strip():
        statement = research_statement.strip()
        if len(statement) > 500:
            statement = statement[:500].rstrip() + "..."
        parts.append("The user research statement says: " + statement)

    source_total = len(sources_used)
    return {
        "profile_name": profile_name.strip() or "Untitled profile",
        "sources_used": sources_used,
        "source_count": source_total,
        "profile_strength": get_profile_strength(source_total),
        "combined_profile_text": " ".join(parts) if parts else "No profile sources were provided yet.",
        "selected_topics": selected_topics,
        "keywords": keyword_list,
        "research_statement": research_statement.strip(),
        "seed_paper_names": seed_filenames,
    }


async def preview_profile(profile_name: str, selected_topics: list[str], keywords: str, research_statement: str, seed_papers: list[UploadFile], temp_dir: Path) -> dict:
    paths = await save_uploads(seed_papers, temp_dir)
    counts = []
    for path in paths:
        try:
            counts.append(len(extract_seed_text(path, max_sentences=70)))
        except Exception:
            counts.append(0)
    return make_preview(profile_name, selected_topics, keywords, research_statement, [p.name for p in paths], counts)


async def save_profile(profile_name: str, selected_topics: list[str], keywords: str, research_statement: str, seed_papers: list[UploadFile], profiles_dir: Path) -> dict:
    profile_id = "profile_" + uuid.uuid4().hex[:10]
    profile_dir = profiles_dir / profile_id
    seed_dir = profile_dir / "seed_papers"
    profile_dir.mkdir(parents=True, exist_ok=True)

    paths = await save_uploads(seed_papers, seed_dir)

    seed_paper_texts = []
    for path in paths:
        try:
            sentences = extract_seed_text(path, max_sentences=70)
        except Exception:
            sentences = []
        if sentences:
            seed_paper_texts.append({
                "filename": path.name,
                "path": str(path),
                "sentences": sentences,
            })

    source_items = build_source_items(profile_name, selected_topics, keywords, research_statement, seed_paper_texts)

    if len(source_items) < 2:
        preview = make_preview(
            profile_name,
            selected_topics,
            keywords,
            research_statement,
            [p.name for p in paths],
            [len(item["sentences"]) for item in seed_paper_texts],
        )
        return {
            "ok": False,
            "message": "Please provide at least two usable profile sources before saving.",
            "profile": preview,
        }

    combined_text = build_combined_profile_text(source_items)
    profile_embedding, source_embeddings = weighted_average_embeddings(source_items)

    np.save(profile_dir / "profile_embedding.npy", profile_embedding)
    np.savez(profile_dir / "source_embeddings.npz", embeddings=source_embeddings)

    preview = make_preview(
        profile_name,
        selected_topics,
        keywords,
        research_statement,
        [p.name for p in paths],
        [len(item["sentences"]) for item in seed_paper_texts],
    )

    profile = {
        "profile_id": profile_id,
        "schema_version": "v4.2.1_real_profile",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_name": get_model_name(),
        "device": get_device(),
        **preview,
        "combined_profile_text": combined_text,
        "source_items": source_items,
        "profile_dir": str(profile_dir),
    }

    (profile_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    return {"ok": True, "message": "Real profile saved with embeddings.", "profile": profile}


def load_profile(profile_id: str, profiles_dir: Path) -> dict:
    path = profiles_dir / profile_id / "profile.json"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_profiles(profiles_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(profiles_dir.glob("profile_*/profile.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "profile_id": data["profile_id"],
                "profile_name": data["profile_name"],
                "profile_strength": data.get("profile_strength", "Unknown"),
                "sources_used": data.get("sources_used", []),
                "created_at": data.get("created_at", ""),
                "model_name": data.get("model_name", ""),
            })
        except Exception:
            continue
    return rows
