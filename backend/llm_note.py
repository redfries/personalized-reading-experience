import json
import os
import re
from typing import Any


def env_true(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def short_text(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def disabled_note(reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "disabled",
        "title": "AI relevance note",
        "overall_match": "Not generated",
        "summary": reason,
        "why_relevant": [],
        "weak_or_missing_connections": [],
        "reading_advice": "The app is using deterministic profile matching, embeddings, adaptive rescue, and PDF highlights only.",
        "sections_to_focus": [],
        "bridge_points": [],
        "caution": "No Gemini note was generated.",
        "provider": "gemini",
    }


def error_note(message: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": "error",
        "title": "AI relevance note",
        "overall_match": "Not generated",
        "summary": "Gemini note could not be generated.",
        "why_relevant": [],
        "weak_or_missing_connections": [],
        "reading_advice": "The normal profile matching and PDF highlights still work.",
        "sections_to_focus": [],
        "bridge_points": [],
        "caution": short_text(message, 500),
        "provider": "gemini",
    }


def safe_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None

    return None


def normalize_note(parsed: dict[str, Any], provider: str, model_name: str) -> dict[str, Any]:
    why_relevant = parsed.get("why_relevant", [])
    if not isinstance(why_relevant, list):
        why_relevant = []

    normalized_why = []
    for item in why_relevant[:5]:
        if isinstance(item, dict):
            normalized_why.append({
                "point": short_text(str(item.get("point", "")), 260),
                "pdf_evidence": short_text(str(item.get("pdf_evidence", "")), 600),
                "section": short_text(str(item.get("section", "")), 80),
            })
        elif str(item).strip():
            normalized_why.append({
                "point": short_text(str(item), 260),
                "pdf_evidence": "",
                "section": "",
            })

    sections = parsed.get("sections_to_focus", [])
    if not isinstance(sections, list):
        sections = []

    normalized_sections = []
    for item in sections[:5]:
        if isinstance(item, dict):
            normalized_sections.append({
                "section": short_text(str(item.get("section", "")), 80),
                "reason": short_text(str(item.get("reason", "")), 280),
            })
        elif str(item).strip():
            normalized_sections.append({
                "section": short_text(str(item), 80),
                "reason": "",
            })

    weak_missing = parsed.get("weak_or_missing_connections", [])
    if not isinstance(weak_missing, list):
        weak_missing = []

    bridge_points = parsed.get("bridge_points", [])
    if not isinstance(bridge_points, list):
        bridge_points = []

    return {
        "enabled": True,
        "status": "ok",
        "title": short_text(str(parsed.get("title", "AI relevance note")), 80),
        "overall_match": short_text(str(parsed.get("overall_match", "Not specified")), 80),
        "summary": short_text(str(parsed.get("summary", "")), 1200),
        "why_relevant": normalized_why,
        "weak_or_missing_connections": [
            short_text(str(x), 300) for x in weak_missing if str(x).strip()
        ][:5],
        "reading_advice": short_text(str(parsed.get("reading_advice", "")), 700),
        "sections_to_focus": normalized_sections,
        "bridge_points": [
            short_text(str(x), 260) for x in bridge_points if str(x).strip()
        ][:5],
        "caution": short_text(str(parsed.get("caution", "")), 600),
        "provider": provider,
        "model": model_name,
    }


def build_prompt(profile: dict[str, Any], analysis: dict[str, Any]) -> str:
    profile_match = analysis.get("profile_match", {})
    summary = analysis.get("summary", {})
    llm_evidence = analysis.get("llm_evidence", {})

    compact_analysis = {
        "paper_title": analysis.get("paper_title", ""),
        "profile_match": profile_match,
        "analysis_mode_used": analysis.get("analysis_mode_used", ""),
        "requested_density": summary.get("density", ""),
        "visible_highlights": summary.get("visible_highlights", 0),
        "pdf_matches": summary.get("pdf_matches", 0),
        "top_sections": summary.get("top_sections", []),
        "warnings": summary.get("warnings", []),
    }

    user_profile = {
        "profile_name": profile.get("profile_name", ""),
        "selected_topics": profile.get("selected_topics", []),
        "keywords": profile.get("keywords", []),
        "sources_used": profile.get("sources_used", []),
        "research_statement": short_text(profile.get("research_statement", ""), 800),
        "combined_profile_text": short_text(profile.get("combined_profile_text", ""), 1600),
    }

    target_pdf_evidence = {
        "paper_title": llm_evidence.get("paper_title", analysis.get("paper_title", "")),
        "abstract": short_text(llm_evidence.get("abstract", ""), 1600),
        "section_samples": llm_evidence.get("section_samples", [])[:8],
        "top_embedding_candidates": llm_evidence.get("top_embedding_candidates", [])[:12],
        "highlight_evidence": llm_evidence.get("highlight_evidence", [])[:10],
    }

    evidence = {
        "user_profile": user_profile,
        "target_pdf_extracted_evidence": target_pdf_evidence,
        "current_embedding_based_analysis": compact_analysis,
    }

    return (
        "You are an academic reading assistant.\n\n"
        "Your task is to compare a user's research profile with a target PDF using ONLY the evidence provided below.\n\n"
        "Important rules:\n"
        "1. Use only the provided profile data and extracted PDF text.\n"
        "2. Do not use outside knowledge.\n"
        "3. Do not invent paper claims, methods, datasets, results, or limitations.\n"
        "4. If the target paper is weakly related to the profile, clearly say so.\n"
        "5. If the connection is weak, describe possible bridge points instead of pretending the paper is highly relevant.\n"
        "6. When you mention a reason, tie it to an extracted sentence, paragraph, section, or highlight from the provided PDF evidence.\n"
        "7. If there is not enough evidence, say that there is not enough evidence.\n"
        "8. Keep the answer useful for a student deciding whether to read, skim, or skip the paper.\n"
        "9. The field pdf_evidence must contain exact text copied from the provided evidence JSON, not a paraphrase.\n\n"
        "Return JSON only with this exact structure:\n"
        "{\n"
        '  "title": "AI relevance note",\n'
        '  "overall_match": "Strong match | Moderate match | Weak match | Off-topic",\n'
        '  "summary": "A short 3-5 sentence explanation of how this PDF relates or does not relate to the user profile.",\n'
        '  "why_relevant": [\n'
        "    {\n"
        '      "point": "Specific reason this paper is relevant or possibly relevant.",\n'
        '      "pdf_evidence": "Exact sentence or short paragraph from the provided extracted PDF evidence.",\n'
        '      "section": "Section name if available"\n'
        "    }\n"
        "  ],\n"
        '  "weak_or_missing_connections": [\n'
        '    "Important profile interests that are missing, weak, or not clearly supported by the PDF evidence."\n'
        "  ],\n"
        '  "reading_advice": "Tell the user whether to read fully, skim specific sections, or skip.",\n'
        '  "sections_to_focus": [\n'
        "    {\n"
        '      "section": "Section name",\n'
        '      "reason": "Why this section matters for the user profile."\n'
        "    }\n"
        "  ],\n"
        '  "bridge_points": [\n'
        '    "Possible connection point if the paper is weakly related."\n'
        "  ],\n"
        '  "caution": "Short warning about uncertainty, weak evidence, or extraction limitations."\n'
        "}\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def generate_llm_note(profile: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if not env_true("ENABLE_LLM_NOTE", "false"):
        return disabled_note("AI relevance note is disabled. Set ENABLE_LLM_NOTE=true and GEMINI_API_KEY to enable it.")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return disabled_note("Gemini API key is missing. Add GEMINI_API_KEY to .env.local or export it before running the app.")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    try:
        from google import genai
    except Exception as exc:
        return error_note(f"google-genai is not installed or could not be imported: {exc}")

    try:
        client = genai.Client(api_key=api_key)
        prompt = build_prompt(profile, analysis)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        text = getattr(response, "text", "") or ""
        parsed = safe_parse_json(text)

        if parsed is None:
            return {
                "enabled": True,
                "status": "ok",
                "title": "AI relevance note",
                "overall_match": analysis.get("profile_match", {}).get("label", "Not specified"),
                "summary": short_text(text, 1000) if text else "Gemini returned an empty note.",
                "why_relevant": [],
                "weak_or_missing_connections": [],
                "reading_advice": "Use the PDF highlights and scores as the main grounded evidence.",
                "sections_to_focus": [],
                "bridge_points": [],
                "caution": "The note was returned as plain text instead of JSON.",
                "provider": "gemini",
                "model": model_name,
            }

        return normalize_note(parsed, provider="gemini", model_name=model_name)

    except Exception as exc:
        return error_note(str(exc))
