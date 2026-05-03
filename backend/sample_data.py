SAMPLE_HIGHLIGHTS = [
    {
        "highlight_id": "h001",
        "rank": 1,
        "page": 2,
        "section": "Introduction",
        "tag": "Background",
        "scope": "Sentence",
        "label": "Relevant",
        "sentence_score": 0.5375,
        "paragraph_score": 0.5299,
        "matched_sources": ["Keywords", "Topics"],
        "explanation": "This highlight relates to your profile because it discusses researcher profile construction and content-based recommendation, which match your selected topics and keyword evidence.",
        "highlighted_sentence": "The CBF approach analyzes the content of a scientific paper in order to construct a researcher profile and recommend papers similar to the profile.",
        "paragraph_context": "Several scientific paper recommendation approaches have been proposed in the literature. The CBF approach analyzes the content of a scientific paper in order to construct a researcher profile and recommend papers similar to the profile.",
        "rects": [{"page": 2, "x": 13, "y": 28, "width": 68, "height": 8}]
    },
    {
        "highlight_id": "h002",
        "rank": 2,
        "page": 3,
        "section": "Related Work",
        "tag": "Method",
        "scope": "Paragraph",
        "label": "Relevant",
        "sentence_score": 0.5226,
        "paragraph_score": 0.5179,
        "matched_sources": ["Keywords", "Topics"],
        "explanation": "This highlight relates to your work because it describes user profile construction from academic signals, which is close to your profile keywords and selected recommendation topics.",
        "highlighted_sentence": "Kaya introduced a recommendation model that builds a user profile based on contextual data such as citations, publication year, and article keywords.",
        "paragraph_context": "Bhagavatula et al. proposed a model that embedded articles into a vector space by encoding textual content. Kaya introduced a recommendation model that builds a user profile based on contextual data such as citations, publication year, and article keywords.",
        "rects": [{"page": 3, "x": 12, "y": 50, "width": 70, "height": 15}]
    },
    {
        "highlight_id": "h003",
        "rank": 3,
        "page": 5,
        "section": "Methodology",
        "tag": "Method",
        "scope": "Sentence",
        "label": "Relevant",
        "sentence_score": 0.5087,
        "paragraph_score": 0.5028,
        "matched_sources": ["Seed papers", "Keywords"],
        "explanation": "This highlight matches the seed-paper evidence and keyword signal because it focuses on metadata, similarity, and academic-paper recommendation.",
        "highlighted_sentence": "The model computes similarities between a paper of interest and candidate papers based on public contextual metadata including title, keywords, and abstract.",
        "paragraph_context": "The CBR model computes similarities between a paper of interest and each qualified candidate paper based on public contextual metadata that includes title, keywords, and abstract.",
        "rects": [{"page": 5, "x": 14, "y": 34, "width": 66, "height": 7}]
    }
]

SAMPLE_ANALYSIS = {
    "analysis_id": "mock_analysis_001",
    "profile_name": "AI Reading Assistant Profile",
    "paper_title": "Sample academic paper",
    "summary": {
        "mode": "Profile match",
        "density": "Standard",
        "visible_highlights": 3,
        "pdf_matches": 3,
        "top_sections": ["Related Work", "Introduction", "Methodology"]
    },
    "highlights": SAMPLE_HIGHLIGHTS
}
