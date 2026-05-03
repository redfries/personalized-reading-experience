import os
import torch
from sentence_transformers import SentenceTransformer

_MODEL = None
_MODEL_NAME = None
_DEVICE = None


def get_device() -> str:
    requested = os.environ.get("EMBEDDING_DEVICE", "cuda").strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model_name() -> str:
    return os.environ.get(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-mpnet-base-v2",
    ).strip()


def get_model() -> SentenceTransformer:
    global _MODEL, _MODEL_NAME, _DEVICE

    model_name = get_model_name()
    device = get_device()

    if _MODEL is not None and _MODEL_NAME == model_name and _DEVICE == device:
        return _MODEL

    print(f"Loading embedding model: {model_name} on {device}")
    _MODEL = SentenceTransformer(model_name, device=device)
    _MODEL_NAME = model_name
    _DEVICE = device
    print("Embedding model loaded.")
    return _MODEL


def encode_texts(texts, batch_size: int = 16):
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return embeddings.float()


def encode_single(text: str):
    return encode_texts([text], batch_size=1)[0]
