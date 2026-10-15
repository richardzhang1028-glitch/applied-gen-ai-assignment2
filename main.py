"""FastAPI service that returns spaCy word embeddings."""
from functools import lru_cache

import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Word Embedding API",
    description="Look up a word's vector using spaCy's en_core_web_md model.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def load_nlp():
    """Load the medium English model once per application process."""
    try:
        return spacy.load("en_core_web_md")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_md' is not installed. "
            "Install it with: python -m spacy download en_core_web_md"
        ) from exc


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, description="Word to embed")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = request.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="word must not be blank")

    try:
        nlp = load_nlp()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    token = nlp(word)[0]
    if not token.has_vector or token.vector_norm == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding is available for '{word}'. Try a common English word.",
        )

    return {
        "word": word,
        "embedding": token.vector.tolist(),
        "dimensions": int(token.vector.shape[0]),
        "model": "en_core_web_md",
    }
