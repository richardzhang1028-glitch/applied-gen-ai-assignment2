"""FastAPI service for bigram text generation and spaCy word embeddings."""

from functools import lru_cache

import spacy
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.cnn_model import AssignmentCNN

app = FastAPI(
    title="Applied Generative AI API",
    description="Generate text with a bigram model and retrieve spaCy word embeddings.",
    version="1.0.0",
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "cifar10_cnn.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

classifier = AssignmentCNN().to(device)
classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.eval()

class_names = checkpoint["classes"]

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

CORPUS = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(CORPUS)


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


class TextGenerationRequest(BaseModel):
    start_word: str = Field(..., min_length=1, description="First word to generate from")
    length: int = Field(..., ge=1, le=100, description="Maximum number of words to return")


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, description="Word to embed")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = request.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="word must not be blank")

    try:
        nlp = load_nlp()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    doc = nlp(word)
    if len(doc) != 1:
        raise HTTPException(status_code=422, detail="Please provide one word per request")

    token = doc[0]
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
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        ) from exc

    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = classifier(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = int(probabilities.argmax())

    return {
        "class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
    }
