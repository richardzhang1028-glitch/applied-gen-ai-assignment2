# Assignment 1: Coding Environment Setup and Simple API Implementation

This project adds spaCy word embeddings to a FastAPI service, following the Module 3 API and Docker class activity.

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Run with Docker

```bash
docker build -t word-embedding-api .
docker run --rm -p 8000:8000 word-embedding-api
```

## API

- `GET /health` returns `{"status":"ok"}`.
- `POST /embedding` accepts a JSON body such as `{"word":"computer"}` and returns the word's vector, its dimension count, and the spaCy model used.

Example response (the vector is shortened here):

```json
{
  "word": "computer",
  "embedding": [0.1, -0.2, 0.3],
  "dimensions": 300,
  "model": "en_core_web_md"
}
```

The actual embedding contains all 300 numeric components. If the model is missing, the API returns HTTP 503 with setup instructions. If spaCy has no nonzero vector for the submitted word, it returns HTTP 404.

## Implementation notes

The model is loaded once and cached for reuse. The vector comes from the spaCy `Token.vector` property. The `/embedding` endpoint accepts one query word per request. This interface is a reasonable interpretation of the assignment's instruction to choose the API parameters.

## GitHub submission

Create a GitHub repository, then from this directory run:

```bash
git init
git add main.py requirements.txt Dockerfile .dockerignore .gitignore README.md
git commit -m "Add spaCy word embedding API"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

Replace `<your-repository-url>` with the URL of your own GitHub repository.
