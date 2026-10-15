# Assignment 1: Simple API and spaCy Word Embeddings

This project follows the Module 3 activity's FastAPI structure and adds a spaCy word-embedding endpoint.

## Run locally

Requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Run with Docker

```bash
docker build -t applied-gen-ai-api .
docker run --rm -p 8000:8000 applied-gen-ai-api
```

## Endpoints

- `GET /` returns the greeting shown in the class activity.
- `GET /health` returns `{"status":"ok"}`.
- `POST /generate` accepts `{"start_word":"we","length":10}` and returns generated text. The length includes the starting word and is capped at 100.
- `POST /embedding` accepts `{"word":"computer"}` and returns the spaCy vector, its dimension count, and model name. The `en_core_web_md` model returns 300 dimensions.

Example embedding response (shortened):

```json
{
  "word": "computer",
  "embedding": [0.1, -0.2, 0.3],
  "dimensions": 300,
  "model": "en_core_web_md"
}
```

## How this relates to the class activity

The activity PDF gives the `BigramModel` interface and sample corpus, but asks students to bring the actual bigram logic over from lecture code. This repository includes a small word-transition implementation in `app/bigram_model.py` so `/generate` can run. If the course provided a specific `BigramModel` implementation, use that version instead.

The assignment PDF describes Docker as optional, while the CourseWorks rubric also lists Docker deployment as a criterion. This project includes a Dockerfile for that criterion.

## GitHub repository

https://github.com/richardzhang1028-glitch/applied-gen-ai-assignment1
