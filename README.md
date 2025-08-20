````markdown
# AI Fake News Detector

A tiny FastAPI app that scores pasted articles as **real news**, **fake news**, **misleading**, or **uncertain** using a zero-shot NLI model from Hugging Face. It includes publisher-agnostic boilerplate stripping, article-window selection, chunked ensembling, a document-type guard, optional priors (bylines/timestamps & attribution verbs), quick token-level explanations, and a SQLite feedback store.

> ⚠️ Demo only — don’t rely on it as your sole source of truth.

---

## Features

- **Zero-shot classification** with Hugging Face (default: `facebook/bart-large-mnli`; recommended: `roberta-large-mnli`).
- **Publisher-agnostic cleanup** of nav/ads/footer text.
- **Article windowing** to keep the most “news-like” slice.
- **Chunk + ensemble** scoring of overlapping windows.
- **Doc-type guard** to confidently reject non-news (opinion/blog/ads/etc.).
- **Optional priors**:
  - *Metadata prior*: byline + timestamp → gentle nudge toward “real news”.
  - *Reporting prior*: attribution verbs density → gentle nudge toward “real news”.
  - *Newsiness penalty*: downweights “misleading” if the doc strongly looks like news.
- **Explanations**: quick occlusion saliency for influential tokens.
- **SQLite feedback** store and **per-IP rate limiting**.

---

## Quickstart

### 1) Clone & enter the project

```bash
git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
cd <REPO_NAME>
````

### 2) Create & activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -U fastapi uvicorn[standard] transformers torch pydantic
```

> On Apple Silicon or if PyTorch wheels fail:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 4) (Optional) Add a `.env` to tune behavior

```bash
cat > .env <<'EOF'
MODEL_ID=roberta-large-mnli

# Labels shown in the UI (classifier uses all except "uncertain")
LABELS=real news, fake news, misleading, uncertain

# Decision thresholds
MIN_CONFIDENCE=0.55
MARGIN_UNCERTAIN=0.08
UNCERTAIN_SCALE=0.6

# Doc-type guard & ensembling
DOC_TYPE_ON=true
USE_DT_FOR_WEIGHT=true
CHUNK_WORDS=220
CHUNK_STRIDE=180
ENSEMBLE_TOP_K=6

# Boilerplate stripping
BOILERPLATE_ON=true
BOILERPLATE_MIN_KEEP_RATIO=0.20
BOILERPLATE_MIN_KEEP_WORDS=80

# Priors
METADATA_PRIOR_ON=true
METADATA_PRIOR_BOOST=1.08

REPORTING_PRIOR_ON=true
REPORTING_PRIOR_BOOST=1.12

MISLEADING_NEWSINESS_PENALTY=0.75
MISLEADING_NEWSINESS_THRESH=0.60

# Label phrasing / hypotheses
USE_NATURAL_LABELS=false
HYPOTHESIS_TEMPLATES="This news report contains {}.,The reporting in this text is {}.,This article is {}."
EOF
```

Load it into your shell (or export variables manually):

```bash
export $(grep -v '^#' .env | xargs)
```

### 5) Run the server

```bash
uvicorn app:app --reload
```

Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Endpoints

### `GET /`

Minimal UI to paste text and view scores.

### `POST /analyze`

Analyze raw text.

**Request**

```json
{ "text": "Paste your article text here..." }
```

**Response**

```json
{
  "label": "real news",
  "scores": {
    "real news": 0.505,
    "fake news": 0.157,
    "misleading": 0.308,
    "uncertain": 0.031
  },
  "explanations": [
    {"token": "minister", "delta": 0.012},
    {"token": "said", "delta": 0.010}
  ]
}
```

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Your article text here..."}' | jq
```

### `POST /feedback`

Store thumbs-up/down and optional corrections/notes to SQLite.

**Request**

```json
{
  "text": "...",
  "predicted_label": "real news",
  "scores": { "real news": 0.5, "fake news": 0.1, "misleading": 0.3, "uncertain": 0.1 },
  "thumbs_up": true,
  "correct_label": null,
  "notes": null
}
```

### `GET /health`

Returns current config (from env) and model info.

### `GET /favicon.ico`

204.

---

## How it works (high level)

1. **Clean & strip boilerplate** (publisher-agnostic).
2. **Select article window** — densest “news-like” slice.
3. **Chunk & ensemble** — zero-shot over overlapping chunks, then weighted average.
4. **Doc-type guard** — if confidently not “news report”, override label.
5. **Apply priors** — byline/timestamp & attribution verbs; penalize “misleading” if doc looks like news.
6. **Uncertainty bar** — derived from top prob vs threshold/gap; dampened when top label is “real news” and doc looks like news.
7. **Explanations** — occlusion saliency over the best chunk.

---

## Configuration (environment variables)

| Variable                   |                                       Default | Description                                             |
| -------------------------- | --------------------------------------------: | ------------------------------------------------------- |
| `MODEL_ID`                 |                    `facebook/bart-large-mnli` | HF model id (try `roberta-large-mnli` for crisper NLI). |
| `LABELS`                   | `real news, fake news, misleading, uncertain` | UI labels; “uncertain” is display-only.                 |
| `USE_NATURAL_LABELS`       |                                  `false/true` | Map labels to phrasing; often better **off** for MNLI.  |
| `HYPOTHESIS_TEMPLATES`     |                                    see `.env` | Templates for zero-shot hypotheses.                     |
| `MIN_CONFIDENCE`           |                                        `0.60` | Threshold for avoiding “uncertain”.                     |
| `MARGIN_UNCERTAIN`         |                                        `0.12` | If top–second < margin and top < MIN → “uncertain”.     |
| `UNCERTAIN_SCALE`          |                                         `1.0` | Scales how shortfall contributes to “uncertain”.        |
| `DOC_TYPE_ON`              |                                        `true` | Enable doc-type guard.                                  |
| `USE_DT_FOR_WEIGHT`        |                                  `false/true` | Weight chunk votes by “news report” probability.        |
| `CHUNK_WORDS`              |                                         `200` | Words per chunk.                                        |
| `CHUNK_STRIDE`             |                                         `170` | Overlap stride.                                         |
| `ENSEMBLE_TOP_K`           |                                           `4` | Number of top “newsiest” chunks to ensemble.            |
| `BOILERPLATE_*`            |                                    see `.env` | Aggressiveness & fallbacks for stripping.               |
| `METADATA_PRIOR_ON/BOOST`  |                                 `true / 1.08` | Boost “real news” when byline + timestamp present.      |
| `REPORTING_PRIOR_ON/BOOST` |                                 `true / 1.12` | Boost “real news” with attribution-verb density.        |
| `MISLEADING_NEWSINESS_*`   |                                 `0.75 / 0.60` | Downweight “misleading” if doc looks like news.         |
| `RATE_LIMIT_PER_MIN`       |                                          `20` | Requests per IP per minute.                             |

---

## Project layout

```
.
├─ app.py                 # FastAPI app with all logic
├─ data/                  # SQLite DB lives here (gitignored, with .gitkeep)
├─ .gitignore
└─ README.md
```

---

## Dev tips

* Suppress tokenizer parallelism warnings:

  ```bash
  export TOKENIZERS_PARALLELISM=false
  ```
* A one-off `resource_tracker` semaphore warning at exit can occur in dev; keeping `transformers`/`torch` current usually helps.

---

## License

MIT (or add your preferred license).

## Acknowledgements

* [FastAPI](https://fastapi.tiangolo.com/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* [PyTorch](https://pytorch.org/)

```
```
