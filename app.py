import os
import re
import time
import json
import sqlite3
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

# -------------------
# Config via env vars
# -------------------
# Use a crisper NLI backbone by default
MODEL_ID = os.getenv("MODEL_ID", "roberta-large-mnli")

# UI labels (order controls bar order). Keep "uncertain" here for display,
# but we do NOT include it in classifier labels anymore.
LABELS_ENV = os.getenv("LABELS", "real news, fake news, misleading, uncertain")
DISPLAY_LABELS = [l.strip() for l in LABELS_ENV.split(",") if l.strip()]

# Internal phrasing map for zero-shot hypotheses.
# (Kept for optional natural phrasing; disabled by default below.)
DEFAULT_NATURAL_MAP = {
    "real news": "reliable factual reporting",
    "fake news": "fabricated false information",
    "misleading": "misleading or cherry-picked information",
}
# Default: use raw labels to avoid making “misleading” sound news-like
USE_NATURAL_LABELS = os.getenv("USE_NATURAL_LABELS", "false").lower() in {"1", "true", "yes"}

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "18000"))
EXPLAIN_MAX_TOKENS = int(os.getenv("EXPLAIN_MAX_TOKENS", "40"))

# Decision thresholds
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.60"))
MARGIN_UNCERTAIN = float(os.getenv("MARGIN_UNCERTAIN", "0.12"))
UNCERTAIN_SCALE = float(os.getenv("UNCERTAIN_SCALE", "1.0"))

# Hypothesis templates (ensembled) — trimmed to reduce generic entailments
HT_ENV = os.getenv(
    "HYPOTHESIS_TEMPLATES",
    "This news report contains {}.,The reporting in this text is {}.,This article is {}.",
)
HYPOTHESIS_TEMPLATES = [t.strip() for t in HT_ENV.split(",") if t.strip()]

# Doc-type guard (used to confidently reject non-news and/or weight chunks)
DOC_TYPE_ON = os.getenv("DOC_TYPE_ON", "true").lower() in {"1", "true", "yes"}
DOC_TYPE_LABELS_ENV = os.getenv(
    "DOC_TYPE_LABELS",
    "news report, reference page, opinion piece, blog post, press release, advertisement",
)
DOC_TYPE_LABELS = [l.strip() for l in DOC_TYPE_LABELS_ENV.split(",") if l.strip()]
DOC_GUARD_STRICTNESS = float(os.getenv("DOC_GUARD_STRICTNESS", "0.85"))
USE_DT_FOR_WEIGHT = os.getenv("USE_DT_FOR_WEIGHT", "false").lower() in {"1", "true", "yes"}

# Optional metadata prior: small nudge to "real news" if byline + timestamp patterns exist
METADATA_PRIOR_ON = os.getenv("METADATA_PRIOR_ON", "true").lower() in {"1","true","yes"}
METADATA_PRIOR_BOOST = float(os.getenv("METADATA_PRIOR_BOOST", "1.08"))

# Reporting-style prior (attribution verbs density -> boost "real news")
REPORTING_PRIOR_ON = os.getenv("REPORTING_PRIOR_ON", "true").lower() in {"1","true","yes"}
REPORTING_PRIOR_BOOST = float(os.getenv("REPORTING_PRIOR_BOOST", "1.12"))

# Penalize "misleading" when overall looks like news
MISLEADING_NEWSINESS_PENALTY = float(os.getenv("MISLEADING_NEWSINESS_PENALTY", "0.75"))  # <1.0 = downweight
MISLEADING_NEWSINESS_THRESH = float(os.getenv("MISLEADING_NEWSINESS_THRESH", "0.60"))

# Chunking/ensembling params
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "220"))
CHUNK_STRIDE = int(os.getenv("CHUNK_STRIDE", "180"))
ENSEMBLE_TOP_K = int(os.getenv("ENSEMBLE_TOP_K", "6"))

# Boilerplate stripping safety knobs
BOILERPLATE_ON = os.getenv("BOILERPLATE_ON", "true").lower() in {"1", "true", "yes"}
BOILERPLATE_MIN_KEEP_WORDS = int(os.getenv("BOILERPLATE_MIN_KEEP_WORDS", "80"))
BOILERPLATE_MIN_KEEP_RATIO = float(os.getenv("BOILERPLATE_MIN_KEEP_RATIO", "0.20"))

# -------------
# HF pipeline
# -------------
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    DEVICE = -1

from transformers import pipeline
classifier = pipeline("zero-shot-classification", model=MODEL_ID, device=DEVICE)

# -------------
# SQLite store
# -------------
os.makedirs("data", exist_ok=True)
DB_PATH = os.getenv("DB_PATH", "data/app.db")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT (datetime('now')),
        text TEXT,
        predicted_label TEXT,
        scores_json TEXT,
        thumbs_up INTEGER,
        correct_label TEXT,
        notes TEXT
    );
    """
)
conn.commit()

# --------------------------
# Tiny in-memory rate limit
# --------------------------
REQUEST_LOG: Dict[str, List[float]] = {}
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))

def rate_limit(request: Request):
    ip = request.client.host if request.client else "local"
    now = time.time()
    window = 60.0
    times = REQUEST_LOG.get(ip, [])
    times = [t for t in times if now - t < window]
    if len(times) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Too many requests, slow down.")
    times.append(now)
    REQUEST_LOG[ip] = times

# -------------
# Text helpers
# -------------
TAG_RE = re.compile(r"<[^>]+>")
MULTI_WS = re.compile(r"\s+")
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","to","of","in","on","for","by","with","as","at",
    "is","are","was","were","be","been","being","it","its","this","that","these","those","from","into","about","over",
    "under","after","before","above","below","between","within","without","through","during","per","via","also"
}

# Generic, publisher-agnostic boilerplate hints
BOILERPLATE_HINTS = [
    # nav + auth
    "privacy", "cookie", "preferences", "consent", "sign in", "log in", "subscribe", "menu", "search",
    # share / social
    "share", "follow us", "facebook", "twitter", "x ", "instagram", "youtube", "rss", "podcast",
    # content chrome
    "trending", "popular", "most read", "related", "recommended", "discover more", "read more",
    "next up", "up next", "watch", "video", "play", "live",
    # footer / corporate
    "footer", "contact", "about", "careers", "advertising", "ad choices", "adchoices", "terms of use",
    "terms and conditions", "privacy policy", "help centre", "help center", "sitemap", "©", "copyright",
    # misc site chrome
    "newsletter", "comments", "commenting", "sponsored", "advertorial", "promotion", "commercial services",
    # misc lists/boxes
    "discover more from", "related stories", "popular now", "recommended for you",
    # cleanup of common tails
    "about the author", "corrections and clarifications", "footer links", "trending videos"
]

# Classifier labels (exclude "uncertain"!)
CLASSIFY_LABELS = [lbl for lbl in DISPLAY_LABELS if lbl.lower() != "uncertain"]
if not CLASSIFY_LABELS:
    CLASSIFY_LABELS = ["real news", "fake news", "misleading"]

def clean_text(text: str) -> str:
    text = text.strip()
    text = TAG_RE.sub(" ", text)
    text = MULTI_WS.sub(" ", text)
    return text[:MAX_INPUT_CHARS]

def strip_boilerplate(text: str) -> str:
    """Publisher-agnostic removal of obvious UI/ads lines; falls back if it strips too much."""
    if not BOILERPLATE_ON:
        return text

    original = text
    orig_words = len(original.split())

    # Prefer real line breaks; else approximate using sentence boundaries.
    raw_lines = re.split(r"[\r\n]+", original)
    if len(raw_lines) < 4:
        raw_lines = re.split(r"(?<=[.!?])\s+(?=[A-Z])", original)

    lines = [ln.strip() for ln in raw_lines]
    keep: List[str] = []
    for ln in lines:
        if not ln or len(ln) <= 2:
            continue

        # Drop very short ALL-CAPS (button-like)
        if ln.isupper() and len(ln) <= 20:
            continue

        low = ln.lower()

        # Strong UI hints -> drop
        if any(h in low for h in BOILERPLATE_HINTS):
            continue

        # Solo "AD" / Mostly non-letters -> drop
        if re.fullmatch(r"(ad|ads?|advertisement|sponsored)\b.*", low):
            continue
        letters = sum(c.isalpha() for c in ln)
        if letters and (letters / max(1, len(ln))) < 0.35:
            continue

        keep.append(ln)

    cleaned = " ".join(keep).strip()
    cleaned = re.sub(r'\bloaded\s+ad\b', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\badvertisement\b|\bad\b', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = MULTI_WS.sub(" ", cleaned).strip()

    # Fallback: if we stripped too much, return the original (already de-tagged) text.
    kept_words = len(cleaned.split())
    min_keep = max(BOILERPLATE_MIN_KEEP_WORDS, int(orig_words * BOILERPLATE_MIN_KEEP_RATIO))
    if kept_words < min_keep:
        return MULTI_WS.sub(" ", original).strip()

    return cleaned

# ---------- pick the most article-like window (publisher-agnostic) ----------
def _window_score(txt: str) -> float:
    """Heuristic 'newsiness' score for a chunk."""
    if not txt:
        return 0.0
    words = txt.split()
    n = len(words)
    if n == 0:
        return 0.0

    punct = len(re.findall(r"[\.!?;:]", txt))
    quotes = txt.count('"') + txt.count('“') + txt.count('”')
    digits = len(re.findall(r"\d", txt))
    dateish = len(re.findall(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|monday|tuesday|wednesday|thursday|friday|saturday|sunday|updated|posted|published)\b",
        txt, re.I))
    alpha = sum(c.isalpha() for c in txt)
    lower_ratio = (sum(c.islower() for c in txt) / max(1, alpha)) if alpha else 0.0

    per_tok = lambda x: x / n
    score = (
        1.5 * per_tok(punct) +
        1.0 * per_tok(quotes) +
        0.6 * per_tok(digits) +
        0.8 * per_tok(dateish) +
        0.5 * lower_ratio
    )
    return float(score)

def select_main_window(text: str, window_words: int = 800, step_words: int = 200) -> str:
    """Slide a window; keep the densest, most article-like slice."""
    words = text.split()
    if len(words) <= window_words:
        return text
    best_score, best_slice = -1.0, (0, window_words)
    i = 0
    while i < len(words):
        j = min(i + window_words, len(words))
        chunk = " ".join(words[i:j])
        sc = _window_score(chunk)
        if sc > best_score:
            best_score, best_slice = sc, (i, j)
        if j == len(words):
            break
        i += step_words
    i, j = best_slice
    return " ".join(words[i:j])

# ---- label phrasing helpers ----
def build_label_phrases(labels: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return (phrases, phrase->display_label map)."""
    phrases = []
    backmap = {}
    for lbl in labels:
        phrase = DEFAULT_NATURAL_MAP.get(lbl, lbl) if USE_NATURAL_LABELS else lbl
        phrases.append(phrase)
        backmap[phrase] = lbl
    return phrases, backmap

# ---- zero-shot scoring with template and label-phrase ensembling ----
def classify_once(text: str, candidate_phrases: List[str], template: str) -> Dict[str, float]:
    result = classifier(
        text,
        candidate_labels=candidate_phrases,
        multi_label=False,
        hypothesis_template=template,
    )
    labels_out = result["labels"]
    scores_out = result["scores"]
    return {lbl: float(sc) for lbl, sc in zip(labels_out, scores_out)}

def normalize_prob_dict(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    return {k: (v / s) if s > 0 else 0.0 for k, v in d.items()}

def classify_with_ensembles(text: str, labels: List[str]) -> Dict[str, float]:
    phrases, backmap = build_label_phrases(labels)
    accum: Dict[str, float] = {lbl: 0.0 for lbl in labels}
    used = 0
    for tmpl in HYPOTHESIS_TEMPLATES:
        try:
            raw = classify_once(text, phrases, tmpl)
            used += 1
        except Exception:
            continue
        raw = normalize_prob_dict(raw)
        mapped = {}
        for ph, p in raw.items():
            mapped[backmap.get(ph, ph)] = p
        for k, v in mapped.items():
            accum[k] += v
    if used == 0:
        return {lbl: (1.0 / len(labels)) for lbl in labels}
    averaged = {k: v / used for k, v in accum.items()}
    return normalize_prob_dict(averaged)

def top_label(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]

# ---- doc-type check ----
def doc_type_scores(text: str) -> Dict[str, float]:
    result = classifier(
        text,
        candidate_labels=DOC_TYPE_LABELS,
        multi_label=False,
        hypothesis_template="The following text is a {}.",
    )
    out = {lbl: float(sc) for lbl, sc in zip(result["labels"], result["scores"])}
    return normalize_prob_dict(out)

def doc_type_check(text: str) -> Optional[Tuple[str, float]]:
    if not DOC_TYPE_ON:
        return None
    sc = doc_type_scores(text)
    lbl = top_label(sc)
    return (lbl, sc.get(lbl, 0.0))

# ---- chunk & ensemble ----
def split_into_chunks(text: str, max_words: int = CHUNK_WORDS, stride: int = CHUNK_STRIDE) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
        if i + max_words >= len(words):
            break
        i += stride
    return chunks

def ensemble_article_scores(text: str) -> Tuple[Dict[str, float], Optional[str], Optional[float], str]:
    """
    Returns (scores3, not_news_label, not_news_conf, best_chunk_for_explain)
    - scores3: averaged label scores over top chunks (only among CLASSIFY_LABELS)
    - not_news_label/conf: if we’re very confident it's not news
    - best_chunk_for_explain: chunk we’ll use for token highlights
    """
    chunks = split_into_chunks(text)
    if not chunks:
        return {lbl: 0.0 for lbl in CLASSIFY_LABELS}, None, None, text

    # score "newsiness" of each chunk
    chunk_info = []
    for ch in chunks:
        if DOC_TYPE_ON and USE_DT_FOR_WEIGHT:
            dt = doc_type_scores(ch)
            newsiness = dt.get("news report", 0.0)
        else:
            w = len(ch.split())
            newsiness = min(1.0, max(0.1, w / CHUNK_WORDS))
        chunk_info.append((ch, newsiness))

    # take top-K
    chunk_info.sort(key=lambda t: t[1], reverse=True)
    top = chunk_info[:max(1, ENSEMBLE_TOP_K)]

    # weighted average of label scores
    accum = {lbl: 0.0 for lbl in CLASSIFY_LABELS}
    weight_sum = 0.0
    for ch, w in top:
        sc = classify_with_ensembles(ch, CLASSIFY_LABELS)
        for k, v in sc.items():
            accum[k] += w * v
        weight_sum += w
    scores3 = {k: (v / weight_sum if weight_sum > 0 else 0.0) for k, v in accum.items()}
    scores3 = normalize_prob_dict(scores3)

    # not-news guard on the whole text
    not_news_label, not_news_conf = None, None
    if DOC_TYPE_ON:
        dt_lbl, dt_conf = doc_type_check(text) or (None, None)
        if dt_lbl and dt_lbl != "news report" and (dt_conf or 0.0) >= DOC_GUARD_STRICTNESS:
            not_news_label, not_news_conf = dt_lbl, dt_conf

    best_chunk = top[0][0]
    return scores3, not_news_label, not_news_conf, best_chunk

# ---- small prior on metadata patterns (byline + timestamp) ----
_BYLINE_NEWSROOM_SUFFIX = r"(News|Times|Post|Press|Daily|Herald|Telegraph|Journal|Gazette|Globe|Mail|Standard|Tribune|Record|Reporter)"
def apply_metadata_prior(text: str, scores3: Dict[str, float]) -> Dict[str, float]:
    if not METADATA_PRIOR_ON:
        return scores3

    # (1) classic "By Jane Doe"
    has_byline = re.search(r"\bBy\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text, re.I) is not None

    # (2) "Jane Doe · Foo News" or "Jane Doe, Foo News"
    if not has_byline:
        has_byline = re.search(
            rf"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){{0,3}}\s*[·,]\s*[A-Za-z][A-Za-z &\-]*\s+{_BYLINE_NEWSROOM_SUFFIX}\b",
            text, re.I
        ) is not None

    # (3) "Jane Doe — Foo News"
    if not has_byline:
        has_byline = re.search(
            rf"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){{0,3}}\s*[—-]\s*[A-Za-z][A-Za-z &\-]*\s+{_BYLINE_NEWSROOM_SUFFIX}\b",
            text, re.I
        ) is not None

    has_stamp = re.search(r"\b(published|posted|updated|last updated)\b", text, re.I) is not None

    if has_byline and has_stamp and "real news" in scores3:
        s = dict(scores3)
        s["real news"] *= METADATA_PRIOR_BOOST
        return normalize_prob_dict(s)
    return scores3

# ---- attribution/reporting prior (generic) ----
_ATTRIB_WORDS = r"(said|told|according to|stated|reported|reports|record[s]? show|court documents|witness|minister|spokesperson|police|officials?)"
def apply_reporting_prior(text: str, scores3: Dict[str, float]) -> Dict[str, float]:
    if not REPORTING_PRIOR_ON:
        return scores3
    tokens = max(1, len(text.split()))
    hits = len(re.findall(rf"\b{_ATTRIB_WORDS}\b", text, re.I))
    density = hits / tokens
    if hits >= 3 or density >= 0.006:
        s = dict(scores3)
        if "real news" in s:
            s["real news"] *= REPORTING_PRIOR_BOOST
        return normalize_prob_dict(s)
    return scores3

# ---- newsiness-based penalty to "misleading" ----
def apply_newsiness_penalty(scores3: Dict[str, float], newsiness: float) -> Dict[str, float]:
    if newsiness >= MISLEADING_NEWSINESS_THRESH and "misleading" in scores3 and MISLEADING_NEWSINESS_PENALTY < 1.0:
        s = dict(scores3)
        s["misleading"] *= MISLEADING_NEWSINESS_PENALTY
        return normalize_prob_dict(s)
    return scores3

# ---- derive uncertain & final label ----
def with_uncertain_bar(scores3: Dict[str, float], newsiness: float = 0.0) -> Dict[str, float]:
    items = sorted(scores3.items(), key=lambda kv: kv[1], reverse=True)
    best_lbl, best_val = items[0]
    second_val = items[1][1] if len(items) > 1 else 0.0
    gap = best_val - second_val

    shortfall = max(0.0, MIN_CONFIDENCE - best_val) / max(1e-6, MIN_CONFIDENCE)
    uncertain_raw = min(1.0, shortfall * UNCERTAIN_SCALE)

    if gap < MARGIN_UNCERTAIN:
        uncertain_raw = min(1.0, uncertain_raw + 0.10)

    # If it looks like news and top is real, dampen uncertainty a bit
    if best_lbl == "real news" and newsiness >= 0.65:
        uncertain_raw *= 0.5

    out = dict(scores3)
    out["uncertain"] = max(0.0, float(uncertain_raw))
    for lbl in DISPLAY_LABELS:
        out.setdefault(lbl, 0.0)
    total = sum(out[lbl] for lbl in DISPLAY_LABELS)
    if total > 0:
        out = {lbl: out[lbl] / total for lbl in DISPLAY_LABELS}
    else:
        out = {lbl: (1.0 / len(DISPLAY_LABELS)) for lbl in DISPLAY_LABELS}
    return out

def decide_label(core_scores: Dict[str, float], newsiness: float) -> str:
    items = sorted(core_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_lbl, best_val = items[0]
    second_val = items[1][1] if len(items) > 1 else 0.0
    gap = best_val - second_val

    # Prefer "real news" when doc-type strongly indicates "news report"
    if best_lbl == "real news" and newsiness >= 0.65:
        if gap >= max(0.5 * MARGIN_UNCERTAIN, 0.01) or best_val >= 0.85 * MIN_CONFIDENCE:
            return "real news"

    if (best_val < MIN_CONFIDENCE) and (gap < MARGIN_UNCERTAIN):
        return "uncertain"
    return best_lbl

# ---- explanations ----
def explain_tokens(text: str, target_label: str, max_tokens: int = EXPLAIN_MAX_TOKENS) -> List[Dict[str, float]]:
    words = re.findall(r"\w+", text)
    if not words:
        return []
    filtered = [w for w in words if len(w) >= 3 and w.lower() not in STOPWORDS]
    filtered = filtered[:max_tokens]

    baseline_scores = classify_with_ensembles(text, CLASSIFY_LABELS)
    baseline = baseline_scores.get(target_label, 0.0)

    impacts: List[Dict[str, float]] = []
    for w in filtered:
        removed_once = False
        def _remove_once(m):
            nonlocal removed_once
            if not removed_once:
                removed_once = True
                return ""
            return m.group(0)

        pattern = re.compile(rf"\b{re.escape(w)}\b", flags=re.IGNORECASE)
        reduced_text = pattern.sub(_remove_once, text)
        try:
            sc = classify_with_ensembles(reduced_text, CLASSIFY_LABELS).get(target_label, 0.0)
        except Exception:
            sc = baseline
        delta = max(0.0, baseline - sc)
        impacts.append({"token": w, "delta": round(float(delta), 6)})

    impacts.sort(key=lambda d: d["delta"], reverse=True)
    return impacts[:8]

# -------------
# API models
# -------------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Article text to analyze")

class ExplanationItem(BaseModel):
    token: str
    delta: float

class AnalyzeResponse(BaseModel):
    label: str
    scores: Dict[str, float]
    explanations: List[ExplanationItem]

class FeedbackRequest(BaseModel):
    text: str
    predicted_label: str
    scores: Optional[Dict[str, float]] = None
    thumbs_up: Optional[bool] = None
    correct_label: Optional[str] = None
    notes: Optional[str] = None

# -------------
# FastAPI app
# -------------
app = FastAPI(title="AI Fake News Detector", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
async def index():
    labels_js = json.dumps(DISPLAY_LABELS)
    html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>AI Fake News Detector</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; max-width: 920px; }
        textarea { width: 100%; height: 220px; }
        .row { margin: 0.5rem 0; }
        .btn { padding: .6rem 1rem; border: 1px solid #ccc; background: #f7f7f7; cursor: pointer; border-radius: .5rem; }
        .card { border: 1px solid #eee; padding: 1rem; border-radius: .75rem; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.06); }
        .bar { height: 10px; background: #e5e7eb; border-radius: 6px; position: relative; }
        .fill { position:absolute; left:0; top:0; bottom:0; background:#3b82f6; border-radius:6px; }
        .pill { display:inline-block; padding:.25rem .5rem; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:600; }
        .muted { color:#6b7280; }
        .good { background:#dcfce7; color:#065f46; }
        .explain span { background: #fff8c5; padding: 0 .15rem; border-radius: .25rem; margin: 0 .05rem; }
      </style>
    </head>
    <body>
      <h1>AI Fake News Detector</h1>
      <p class="muted">Paste a news article or paragraph. This is a demo; do not rely on it as the sole source of truth.</p>

      <div class="card">
        <textarea id="text" placeholder="Paste article text here..."></textarea>
        <div class="row">
          <button id="analyze" class="btn">Analyze</button>
          <span id="status" class="muted"></span>
        </div>
      </div>

      <div id="results" style="margin-top:1rem; display:none;" class="card"></div>

      <script>
        const LABELS = __LABELS__;
        const textEl = document.getElementById('text');
        const statusEl = document.getElementById('status');
        const resultsEl = document.getElementById('results');
        document.getElementById('analyze').onclick = async () => {
          const text = textEl.value.trim();
          if (!text) return;
          statusEl.textContent = 'Analyzing...';
          resultsEl.style.display = 'none';
          try {
            const r = await fetch('/analyze', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify({text})
            });
            if (!r.ok) throw new Error('Request failed: ' + r.status);
            const data = await r.json();
            render(data, text);
            statusEl.textContent = '';
          } catch (e) {
            console.error(e);
            statusEl.textContent = 'Error: ' + e.message;
          }
        };

        function pct(x) { return (x*100).toFixed(1) + '%'; }

        function render(data, originalText) {
          const scores = data.scores;
          const best = data.label;
          let bars = LABELS.map(lbl => {
            const v = scores[lbl] || 0;
            return `
              <div class="row">
                <div><span class="pill ${best===lbl ? 'good' : ''}">${lbl}</span> <span class="muted">${pct(v)}</span></div>
                <div class="bar"><div class="fill" style="width:${pct(v)}"></div></div>
              </div>`;
          }).join('');

          let expl = '';
          if (data.explanations && data.explanations.length) {
            const topTokens = data.explanations.map(e => e.token.toLowerCase());
            const highlighted = originalText.replace(/\\b(\\w+)\\b/g, (m, w) =>
              topTokens.includes(w.toLowerCase()) ? `<span>${m}</span>` : m
            );
            expl = `
              <h3>Why (quick & rough)</h3>
              <p class="muted">Highlighted tokens increased the probability of the predicted label when present.</p>
              <div class="explain">${highlighted}</div>
            `;
          }

          resultsEl.innerHTML = `
            <h2>Prediction: <span class="pill">${best}</span></h2>
            ${bars}
            ${expl}
            <div class="row">
              <button class="btn" onclick="sendFeedback(true)">👍 Seems right</button>
              <button class="btn" onclick="sendFeedback(false)">👎 Seems wrong</button>
            </div>
          `;
          resultsEl.style.display = 'block';
          window._last = { data, originalText };
        }

        async function sendFeedback(ok) {
          if (!window._last) return;
          const payload = {
            text: window._last.originalText,
            predicted_label: window._last.data.label,
            scores: window._last.data.scores,
            thumbs_up: ok
          };
          try {
            await fetch('/feedback', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify(payload)
            });
            alert('Thanks for the feedback!');
          } catch (e) {
            alert('Could not submit feedback.');
          }
        }
      </script>
    </body>
    </html>
    """
    html = html.replace("__LABELS__", labels_js)
    return HTMLResponse(html)

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# ---------------
# API endpoints
# ---------------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Article text to analyze")

class ExplanationItem(BaseModel):
    token: str
    delta: float

class AnalyzeResponse(BaseModel):
    label: str
    scores: Dict[str, float]
    explanations: List[ExplanationItem]

def build_display_scores(scores3: Dict[str, float], newsiness: float = 0.0) -> Dict[str, float]:
    return with_uncertain_bar(scores3, newsiness=newsiness)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, request: Request):
    rate_limit(request)
    raw = (req.text or "").strip()
    if len(raw) < 40:
        raise HTTPException(status_code=400, detail="Please provide at least 40 characters of text.")

    # 1) Clean + safely strip boilerplate
    cleaned = clean_text(raw)
    stripped = strip_boilerplate(cleaned)
    if len(stripped.split()) < 40:
        stripped = cleaned

    # 2) Focus on the most article-like section (publisher-agnostic)
    core = select_main_window(stripped, window_words=800, step_words=200)

    # 3) Chunk + ensemble (among CLASSIFY_LABELS only)
    scores3, not_news_label, not_news_conf, best_chunk = ensemble_article_scores(core)

    # 4) Doc-type "newsiness" hint from the whole (core) text
    newsiness = 0.0
    if DOC_TYPE_ON:
        try:
            dt = doc_type_scores(core)
            newsiness = dt.get("news report", 0.0)
        except Exception:
            newsiness = 0.0

    # 5) Optional priors / penalties
    scores3 = apply_metadata_prior(core, scores3)
    scores3 = apply_reporting_prior(core, scores3)
    scores3 = apply_newsiness_penalty(scores3, newsiness)

    # 6) Build display scores (adds derived 'uncertain' bar) with newsiness
    final_scores = build_display_scores(scores3, newsiness=newsiness)

    # 7) Decide headline label using core scores and newsiness
    label = decide_label(scores3, newsiness)

    # 8) If VERY confident it's not news at all, override
    explanations: List[Dict[str, float]] = []
    if not_news_label and not_news_conf and not_news_conf >= DOC_GUARD_STRICTNESS:
        label = f"not a news article ({not_news_label})"
        final_scores = {lbl: 0.0 for lbl in DISPLAY_LABELS}
        explanations = []
    else:
        chosen = label
        if chosen.lower() == "uncertain":
            chosen = top_label(scores3)
        try:
            explanations = explain_tokens(best_chunk, chosen, EXPLAIN_MAX_TOKENS)
        except Exception:
            explanations = []

    return AnalyzeResponse(label=label, scores=final_scores, explanations=explanations)

@app.post("/feedback")
async def feedback(req: dict, request: Request):
    rate_limit(request)
    try:
        cur.execute(
            "INSERT INTO feedback(text, predicted_label, scores_json, thumbs_up, correct_label, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (
                req.get("text"),
                req.get("predicted_label"),
                json.dumps(req.get("scores") or {}),
                1 if req.get("thumbs_up") else 0 if req.get("thumbs_up") is not None else None,
                req.get("correct_label"),
                req.get("notes"),
            ),
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    return {"ok": True}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "display_labels": DISPLAY_LABELS,
        "classify_labels": CLASSIFY_LABELS,
        "use_natural_labels": USE_NATURAL_LABELS,
        "doc_type_on": DOC_TYPE_ON,
        "doc_type_labels": DOC_TYPE_LABELS,
        "doc_guard_strictness": DOC_GUARD_STRICTNESS,
        "use_dt_for_weight": USE_DT_FOR_WEIGHT,
        "chunk_words": CHUNK_WORDS,
        "chunk_stride": CHUNK_STRIDE,
        "ensemble_top_k": ENSEMBLE_TOP_K,
        "hypothesis_templates": HYPOTHESIS_TEMPLATES,
        "min_confidence": MIN_CONFIDENCE,
        "margin_uncertain": MARGIN_UNCERTAIN,
        "uncertain_scale": UNCERTAIN_SCALE,
        "boilerplate_on": BOILERPLATE_ON,
        "boilerplate_min_keep_words": BOILERPLATE_MIN_KEEP_WORDS,
        "boilerplate_min_keep_ratio": BOILERPLATE_MIN_KEEP_RATIO,
        "metadata_prior_on": METADATA_PRIOR_ON,
        "metadata_prior_boost": METADATA_PRIOR_BOOST,
        "reporting_prior_on": REPORTING_PRIOR_ON,
        "reporting_prior_boost": REPORTING_PRIOR_BOOST,
        "misleading_newsiness_penalty": MISLEADING_NEWSINESS_PENALTY,
        "misleading_newsiness_thresh": MISLEADING_NEWSINESS_THRESH,
    }
