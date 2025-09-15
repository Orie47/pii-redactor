import os
import logging
import re
from fastapi import FastAPI
from pydantic import BaseModel
from gliner import GLiNER
from huggingface_hub import snapshot_download

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info(f">>> Render PORT env: {os.environ.get('PORT')}")

app = FastAPI()

# ------------------------------
# Model Setup
# ------------------------------
CACHE_DIR = os.environ.get("MODEL_CACHE", "./models")
MODEL_ID = "urchade/gliner_small-v2"

logging.info(">>> Downloading/loading model...")
model_path = snapshot_download(
    MODEL_ID,
    local_dir=CACHE_DIR,
    local_dir_use_symlinks=False
)
model = GLiNER.from_pretrained(model_path)
logging.info(">>> Model loaded successfully")

# ------------------------------
# Request Schema
# ------------------------------
class Prompt(BaseModel):
    text: str

# ------------------------------
# Regex Patterns
# ------------------------------
REGEX_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:\.[A-Za-z]{2,})?\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){1,3}\d{3,4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ZIP_CODE": re.compile(r"(?<!#)\b\d{5}(?:-\d{4})?\b"),
    "DATE": re.compile(
        r"\b(?:\d{1,2}[-/]){2}\d{2,4}\b|"
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"[-\s]?\d{1,2},?\s?\d{2,4}\b",
        re.IGNORECASE,
    ),
}

# ------------------------------
# Helper Functions
# ------------------------------
BANNED_WORDS = {
    "i", "me", "my", "mine",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs",
    "name", "person", "number", "email"
}

def regex_fallback(text: str):
    ents = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            match_text = m.group()
            # Skip phone if it's actually SSN or credit card
            if label == "PHONE":
                if REGEX_PATTERNS["SSN"].fullmatch(match_text) or REGEX_PATTERNS["CREDIT_CARD"].fullmatch(match_text):
                    continue
                if sum(c.isdigit() for c in match_text) < 7:
                    continue
            ents.append({
                "start": m.start(),
                "end": m.end(),
                "text": match_text,
                "label": label,
                "score": 1.0
            })
    return ents

def merge_entities(entities):
    entities = sorted(entities, key=lambda x: (x["start"], -x["end"]))
    merged = []
    for ent in entities:
        if merged and ent["start"] < merged[-1]["end"]:
            if ent["end"] > merged[-1]["end"]:
                merged[-1] = ent
        else:
            merged.append(ent)
    return merged

def normalize_labels(entities):
    """Force regex priority: SSN > PHONE, CREDIT_CARD > PHONE."""
    for ent in entities:
        txt = ent["text"]
        if REGEX_PATTERNS["SSN"].fullmatch(txt):
            ent["label"] = "SSN"
        elif REGEX_PATTERNS["CREDIT_CARD"].fullmatch(txt):
            ent["label"] = "CREDIT_CARD"
    return entities

def filter_entities(entities):
    return [
        ent for ent in entities
        if ent["text"].lower() not in BANNED_WORDS
    ]

def redact_text(text: str, entities: list):
    entities = sorted(entities, key=lambda x: x["start"], reverse=True)
    redacted = text
    for ent in entities:
        redacted = redacted[:ent["start"]] + f"[{ent['label'].upper()}]" + redacted[ent["end"]:]
    return redacted

def enforce_final_redaction(text: str):
    """Final regex sweep to ensure nothing leaks."""
    redacted = REGEX_PATTERNS["EMAIL"].sub("[EMAIL]", text)

    def phone_replacer(m):
        txt = m.group()
        if REGEX_PATTERNS["SSN"].fullmatch(txt) or REGEX_PATTERNS["CREDIT_CARD"].fullmatch(txt):
            return txt  # let SSN/CC overwrite later
        return "[PHONE]"
    redacted = REGEX_PATTERNS["PHONE"].sub(phone_replacer, redacted)

    redacted = REGEX_PATTERNS["SSN"].sub("[SSN]", redacted)
    redacted = REGEX_PATTERNS["CREDIT_CARD"].sub("[CREDIT_CARD]", redacted)
    redacted = REGEX_PATTERNS["DATE"].sub("[DATE]", redacted)
    return redacted

# ------------------------------
# Routes
# ------------------------------
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def health():
    """Health check for Render (supports GET + HEAD)."""
    return {"status": "ok", "message": "PII redactor service is running"}

@app.post("/redact")
def redact(prompt: Prompt):
    text = prompt.text

    labels = [
        "first name", "last name", "person",
        "email", "phone number", "location",
        "organization", "ID number", "passport number",
        "credit card", "social security number",
        "date", "date of birth", "death date", "appointment date"
    ]
    model_entities = model.predict_entities(text, labels=labels)

    regex_entities = regex_fallback(text)
    all_entities = merge_entities(model_entities + regex_entities)
    all_entities = normalize_labels(all_entities)
    all_entities = filter_entities(all_entities)

    redacted_text = redact_text(text, all_entities)
    redacted_text = enforce_final_redaction(redacted_text)

    return {
        "original": text,
        "redacted": redacted_text,
        "entities": all_entities
    }

# ------------------------------
# Notes
# ------------------------------
# Do NOT include uvicorn.run() here for Render
# Render uses the Start Command:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
# For local testing:
#   uvicorn main:app --reload
