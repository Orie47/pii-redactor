import os
import logging
import re
from fastapi import FastAPI
from pydantic import BaseModel
from gliner import GLiNER
from huggingface_hub import snapshot_download
from wordfreq import zipf_frequency

# ------------------------------
# Logging (Render-friendly)
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info(f">>> Render PORT env: {os.environ.get('PORT')}")

# ------------------------------
# FastAPI app
# ------------------------------
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
# Config & Constants
# ------------------------------
CONFIDENCE_THRESHOLD = 0.6
ZIPF_THRESHOLD = 4.5

DO_NOT_REDACT = {
    "name", "person", "street", "road", "address",
    "phone", "number", "email", "user", "customer",
    "manager", "employee", "doctor", "teacher", "patient"
}
DO_NOT_REDACT_EXTRA = os.getenv("DO_NOT_REDACT_EXTRA", "")
if DO_NOT_REDACT_EXTRA:
    DO_NOT_REDACT.update({w.strip().lower() for w in DO_NOT_REDACT_EXTRA.split(",")})

# ------------------------------
# Regex Patterns (PII Formats)
# ------------------------------
REGEX_PATTERNS = {
    # Contact
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:\.[A-Za-z]{2,})?\b"
    ),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3})?[-.\s]?(?:\(?\d{2,3}\)?[-.\s]?){1,4}\d{2,4}\b"),
    "URL": re.compile(r"\bhttps?://[^\s/$.?#].[^\s]*\b", re.IGNORECASE),

    # IDs
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b"),
    "PASSPORT": re.compile(r"\b[A-Z]{1}[0-9]{6,8}\b"),
    "GENERIC_ID": re.compile(r"\b\d{8,12}\b"),

    # Dates
    "DATE": re.compile(r"\b(?:\d{1,2}[-/]){2}\d{2,4}\b"),
    "DATE_TEXT": re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"[-\s]?\d{1,2},?\s?\d{2,4}\b",
        re.IGNORECASE,
    ),

    # Network
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "IPV6": re.compile(r"\b(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}\b", re.IGNORECASE),
    "MAC_ADDRESS": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b"),

    # Financial
    "CRYPTO_WALLET": re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b"),

    # Postal
    "ZIP_CODE": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
}

# ------------------------------
# Helper Functions
# ------------------------------
def is_valid_entity(ent: dict) -> bool:
    text = ent["text"].strip()
    label = ent["label"].strip()
    score = ent.get("score", 0.0)

    if score < CONFIDENCE_THRESHOLD:
        return False
    if text.lower() == label.lower():
        return False
    if text.lower() in DO_NOT_REDACT:
        return False
    if len(text) <= 2:
        return False
    if text.isdigit() and len(text) < 6:
        return False
    if zipf_frequency(text, "en") >= ZIPF_THRESHOLD:
        return False
    return True

def regex_fallback(text: str):
    ents = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            ents.append({
                "start": m.start(),
                "end": m.end(),
                "text": m.group(),
                "label": label,
                "score": 1.0
            })
    return ents

def drop_overlaps_with_regex(text, entities, regex_patterns):
    """Remove model entities that overlap with regex-detected spans (esp. EMAIL)."""
    protected_spans = []
    for label, pattern in regex_patterns.items():
        if label == "EMAIL":
            for m in pattern.finditer(text):
                protected_spans.append((m.start(), m.end()))
    return [
        ent for ent in entities
        if not any(ent["start"] < end and ent["end"] > start for start, end in protected_spans)
    ]

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

def redact_text(text: str, entities: list):
    entities = sorted(entities, key=lambda x: x["start"], reverse=True)
    redacted = text
    for ent in entities:
        redacted = redacted[:ent["start"]] + f"[{ent['label'].upper()}]" + redacted[ent["end"]:]
    return redacted

def enforce_final_email_redaction(text: str):
    """Final sweep to ensure no raw emails remain."""
    return REGEX_PATTERNS["EMAIL"].sub("[EMAIL]", text)

# ------------------------------
# Routes
# ------------------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "PII redactor service is running"}

@app.post("/redact")
def redact(prompt: Prompt):
    text = prompt.text

    # Step 1: Model predictions
    labels = [
        "first name", "last name", "person",
        "email", "phone number", "location",
        "organization", "ID number", "passport number",
        "credit card", "social security number",
        "date", "date of birth", "death date", "appointment date"
    ]
    raw_entities = model.predict_entities(text, labels=labels)

    # Step 2: Filter entities
    valid_entities = [ent for ent in raw_entities if is_valid_entity(ent)]

    # Step 3: Regex fallback
    regex_entities = regex_fallback(text)

    # Step 4: Drop model entities overlapping regex EMAILs
    safe_entities = drop_overlaps_with_regex(text, valid_entities, REGEX_PATTERNS)

    # Step 5: Combine + merge
    all_entities = merge_entities(safe_entities + regex_entities)

    # Step 6: Redact
    redacted_text = redact_text(text, all_entities)

    # Step 7: Final safety net for emails
    redacted_text = enforce_final_email_redaction(redacted_text)

    return {
        "original": text,
        "redacted": redacted_text,
        "entities": all_entities
    }

# ------------------------------
# Notes
# ------------------------------
# ❌ Do NOT include uvicorn.run() here for Render
# Render uses the Start Command:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
# For local testing:
#   uvicorn main:app --reload
