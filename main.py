import os
from fastapi import FastAPI
from pydantic import BaseModel
from gliner import GLiNER
from huggingface_hub import snapshot_download

app = FastAPI()

# ----------- Model Loading -----------

# Preferred location for caching model (Render disk if available, otherwise local)
CACHE_DIR = os.environ.get("MODEL_CACHE", "./models")

# Download/cached path
model_path = snapshot_download(
    "urchade/gliner_small-v2",
    local_dir=CACHE_DIR,
    local_dir_use_symlinks=False  # ensures files really go into this dir
)

# Load the lightweight GLiNER model
model = GLiNER.from_pretrained(model_path)

# ----------- Request Schema -----------

class Prompt(BaseModel):
    text: str

# ----------- API Route -----------

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

    entities = model.predict_entities(text, labels=labels)

    redacted = text
    for entity in sorted(entities, key=lambda e: -e["start"]):
        start = entity["start"]
        end = entity["end"]
        label = entity["label"].upper()
        redacted = redacted[:start] + f"[{label}]" + redacted[end:]

    return {
        "original": text,
        "redacted": redacted,
        "entities": entities
    }

# ----------- Notes -----------
# ❌ Do NOT include uvicorn.run() here for Render
# Render uses the Start Command:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
# For local testing:
#   uvicorn main:app --reload
