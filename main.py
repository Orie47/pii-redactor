from fastapi import FastAPI
from pydantic import BaseModel
from gliner import GLiNER

app = FastAPI()
model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

class Prompt(BaseModel):
    text: str

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
