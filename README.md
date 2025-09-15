# PII Redactor API

This project provides an API for redacting personally identifiable information (PII) from user input text.
It combines a machine learning model (GLiNER) with regular expressions to identify and mask sensitive data.
The application is built with FastAPI and is suitable for local use or cloud deployment (e.g., Render).

## Features

- Entity detection using a pre-trained GLiNER model from Hugging Face
- Regex-based fallback for common PII patterns (email, phone, SSN, credit card, etc.)
- Final enforcement sweep to minimize residual leakage
- Simple REST API with a `/redact` endpoint
- Ready for deployment via `render.yaml`

## API Endpoints

### `GET /`

Health check endpoint to verify the service is running.

Response:
```json
{
  "status": "ok",
  "message": "PII redactor service is running"
}
