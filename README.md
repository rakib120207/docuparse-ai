# DocuParse AI

**Intelligent Invoice Extraction for Ecommerce** - A production-grade FastAPI service that extracts vendor, date, amount, and confidence from invoices (PDF, PNG, JPEG) using Llama Vision. Built for ecommerce operators — Shopify stores, Amazon sellers, and ops teams who process supplier invoices daily.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Security](#security)
- [Architecture](#architecture)
- [Development](#development)
- [Deployment](#deployment)
- [License](#license)

## Features

- **Template-Free Extraction** — Works with any invoice layout using Llama Vision AI
- **Fast & Accurate** — Structured JSON + confidence score in under 3 seconds
- **Ecommerce-First UX** — Landing page, extractor app, and admin dashboard built-in
- **Secure by Default** — API-key auth, per-key rate limits, HSTS/CSP, HTTPS-only headers
- **Smart Rate Limiting** — Per-IP (30 req/min), per-key daily quotas, global RPM guard (28/min)
- **Demo Mode** — 5 free extractions/day per IP, no signup required
- **SQLite Storage** — Full extraction history with indexed queries
- **Sentry & Monitoring** — Optional error tracking and observability
- **Multi-Format** — PDF, PNG, JPEG with magic-byte validation

## Quick Start

### Prerequisites

- Python 3.10 or higher
- A Groq API key (for Llama Vision) — [get one here](https://console.groq.com/keys)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd InvoiceScanner

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with the following variables:

```bash
# Required
LLAMA_API_KEY=gsk_your_groq_api_key_here
LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions
LLAMA_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# Recommended
ADMIN_PASSWORD=your_secure_admin_password
CONTACT_EMAIL=your@email.com

# Optional
MAX_FILE_MB=10
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
KEYS_FILE_PATH=api_keys.json
DB_PATH=extractions.db
DEMO_DAILY_LIMIT=5

# Monitoring (optional)
SENTRY_DSN=your_sentry_dsn_here
POSTHOG_KEY=your_posthog_key_here
```

### Run the Server

```bash
# Standard
uvicorn main:app --host 0.0.0.0 --port 8000

# Development with hot-reload
uvicorn main:app --reload --port 8000
```

## Usage

### Web Interface

| URL | Purpose |
|-----|---------|
| `http://localhost:8000` | Landing page — request access or try demo |
| `http://localhost:8000/app` | Invoice extractor (demo or API-key mode) |
| `http://localhost:8000/admin` | Admin dashboard (password-protected) |

### Demo Mode

Visit `http://localhost:8000/app?demo=1` for 5 free extractions per day (per IP). No signup required.

## API Reference

### Extract Invoice

```bash
curl -X POST http://localhost:8000/extract \
  -H "X-API-Key: your_api_key_here" \
  -F "file=@invoice.pdf"
```

**Response (200)**

```json
{
  "status": "ok",
  "vendor": "Shenzhen Global Electronics Co.",
  "date": "2025-01-15",
  "amount": "$4,820.00",
  "confidence": 97,
  "usage": {
    "used": 3,
    "limit": 50
  }
}
```

### Verify API Key

```bash
curl -X POST http://localhost:8000/verify \
  -H "X-API-Key: your_api_key_here"
```

## Security

- **SHA-256 Key Storage** — API keys are hashed before storage; plaintext is never persisted
- **Constant-Time Comparison** — HMAC-based secret comparison prevents timing attacks
- **Security Headers** — HSTS, CSP, X-XSS-Protection, X-Frame-Options, Referrer-Policy
- **Rate Limiting** — Per-IP (30 req/min), per-key daily quotas, global RPM guard (28/min)
- **File Validation** — Magic byte verification on all uploads
- **Document Privacy** — Files processed in memory and immediately discarded

## Architecture

```
├── main.py                 # FastAPI app, routes, middleware, HTML pages
├── Procfile                # Deployment process definition
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignore .env, db, keys, cache
├── api_keys.json           # Generated: hashed API keys (never commit!)
├── extractions.db          # SQLite extraction history
└── README.md               # This file
```

### Database Schema

```sql
CREATE TABLE extractions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    key_name      TEXT    NOT NULL,
    filename      TEXT    NOT NULL,
    file_type     TEXT    NOT NULL,
    vendor        TEXT,
    date          TEXT,
    amount        TEXT,
    status        TEXT    NOT NULL DEFAULT 'ok',
    error_msg     TEXT,
    duration_ms   INTEGER
);
```

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload --port 8000

# Check extraction history
sqlite3 extractions.db "SELECT * FROM extractions ORDER BY created_at DESC LIMIT 10;"
```

### Project Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.115.0 | Web framework |
| uvicorn[standard] | 0.30.6 | ASGI server |
| python-dotenv | 1.0.1 | Environment loading |
| httpx | 0.27.2 | Async HTTP client |
| PyMuPDF | 1.24.10 | PDF to PNG rendering |
| aiosqlite | 0.20.0 | Async SQLite |
| python-multipart | 0.0.9 | File uploads |
| sentry-sdk | 2.14.0 | Error tracking |
| posthog | 3.5.0 | Analytics |

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Render / Fly.io / Heroku

```bash
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_API_KEY` | — | Groq API key (required) |
| `LLAMA_API_URL` | Groq endpoint | Vision API URL |
| `LLAMA_MODEL` | Llama-4 Scout | Vision model |
| `ADMIN_PASSWORD` | — | Admin panel password |
| `CONTACT_EMAIL` | your@email.com | Contact email in UI |
| `MAX_FILE_MB` | 10 | Max upload size |
| `DEMO_DAILY_LIMIT` | 5 | Free demo extractions per IP/day |

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

**Powered by Llama Vision** — Intelligent document understanding without templates. Built for ecommerce invoice automation.