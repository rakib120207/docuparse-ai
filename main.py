"""
DocuParse AI — Production-Grade FastAPI Application
Fixes: JS syntax errors from {{ }} in non-f-strings, async HTTP, security hardening
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import aiosqlite
import fitz  # pymupdf — converts PDF pages to PNG for vision API
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── DATABASE ─────────────────────────────────────────────────────────────────────
DB_PATH = Path(os.getenv("DB_PATH", "extractions.db"))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS extractions (
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
CREATE INDEX IF NOT EXISTS idx_extractions_created ON extractions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_extractions_key     ON extractions(key_name);
"""

async def db_init() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_TABLE_SQL)
        await db.commit()
    logger.info("Database ready: %s", DB_PATH)

async def db_log(
    *,
    key_name:    str,
    filename:    str,
    file_type:   str,
    vendor:      str | None,
    date:        str | None,
    amount:      str | None,
    status:      str,
    error_msg:   str | None,
    duration_ms: int,
) -> int:
    """Insert one extraction record; return the new row id."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """INSERT INTO extractions
               (created_at, key_name, filename, file_type,
                vendor, date, amount, status, error_msg, duration_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                key_name, filename, file_type,
                vendor, date, amount, status, error_msg, duration_ms,
            ),
        )
        await db.commit()
        return cur.lastrowid

async def db_get_extractions(
    limit:    int  = 100,
    offset:   int  = 0,
    key_name: str | None = None,
    status:   str | None = None,
) -> list[dict]:
    filters, params = [], []
    if key_name:
        filters.append("key_name = ?"); params.append(key_name)
    if status:
        filters.append("status = ?");   params.append(status)
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params += [limit, offset]
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            f"SELECT * FROM extractions {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

async def db_stats() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("""
            SELECT
                COUNT(*)                                    AS total,
                SUM(CASE WHEN status='ok'    THEN 1 END)   AS success,
                SUM(CASE WHEN status='error' THEN 1 END)   AS errors,
                ROUND(AVG(duration_ms))                    AS avg_ms,
                COUNT(DISTINCT key_name)                   AS unique_keys
            FROM extractions
        """)
        return dict(await cur.fetchone())

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────────
LLAMA_API_KEY  = os.getenv("LLAMA_API_KEY", "")
LLAMA_API_URL  = os.getenv("LLAMA_API_URL", "https://api.groq.com/openai/v1/chat/completions")
LLAMA_MODEL    = os.getenv("LLAMA_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")          # MUST be set in .env
CONTACT_EMAIL  = os.getenv("CONTACT_EMAIL", "your@email.com")
MAX_FILE_MB    = int(os.getenv("MAX_FILE_MB", "10"))

_raw_origins   = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins else ["*"]
)

KEYS_FILE = Path("api_keys.json")

# ── SECURITY UTILITIES ───────────────────────────────────────────────────────────

def _hash_secret(value: str) -> str:
    """SHA-256 hex digest — used for both admin password and key storage index."""
    return hashlib.sha256(value.encode()).hexdigest()


def _secure_equal(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


# ── ADMIN PASSWORD VALIDATION ────────────────────────────────────────────────────

def _admin_auth(pwd: str | None) -> None:
    """Validate admin password; raises 403 on failure."""
    if not pwd or not ADMIN_PASSWORD:
        raise HTTPException(403, "Admin access denied.")
    if not _secure_equal(_hash_secret(pwd), _hash_secret(ADMIN_PASSWORD)):
        raise HTTPException(403, "Admin access denied.")


# ── IP RATE LIMITER (in-memory, per-endpoint) ────────────────────────────────────

_ip_hits: dict[str, list[float]] = defaultdict(list)
_IP_WINDOW  = 60   # seconds
_IP_MAX_REQ = 30   # max requests per window per IP


def _ip_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - _IP_WINDOW
    hits = [t for t in _ip_hits[ip] if t > window_start]
    if len(hits) >= _IP_MAX_REQ:
        raise HTTPException(429, "Too many requests. Slow down.")
    hits.append(now)
    _ip_hits[ip] = hits



# ── PDF → PNG CONVERSION ──────────────────────────────────────────────────────────

def _pdf_first_page_to_png(pdf_bytes: bytes, dpi: int = 150) -> bytes:
    """
    Render the first page of a PDF at `dpi` resolution and return PNG bytes.
    Raises HTTPException if the PDF is invalid or has no pages.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise HTTPException(400, "PDF has no pages.")
        page = doc.load_page(0)
        zoom = dpi / 72          # 72 is PDF base DPI
        mat  = fitz.Matrix(zoom, zoom)
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("PDF render failed: %s", exc)
        raise HTTPException(400, "Could not render PDF. File may be corrupted or password-protected.")


# ── GLOBAL RPM GUARD ─────────────────────────────────────────────────────────────
_rpm_lock = threading.Lock()
_rpm_timestamps: list[float] = []
RPM_LIMIT = 28  # 2 below Groq's 30 — safety buffer


def _check_global_rpm() -> None:
    now = time.monotonic()
    with _rpm_lock:
        window = [t for t in _rpm_timestamps if now - t < 60]
        if len(window) >= RPM_LIMIT:
            raise HTTPException(429, "Service is busy. Please retry in a few seconds.")
        window.append(now)
        _rpm_timestamps.clear()
        _rpm_timestamps.extend(window)

# ── MAGIC BYTE FILE VALIDATION ───────────────────────────────────────────────────

_MAGIC: dict[str, tuple[bytes, ...]] = {
    "pdf":  (b"%PDF",),
    "png":  (b"\x89PNG",),
    "jpg":  (b"\xff\xd8\xff",),
    "jpeg": (b"\xff\xd8\xff",),
}


def _validate_file_bytes(filename: str, data: bytes) -> str:
    """
    Verify extension AND magic bytes match.
    Returns MIME type string on success; raises HTTPException on failure.
    """
    ext = (filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if ext not in _MAGIC:
        raise HTTPException(400, "Unsupported file type. Accepted: PDF, PNG, JPEG.")
    if not any(data.startswith(sig) for sig in _MAGIC[ext]):
        raise HTTPException(400, "File content does not match its extension.")
    mime_map = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    return mime_map[ext]


# ── KEY STORAGE (SHA-256 keyed) ──────────────────────────────────────────────────
# Raw keys are NEVER stored on disk. Only their SHA-256 digest is the dict key.
# The plaintext key is returned once on creation and never persisted.

def _load() -> dict[str, Any]:
    try:
        if KEYS_FILE.exists():
            return json.loads(KEYS_FILE.read_text())
    except Exception:
        pass
    return {}


def _save(keys: dict[str, Any]) -> None:
    KEYS_FILE.write_text(json.dumps(keys, indent=2))


def _init() -> None:
    # ── Startup config validation ──────────────────────────────────────
    errors   = []
    warnings = []

    if not LLAMA_API_KEY:
        errors.append("LLAMA_API_KEY is not set in .env")

    if not LLAMA_API_URL or "console.groq.com" in LLAMA_API_URL:
        errors.append(
            f"LLAMA_API_URL looks wrong: {LLAMA_API_URL!r}\n"
            "     ✅ Correct value: https://api.groq.com/openai/v1/chat/completions"
        )
    elif not LLAMA_API_URL.startswith("https://"):
        warnings.append(f"LLAMA_API_URL does not use HTTPS: {LLAMA_API_URL!r}")

    if not ADMIN_PASSWORD:
        warnings.append("ADMIN_PASSWORD is not set — admin panel is disabled")

    if errors:
        print("\n" + "=" * 64)
        print("  ❌  CONFIGURATION ERROR — server cannot process extractions:")
        for e in errors:
            print(f"     • {e}")
        print("=" * 64 + "\n")
    for w in warnings:
        print(f"\n⚠️   WARNING: {w}\n")
    keys = _load()
    if not keys:
        raw = secrets.token_urlsafe(40)
        hashed = _hash_secret(raw)
        keys[hashed] = dict(
            name="Default Key",
            active=True,
            daily_limit=50,
            usage_today=0,
            last_reset=str(date.today()),
            created_at=datetime.now().isoformat(),
        )
        _save(keys)
        print("=" * 64)
        print("  🔑  FIRST RUN — your API key (shown ONCE, store it now):")
        print(f"     {raw}")
        print(f"  🛡   Admin panel : http://localhost:8000/admin")
        print("  ⚙️   Set ADMIN_PASSWORD in your .env file!")
        print("=" * 64)


def _check_rate_limit(kd: dict, hashed_key: str, keys: dict) -> None:
    today = str(date.today())
    if kd.get("last_reset") != today:
        kd["usage_today"] = 0
        kd["last_reset"] = today
    if kd["usage_today"] >= kd.get("daily_limit", 50):
        raise HTTPException(429, f"Daily limit ({kd['daily_limit']}) reached. Resets tomorrow.")


def _validate_key(raw_key: str | None, consume: bool = True) -> tuple[dict, str, dict]:
    """
    Validate an API key. Returns (key_data, hashed_key, all_keys).
    Raises HTTPException on auth/rate failure.
    """
    if not raw_key:
        raise HTTPException(401, "API key required.")
    hashed = _hash_secret(raw_key)
    keys = _load()
    if hashed not in keys:
        raise HTTPException(403, "Invalid API key.")
    kd = keys[hashed]
    if not kd.get("active", True):
        raise HTTPException(403, "API key is inactive.")
    _check_rate_limit(kd, hashed, keys)
    if consume:
        kd["usage_today"] = kd.get("usage_today", 0) + 1
        keys[hashed] = kd
        _save(keys)
    return kd, hashed, keys


# ── FASTAPI APP ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_init()
    yield

app = FastAPI(
    title="DocuParse AI",
    lifespan=lifespan,
    docs_url=None,    # hide Swagger in production
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "X-Admin-Password"],
    allow_credentials=False,
)


@app.middleware("http")
async def apply_security_headers(request: Request, call_next):
    response = await call_next(request)
    hdrs = {
        "X-Content-Type-Options":    "nosniff",
        "X-Frame-Options":           "DENY",
        "X-XSS-Protection":          "1; mode=block",
        "Referrer-Policy":           "strict-origin-when-cross-origin",
        "Cache-Control":             "no-store, no-cache, must-revalidate",
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
        "Permissions-Policy":        "camera=(), microphone=(), geolocation=()",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self';"
        ),
    }
    for k, v in hdrs.items():
        response.headers[k] = v
    return response


_init()


# ── HTML PAGES ───────────────────────────────────────────────────────────────────
# NOTE: Only LANDING uses an f-string (needs {CONTACT_EMAIL}).
# APP_PAGE and ADMIN_PAGE are plain strings — NO {{ }} escaping needed,
# which is exactly why the previous version had broken JS.

LANDING = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DocuParse AI — Intelligent Invoice Extraction</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<script>tailwind.config={{theme:{{extend:{{colors:{{accent:'#D2B48C','accent-h':'#C4A67D'}},fontFamily:{{sans:['Inter','system-ui','sans-serif']}}}}}}}}</script>
<style>html{{scroll-behavior:smooth}}</style>
</head>
<body class="bg-[#fafafa] font-sans text-zinc-900 antialiased">

<nav class="fixed top-0 w-full bg-white/95 backdrop-blur-sm border-b border-black/5 z-50">
<div class="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
  <span class="text-lg font-medium tracking-tight">DocuParse AI</span>
  <div class="hidden md:flex items-center gap-8">
    <a href="#features" class="text-sm text-zinc-500 hover:text-zinc-900 transition-colors">Features</a>
    <a href="#how" class="text-sm text-zinc-500 hover:text-zinc-900 transition-colors">How It Works</a>
    <a href="#contact" class="text-sm text-zinc-500 hover:text-zinc-900 transition-colors">Contact</a>
    <a href="/app" class="bg-zinc-900 text-white text-sm px-5 py-2.5 rounded-lg hover:bg-zinc-800 transition-colors">Get Access</a>
  </div>
  <a href="/app" class="md:hidden bg-zinc-900 text-white text-sm px-4 py-2 rounded-lg">Access</a>
</div>
</nav>

<section class="relative min-h-screen flex items-center justify-center bg-zinc-900 pt-20 overflow-hidden">
  <div class="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[#fafafa]"></div>
  <div class="relative z-10 text-center px-6 max-w-4xl mx-auto">
    <p class="text-accent text-xs font-semibold uppercase tracking-[0.2em] mb-6">AI-Powered Document Intelligence</p>
    <h1 class="text-5xl md:text-7xl lg:text-8xl font-light text-white leading-[0.9] tracking-tighter mb-8">
      Extract Data<br><span class="text-accent">Instantly</span>
    </h1>
    <p class="text-zinc-400 text-lg md:text-xl font-light max-w-2xl mx-auto mb-12 leading-relaxed">
      Upload any invoice and get structured data in seconds. Powered by Llama Vision — no templates, no training, just results.
    </p>
    <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
      <a href="/app" class="bg-accent text-zinc-900 px-8 py-3.5 rounded-lg text-sm font-medium hover:bg-accent-h transition-colors">Start Extracting →</a>
      <a href="#how" class="text-zinc-400 text-sm hover:text-white transition-colors">See how it works</a>
    </div>
  </div>
</section>

<section id="features" class="py-24 md:py-32 px-6">
  <div class="max-w-7xl mx-auto">
    <div class="text-center mb-16">
      <p class="text-accent text-xs font-semibold uppercase tracking-[0.2em] mb-4">Features</p>
      <h2 class="text-3xl md:text-4xl font-light tracking-tight">Built for Accuracy</h2>
    </div>
    <div class="grid md:grid-cols-3 gap-8">
      <div class="bg-white rounded-xl border border-black/5 p-8 hover:shadow-md transition-shadow">
        <div class="w-12 h-12 bg-zinc-100 rounded-lg flex items-center justify-center mb-6 text-2xl">⚡</div>
        <h3 class="text-lg font-medium tracking-tight mb-3">Lightning Fast</h3>
        <p class="text-zinc-500 text-sm leading-relaxed">Get structured JSON from any invoice in under 3 seconds. No queues, no waiting.</p>
      </div>
      <div class="bg-white rounded-xl border border-black/5 p-8 hover:shadow-md transition-shadow">
        <div class="w-12 h-12 bg-zinc-100 rounded-lg flex items-center justify-center mb-6 text-2xl">🎯</div>
        <h3 class="text-lg font-medium tracking-tight mb-3">Template-Free</h3>
        <p class="text-zinc-500 text-sm leading-relaxed">Works with any invoice format. No templates or model training required.</p>
      </div>
      <div class="bg-white rounded-xl border border-black/5 p-8 hover:shadow-md transition-shadow">
        <div class="w-12 h-12 bg-zinc-100 rounded-lg flex items-center justify-center mb-6 text-2xl">🔒</div>
        <h3 class="text-lg font-medium tracking-tight mb-3">Secure &amp; Controlled</h3>
        <p class="text-zinc-500 text-sm leading-relaxed">API-key auth, per-key rate limits, and documents are processed then discarded.</p>
      </div>
    </div>
  </div>
</section>

<section id="how" class="py-24 md:py-32 px-6 bg-white">
  <div class="max-w-4xl mx-auto">
    <div class="text-center mb-16">
      <p class="text-accent text-xs font-semibold uppercase tracking-[0.2em] mb-4">Process</p>
      <h2 class="text-3xl md:text-4xl font-light tracking-tight">How It Works</h2>
    </div>
    <div class="space-y-12">
      <div class="flex gap-6 items-start"><div class="flex-shrink-0 w-10 h-10 rounded-full bg-zinc-900 text-white flex items-center justify-center text-sm font-medium">1</div><div><h3 class="text-lg font-medium tracking-tight mb-2">Upload Your Invoice</h3><p class="text-zinc-500 text-sm leading-relaxed">Drop a PDF, PNG, or JPEG. Any vendor, any layout.</p></div></div>
      <div class="flex gap-6 items-start"><div class="flex-shrink-0 w-10 h-10 rounded-full bg-zinc-900 text-white flex items-center justify-center text-sm font-medium">2</div><div><h3 class="text-lg font-medium tracking-tight mb-2">AI Analyzes the Document</h3><p class="text-zinc-500 text-sm leading-relaxed">Llama Vision reads and understands the document, extracting vendor, date, and amount.</p></div></div>
      <div class="flex gap-6 items-start"><div class="flex-shrink-0 w-10 h-10 rounded-full bg-zinc-900 text-white flex items-center justify-center text-sm font-medium">3</div><div><h3 class="text-lg font-medium tracking-tight mb-2">Get Structured JSON</h3><p class="text-zinc-500 text-sm leading-relaxed">Clean, validated JSON ready for your database, ERP, or accounting software.</p></div></div>
    </div>
  </div>
</section>

<section class="py-24 md:py-32 px-6">
  <div class="max-w-4xl mx-auto">
    <div class="text-center mb-16">
      <p class="text-accent text-xs font-semibold uppercase tracking-[0.2em] mb-4">Output</p>
      <h2 class="text-3xl md:text-4xl font-light tracking-tight">Clean JSON, Every Time</h2>
    </div>
    <div class="bg-zinc-900 rounded-xl p-8 overflow-x-auto">
      <pre class="text-sm text-zinc-300 font-mono leading-relaxed"><code>{{
  "vendor": "Acme Corporation",
  "date": "2025-01-15",
  "amount": "$2,450.00"
}}</code></pre>
    </div>
  </div>
</section>

<section id="contact" class="py-24 md:py-32 px-6 bg-white">
  <div class="max-w-2xl mx-auto text-center">
    <p class="text-accent text-xs font-semibold uppercase tracking-[0.2em] mb-4">Get Access</p>
    <h2 class="text-3xl md:text-4xl font-light tracking-tight mb-6">Need an API Key?</h2>
    <p class="text-zinc-500 text-lg font-light leading-relaxed mb-4">
      This service is available on request. Reach out with your use case and we'll set you up with a key and usage limits.
    </p>
    <p class="text-zinc-400 text-sm mb-10">Typical response time: within 24 hours</p>
    <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
      <a href="mailto:{CONTACT_EMAIL}?subject=DocuParse%20AI%20Access%20Request"
         class="bg-zinc-900 text-white px-8 py-3.5 rounded-lg text-sm font-medium hover:bg-zinc-800 transition-colors inline-flex items-center gap-2">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75"/>
        </svg>
        Contact Me
      </a>
      <a href="/app" class="text-zinc-500 text-sm hover:text-zinc-900 transition-colors">I already have a key →</a>
    </div>
  </div>
</section>

<footer class="py-12 px-6 border-t border-black/5">
  <div class="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
    <span class="text-sm text-zinc-400">&copy; 2025 DocuParse AI</span>
    <div class="flex items-center gap-6">
      <a href="mailto:{CONTACT_EMAIL}" class="text-sm text-zinc-400 hover:text-zinc-900 transition-colors">Email</a>
      <span class="text-sm text-zinc-400">Powered by Llama Vision</span>
    </div>
  </div>
</footer>
</body></html>"""


# ── APP PAGE (plain string — no f-prefix, no {{ }} escaping) ─────────────────────
APP_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DocuParse AI — Extractor</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<script>
  tailwind.config = {
    theme: { extend: { colors: { accent: '#D2B48C' }, fontFamily: { sans: ['Inter','system-ui','sans-serif'] } } }
  };
</script>
</head>
<body class="bg-[#fafafa] font-sans text-zinc-900 antialiased min-h-screen">

<!-- AUTH GATE -->
<div id="authGate" class="min-h-screen flex items-center justify-center px-6">
  <div class="bg-white rounded-xl border border-black/5 p-8 w-full max-w-md shadow-sm">
    <div class="text-center mb-6">
      <h1 class="text-xl font-medium tracking-tight mb-1">DocuParse AI</h1>
      <p class="text-sm text-zinc-400">Enter your API key to continue</p>
    </div>
    <div class="space-y-3">
      <input id="keyInput" type="password" placeholder="API key..."
        autocomplete="off" spellcheck="false"
        class="w-full border border-black/10 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-zinc-400 transition-colors font-mono"
        onkeydown="if(event.key==='Enter') verifyKey()">
      <button onclick="verifyKey()" id="verifyBtn"
        class="w-full bg-zinc-900 text-white text-sm py-3 rounded-lg font-medium hover:bg-zinc-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
        Verify &amp; Enter
      </button>
    </div>
    <p id="authError" class="text-red-500 text-xs mt-3 text-center hidden"></p>
    <p class="text-xs text-zinc-400 text-center mt-6">Don't have a key?
      <a href="/" class="text-amber-700 hover:underline">Request access</a>
    </p>
  </div>
</div>

<!-- MAIN APP (hidden until auth) -->
<div id="mainApp" class="hidden">
  <nav class="bg-white/95 backdrop-blur-sm border-b border-black/5">
    <div class="max-w-3xl mx-auto px-6 h-16 flex items-center justify-between">
      <a href="/" class="text-sm font-medium tracking-tight">DocuParse AI</a>
      <div class="flex items-center gap-4">
        <div id="usageBar" class="hidden items-center gap-2 text-xs text-zinc-400">
          <span id="usageText"></span>
          <div class="w-16 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
            <div id="usageFill" class="h-full bg-amber-700 rounded-full transition-all duration-300"></div>
          </div>
        </div>
        <span id="keyName" class="text-xs text-zinc-400 hidden"></span>
        <button onclick="logout()" class="text-xs text-zinc-400 hover:text-zinc-900 transition-colors">Logout</button>
      </div>
    </div>
  </nav>

  <main class="max-w-3xl mx-auto px-6 py-12">
    <div class="bg-white rounded-xl border border-black/5 p-8 shadow-sm">
      <h2 class="text-lg font-medium tracking-tight mb-1">Invoice Extractor</h2>
      <p class="text-sm text-zinc-400 mb-6">Upload a PDF, PNG, or JPEG invoice to extract vendor, date, and amount.</p>

      <div id="dropZone"
        class="border-2 border-dashed border-black/10 rounded-lg p-8 text-center mb-6 hover:border-zinc-300 transition-colors cursor-pointer"
        onclick="document.getElementById('fileInput').click()"
        ondragover="event.preventDefault(); this.classList.add('border-zinc-400')"
        ondragleave="this.classList.remove('border-zinc-400')"
        ondrop="handleDrop(event)">
        <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg" class="hidden" onchange="fileSelected(this.files[0])">
        <div id="dropText">
          <svg class="w-8 h-8 mx-auto mb-3 text-zinc-300" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
          </svg>
          <p class="text-sm text-zinc-400">Click to upload or drag &amp; drop</p>
          <p class="text-xs text-zinc-300 mt-1">PDF, PNG, JPEG — max 10 MB</p>
        </div>
        <p id="fileName" class="text-sm text-zinc-600 hidden font-medium"></p>
      </div>

      <div class="flex gap-3">
        <button id="extractBtn" onclick="extractData()" disabled
          class="flex-1 bg-zinc-900 text-white text-sm py-3 rounded-lg font-medium hover:bg-zinc-800 transition-colors disabled:bg-zinc-200 disabled:text-zinc-400 disabled:cursor-not-allowed">
          Extract Data
        </button>
        <button id="cancelBtn" onclick="resetForm()"
          class="hidden px-4 py-3 border border-black/10 rounded-lg text-sm text-zinc-500 hover:bg-zinc-50 transition-colors">
          Clear
        </button>
      </div>
      <p id="extractError" class="text-red-500 text-xs mt-3 hidden"></p>

      <div id="resultBox" class="hidden mt-6">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-medium text-zinc-400 uppercase tracking-wider">Result</span>
          <button id="copyBtn" onclick="copyResult()" class="text-xs text-zinc-400 hover:text-zinc-900 transition-colors">Copy JSON</button>
        </div>
        <pre id="resultJson" class="bg-zinc-900 text-green-400 p-6 rounded-lg text-sm font-mono overflow-x-auto leading-relaxed whitespace-pre-wrap"></pre>
      </div>
    </div>

    <div class="text-center mt-8">
      <p class="text-xs text-zinc-400">Need bulk processing or custom fields?
        <a href="/" class="text-amber-700 hover:underline">Contact us</a>
      </p>
    </div>
  </main>
</div>

<script>
  'use strict';

  let apiKey = '';
  let keyInfo = null;
  let selectedFile = null;

  // ── INIT ──────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', () => {
    const stored = sessionStorage.getItem('dp_key');
    const param  = new URLSearchParams(location.search).get('key');
    if (param) {
      document.getElementById('keyInput').value = param;
      history.replaceState({}, '', '/app'); // remove key from URL immediately
    }
    if (stored) {
      apiKey = stored;
      verifyKey(false);
    }
  });

  // ── AUTH ──────────────────────────────────────────────────────────────
  async function verifyKey(showError = true) {
    const input = document.getElementById('keyInput').value.trim();
    if (input) apiKey = input;
    if (!apiKey) return;

    const btn = document.getElementById('verifyBtn');
    const errEl = document.getElementById('authError');
    btn.disabled = true;
    btn.textContent = 'Verifying…';
    errEl.classList.add('hidden');

    try {
      const res = await fetch('/verify', {
        method: 'POST',
        headers: { 'X-API-Key': apiKey }
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || 'Invalid API key.');
      }
      keyInfo = await res.json();
      sessionStorage.setItem('dp_key', apiKey);
      enterApp();
    } catch (err) {
      apiKey = '';
      sessionStorage.removeItem('dp_key');
      if (showError) {
        errEl.textContent = err.message;
        errEl.classList.remove('hidden');
      }
    } finally {
      btn.disabled = false;
      btn.textContent = 'Verify & Enter';
    }
  }

  function enterApp() {
    document.getElementById('authGate').classList.add('hidden');
    document.getElementById('mainApp').classList.remove('hidden');
    renderUsage();
    if (keyInfo && keyInfo.name) {
      const nameEl = document.getElementById('keyName');
      nameEl.textContent = keyInfo.name;
      nameEl.classList.remove('hidden');
    }
  }

  function renderUsage() {
    if (!keyInfo) return;
    const bar = document.getElementById('usageBar');
    bar.classList.remove('hidden');
    bar.classList.add('flex');
    const used  = keyInfo.usage_today  || 0;
    const limit = keyInfo.daily_limit  || 50;
    const pct   = Math.min(100, (used / limit) * 100);
    document.getElementById('usageText').textContent = `${used}/${limit}`;
    const fill = document.getElementById('usageFill');
    fill.style.width = pct + '%';
    fill.className = `h-full rounded-full transition-all duration-300 ${pct > 80 ? 'bg-red-400' : 'bg-amber-700'}`;
  }

  function logout() {
    sessionStorage.removeItem('dp_key');
    apiKey = '';
    keyInfo = null;
    location.reload();
  }

  // ── FILE HANDLING ─────────────────────────────────────────────────────
  function handleDrop(event) {
    event.preventDefault();
    document.getElementById('dropZone').classList.remove('border-zinc-400');
    const file = event.dataTransfer.files[0];
    if (file) fileSelected(file);
  }

  function fileSelected(file) {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      showExtractError('File too large. Maximum size is 10 MB.');
      return;
    }
    if (!/[.](pdf|png|jpe?g)$/i.test(file.name)) {
      showExtractError('Invalid file type. Please upload a PDF, PNG, or JPEG.');
      return;
    }
    selectedFile = file;
    document.getElementById('dropText').classList.add('hidden');
    const nameEl = document.getElementById('fileName');
    nameEl.textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
    nameEl.classList.remove('hidden');
    document.getElementById('extractBtn').disabled = false;
    document.getElementById('cancelBtn').classList.remove('hidden');
    document.getElementById('extractError').classList.add('hidden');
    document.getElementById('resultBox').classList.add('hidden');
  }

  function resetForm() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('dropText').classList.remove('hidden');
    document.getElementById('fileName').classList.add('hidden');
    document.getElementById('extractBtn').disabled = true;
    document.getElementById('cancelBtn').classList.add('hidden');
    document.getElementById('resultBox').classList.add('hidden');
    document.getElementById('extractError').classList.add('hidden');
  }

  function showExtractError(msg) {
    const el = document.getElementById('extractError');
    el.textContent = msg;
    el.classList.remove('hidden');
  }

  // ── EXTRACT ───────────────────────────────────────────────────────────
  async function extractData() {
    if (!selectedFile || !apiKey) return;

    const btn = document.getElementById('extractBtn');
    btn.disabled = true;
    btn.textContent = 'Analyzing…';
    document.getElementById('extractError').classList.add('hidden');
    document.getElementById('resultBox').classList.add('hidden');

    const fd = new FormData();
    fd.append('file', selectedFile);

    try {
      const res = await fetch('/extract', {
        method: 'POST',
        body: fd,
        headers: { 'X-API-Key': apiKey }
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Extraction failed.');

      document.getElementById('resultJson').textContent =
        JSON.stringify({ vendor: data.vendor, date: data.date, amount: data.amount }, null, 2);
      document.getElementById('resultBox').classList.remove('hidden');

      if (data.usage && keyInfo) {
        keyInfo.usage_today = data.usage.used;
        renderUsage();
      }
    } catch (err) {
      showExtractError(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Extract Data';
    }
  }

  // ── COPY ──────────────────────────────────────────────────────────────
  async function copyResult() {
    const text = document.getElementById('resultJson').textContent;
    const btn  = document.getElementById('copyBtn');
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = 'Copied!';
    } catch {
      btn.textContent = 'Failed';
    }
    setTimeout(() => { btn.textContent = 'Copy JSON'; }, 1500);
  }
</script>
</body></html>"""


# ── ADMIN PAGE (plain string — no f-prefix) ──────────────────────────────────────
ADMIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DocuParse AI — Admin</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<script>
  tailwind.config = {
    theme: { extend: { fontFamily: { sans: ['Inter','system-ui','sans-serif'] } } }
  };
</script>
</head>
<body class="bg-[#fafafa] font-sans text-zinc-900 antialiased min-h-screen">

<!-- LOGIN GATE -->
<div id="loginGate" class="min-h-screen flex items-center justify-center px-6">
  <div class="bg-white rounded-xl border border-black/5 p-8 w-full max-w-sm shadow-sm text-center">
    <h1 class="text-xl font-medium tracking-tight mb-1">Admin Access</h1>
    <p class="text-sm text-zinc-400 mb-6">Enter the admin password</p>
    <input id="pwdInput" type="password" placeholder="Password"
      class="w-full border border-black/10 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-zinc-400 mb-3"
      onkeydown="if(event.key==='Enter') unlock()">
    <button onclick="unlock()" class="w-full bg-zinc-900 text-white text-sm py-3 rounded-lg font-medium hover:bg-zinc-800 transition-colors">
      Unlock
    </button>
    <p id="loginErr" class="text-red-500 text-xs mt-3 hidden"></p>
  </div>
</div>

<!-- DASHBOARD -->
<div id="dashboard" class="hidden">
  <nav class="bg-white/95 backdrop-blur-sm border-b border-black/5">
    <div class="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span class="text-sm font-medium tracking-tight">Admin Panel</span>
        <span class="text-xs bg-zinc-100 text-zinc-500 px-2 py-0.5 rounded">DocuParse AI</span>
      </div>
      <a href="/" class="text-xs text-zinc-400 hover:text-zinc-900 transition-colors">← Back to site</a>
    </div>
  </nav>

  <!-- TABS -->
  <div class="border-b border-black/5 bg-white">
    <div class="max-w-5xl mx-auto px-6 flex gap-0">
      <button id="tabKeys" onclick="switchTab('Keys')"
        class="px-5 py-4 text-sm font-medium border-b-2 border-zinc-900 text-zinc-900 transition-colors">
        API Keys
      </button>
      <button id="tabExtractions" onclick="switchTab('Extractions')"
        class="px-5 py-4 text-sm font-medium text-zinc-400 hover:text-zinc-700 transition-colors">
        Extraction History
      </button>
    </div>
  </div>

  <main class="max-w-5xl mx-auto px-6 py-10">

  <!-- PANEL: KEYS -->
  <div id="panelKeys">

    <!-- CREATE KEY -->
    <div class="bg-white rounded-xl border border-black/5 p-6 mb-8">
      <h2 class="text-sm font-medium mb-4">Create New API Key</h2>
      <div class="flex flex-col sm:flex-row gap-3">
        <input id="newName" type="text" placeholder="Key name (e.g. Client A)"
          class="flex-1 border border-black/10 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-zinc-400">
        <input id="newLimit" type="number" value="50" min="1" max="10000"
          class="w-32 border border-black/10 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-zinc-400"
          title="Daily request limit">
        <button onclick="createKey()"
          class="bg-zinc-900 text-white text-sm px-6 py-2.5 rounded-lg font-medium hover:bg-zinc-800 whitespace-nowrap transition-colors">
          Create Key
        </button>
      </div>
      <div id="newKeyResult" class="hidden mt-4 bg-zinc-900 rounded-lg p-4">
        <p class="text-xs text-zinc-400 mb-1">Copy this key — it will never be shown again:</p>
        <div class="flex items-center gap-2">
          <code id="newKeyVal" class="text-sm text-green-400 font-mono break-all flex-1"></code>
          <button onclick="copyNewKey()" id="copyNewBtn"
            class="text-xs text-zinc-400 hover:text-white whitespace-nowrap transition-colors">Copy</button>
        </div>
      </div>
    </div>

    <!-- KEYS TABLE -->
    <div class="bg-white rounded-xl border border-black/5 overflow-hidden">
      <div class="px-6 py-4 border-b border-black/5 flex items-center justify-between">
        <h2 class="text-sm font-medium">API Keys</h2>
        <button onclick="loadKeys()" class="text-xs text-zinc-400 hover:text-zinc-900 transition-colors">↻ Refresh</button>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-black/5 text-left text-xs text-zinc-400 uppercase tracking-wider">
              <th class="px-6 py-3">Key (masked)</th>
              <th class="px-6 py-3">Name</th>
              <th class="px-6 py-3">Status</th>
              <th class="px-6 py-3">Usage today</th>
              <th class="px-6 py-3">Daily limit</th>
              <th class="px-6 py-3">Created</th>
              <th class="px-6 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody id="keysBody"></tbody>
        </table>
      </div>
      <div id="emptyState" class="hidden px-6 py-12 text-center text-sm text-zinc-400">No API keys found.</div>
    </div>
  </div> <!-- /panelKeys -->

  <!-- PANEL: EXTRACTIONS -->
  <div id="panelExtractions" class="hidden">

    <!-- STATS ROW -->
    <div class="grid grid-cols-2 sm:grid-cols-5 gap-4 mb-8">
      <div class="bg-white rounded-xl border border-black/5 p-4 text-center">
        <p class="text-xs text-zinc-400 mb-1">Total</p>
        <p id="statTotal" class="text-2xl font-light">—</p>
      </div>
      <div class="bg-white rounded-xl border border-black/5 p-4 text-center">
        <p class="text-xs text-zinc-400 mb-1">Success</p>
        <p id="statSuccess" class="text-2xl font-light text-green-600">—</p>
      </div>
      <div class="bg-white rounded-xl border border-black/5 p-4 text-center">
        <p class="text-xs text-zinc-400 mb-1">Errors</p>
        <p id="statErrors" class="text-2xl font-light text-red-500">—</p>
      </div>
      <div class="bg-white rounded-xl border border-black/5 p-4 text-center">
        <p class="text-xs text-zinc-400 mb-1">Avg time</p>
        <p id="statAvgMs" class="text-2xl font-light">—</p>
      </div>
      <div class="bg-white rounded-xl border border-black/5 p-4 text-center">
        <p class="text-xs text-zinc-400 mb-1">Clients</p>
        <p id="statKeys" class="text-2xl font-light">—</p>
      </div>
    </div>

    <!-- FILTERS -->
    <div class="bg-white rounded-xl border border-black/5 p-4 mb-4 flex flex-col sm:flex-row gap-3 items-end">
      <div class="flex-1">
        <label class="text-xs text-zinc-400 block mb-1">Filter by key name</label>
        <input id="exFilter" type="text" placeholder="e.g. Acme Corp"
          class="w-full border border-black/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-zinc-400">
      </div>
      <div>
        <label class="text-xs text-zinc-400 block mb-1">Status</label>
        <select id="exStatus" class="border border-black/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-zinc-400 bg-white">
          <option value="">All</option>
          <option value="ok">Success</option>
          <option value="error">Error</option>
        </select>
      </div>
      <button onclick="loadExtractions()"
        class="bg-zinc-900 text-white text-sm px-5 py-2 rounded-lg font-medium hover:bg-zinc-800 transition-colors">
        Search
      </button>
    </div>

    <!-- TABLE -->
    <div class="bg-white rounded-xl border border-black/5 overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-black/5 text-left text-xs text-zinc-400 uppercase tracking-wider bg-zinc-50">
              <th class="px-4 py-3">Time (UTC)</th>
              <th class="px-4 py-3">Key</th>
              <th class="px-4 py-3">File</th>
              <th class="px-4 py-3">Type</th>
              <th class="px-4 py-3">Vendor</th>
              <th class="px-4 py-3">Date</th>
              <th class="px-4 py-3">Amount</th>
              <th class="px-4 py-3">Status</th>
              <th class="px-4 py-3 text-right">Duration</th>
            </tr>
          </thead>
          <tbody id="exBody"></tbody>
        </table>
      </div>
      <div id="exEmpty" class="hidden px-6 py-12 text-center text-sm text-zinc-400">No extractions yet.</div>
      <!-- PAGINATION -->
      <div class="px-6 py-3 border-t border-black/5 flex items-center justify-between">
        <button id="exPrev" onclick="exPrevPage()" disabled
          class="text-xs text-zinc-400 hover:text-zinc-900 disabled:opacity-30 transition-colors">← Prev</button>
        <span id="exPageNum" class="text-xs text-zinc-400">Page 1</span>
        <button id="exNext" onclick="exNextPage()" disabled
          class="text-xs text-zinc-400 hover:text-zinc-900 disabled:opacity-30 transition-colors">Next →</button>
      </div>
    </div>

  </div> <!-- /panelExtractions -->

  </main>
</div>

<script>
  'use strict';

  let pwd = '';

  // ── UNLOCK ────────────────────────────────────────────────────────────
  async function unlock() {
    pwd = document.getElementById('pwdInput').value;
    const errEl = document.getElementById('loginErr');
    errEl.classList.add('hidden');

    try {
      const res = await fetch('/admin/api/keys', { headers: { 'X-Admin-Password': pwd } });
      if (!res.ok) throw new Error();
      document.getElementById('loginGate').classList.add('hidden');
      document.getElementById('dashboard').classList.remove('hidden');
      await loadKeys();
    } catch {
      errEl.textContent = 'Invalid password.';
      errEl.classList.remove('hidden');
    }
  }

  // ── LOAD KEYS ─────────────────────────────────────────────────────────
  async function loadKeys() {
    const res  = await fetch('/admin/api/keys', { headers: { 'X-Admin-Password': pwd } });
    const keys = await res.json();
    const body = document.getElementById('keysBody');
    const empty = document.getElementById('emptyState');

    if (!Array.isArray(keys) || keys.length === 0) {
      body.innerHTML = '';
      empty.classList.remove('hidden');
      return;
    }
    empty.classList.add('hidden');

    body.innerHTML = keys.map(k => {
      const statusBadge = k.active
        ? '<span class="inline-flex items-center gap-1 text-xs text-green-600"><span class="w-1.5 h-1.5 bg-green-500 rounded-full"></span>Active</span>'
        : '<span class="inline-flex items-center gap-1 text-xs text-zinc-400"><span class="w-1.5 h-1.5 bg-zinc-300 rounded-full"></span>Inactive</span>';

      const usagePct = Math.min(100, ((k.usage_today || 0) / (k.daily_limit || 50)) * 100);
      const barColor = usagePct > 80 ? 'bg-red-400' : 'bg-amber-600';

      return `<tr class="border-b border-black/5 hover:bg-zinc-50/50">
        <td class="px-6 py-3 font-mono text-xs text-zinc-500">${escHtml(k.key)}</td>
        <td class="px-6 py-3 text-sm">${escHtml(k.name || '—')}</td>
        <td class="px-6 py-3">${statusBadge}</td>
        <td class="px-6 py-3">
          <div class="flex items-center gap-2">
            <span class="font-mono text-xs">${k.usage_today || 0}/${k.daily_limit || 50}</span>
            <div class="w-14 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
              <div class="h-full ${barColor} rounded-full" style="width:${usagePct}%"></div>
            </div>
          </div>
        </td>
        <td class="px-6 py-3 font-mono text-xs">${k.daily_limit || 50}/day</td>
        <td class="px-6 py-3 text-xs text-zinc-400">${k.created_at ? k.created_at.split('T')[0] : '—'}</td>
        <td class="px-6 py-3 text-right space-x-2">
          <button onclick="toggleKey('${escAttr(k.key_full)}')"
            class="text-xs text-zinc-400 hover:text-zinc-900 transition-colors">
            ${k.active ? 'Deactivate' : 'Activate'}
          </button>
          <button onclick="resetUsage('${escAttr(k.key_full)}')"
            class="text-xs text-zinc-400 hover:text-zinc-900 transition-colors">Reset</button>
          <button onclick="deleteKey('${escAttr(k.key_full)}', '${escAttr(k.name || '')}')"
            class="text-xs text-red-400 hover:text-red-600 transition-colors">Delete</button>
        </td>
      </tr>`;
    }).join('');
  }

  // ── HELPERS ───────────────────────────────────────────────────────────
  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function escAttr(str) {
    return String(str).replace(/'/g, "\\'");
  }

  // ── CREATE KEY ────────────────────────────────────────────────────────
  async function createKey() {
    const name  = document.getElementById('newName').value.trim() || 'Unnamed';
    const limit = parseInt(document.getElementById('newLimit').value, 10) || 50;

    const res = await fetch('/admin/api/keys', {
      method: 'POST',
      headers: { 'X-Admin-Password': pwd, 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, daily_limit: limit })
    });
    if (!res.ok) { alert('Error creating key. Check admin password.'); return; }

    const data = await res.json();
    document.getElementById('newKeyVal').textContent = data.key;
    document.getElementById('newKeyResult').classList.remove('hidden');
    document.getElementById('newName').value = '';
    await loadKeys();
  }

  async function copyNewKey() {
    const text = document.getElementById('newKeyVal').textContent;
    const btn  = document.getElementById('copyNewBtn');
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = 'Copied!';
    } catch {
      btn.textContent = 'Failed';
    }
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  }

  // ── ACTIONS ───────────────────────────────────────────────────────────
  async function toggleKey(key)  { await _patch(key, 'toggle');  await loadKeys(); }
  async function resetUsage(key) { await _patch(key, 'reset');   await loadKeys(); }

  async function _patch(key, action) {
    await fetch(`/admin/api/keys/${encodeURIComponent(key)}`, {
      method: 'PATCH',
      headers: { 'X-Admin-Password': pwd, 'Content-Type': 'application/json' },
      body: JSON.stringify({ action })
    });
  }

  async function deleteKey(key, name) {
    if (!confirm(`Delete key "${name}"? This cannot be undone.`)) return;
    await fetch(`/admin/api/keys/${encodeURIComponent(key)}`, {
      method: 'DELETE',
      headers: { 'X-Admin-Password': pwd }
    });
    await loadKeys();
  }

  // ── EXTRACTIONS TAB ───────────────────────────────────────────────────
  let exPage = 0;
  const EX_PAGE_SIZE = 50;

  async function loadStats() {
    try {
      const res  = await fetch('/admin/api/stats', { headers: { 'X-Admin-Password': pwd } });
      const s    = await res.json();
      document.getElementById('statTotal').textContent   = s.total   ?? '—';
      document.getElementById('statSuccess').textContent = s.success ?? '—';
      document.getElementById('statErrors').textContent  = s.errors  ?? '—';
      document.getElementById('statAvgMs').textContent   = s.avg_ms  ? s.avg_ms + ' ms' : '—';
      document.getElementById('statKeys').textContent    = s.unique_keys ?? '—';
    } catch(e) { console.error('Stats load failed', e); }
  }

  async function loadExtractions(reset = true) {
    if (reset) exPage = 0;
    const filter = document.getElementById('exFilter').value.trim();
    const status = document.getElementById('exStatus').value;
    const params = new URLSearchParams({
      limit: EX_PAGE_SIZE,
      offset: exPage * EX_PAGE_SIZE,
      ...(filter ? { key_name: filter } : {}),
      ...(status  ? { status }          : {}),
    });
    const res  = await fetch(`/admin/api/extractions?${params}`, { headers: { 'X-Admin-Password': pwd } });
    const rows = await res.json();
    const tbody = document.getElementById('exBody');
    const empty = document.getElementById('exEmpty');

    if (!rows.length && exPage === 0) {
      tbody.innerHTML = '';
      empty.classList.remove('hidden');
      document.getElementById('exPrev').disabled = true;
      document.getElementById('exNext').disabled = true;
      return;
    }
    empty.classList.add('hidden');

    tbody.innerHTML = rows.map(r => {
      const statusBadge = r.status === 'ok'
        ? '<span class="text-xs text-green-600 font-medium">ok</span>'
        : `<span class="text-xs text-red-500 font-medium" title="${escHtml(r.error_msg || '')}">error</span>`;
      const ts = r.created_at ? r.created_at.replace('T',' ').replace('Z','') : '—';
      return `<tr class="border-b border-black/5 hover:bg-zinc-50/50 text-sm">
        <td class="px-4 py-2.5 text-xs text-zinc-400 whitespace-nowrap">${ts}</td>
        <td class="px-4 py-2.5 text-xs font-medium">${escHtml(r.key_name)}</td>
        <td class="px-4 py-2.5 text-xs text-zinc-500 max-w-[120px] truncate" title="${escHtml(r.filename)}">${escHtml(r.filename)}</td>
        <td class="px-4 py-2.5 text-xs uppercase text-zinc-400">${escHtml(r.file_type.split('/')[1] || r.file_type)}</td>
        <td class="px-4 py-2.5 text-xs">${escHtml(r.vendor || '—')}</td>
        <td class="px-4 py-2.5 text-xs text-zinc-500">${escHtml(r.date || '—')}</td>
        <td class="px-4 py-2.5 text-xs font-mono">${escHtml(r.amount || '—')}</td>
        <td class="px-4 py-2.5">${statusBadge}</td>
        <td class="px-4 py-2.5 text-xs text-zinc-400 text-right">${r.duration_ms != null ? r.duration_ms + ' ms' : '—'}</td>
      </tr>`;
    }).join('');

    document.getElementById('exPrev').disabled = exPage === 0;
    document.getElementById('exNext').disabled = rows.length < EX_PAGE_SIZE;
    document.getElementById('exPageNum').textContent = `Page ${exPage + 1}`;
  }

  function exNextPage() { exPage++; loadExtractions(false); }
  function exPrevPage() { if (exPage > 0) { exPage--; loadExtractions(false); } }

  function switchTab(tab) {
    ['tabKeys','tabExtractions'].forEach(id => {
      document.getElementById(id).classList.toggle(
        'border-b-2', id === 'tab' + tab
      );
      document.getElementById(id).classList.toggle(
        'border-zinc-900', id === 'tab' + tab
      );
      document.getElementById(id).classList.toggle(
        'text-zinc-900', id === 'tab' + tab
      );
      document.getElementById(id).classList.toggle(
        'text-zinc-400', id !== 'tab' + tab
      );
    });
    document.getElementById('panelKeys').classList.toggle('hidden', tab !== 'Keys');
    document.getElementById('panelExtractions').classList.toggle('hidden', tab !== 'Extractions');
    if (tab === 'Extractions') { loadStats(); loadExtractions(); }
  }
</script>
</body></html>"""


# ── ROUTES — PUBLIC ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def landing():
    return HTMLResponse(content=LANDING)


@app.get("/app", response_class=HTMLResponse)
async def app_page():
    return HTMLResponse(content=APP_PAGE)


@app.get("/admin", response_class=HTMLResponse)
async def admin_page_route():
    return HTMLResponse(content=ADMIN_PAGE)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llama_configured": bool(LLAMA_API_KEY),
        "admin_configured": bool(ADMIN_PASSWORD),
    }


# ── ROUTES — API ─────────────────────────────────────────────────────────────────

@app.post("/verify")
async def verify_key(request: Request, x_api_key: str = Header(None)):
    _ip_rate_limit(request)
    kd, _, _ = _validate_key(x_api_key, consume=False)
    return JSONResponse({
        "valid":       True,
        "name":        kd.get("name"),
        "usage_today": kd.get("usage_today", 0),
        "daily_limit": kd.get("daily_limit", 50),
    })


@app.post("/extract")
async def extract_file(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str   = Header(None),
):
    _ip_rate_limit(request)

    if not LLAMA_API_KEY:
        raise HTTPException(500, "Service not configured. Contact the administrator.")

    kd, hashed, _ = _validate_key(x_api_key, consume=True)
    _t_start = time.monotonic()

    # ── File validation & normalisation ──────────────────────────────
    filename  = file.filename or ""
    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {MAX_FILE_MB} MB limit.")
    if len(raw_bytes) < 8:
        raise HTTPException(400, "File is too small to be valid.")

    mime = _validate_file_bytes(filename, raw_bytes)
    original_mime = mime  # remember original for DB logging

    # ── Convert PDF → PNG (vision API only accepts images) ────────────
    # Groq / most vision APIs reject PDFs sent as image_url.
    # We render page 1 at 150 DPI and send the PNG instead.
    if mime == "application/pdf":
        logger.info("Converting PDF to PNG for vision API")
        raw_bytes = _pdf_first_page_to_png(raw_bytes, dpi=150)
        mime = "image/png"

    # ── Call AI provider ───────────────────────────────────────────────
    b64 = base64.b64encode(raw_bytes).decode()
    payload = {
        "model": LLAMA_MODEL,
        "temperature": 0.1,
        "max_tokens": 200,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This is an invoice image. Extract the invoice details. "
                        "Respond with ONLY a valid JSON object — no markdown, no explanation, no extra text. "
                        "Use exactly these three keys: "
                        "\"vendor\": the company or person who issued the invoice (string or null), "
                        "\"date\": the invoice date in ISO 8601 format YYYY-MM-DD (string or null), "
                        "\"amount\": the total amount due including currency symbol (string or null). "
                        "Example: {\"vendor\":\"Acme Corp\",\"date\":\"2025-01-15\",\"amount\":\"$2,450.00\"}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                        "detail": "high",
                    },
                },
            ],
        }],
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                LLAMA_API_URL,
                headers={
                    "Authorization": f"Bearer {LLAMA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(504, "AI provider timed out. Please try again.")
    except httpx.RequestError as exc:
        logger.error("httpx request error: %s", exc)
        raise HTTPException(502, "Could not reach AI provider.")

    if resp.status_code != 200:
        # Log the FULL Groq error so you can see exactly what went wrong
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text
        logger.error("Groq error %s: %s", resp.status_code, err_body)
        # Surface a useful message to the caller
        detail = "Unknown error"
        if isinstance(err_body, dict):
            detail = (err_body.get("error", {}) or {}).get("message", str(err_body))
        raise HTTPException(502, f"AI provider error: {detail}")

    try:
        resp_json = resp.json()
        raw_text  = resp_json["choices"][0]["message"]["content"]
    except json.JSONDecodeError:
        # The API returned non-JSON (e.g. HTML from a wrong URL)
        preview = resp.text[:300].replace("\n", " ")
        logger.error(
            "AI provider returned non-JSON (status %s). "
            "Check LLAMA_API_URL in .env. Response preview: %s",
            resp.status_code, preview,
        )
        raise HTTPException(
            502,
            "AI provider returned an unexpected response. "
            "Check LLAMA_API_URL in your .env — it must be "
            "https://api.groq.com/openai/v1/chat/completions"
        )
    except (KeyError, IndexError) as exc:
        logger.error("AI response missing expected keys (%s). Full body: %s", exc, resp.text[:500])
        raise HTTPException(502, f"AI response was missing expected field: {exc}")

    # Strip optional markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        raise HTTPException(502, "AI returned invalid JSON. Please try again.")

    if not isinstance(data, dict):
        raise HTTPException(502, "AI response was not a JSON object.")

    vendor = data.get("vendor")
    inv_date = data.get("date")
    amount = data.get("amount")

    await db_log(
        key_name    = kd.get("name", "unknown"),
        filename    = filename,
        file_type   = original_mime,
        vendor      = vendor,
        date        = inv_date,
        amount      = amount,
        status      = "ok",
        error_msg   = None,
        duration_ms = int((time.monotonic() - _t_start) * 1000),
    )

    return JSONResponse({
        "vendor": vendor,
        "date":   inv_date,
        "amount": amount,
        "usage": {
            "used":  kd.get("usage_today", 0),
            "limit": kd.get("daily_limit", 50),
        },
    })


# ── ROUTES — ADMIN API ────────────────────────────────────────────────────────────

@app.get("/admin/api/keys")
async def list_keys(request: Request, x_admin_password: str = Header(None)):
    _ip_rate_limit(request)
    _admin_auth(x_admin_password)
    keys = _load()
    return [
        {
            "key":         k[:6] + "…" + k[-4:],   # masked display
            "key_full":    k,                        # hashed ID used for actions
            **{field: v for field, v in meta.items()
               if field not in ("usage_today", "last_reset", "created_at",
                                "active", "daily_limit", "name")},
            "name":        meta.get("name", ""),
            "active":      meta.get("active", True),
            "daily_limit": meta.get("daily_limit", 50),
            "usage_today": meta.get("usage_today", 0),
            "created_at":  meta.get("created_at", ""),
        }
        for k, meta in keys.items()
    ]


@app.post("/admin/api/keys")
async def create_key_admin(
    request: Request,
    x_admin_password: str = Header(None),
):
    _ip_rate_limit(request)
    _admin_auth(x_admin_password)

    body = await request.json()
    raw_key = secrets.token_urlsafe(40)
    hashed  = _hash_secret(raw_key)
    keys    = _load()
    keys[hashed] = dict(
        name        = str(body.get("name", "Unnamed"))[:80],
        active      = True,
        daily_limit = max(1, min(int(body.get("daily_limit", 50)), 10_000)),
        usage_today = 0,
        last_reset  = str(date.today()),
        created_at  = datetime.now().isoformat(),
    )
    _save(keys)
    # Return the raw key ONCE — it is never stored
    return JSONResponse({
        "key":         raw_key,
        "name":        keys[hashed]["name"],
        "daily_limit": keys[hashed]["daily_limit"],
    })


@app.patch("/admin/api/keys/{key}")
async def update_key_admin(
    key: str,
    request: Request,
    x_admin_password: str = Header(None),
):
    _ip_rate_limit(request)
    _admin_auth(x_admin_password)

    body = await request.json()
    keys = _load()
    if key not in keys:
        raise HTTPException(404, "Key not found.")

    action = body.get("action")
    if action == "toggle":
        keys[key]["active"] = not keys[key].get("active", True)
    elif action == "reset":
        keys[key]["usage_today"] = 0
    elif action == "limit":
        keys[key]["daily_limit"] = max(1, min(int(body.get("daily_limit", 50)), 10_000))
    else:
        raise HTTPException(400, f"Unknown action: {action!r}")

    _save(keys)
    return {"status": "ok"}


@app.delete("/admin/api/keys/{key}")
async def delete_key_admin(
    key: str,
    request: Request,
    x_admin_password: str = Header(None),
):
    _ip_rate_limit(request)
    _admin_auth(x_admin_password)

    keys = _load()
    if key not in keys:
        raise HTTPException(404, "Key not found.")
    del keys[key]
    _save(keys)
    return {"status": "deleted"}


# ── ADMIN — EXTRACTION HISTORY ────────────────────────────────────────────────

@app.get("/admin/api/extractions")
async def get_extractions(
    request:           Request,
    x_admin_password:  str = Header(None),
    limit:             int = 100,
    offset:            int = 0,
    key_name:          str = None,
    status:            str = None,
):
    _ip_rate_limit(request)
    _admin_auth(x_admin_password)
    rows = await db_get_extractions(
        limit=min(limit, 500),
        offset=offset,
        key_name=key_name or None,
        status=status or None,
    )
    return rows


@app.get("/admin/api/stats")
async def get_stats(
    request:          Request,
    x_admin_password: str = Header(None),
):
    _ip_rate_limit(request)
    _admin_auth(x_admin_password)
    return await db_stats()