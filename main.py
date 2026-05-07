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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import fitz  # pymupdf — converts PDF pages to PNG for vision API
import httpx
from dotenv import load_dotenv

# Load .env FIRST — before any os.getenv() calls anywhere in this file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── SENTRY (error monitoring) ─────────────────────────────────────────────────
# Set SENTRY_DSN in .env to enable — free at sentry.io
def _init_sentry() -> None:
    dsn = os.getenv("SENTRY_DSN", "")
    if not dsn:
        return
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=dsn, traces_sample_rate=0.1)
        logger.info("Sentry enabled")
    except ImportError:
        logger.warning("sentry-sdk not installed — add it to requirements.txt")

_init_sentry()

# ── POSTHOG (product analytics) ──────────────────────────────────────────────
_posthog: Any | None = None  # posthog.Posthog instance, lazily imported

def _init_posthog() -> None:
    """Lazy-import posthog so a missing package never crashes the app."""
    global _posthog
    api_key = os.getenv("POSTHOG_KEY", "")
    host    = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")
    if not api_key:
        return
    try:
        from posthog import Posthog
        _posthog = Posthog(api_key=api_key, host=host)
        logger.info("PostHog enabled (host: %s)", host)
    except ImportError:
        logger.warning("posthog package not installed — add posthog to requirements.txt")

_init_posthog()

# ── DEMO MODE rate tracker (in-memory, per IP, resets daily) ─────────────────
_demo_hits: dict[str, tuple[int, str]] = {}  # ip -> (count, date)

def _check_demo_limit(request: Request) -> None:
    ip    = request.client.host if request.client else "unknown"
    today = str(date.today())
    count, last_date = _demo_hits.get(ip, (0, today))
    if last_date != today:
        count = 0
    if count >= DEMO_DAILY_LIMIT:
        raise HTTPException(429, f"Demo limit ({DEMO_DAILY_LIMIT} extractions/day) reached. Get a free API key to continue.")
    _demo_hits[ip] = (count + 1, today)


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
                datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z"),
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

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

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

KEYS_FILE    = Path(os.getenv("KEYS_FILE_PATH", "/tmp/api_keys.json"))  # /tmp persists across restarts on Render
POSTHOG_KEY  = os.getenv("POSTHOG_KEY", "")   # get free at posthog.com
SENTRY_DSN   = os.getenv("SENTRY_DSN",  "")   # get free at sentry.io
DEMO_DAILY_LIMIT = int(os.getenv("DEMO_DAILY_LIMIT", "5"))  # free demo extractions per IP/day

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


def _check_rate_limit(kd: dict) -> None:  # hashed_key/keys removed — unused
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
    _check_rate_limit(kd)
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
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DocuParse AI — Invoice Automation for Ecommerce Brands</title>
<meta name="description" content="Automatically extract supplier invoice data for ecommerce operations. Built for Shopify stores, Amazon sellers, and ops teams who process invoices daily.">
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script>
  tailwind.config = {{
    theme: {{
      extend: {{
        colors: {{ accent: '#2563eb', 'accent-h': '#1d4ed8', muted: '#f8fafc' }},
        fontFamily: {{ sans: ['Inter','system-ui','sans-serif'] }}
      }}
    }}
  }};
</script>
<style>
  html {{ scroll-behavior: smooth }}
  .gradient-text {{ background: linear-gradient(135deg,#2563eb,#7c3aed); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }}
  @keyframes fadeUp {{ from {{ opacity:0;transform:translateY(16px) }} to {{ opacity:1;transform:translateY(0) }} }}
  .fade-up {{ animation: fadeUp 0.5s ease forwards }}
</style>
</head>
<body class="bg-white font-sans text-slate-900 antialiased">

<!-- NAV -->
<nav class="fixed top-0 w-full bg-white/95 backdrop-blur-sm border-b border-slate-100 z-50">
  <div class="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
    <div class="flex items-center gap-2">
      <div class="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
      </div>
      <span class="font-semibold text-slate-900">DocuParse AI</span>
    </div>
    <div class="hidden md:flex items-center gap-8">
      <a href="#how" class="text-sm text-slate-500 hover:text-slate-900 transition-colors">How it works</a>
      <a href="#who" class="text-sm text-slate-500 hover:text-slate-900 transition-colors">Who it's for</a>
      <a href="#pricing" class="text-sm text-slate-500 hover:text-slate-900 transition-colors">Pricing</a>
      <a href="/app?demo=1" class="text-sm bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors font-medium">Try Free Demo →</a>
    </div>
    <a href="/app?demo=1" class="md:hidden text-sm bg-blue-600 text-white px-4 py-2 rounded-lg font-medium">Try Demo</a>
  </div>
</nav>

<!-- HERO -->
<section class="pt-32 pb-20 px-6">
  <div class="max-w-4xl mx-auto text-center fade-up">
    <div class="inline-flex items-center gap-2 bg-blue-50 text-blue-700 text-xs font-semibold px-3 py-1.5 rounded-full mb-6 border border-blue-100">
      <span class="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>
      Built for ecommerce operators
    </div>
    <h1 class="text-4xl md:text-6xl font-bold text-slate-900 leading-tight mb-6">
      Stop manually entering<br>
      <span class="gradient-text">supplier invoice data</span>
    </h1>
    <p class="text-lg md:text-xl text-slate-500 max-w-2xl mx-auto mb-10 leading-relaxed">
      Upload any supplier invoice and instantly get structured data — vendor, date, amount — ready for your spreadsheet, ERP, or accounting software. No templates. No setup.
    </p>
    <div class="flex flex-col sm:flex-row items-center justify-center gap-3 mb-6">
      <a href="/app?demo=1"
         class="w-full sm:w-auto bg-blue-600 text-white px-8 py-3.5 rounded-xl text-sm font-semibold hover:bg-blue-700 transition-colors shadow-lg shadow-blue-100 inline-flex items-center justify-center gap-2">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
        </svg>
        Try with your invoice — free
      </a>
      <a href="mailto:{CONTACT_EMAIL}?subject=DocuParse%20AI%20Access"
         class="w-full sm:w-auto border border-slate-200 text-slate-600 px-8 py-3.5 rounded-xl text-sm font-semibold hover:border-slate-300 hover:bg-slate-50 transition-colors inline-flex items-center justify-center gap-2">
        Get API access →
      </a>
    </div>
    <p class="text-xs text-slate-400">No signup required for demo · 5 free extractions per day</p>
  </div>
</section>

<!-- LIVE DEMO PREVIEW (static example) -->
<section class="pb-20 px-6">
  <div class="max-w-3xl mx-auto">
    <div class="bg-slate-900 rounded-2xl overflow-hidden shadow-2xl">
      <div class="flex items-center gap-2 px-4 py-3 bg-slate-800 border-b border-slate-700">
        <div class="w-3 h-3 rounded-full bg-red-400"></div>
        <div class="w-3 h-3 rounded-full bg-yellow-400"></div>
        <div class="w-3 h-3 rounded-full bg-green-400"></div>
        <span class="ml-2 text-xs text-slate-400 font-mono">invoice_amazon_supplier_jan2025.pdf → extracted in 1.8s</span>
      </div>
      <div class="p-6 font-mono text-sm">
        <div class="text-slate-400 mb-3">// Result</div>
        <div class="space-y-1">
          <div><span class="text-blue-400">"vendor"</span><span class="text-slate-400">: </span><span class="text-green-400">"Shenzhen Global Electronics Co."</span><span class="text-slate-500">,</span></div>
          <div><span class="text-blue-400">"date"</span><span class="text-slate-400">: </span><span class="text-green-400">"2025-01-15"</span><span class="text-slate-500">,</span></div>
          <div><span class="text-blue-400">"amount"</span><span class="text-slate-400">: </span><span class="text-green-400">"$4,820.00"</span><span class="text-slate-500">,</span></div>
          <div><span class="text-blue-400">"confidence"</span><span class="text-slate-400">: </span><span class="text-yellow-400">97</span></div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- WHO IT'S FOR -->
<section id="who" class="py-20 px-6 bg-slate-50">
  <div class="max-w-5xl mx-auto">
    <div class="text-center mb-14">
      <p class="text-xs font-semibold text-blue-600 uppercase tracking-widest mb-3">Who this is for</p>
      <h2 class="text-3xl md:text-4xl font-bold text-slate-900">Built specifically for ecommerce teams</h2>
    </div>
    <div class="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
      <div class="bg-white rounded-xl border border-slate-100 p-6 hover:shadow-md transition-shadow">
        <div class="text-3xl mb-3">🛒</div>
        <h3 class="font-semibold text-slate-900 mb-2">Shopify Stores</h3>
        <p class="text-sm text-slate-500 leading-relaxed">Processing 10–100+ supplier invoices per month and tired of manual data entry into your books.</p>
      </div>
      <div class="bg-white rounded-xl border border-slate-100 p-6 hover:shadow-md transition-shadow">
        <div class="text-3xl mb-3">📦</div>
        <h3 class="font-semibold text-slate-900 mb-2">Amazon Sellers</h3>
        <p class="text-sm text-slate-500 leading-relaxed">Managing multiple suppliers across countries with invoices in different formats and languages.</p>
      </div>
      <div class="bg-white rounded-xl border border-slate-100 p-6 hover:shadow-md transition-shadow">
        <div class="text-3xl mb-3">🏭</div>
        <h3 class="font-semibold text-slate-900 mb-2">Ecommerce Ops Teams</h3>
        <p class="text-sm text-slate-500 leading-relaxed">Operations managers who need invoice data flowing into Xero, QuickBooks, or custom ERPs automatically.</p>
      </div>
      <div class="bg-white rounded-xl border border-slate-100 p-6 hover:shadow-md transition-shadow">
        <div class="text-3xl mb-3">📊</div>
        <h3 class="font-semibold text-slate-900 mb-2">Ecommerce Bookkeepers</h3>
        <p class="text-sm text-slate-500 leading-relaxed">Bookkeeping freelancers who serve ecommerce clients and need to process invoices faster without errors.</p>
      </div>
    </div>
  </div>
</section>

<!-- HOW IT WORKS -->
<section id="how" class="py-20 px-6">
  <div class="max-w-4xl mx-auto">
    <div class="text-center mb-14">
      <p class="text-xs font-semibold text-blue-600 uppercase tracking-widest mb-3">How it works</p>
      <h2 class="text-3xl md:text-4xl font-bold text-slate-900">Three steps, under 5 seconds</h2>
    </div>
    <div class="grid md:grid-cols-3 gap-8">
      <div class="text-center">
        <div class="w-14 h-14 rounded-2xl bg-blue-600 text-white flex items-center justify-center text-xl font-bold mx-auto mb-5">1</div>
        <h3 class="font-semibold text-slate-900 mb-2">Upload your invoice</h3>
        <p class="text-sm text-slate-500 leading-relaxed">Drag and drop any PDF, PNG, or JPEG invoice. Any vendor, any country, any layout.</p>
      </div>
      <div class="text-center">
        <div class="w-14 h-14 rounded-2xl bg-blue-600 text-white flex items-center justify-center text-xl font-bold mx-auto mb-5">2</div>
        <h3 class="font-semibold text-slate-900 mb-2">AI reads it instantly</h3>
        <p class="text-sm text-slate-500 leading-relaxed">Vision AI scans the document and identifies vendor name, invoice date, and total amount due.</p>
      </div>
      <div class="text-center">
        <div class="w-14 h-14 rounded-2xl bg-blue-600 text-white flex items-center justify-center text-xl font-bold mx-auto mb-5">3</div>
        <h3 class="font-semibold text-slate-900 mb-2">Get structured data</h3>
        <p class="text-sm text-slate-500 leading-relaxed">Receive clean JSON with a confidence score. Paste into your spreadsheet or pipe directly into your ERP via API.</p>
      </div>
    </div>
  </div>
</section>

<!-- FEATURES -->
<section class="py-20 px-6 bg-slate-50">
  <div class="max-w-5xl mx-auto">
    <div class="text-center mb-14">
      <p class="text-xs font-semibold text-blue-600 uppercase tracking-widest mb-3">Features</p>
      <h2 class="text-3xl md:text-4xl font-bold text-slate-900">Everything you need, nothing you don't</h2>
    </div>
    <div class="grid md:grid-cols-3 gap-6">
      <div class="bg-white rounded-xl border border-slate-100 p-6">
        <div class="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center mb-4">
          <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
        </div>
        <h3 class="font-semibold text-slate-900 mb-2">Under 3 seconds</h3>
        <p class="text-sm text-slate-500">Faster than you can open the invoice manually. No queues, no batching.</p>
      </div>
      <div class="bg-white rounded-xl border border-slate-100 p-6">
        <div class="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center mb-4">
          <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        </div>
        <h3 class="font-semibold text-slate-900 mb-2">Confidence score</h3>
        <p class="text-sm text-slate-500">Every result includes a 0–100 confidence rating so you know when to double-check.</p>
      </div>
      <div class="bg-white rounded-xl border border-slate-100 p-6">
        <div class="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center mb-4">
          <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
        </div>
        <h3 class="font-semibold text-slate-900 mb-2">No templates needed</h3>
        <p class="text-sm text-slate-500">Works with any invoice format from any supplier worldwide. Zero setup or training.</p>
      </div>
      <div class="bg-white rounded-xl border border-slate-100 p-6">
        <div class="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center mb-4">
          <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
        </div>
        <h3 class="font-semibold text-slate-900 mb-2">API-ready JSON</h3>
        <p class="text-sm text-slate-500">Clean structured output you can pipe directly into Xero, QuickBooks, Sheets, or your own code.</p>
      </div>
      <div class="bg-white rounded-xl border border-slate-100 p-6">
        <div class="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center mb-4">
          <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg>
        </div>
        <h3 class="font-semibold text-slate-900 mb-2">Secure by design</h3>
        <p class="text-sm text-slate-500">API-key auth, per-key rate limits, HTTPS only. Documents processed and immediately discarded.</p>
      </div>
      <div class="bg-white rounded-xl border border-slate-100 p-6">
        <div class="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center mb-4">
          <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
        </div>
        <h3 class="font-semibold text-slate-900 mb-2">PDF, PNG &amp; JPEG</h3>
        <p class="text-sm text-slate-500">Upload scanned images or digital PDFs — both work. Multi-page PDFs automatically use the first page.</p>
      </div>
    </div>
  </div>
</section>

<!-- PRICING -->
<section id="pricing" class="py-20 px-6">
  <div class="max-w-4xl mx-auto">
    <div class="text-center mb-14">
      <p class="text-xs font-semibold text-blue-600 uppercase tracking-widest mb-3">Pricing</p>
      <h2 class="text-3xl md:text-4xl font-bold text-slate-900">Simple, usage-based pricing</h2>
      <p class="text-slate-500 mt-3 text-sm">All plans include confidence scoring, API access, and extraction history.</p>
    </div>
    <div class="grid md:grid-cols-3 gap-6">
      <div class="rounded-2xl border border-slate-200 p-7">
        <p class="text-sm font-semibold text-slate-500 mb-1">Starter</p>
        <p class="text-3xl font-bold text-slate-900 mb-1">$19<span class="text-base font-normal text-slate-400">/mo</span></p>
        <p class="text-xs text-slate-400 mb-6">~900 extractions/month</p>
        <ul class="space-y-2.5 text-sm text-slate-600 mb-7">
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>30 extractions/day</li>
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>API access</li>
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>Web app included</li>
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>Email support</li>
        </ul>
        <a href="mailto:{CONTACT_EMAIL}?subject=DocuParse%20Starter%20Plan" class="block text-center border border-slate-200 text-slate-700 py-2.5 rounded-lg text-sm font-medium hover:bg-slate-50 transition-colors">Get started</a>
      </div>
      <div class="rounded-2xl border-2 border-blue-600 p-7 relative shadow-lg shadow-blue-50">
        <div class="absolute -top-3 left-1/2 -translate-x-1/2 bg-blue-600 text-white text-xs font-semibold px-3 py-1 rounded-full">Most popular</div>
        <p class="text-sm font-semibold text-blue-600 mb-1">Growth</p>
        <p class="text-3xl font-bold text-slate-900 mb-1">$49<span class="text-base font-normal text-slate-400">/mo</span></p>
        <p class="text-xs text-slate-400 mb-6">~3,000 extractions/month</p>
        <ul class="space-y-2.5 text-sm text-slate-600 mb-7">
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>100 extractions/day</li>
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>API access</li>
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>Extraction history dashboard</li>
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>Priority support</li>
        </ul>
        <a href="mailto:{CONTACT_EMAIL}?subject=DocuParse%20Growth%20Plan" class="block text-center bg-blue-600 text-white py-2.5 rounded-lg text-sm font-semibold hover:bg-blue-700 transition-colors">Get started</a>
      </div>
      <div class="rounded-2xl border border-slate-200 p-7">
        <p class="text-sm font-semibold text-slate-500 mb-1">Business</p>
        <p class="text-3xl font-bold text-slate-900 mb-1">$99<span class="text-base font-normal text-slate-400">/mo</span></p>
        <p class="text-xs text-slate-400 mb-6">~9,000 extractions/month</p>
        <ul class="space-y-2.5 text-sm text-slate-600 mb-7">
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>300 extractions/day</li>
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>API access + webhooks</li>
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>Custom field extraction</li>
          <li class="flex items-center gap-2"><svg class="w-4 h-4 text-green-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>Dedicated support</li>
        </ul>
        <a href="mailto:{CONTACT_EMAIL}?subject=DocuParse%20Business%20Plan" class="block text-center border border-slate-200 text-slate-700 py-2.5 rounded-lg text-sm font-medium hover:bg-slate-50 transition-colors">Get started</a>
      </div>
    </div>
  </div>
</section>

<!-- CTA -->
<section class="py-20 px-6 bg-slate-900">
  <div class="max-w-2xl mx-auto text-center">
    <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">Try it on your invoice right now</h2>
    <p class="text-slate-400 mb-8">No signup. No credit card. Upload any invoice and see the result in seconds.</p>
    <a href="/app?demo=1"
       class="inline-flex items-center gap-2 bg-blue-600 text-white px-10 py-4 rounded-xl text-sm font-semibold hover:bg-blue-700 transition-colors shadow-lg">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
      </svg>
      Start free demo →
    </a>
    <p class="text-slate-500 text-xs mt-4">Need more? <a href="mailto:{CONTACT_EMAIL}?subject=DocuParse%20Access" class="text-blue-400 hover:text-blue-300">Email for API access</a></p>
  </div>
</section>

<!-- FOOTER -->
<footer class="py-10 px-6 border-t border-slate-100">
  <div class="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
    <div class="flex items-center gap-2">
      <div class="w-6 h-6 bg-blue-600 rounded-md flex items-center justify-center">
        <svg class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
      </div>
      <span class="text-sm font-semibold text-slate-700">DocuParse AI</span>
      <span class="text-xs text-slate-400 ml-2">Invoice automation for ecommerce</span>
    </div>
    <div class="flex items-center gap-6">
      <a href="mailto:{CONTACT_EMAIL}" class="text-sm text-slate-400 hover:text-slate-700 transition-colors">Contact</a>
      <a href="/app" class="text-sm text-slate-400 hover:text-slate-700 transition-colors">App</a>
      <span class="text-xs text-slate-300">© 2025 DocuParse AI</span>
    </div>
  </div>
</footer>
</body></html>"""

# ── APP PAGE (plain string — no f-prefix) ─────────────────────────────────────
APP_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DocuParse AI — Invoice Extractor</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<script>
  tailwind.config = {
    theme: { extend: { fontFamily: { sans: ['Inter','system-ui','sans-serif'] } } }
  };
</script>
</head>
<body class="bg-slate-50 font-sans text-slate-900 antialiased min-h-screen">

<!-- AUTH GATE -->
<div id="authGate" class="min-h-screen flex items-center justify-center px-6">
  <div class="bg-white rounded-2xl border border-slate-100 shadow-sm p-8 w-full max-w-md">
    <!-- Demo banner -->
    <div id="demoBanner" class="hidden bg-blue-50 border border-blue-100 rounded-xl p-4 mb-6 text-center">
      <p class="text-sm font-semibold text-blue-700 mb-1">🎉 Free Demo Mode</p>
      <p class="text-xs text-blue-500">Upload any invoice — no signup needed. 5 free extractions per day.</p>
    </div>
    <div class="text-center mb-6">
      <div class="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center mx-auto mb-3">
        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
      </div>
      <h1 class="text-lg font-semibold tracking-tight mb-1">DocuParse AI</h1>
      <p id="authSubtitle" class="text-sm text-slate-400">Enter your API key to continue</p>
    </div>
    <!-- Demo quick-enter -->
    <div id="demoQuickEnter" class="hidden">
      <button onclick="startDemo()"
        class="w-full bg-blue-600 text-white text-sm py-3 rounded-xl font-semibold hover:bg-blue-700 transition-colors mb-3">
        Try Demo — Upload Your Invoice
      </button>
      <div class="relative flex items-center gap-3 my-4">
        <div class="flex-1 h-px bg-slate-100"></div>
        <span class="text-xs text-slate-400">or use API key</span>
        <div class="flex-1 h-px bg-slate-100"></div>
      </div>
    </div>
    <div class="space-y-3">
      <input id="keyInput" type="password" placeholder="API key..."
        autocomplete="off" spellcheck="false"
        class="w-full border border-slate-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-blue-400 transition-colors font-mono"
        onkeydown="if(event.key==='Enter') verifyKey()">
      <button onclick="verifyKey()" id="verifyBtn"
        class="w-full bg-slate-900 text-white text-sm py-3 rounded-xl font-semibold hover:bg-slate-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
        Verify &amp; Enter
      </button>
    </div>
    <p id="authError" class="text-red-500 text-xs mt-3 text-center hidden"></p>
    <p class="text-xs text-slate-400 text-center mt-5">Need a key? <a href="/" class="text-blue-600 hover:underline">Request access</a></p>
  </div>
</div>

<!-- MAIN APP -->
<div id="mainApp" class="hidden">
  <nav class="bg-white border-b border-slate-100 sticky top-0 z-10">
    <div class="max-w-2xl mx-auto px-6 h-14 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <div class="w-6 h-6 bg-blue-600 rounded-md flex items-center justify-center">
          <svg class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
        </div>
        <a href="/" class="text-sm font-semibold text-slate-800">DocuParse AI</a>
        <span id="demoPill" class="hidden text-xs bg-blue-100 text-blue-700 font-semibold px-2 py-0.5 rounded-full">DEMO</span>
      </div>
      <div class="flex items-center gap-4">
        <div id="usageBar" class="hidden items-center gap-2 text-xs text-slate-400">
          <span id="usageText"></span>
          <div class="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div id="usageFill" class="h-full bg-blue-500 rounded-full transition-all duration-300"></div>
          </div>
        </div>
        <span id="keyName" class="text-xs text-slate-400 hidden"></span>
        <button onclick="logout()" class="text-xs text-slate-400 hover:text-slate-900 transition-colors">
          <span id="logoutLabel">Logout</span>
        </button>
      </div>
    </div>
  </nav>

  <!-- Demo upgrade banner -->
  <div id="upgradeBar" class="hidden bg-blue-600 text-white text-xs text-center py-2 px-4">
    🎯 You're in demo mode — <a href="/" class="underline font-semibold">get an API key</a> for unlimited access and history tracking.
  </div>

  <main class="max-w-2xl mx-auto px-6 py-10">
    <div class="bg-white rounded-2xl border border-slate-100 shadow-sm p-8">
      <h2 class="text-base font-semibold mb-0.5">Invoice Extractor</h2>
      <p class="text-sm text-slate-400 mb-6">Upload a PDF, PNG, or JPEG — get structured data instantly.</p>

      <div id="dropZone"
        class="border-2 border-dashed border-slate-200 rounded-xl p-10 text-center mb-5 hover:border-blue-300 hover:bg-blue-50/30 transition-all cursor-pointer"
        onclick="document.getElementById('fileInput').click()"
        ondragover="event.preventDefault(); this.classList.add('border-blue-400','bg-blue-50')"
        ondragleave="this.classList.remove('border-blue-400','bg-blue-50')"
        ondrop="handleDrop(event)">
        <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg" class="hidden" onchange="fileSelected(this.files[0])">
        <div id="dropText">
          <svg class="w-9 h-9 mx-auto mb-3 text-slate-300" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
          </svg>
          <p class="text-sm text-slate-500 font-medium">Click or drag invoice here</p>
          <p class="text-xs text-slate-300 mt-1">PDF, PNG, JPEG · max 10 MB</p>
        </div>
        <div id="filePreview" class="hidden">
          <svg class="w-9 h-9 mx-auto mb-2 text-blue-400" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
          </svg>
          <p id="fileName" class="text-sm text-slate-700 font-medium"></p>
          <p id="fileSize" class="text-xs text-slate-400 mt-0.5"></p>
        </div>
      </div>

      <div class="flex gap-2.5">
        <button id="extractBtn" onclick="extractData()" disabled
          class="flex-1 bg-blue-600 text-white text-sm py-3 rounded-xl font-semibold hover:bg-blue-700 transition-colors disabled:bg-slate-100 disabled:text-slate-400 disabled:cursor-not-allowed">
          Extract Invoice Data
        </button>
        <button id="cancelBtn" onclick="resetForm()"
          class="hidden px-4 py-3 border border-slate-200 rounded-xl text-sm text-slate-500 hover:bg-slate-50 transition-colors">
          Clear
        </button>
      </div>
      <p id="extractError" class="text-red-500 text-xs mt-3 hidden"></p>

      <!-- RESULT -->
      <div id="resultBox" class="hidden mt-6">
        <!-- Confidence badge -->
        <div id="confidenceRow" class="flex items-center justify-between mb-4 p-3 bg-slate-50 rounded-xl border border-slate-100">
          <span class="text-xs text-slate-500 font-medium">Extraction confidence</span>
          <div class="flex items-center gap-2">
            <div class="w-24 h-2 bg-slate-200 rounded-full overflow-hidden">
              <div id="confBar" class="h-full rounded-full transition-all duration-500"></div>
            </div>
            <span id="confLabel" class="text-xs font-bold w-10 text-right"></span>
          </div>
        </div>
        <!-- Fields -->
        <div class="grid grid-cols-1 gap-3 mb-4">
          <div class="flex items-center justify-between p-3.5 bg-slate-50 rounded-xl border border-slate-100">
            <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Vendor</span>
            <span id="resVendor" class="text-sm font-medium text-slate-800 text-right max-w-[200px]"></span>
          </div>
          <div class="flex items-center justify-between p-3.5 bg-slate-50 rounded-xl border border-slate-100">
            <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Date</span>
            <span id="resDate" class="text-sm font-medium text-slate-800"></span>
          </div>
          <div class="flex items-center justify-between p-3.5 bg-slate-50 rounded-xl border border-slate-100">
            <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Amount</span>
            <span id="resAmount" class="text-sm font-semibold text-slate-900"></span>
          </div>
        </div>
        <!-- Raw JSON toggle -->
        <div class="flex items-center justify-between mb-2">
          <button onclick="toggleJson()" class="text-xs text-slate-400 hover:text-slate-700 transition-colors">Show raw JSON ↓</button>
          <button id="copyBtn" onclick="copyResult()" class="text-xs text-blue-600 hover:text-blue-700 font-medium transition-colors">Copy JSON</button>
        </div>
        <pre id="resultJson" class="hidden bg-slate-900 text-green-400 p-4 rounded-xl text-xs font-mono overflow-x-auto leading-relaxed whitespace-pre-wrap"></pre>
        <!-- Demo CTA -->
        <div id="demoCta" class="hidden mt-4 bg-blue-50 border border-blue-100 rounded-xl p-4 text-center">
          <p class="text-sm font-semibold text-blue-800 mb-1">Like what you see?</p>
          <p class="text-xs text-blue-600 mb-3">Get an API key for unlimited extractions, history tracking, and direct API access.</p>
          <a href="/" class="inline-block bg-blue-600 text-white text-xs font-semibold px-5 py-2 rounded-lg hover:bg-blue-700 transition-colors">Get API Access →</a>
        </div>
      </div>
    </div>
  </main>
</div>

<script>
  'use strict';

  let apiKey       = '';
  let keyInfo      = null;
  let selectedFile = null;
  let isDemoMode   = false;
  let lastResult   = null;

  // ── INIT ──────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(location.search);
    const isDemo = params.get('demo') === '1';
    const stored = sessionStorage.getItem('dp_key');
    const param  = params.get('key');

    if (param) {
      document.getElementById('keyInput').value = param;
      history.replaceState({}, '', '/app');
    }

    if (isDemo) {
      document.getElementById('demoBanner').classList.remove('hidden');
      document.getElementById('demoQuickEnter').classList.remove('hidden');
      document.getElementById('authSubtitle').textContent = 'Demo mode — no signup needed';
      history.replaceState({}, '', '/app');
    }

    if (stored) {
      apiKey = stored;
      verifyKey(false);
    }
  });

  function startDemo() {
    isDemoMode = true;
    document.getElementById('authGate').classList.add('hidden');
    document.getElementById('mainApp').classList.remove('hidden');
    document.getElementById('demoPill').classList.remove('hidden');
    document.getElementById('upgradeBar').classList.remove('hidden');
    document.getElementById('logoutLabel').textContent = 'Exit Demo';
    const usageBar = document.getElementById('usageBar');
    usageBar.classList.remove('hidden');
    usageBar.classList.add('flex');
    document.getElementById('usageText').textContent = '5 demo/day';
    document.getElementById('usageFill').style.width = '0%';
  }

  // ── AUTH ──────────────────────────────────────────────────────────────
  async function verifyKey(showError = true) {
    const input = document.getElementById('keyInput').value.trim();
    if (input) apiKey = input;
    if (!apiKey) return;

    const btn   = document.getElementById('verifyBtn');
    const errEl = document.getElementById('authError');
    btn.disabled = true;
    btn.textContent = 'Verifying…';
    errEl.classList.add('hidden');

    try {
      const res = await fetch('/verify', { method: 'POST', headers: { 'X-API-Key': apiKey } });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || 'Invalid API key.');
      }
      keyInfo = await res.json();
      sessionStorage.setItem('dp_key', apiKey);
      isDemoMode = false;
      enterApp();
    } catch (err) {
      apiKey = '';
      sessionStorage.removeItem('dp_key');
      if (showError) { errEl.textContent = err.message; errEl.classList.remove('hidden'); }
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
      const el = document.getElementById('keyName');
      el.textContent = keyInfo.name;
      el.classList.remove('hidden');
    }
  }

  function renderUsage() {
    if (!keyInfo) return;
    const bar  = document.getElementById('usageBar');
    bar.classList.remove('hidden'); bar.classList.add('flex');
    const used  = keyInfo.usage_today || 0;
    const limit = keyInfo.daily_limit  || 50;
    const pct   = Math.min(100, (used / limit) * 100);
    document.getElementById('usageText').textContent = `${used}/${limit}`;
    const fill = document.getElementById('usageFill');
    fill.style.width = pct + '%';
    fill.className = `h-full rounded-full transition-all duration-300 ${pct > 80 ? 'bg-red-400' : 'bg-blue-500'}`;
  }

  function logout() {
    sessionStorage.removeItem('dp_key');
    apiKey = ''; keyInfo = null; isDemoMode = false;
    location.href = '/';
  }

  // ── FILE HANDLING ─────────────────────────────────────────────────────
  function handleDrop(e) {
    e.preventDefault();
    document.getElementById('dropZone').classList.remove('border-blue-400','bg-blue-50');
    if (e.dataTransfer.files[0]) fileSelected(e.dataTransfer.files[0]);
  }

  function fileSelected(file) {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) { showErr('File too large. Max 10 MB.'); return; }
    if (!/[.](pdf|png|jpe?g)$/i.test(file.name)) { showErr('Use PDF, PNG, or JPEG.'); return; }
    selectedFile = file;
    document.getElementById('dropText').classList.add('hidden');
    document.getElementById('filePreview').classList.remove('hidden');
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = (file.size / 1024).toFixed(0) + ' KB';
    document.getElementById('extractBtn').disabled = false;
    document.getElementById('cancelBtn').classList.remove('hidden');
    document.getElementById('extractError').classList.add('hidden');
    document.getElementById('resultBox').classList.add('hidden');
  }

  function resetForm() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('dropText').classList.remove('hidden');
    document.getElementById('filePreview').classList.add('hidden');
    document.getElementById('extractBtn').disabled = true;
    document.getElementById('cancelBtn').classList.add('hidden');
    document.getElementById('resultBox').classList.add('hidden');
    document.getElementById('extractError').classList.add('hidden');
  }

  function showErr(msg) {
    const el = document.getElementById('extractError');
    el.textContent = msg; el.classList.remove('hidden');
  }

  // ── EXTRACT ───────────────────────────────────────────────────────────
  async function extractData() {
    if (!selectedFile) return;
    if (!isDemoMode && !apiKey) return;

    const btn = document.getElementById('extractBtn');
    btn.disabled = true; btn.textContent = 'Analyzing…';
    document.getElementById('extractError').classList.add('hidden');
    document.getElementById('resultBox').classList.add('hidden');

    const fd = new FormData();
    fd.append('file', selectedFile);
    const endpoint = isDemoMode ? '/demo' : '/extract';
    const headers  = isDemoMode ? {} : { 'X-API-Key': apiKey };

    try {
      const res  = await fetch(endpoint, { method: 'POST', body: fd, headers });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Extraction failed.');
      lastResult = data;
      renderResult(data);
      if (!isDemoMode && data.usage && keyInfo) {
        keyInfo.usage_today = data.usage.used;
        renderUsage();
      }
      if (isDemoMode && data.demo_remaining !== undefined) {
        document.getElementById('usageText').textContent = `${data.demo_remaining} demo left`;
        const pct = Math.max(0, (data.demo_remaining / 5) * 100);
        document.getElementById('usageFill').style.width = pct + '%';
      }
    } catch (err) {
      showErr(err.message);
    } finally {
      btn.disabled = false; btn.textContent = 'Extract Invoice Data';
    }
  }

  // ── RENDER RESULT ─────────────────────────────────────────────────────
  function renderResult(data) {
    document.getElementById('resVendor').textContent = data.vendor || '—';
    document.getElementById('resDate').textContent   = data.date   || '—';
    document.getElementById('resAmount').textContent = data.amount || '—';

    const conf    = data.confidence ?? 0;
    const confBar = document.getElementById('confBar');
    const confLbl = document.getElementById('confLabel');
    confBar.style.width = conf + '%';
    if (conf >= 85)      { confBar.className = 'h-full rounded-full bg-green-500 transition-all duration-500'; confLbl.className = 'text-xs font-bold w-10 text-right text-green-600'; }
    else if (conf >= 60) { confBar.className = 'h-full rounded-full bg-yellow-400 transition-all duration-500'; confLbl.className = 'text-xs font-bold w-10 text-right text-yellow-600'; }
    else                 { confBar.className = 'h-full rounded-full bg-red-400 transition-all duration-500'; confLbl.className = 'text-xs font-bold w-10 text-right text-red-500'; }
    confLbl.textContent = conf + '%';

    document.getElementById('resultJson').textContent =
      JSON.stringify({ vendor: data.vendor, date: data.date, amount: data.amount, confidence: data.confidence }, null, 2);

    if (isDemoMode) document.getElementById('demoCta').classList.remove('hidden');
    document.getElementById('resultBox').classList.remove('hidden');
  }

  function toggleJson() {
    const el  = document.getElementById('resultJson');
    const btn = event.target;
    const hidden = el.classList.toggle('hidden');
    btn.textContent = hidden ? 'Show raw JSON ↓' : 'Hide raw JSON ↑';
  }

  async function copyResult() {
    const btn = document.getElementById('copyBtn');
    try {
      await navigator.clipboard.writeText(document.getElementById('resultJson').textContent);
      btn.textContent = 'Copied!';
    } catch { btn.textContent = 'Failed'; }
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
        "max_tokens": 250,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This is an invoice image. Extract the invoice details. "
                        "Respond with ONLY a valid JSON object — no markdown, no explanation, no extra text. "
                        "Use exactly these four keys: "
                        "\"vendor\": the company or person who issued the invoice (string or null), "
                        "\"date\": the invoice date in ISO 8601 format YYYY-MM-DD (string or null), "
                        "\"amount\": the total amount due including currency symbol (string or null), "
                        "\"confidence\": integer 0-100 reflecting how clearly readable the invoice was "
                        "(100=perfectly clear, 80+=good, 60-79=some fields uncertain, below 60=poor quality). "
                        "Example: {\"vendor\":\"Acme Corp\",\"date\":\"2025-01-15\",\"amount\":\"$2,450.00\",\"confidence\":95}"
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
        await db_log(key_name=kd.get("name","unknown"), filename=filename,
                     file_type=original_mime, vendor=None, date=None, amount=None,
                     status="error", error_msg="AI provider timed out",
                     duration_ms=int((time.monotonic()-_t_start)*1000))
        raise HTTPException(504, "AI provider timed out. Please try again.")
    except httpx.RequestError as exc:
        logger.error("httpx request error: %s", exc)
        await db_log(key_name=kd.get("name","unknown"), filename=filename,
                     file_type=original_mime, vendor=None, date=None, amount=None,
                     status="error", error_msg=f"Network error: {exc}",
                     duration_ms=int((time.monotonic()-_t_start)*1000))
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
        await db_log(key_name=kd.get("name","unknown"), filename=filename,
                     file_type=original_mime, vendor=None, date=None, amount=None,
                     status="error", error_msg=f"Groq {resp.status_code}: {detail}",
                     duration_ms=int((time.monotonic()-_t_start)*1000))
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
        await db_log(key_name=kd.get("name","unknown"), filename=filename,
                     file_type=original_mime, vendor=None, date=None, amount=None,
                     status="error", error_msg="AI returned invalid JSON",
                     duration_ms=int((time.monotonic()-_t_start)*1000))
        raise HTTPException(502, "AI returned invalid JSON. Please try again.")

    if not isinstance(data, dict):
        await db_log(key_name=kd.get("name","unknown"), filename=filename,
                     file_type=original_mime, vendor=None, date=None, amount=None,
                     status="error", error_msg="AI response was not a JSON object",
                     duration_ms=int((time.monotonic()-_t_start)*1000))
        raise HTTPException(502, "AI response was not a JSON object.")

    vendor     = data.get("vendor")
    inv_date   = data.get("date")
    amount     = data.get("amount")
    raw_conf   = data.get("confidence")
    # Ensure confidence is a valid int 0-100; fall back to field-presence heuristic
    try:
        confidence = max(0, min(100, int(raw_conf))) if raw_conf is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is None:
        filled = sum(1 for v in [vendor, inv_date, amount] if v)
        confidence = [0, 60, 80, 95][filled]

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
        "vendor":     vendor,
        "date":       inv_date,
        "amount":     amount,
        "confidence": confidence,
        "usage": {
            "used":  kd.get("usage_today", 0),
            "limit": kd.get("daily_limit", 50),
        },
    })



# ── DEMO ENDPOINT (no API key needed, IP rate-limited) ───────────────────────
@app.post("/demo")
async def demo_extract(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Free demo — 5 extractions per IP per day, no signup needed.
    Returns same payload as /extract but with demo=True flag.
    """
    _ip_rate_limit(request)
    _check_global_rpm()          # guard Groq RPM across all endpoints including demo
    _check_demo_limit(request)

    if not LLAMA_API_KEY:
        raise HTTPException(500, "Service not configured.")

    _t_start  = time.monotonic()
    filename  = file.filename or "demo_upload"
    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {MAX_FILE_MB} MB limit.")
    if len(raw_bytes) < 8:
        raise HTTPException(400, "File is too small to be valid.")

    mime = _validate_file_bytes(filename, raw_bytes)
    if mime == "application/pdf":
        raw_bytes = _pdf_first_page_to_png(raw_bytes, dpi=150)
        mime = "image/png"

    b64 = base64.b64encode(raw_bytes).decode()
    payload = {
        "model": LLAMA_MODEL,
        "temperature": 0.1,
        "max_tokens": 250,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This is an invoice image. Extract the invoice details. "
                        "Respond with ONLY a valid JSON object — no markdown, no explanation. "
                        "Keys: \"vendor\" (string or null), \"date\" (ISO 8601 or null), "
                        "\"amount\" (with currency symbol or null), "
                        "\"confidence\" (int 0-100 how clearly readable the invoice was). "
                        "Example: {\"vendor\":\"Acme Corp\",\"date\":\"2025-01-15\",\"amount\":\"$2,450.00\",\"confidence\":95}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
            ],
        }],
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                LLAMA_API_URL,
                headers={"Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(504, "AI provider timed out.")
    except httpx.RequestError:
        raise HTTPException(502, "Could not reach AI provider.")

    if resp.status_code != 200:
        raise HTTPException(502, "AI provider error.")

    try:
        raw_text = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError):
        raise HTTPException(502, "Unexpected AI response.")

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        raise HTTPException(502, "AI returned invalid JSON.")

    vendor   = data.get("vendor")
    inv_date = data.get("date")
    amount   = data.get("amount")
    try:
        confidence = max(0, min(100, int(data.get("confidence", 0))))
    except (TypeError, ValueError):
        filled = sum(1 for v in [vendor, inv_date, amount] if v)
        confidence = [0, 60, 80, 95][filled]

    await db_log(
        key_name    = "__demo__",
        filename    = filename,
        file_type   = mime,
        vendor      = vendor,
        date        = inv_date,
        amount      = amount,
        status      = "ok",
        error_msg   = None,
        duration_ms = int((time.monotonic() - _t_start) * 1000),
    )

    ip = request.client.host if request.client else "unknown"
    demo_used, _ = _demo_hits.get(ip, (0, str(date.today())))

    return JSONResponse({
        "vendor":     vendor,
        "date":       inv_date,
        "amount":     amount,
        "confidence": confidence,
        "demo":       True,
        "demo_remaining": max(0, DEMO_DAILY_LIMIT - demo_used),
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