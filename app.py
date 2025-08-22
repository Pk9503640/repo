from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import os
import json
import httpx
import asyncio
import re

# -----------------------
# App & CORS
# -----------------------
app = FastAPI(title="RJC Backend", version="1.0.0")

# Allow all by default; override via ALLOW_ORIGINS env (comma-separated)
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
origins: List[str] = [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Config / ENV
# -----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")  # safe default
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")        # v1beta
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
RETRIES = int(os.getenv("RETRIES", "2"))

# -----------------------
# Utilities
# -----------------------
async def stream_response(text: str):
    # nice small chunks for UI streaming
    for i in range(0, len(text), 60):
        yield text[i:i+60]
        await asyncio.sleep(0.02)

async def http_post_json(client: httpx.AsyncClient, url: str, headers: dict, payload: dict):
    # single request helper with JSON
    return await client.post(url, headers=headers, json=payload)

async def with_retries(coro_factory, retries: int = 2, delay: float = 0.6):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(delay * (attempt + 1))
    raise last_exc

# -----------------------
# Provider Calls (robust)
# -----------------------
async def call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return "[Groq Error] Missing GROQ_API_KEY in environment."

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        async def do_call(path: str):
            resp = await http_post_json(
                client,
                path,
                headers,
                {
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful, concise assistant."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            data = resp.json()
            if resp.status_code >= 400:
                # try to surface server-provided message
                msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
                raise RuntimeError(f"HTTP {resp.status_code} - {msg or data}")
            try:
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                raise RuntimeError(f"Groq parse error: {e} | Raw: {data}")

        # Groq recently uses /openai/v1; some older examples use /v1
        try:
            return await with_retries(lambda: do_call("https://api.groq.com/openai/v1/chat/completions"), retries=RETRIES)
        except Exception:
            # fallback path
            return await with_retries(lambda: do_call("https://api.groq.com/v1/chat/completions"), retries=RETRIES)

async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "[Gemini Error] Missing GEMINI_API_KEY in environment."

    # Gemini expects API key in query param; v1beta models endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await with_retries(lambda: client.post(url, json=payload), retries=RETRIES)
        data = resp.json()
        if resp.status_code >= 400:
            # surface error message if present
            msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
            return f"[Gemini Error] HTTP {resp.status_code} - {msg or data}"
        if not isinstance(data, dict) or "candidates" not in data:
            return f"[Gemini Error] Unexpected response: {data}"
        try:
            # Prefer the first candidate text part
            cand0 = data["candidates"][0]
            parts = cand0.get("content", {}).get("parts", [])
            texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            text = "".join(texts).strip()
            if not text:
                return f"[Gemini Error] Empty text in candidates: {data}"
            return text
        except Exception as e:
            return f"[Gemini Parse Error] {e} | Raw: {data}"

# -----------------------
# Routing & Model Selection
# -----------------------
MATH_HINTS = [
    "solve", "equation", "math", "physics", "theorem", "integral", "derivative", "prove", "proof",
    "numerical", "tensor", "quantum", "matrix", "algebra", "limit", "series", "vector"
]


def choose_model(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in MATH_HINTS):
        return "gemini"  # better for structured/math in many cases
    return "groq"


@app.get("/")
async def root():
    return JSONResponse({
        "name": "RJC Backend",
        "status": "ok",
        "models": {"groq": GROQ_MODEL, "gemini": GEMINI_MODEL},
        "docs": "/docs",
    })


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/solve")
async def solve(
    prompt: str = Form(...),
    model: str = Form("auto"),
    history: str = Form("[]"),
    image: Optional[UploadFile] = None,
):
    # Parse history defensively (not used in provider calls yet, but kept for forward compat)
    try:
        _ = json.loads(history)
    except Exception:
        _ = []

    # Validate prompt fast
    if not prompt or not prompt.strip():
        return StreamingResponse(stream_response("[Error] Empty prompt."), media_type="text/plain")

    # Explicit selection or auto-choice
    chosen = model
    if model == "auto":
        chosen = choose_model(prompt)

    # Call providers with fallback
    text = ""
    try:
        if chosen == "gemini":
            text = await call_gemini(prompt)
            if text.startswith("[Gemini Error]") or text.startswith("[Gemini Parse Error]") or not text.strip():
                # fallback to Groq
                alt = await call_groq(prompt)
                text = text + "\n\n[Fallback → Groq]\n" + alt if alt else text
        elif chosen == "groq" or chosen == "1.0":
            text = await call_groq(prompt)
            if text.startswith("[Groq Error]") or "parse error" in text.lower() or not text.strip():
                # fallback to Gemini
                alt = await call_gemini(prompt)
                text = text + "\n\n[Fallback → Gemini]\n" + alt if alt else text
        elif chosen == "mathica":  # alias to gemini from your UI
            text = await call_gemini(prompt)
            if text.startswith("[Gemini Error]") or text.startswith("[Gemini Parse Error]") or not text.strip():
                alt = await call_groq(prompt)
                text = text + "\n\n[Fallback → Groq]\n" + alt if alt else text
        else:
            # Unknown flag → try auto
            primary = choose_model(prompt)
            text = await (call_gemini(prompt) if primary == "gemini" else call_groq(prompt))
    except Exception as e:
        text = f"[Unhandled Error] {e}"

    if not text:
        text = "[Error] Provider returned empty response."

    return StreamingResponse(stream_response(text), media_type="text/plain")
