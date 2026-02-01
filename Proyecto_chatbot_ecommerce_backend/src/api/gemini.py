from __future__ import annotations
import os
import requests

BASE = "https://generativelanguage.googleapis.com/v1beta"

def list_models():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return []
    url = f"{BASE}/models"
    r = requests.get(url, params={"key": api_key}, timeout=20)
    if not r.ok:
        return []
    data = r.json()
    models = data.get("models") or []
    return [m.get("name","") for m in models if isinstance(m, dict) and m.get("name")]

def _extract_text(data: dict) -> str:
    cand = (data.get("candidates") or [{}])[0]
    content = cand.get("content", {})
    parts = content.get("parts") or []
    texts = [p.get("text","") for p in parts if isinstance(p, dict)]
    out = "\n".join([t for t in texts if t]).strip()
    return out or "No pude generar respuesta (Gemini devolvió vacío)."

def gemini_generate(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    # Try model from env first, then fallbacks that are commonly available.
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()

    if not api_key:
        return "Falta configurar GEMINI_API_KEY en tu .env"

    def call(model_name: str) -> requests.Response:
        url = f"{BASE}/models/{model_name}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 512},
        }
        return requests.post(url, params={"key": api_key}, json=payload, timeout=30)

    # 1) Primary model
    try_models = [model, "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    tried = []
    for m in try_models:
        if m in tried:
            continue
        tried.append(m)
        try:
            r = call(m)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            return _extract_text(r.json())
        except Exception as e:
            # If not 404, return error
            if isinstance(e, requests.HTTPError):
                # if still 404 continue, else report
                if e.response is not None and e.response.status_code == 404:
                    continue
            return f"Error llamando a Gemini: {e}"

    # If all 404, guide user with available models
    available = list_models()
    if available:
        # show a few names
        sample = ", ".join(available[:10])
        return ("Error llamando a Gemini: 404 (modelo no disponible). "
                "Usa un modelo válido de ListModels. Ejemplos: " + sample)
    return "Error llamando a Gemini: 404 (modelo no disponible). Revisa tu GEMINI_MODEL y que tu API key tenga acceso."
