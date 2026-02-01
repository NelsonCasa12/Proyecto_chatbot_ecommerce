import os
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

# OJO: ya NO vamos a bloquear por scope, pero puedes dejarlo importado si quieres.
# from src.utils.safety import is_ecommerce_query, safe_refusal

from src.utils.memory import MemoryStore
from src.utils.rerank import rerank
from src.api.gemini import gemini_generate

load_dotenv()

TEXT_INDEX_PATH = "data/faiss_text.index"
TEXT_META_PATH = "data/meta_text.npy"
IMAGE_INDEX_PATH = "data/faiss_image.index"
IMAGE_META_PATH = "data/meta_image.npy"

RETRIEVE_K = int(os.getenv("RETRIEVE_K", "25"))
RERANK_K = int(os.getenv("RERANK_K", "10"))

app = FastAPI(title="Ecommerce Multimodal RAG API", version="1.1")

# CORS
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def _to_tensor(output):
    # Compatibilidad con distintos outputs del modelo
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    return output


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def embed_text(q: str) -> np.ndarray:
    inputs = processor(text=[q], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        out = _to_tensor(out)
    v = out.detach().cpu().numpy().astype("float32")
    return _norm(v)


def embed_image_bytes(b: bytes) -> np.ndarray:
    img = Image.open(BytesIO(b)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        out = _to_tensor(out)
    v = out.detach().cpu().numpy().astype("float32")
    return _norm(v)


def load_index(path: str):
    return faiss.read_index(path) if os.path.exists(path) else None


def load_meta(path: str):
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


text_index = load_index(TEXT_INDEX_PATH)
text_meta = load_meta(TEXT_META_PATH)

img_index = load_index(IMAGE_INDEX_PATH)
img_meta = load_meta(IMAGE_META_PATH)

memory = MemoryStore()


# -------------------- DEBUG ENDPOINTS (GEMINI) --------------------

@app.get("/debug/models")
def debug_models():
    """Lista modelos disponibles para tu API key (Gemini)."""
    from src.api.gemini import list_models
    return {"models": list_models()}


class DebugGemini(BaseModel):
    prompt: str = "Di hola"


@app.post("/debug/gemini")
def debug_gemini(body: DebugGemini):
    return {"answer": gemini_generate(body.prompt)}


# -------------------- HEALTH --------------------

@app.get("/health")
def health():
    return {
        "text_index": None if text_index is None else int(text_index.ntotal),
        "image_index": None if img_index is None else int(img_index.ntotal),
    }


# -------------------- SEARCH --------------------

class TextSearch(BaseModel):
    query: str
    k: int = 10


@app.post("/search/text")
def search_text(body: TextSearch):
    if text_index is None or text_meta is None:
        return {"error": "Text index no existe. Ejecuta: python -m src.build_index"}
    v = embed_text(body.query)
    D, I = text_index.search(v, body.k)

    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        rec = dict(text_meta[idx])
        rec["score"] = float(score)
        results.append(rec)

    return {"mode": "text", "results": results}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), k: int = 10):
    if img_index is None or img_meta is None:
        return {"error": "Image index no existe. Ejecuta: python -m src.build_index"}
    b = await file.read()
    v = embed_image_bytes(b)
    D, I = img_index.search(v, k)

    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        rec = dict(img_meta[idx])
        rec["score"] = float(score)
        results.append(rec)

    return {"mode": "image", "results": results}


# -------------------- CHAT (RAG + RERANK + MEMORY) --------------------

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    mode: str = "text"
    k: int = 10


def build_context(items):
    lines = []
    for i, it in enumerate(items, start=1):
        name = str(it.get("name", "")).strip()
        brand = str(it.get("brand", "")).strip()
        cats = str(it.get("categories", "")).strip()
        rating = it.get("rating", None)
        if rating is None:
            rating = it.get("avg_rating", None)
        review = str(it.get("review_text", "")).strip()
        if len(review) > 220:
            review = review[:220] + "..."
        lines.append(
            f"[{i}] name={name} | brand={brand} | categories={cats} | rating={rating} | evidence={review}"
        )
    return "\n".join(lines)


def boost_query_for_retrieval(q: str) -> str:
    """
    CLIP-text es semántico, pero a veces necesita ayuda lexical por categoría.
    Esto NO inventa nada; solo mejora el retrieval.
    """
    base = (q or "").strip()
    ql = base.lower()

    # Shoes/boots
    if any(x in ql for x in ["zapato", "zapat", "shoe", "shoes", "bota", "botas", "boot", "boots"]):
        base += " shoes boots footwear black white red blue"

    # Headphones / audio
    if any(x in ql for x in ["audif", "audí", "headphone", "headphones", "earbud", "earbuds", "headset", "auricular"]):
        base += " headphones earbuds headset audio bluetooth wireless"

    # Tablets / Kindle / Fire
    if any(x in ql for x in ["tablet", "kindle", "fire", "ebook", "e-book"]):
        base += " tablet kindle fire amazon"

    # Speakers / Echo
    if any(x in ql for x in ["parlante", "speaker", "speakers", "echo", "alexa"]):
        base += " speaker echo alexa smart speaker audio"

    return base


@app.post("/chat")
def chat(body: ChatRequest):
    if text_index is None or text_meta is None:
        return {"error": "Text index no existe. Ejecuta: python -m src.build_index"}

    # ✅ Ya NO bloqueamos por "scope". Siempre intentamos responder con el corpus.
    # Esto permite "zapatos negros", "xapatos", etc.

    # Update memory
    memory.update_query(body.session_id, body.message)
    memory.update_prefs_from_text(body.session_id, body.message)
    prefs = memory.get(body.session_id).prefs

    # Retrieval (con boosting)
    boosted = boost_query_for_retrieval(body.message)
    v = embed_text(boosted)

    retrieve_k = max(RETRIEVE_K, body.k)
    D, I = text_index.search(v, retrieve_k)

    before = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        rec = dict(text_meta[idx])
        rec["score"] = float(score)
        before.append(rec)

    # Re-ranking (usa query original)
    after = rerank(body.message, before, top_k=max(RERANK_K, body.k), prefs=prefs)

    memory.update_results(body.session_id, after)

    # RAG prompt (mejorado: si no coincide, dar alternativas más cercanas)
    context = build_context(after[:min(8, len(after))])
    mem_line = f"User preferences (memory): {prefs}" if prefs else "User preferences (memory): none"

    prompt = (
        "Eres un asistente de compras de e-commerce.\n"
        "Usa SOLO la EVIDENCIA proporcionada; NO inventes productos, marcas ni características.\n"
        "Si la evidencia NO coincide exactamente con lo que pide el usuario, dilo claramente y ofrece las alternativas más cercanas basadas en la evidencia.\n"
        "Responde en español. Recomienda 2 a 4 opciones con razones cortas.\n\n"
        f"{mem_line}\n"
        f"Consulta del usuario: {body.message}\n\n"
        f"EVIDENCIA:\n{context}\n"
    )

    answer = gemini_generate(prompt)

    return {
        "session_id": body.session_id,
        "answer": answer,
        "prefs": prefs,
        "results_before": before[:body.k],
        "results_after": after[:body.k],
    }


@app.post("/chat/image")
async def chat_image(
    session_id: str = "default",
    file: UploadFile = File(...),
    message: str = "",
    k: int = 10,
):
    if img_index is None or img_meta is None:
        return {"error": "Image index no existe. Ejecuta: python -m src.build_index"}

    user_text = message or "buscar productos similares a la imagen"

    memory.update_query(session_id, user_text)
    memory.update_prefs_from_text(session_id, user_text)
    prefs = memory.get(session_id).prefs

    b = await file.read()
    v = embed_image_bytes(b)

    retrieve_k = max(RETRIEVE_K, k)
    D, I = img_index.search(v, retrieve_k)

    before = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        rec = dict(img_meta[idx])
        rec["score"] = float(score)
        before.append(rec)

    after = rerank(user_text, before, top_k=max(RERANK_K, k), prefs=prefs)
    memory.update_results(session_id, after)

    context = build_context(after[:min(8, len(after))])
    mem_line = f"User preferences (memory): {prefs}" if prefs else "User preferences (memory): none"

    prompt = (
        "Eres un asistente de compras de e-commerce.\n"
        "Usa SOLO la EVIDENCIA proporcionada; NO inventes productos, marcas ni características.\n"
        "Si la evidencia NO coincide exactamente con lo que pide el usuario, dilo claramente y ofrece las alternativas más cercanas basadas en la evidencia.\n"
        "Responde en español. Recomienda 2 a 4 opciones con razones cortas.\n\n"
        f"{mem_line}\n"
        f"Consulta del usuario: {user_text}\n\n"
        f"EVIDENCIA:\n{context}\n"
    )

    answer = gemini_generate(prompt)

    return {
        "session_id": session_id,
        "answer": answer,
        "prefs": prefs,
        "results_before": before[:k],
        "results_after": after[:k],
    }
