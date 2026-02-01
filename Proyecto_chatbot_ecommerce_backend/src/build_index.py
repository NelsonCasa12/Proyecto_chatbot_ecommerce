import os
import numpy as np
import pandas as pd
import faiss
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

load_dotenv()

DATA_REVIEWS = "data/corpus_reviews.parquet"

MAX_INDEX_ITEMS = int(os.getenv("MAX_INDEX_ITEMS", "30000"))  # 0 = todo
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "3000"))            # 0 = todo

TEXT_INDEX_PATH = "data/faiss_text.index"
TEXT_META_PATH = "data/meta_text.npy"

IMAGE_INDEX_PATH = "data/faiss_image.index"
IMAGE_META_PATH = "data/meta_image.npy"

device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def _to_tensor(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    return output

def _norm(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-12)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

def embed_text_batch(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        out = _to_tensor(out)
    emb = out.detach().cpu().numpy()
    emb = _norm(emb).astype("float32")
    return emb

def embed_image_url(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            out = _to_tensor(out)
        emb = out.detach().cpu().numpy()[0]
        emb = _norm(emb).astype("float32")
        return emb
    except:
        return None

def main():
    if not os.path.exists(DATA_REVIEWS):
        raise FileNotFoundError("No existe data/corpus_reviews.parquet. Ejecuta: python -m src.prepare_data")

    # TEXT INDEX
    df_text = pd.read_parquet(DATA_REVIEWS)
    if MAX_INDEX_ITEMS > 0:
        df_text = df_text.head(MAX_INDEX_ITEMS)

    texts = df_text["text_for_embed"].fillna("").astype(str).tolist()
    print("Docs para indexar (texto):", len(texts))

    batch = 128
    text_embs = []
    for i in tqdm(range(0, len(texts), batch), desc="Embeddings texto"):
        text_embs.append(embed_text_batch(texts[i:i+batch]))
    text_embs = np.vstack(text_embs).astype("float32")

    text_index = faiss.IndexFlatIP(text_embs.shape[1])
    text_index.add(text_embs)
    faiss.write_index(text_index, TEXT_INDEX_PATH)
    np.save(TEXT_META_PATH, df_text.drop(columns=["text_for_embed"]).to_dict(orient="records"), allow_pickle=True)
    print("Text index creado:", text_index.ntotal, "->", TEXT_INDEX_PATH)

    # IMAGE INDEX (from full corpus, dedup by URL)
    df_full = pd.read_parquet(DATA_REVIEWS)
    df_img = df_full[df_full["image_url"].notna()].copy()
    df_img = df_img[df_img["image_url"].astype(str).str.startswith("http")]
    df_img = df_img.drop_duplicates(subset=["image_url"]).reset_index(drop=True)

    if MAX_IMAGES > 0:
        df_img = df_img.sample(n=min(MAX_IMAGES, len(df_img)), random_state=42).reset_index(drop=True)

    print("Imágenes únicas candidatas para indexar:", len(df_img))

    img_embs = []
    kept_meta = []
    for rec in tqdm(df_img.to_dict(orient="records"), desc="Embeddings imagen"):
        url = rec.get("image_url")
        emb = embed_image_url(url)
        if emb is None:
            continue
        img_embs.append(emb)
        kept_meta.append({k:v for k,v in rec.items() if k != "text_for_embed"})

    if len(img_embs) == 0:
        print("No se pudo descargar ninguna imagen. Revisa conectividad/URLs.")
        return

    img_embs = np.vstack(img_embs).astype("float32")
    img_index = faiss.IndexFlatIP(img_embs.shape[1])
    img_index.add(img_embs)
    faiss.write_index(img_index, IMAGE_INDEX_PATH)
    np.save(IMAGE_META_PATH, kept_meta, allow_pickle=True)

    print("Image index creado:", img_index.ntotal, "->", IMAGE_INDEX_PATH)

if __name__ == "__main__":
    main()
